"""Support for OpenAI gpt-5 model family."""

from typing import List, Optional, Union

import litellm
from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET,
)
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    _extract_reasoning_content,
)
from litellm.types.llms.anthropic import AnthropicThinkingParam
from litellm.types.llms.openai import (
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionThinkingBlock,
    OpenAIChatCompletionChoices,
)
from litellm.types.utils import Choices, Message

from .gpt_transformation import OpenAIGPTConfig


class OpenAIGPT5Config(OpenAIGPTConfig):
    """Configuration for gpt-5 models including GPT-5-Codex variants.

    Handles OpenAI API quirks for the gpt-5 series like:

    - Mapping ``max_tokens`` -> ``max_completion_tokens``.
    - Dropping unsupported ``temperature`` values when requested.
    - Support for GPT-5-Codex models optimized for code generation.
    - Reverse mapping ``thinking`` -> ``reasoning_effort`` for Claude compatibility.
    """

    @classmethod
    def is_model_gpt_5_model(cls, model: str) -> bool:
        return "gpt-5" in model

    @classmethod
    def is_model_gpt_5_codex_model(cls, model: str) -> bool:
        """Check if the model is specifically a GPT-5 Codex variant."""
        return "gpt-5-codex" in model

    @staticmethod
    def _map_thinking_to_reasoning_effort(
        thinking: Union[AnthropicThinkingParam, dict],
    ) -> Optional[str]:
        """
        Reverse map Claude's thinking parameter to OpenAI's reasoning_effort.
        
        This is the inverse of AnthropicConfig._map_reasoning_effort().
        
        Args:
            thinking: Claude's thinking parameter with type and budget_tokens
            
        Returns:
            reasoning_effort string ("minimal", "low", "medium", or "high")
        """
        if thinking is None:
            return None
            
        # Handle both dict and AnthropicThinkingParam types
        if isinstance(thinking, dict):
            thinking_type = thinking.get("type")
            budget_tokens = thinking.get("budget_tokens")
        else:
            thinking_type = getattr(thinking, "type", None)
            budget_tokens = getattr(thinking, "budget_tokens", None)
            
        # If thinking is disabled or no budget specified, return None
        if thinking_type != "enabled" or budget_tokens is None:
            return None
            
        # Map budget_tokens back to reasoning_effort levels
        # Using the same thresholds as AnthropicConfig._map_reasoning_effort
        if budget_tokens <= DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET:
            return "minimal"
        elif budget_tokens <= DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET:
            return "low"
        elif budget_tokens <= DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET:
            return "medium"
        else:  # budget_tokens >= DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET
            return "high"

    @staticmethod
    def _convert_reasoning_to_thinking_blocks(
        reasoning_content: Optional[str],
    ) -> Optional[
        List[Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]]
    ]:
        """
        Convert OpenAI's reasoning_content to Claude's thinking_blocks format.
        
        This enables bidirectional conversion: when a request comes in with Claude's
        thinking parameter, the response includes thinking_blocks in Claude format.
        
        Args:
            reasoning_content: OpenAI's reasoning content string
            
        Returns:
            thinking_blocks in Claude format, or None if no reasoning content
        """
        if not reasoning_content:
            return None
            
        # OpenAI doesn't provide cryptographic signature, so we omit that field
        thinking_block: ChatCompletionThinkingBlock = {
            "type": "thinking",
            "thinking": reasoning_content,
        }
        return [thinking_block]

    def _transform_choices(
        self,
        choices: List[OpenAIChatCompletionChoices],
        json_mode: Optional[bool] = None,
        optional_params: Optional[dict] = None,
    ) -> List[Choices]:
        """
        Override to convert reasoning_content to thinking_blocks for Claude compatibility.
        
        When routing Claude → GPT-5, we want the response to maintain Claude's format
        with thinking_blocks so the client (like Claude Code) can display reasoning.
        """
        # Get base transformation
        transformed_choices = super()._transform_choices(
            choices, json_mode, optional_params
        )
        
        # Add thinking_blocks conversion for Claude compatibility
        for choice in transformed_choices:
            if choice.message.reasoning_content:
                # Convert OpenAI's reasoning_content to Claude's thinking_blocks
                choice.message.thinking_blocks = self._convert_reasoning_to_thinking_blocks(
                    choice.message.reasoning_content
                )
        
        return transformed_choices

    def get_supported_openai_params(self, model: str) -> list:
        from litellm.utils import supports_tool_choice

        base_gpt_series_params = super().get_supported_openai_params(model=model)
        gpt_5_only_params = ["reasoning_effort", "verbosity", "thinking"]
        base_gpt_series_params.extend(gpt_5_only_params)
        if not supports_tool_choice(model=model):
            base_gpt_series_params.remove("tool_choice")

        non_supported_params = [
            "logprobs",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "top_logprobs",
            "stop",
        ]

        return [
            param
            for param in base_gpt_series_params
            if param not in non_supported_params
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        ################################################################
        # max_tokens is not supported for gpt-5 models on OpenAI API
        # Relevant issue: https://github.com/BerriAI/litellm/issues/13381
        ################################################################
        if "max_tokens" in non_default_params:
            optional_params["max_completion_tokens"] = non_default_params.pop(
                "max_tokens"
            )

        ################################################################
        # Reverse map thinking -> reasoning_effort for Claude compatibility
        # When routing Claude calls to GPT-5, convert the thinking parameter
        ################################################################
        if "thinking" in non_default_params:
            thinking_value = non_default_params.pop("thinking")
            reasoning_effort = self._map_thinking_to_reasoning_effort(thinking_value)
            if reasoning_effort is not None:
                non_default_params["reasoning_effort"] = reasoning_effort

        if "temperature" in non_default_params:
            temperature_value: Optional[float] = non_default_params.pop("temperature")
            if temperature_value is not None:
                if temperature_value == 1:
                    optional_params["temperature"] = temperature_value
                elif litellm.drop_params or drop_params:
                    pass
                else:
                    raise litellm.utils.UnsupportedParamsError(
                        message=(
                            "gpt-5 models (including gpt-5-codex) don't support temperature={}. Only temperature=1 is supported. To drop unsupported params set `litellm.drop_params = True`"
                        ).format(temperature_value),
                        status_code=400,
                    )
        return super()._map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )
