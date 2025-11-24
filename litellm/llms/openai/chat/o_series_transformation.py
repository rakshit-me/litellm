"""
Support for o1/o3 model family 

https://platform.openai.com/docs/guides/reasoning

Translations handled by LiteLLM:
- modalities: image => drop param (if user opts in to dropping param)  
- role: system ==> translate to role 'user' 
- streaming => faked by LiteLLM 
- Tools, response_format =>  drop param (if user opts in to dropping param) 
- Logprobs => drop param (if user opts in to dropping param) 
"""

from typing import Any, Coroutine, List, Literal, Optional, Union, cast, overload

import httpx

import litellm
from litellm import verbose_logger
from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET,
)
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.types.llms.anthropic import AnthropicThinkingParam
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionThinkingBlock,
    ChatCompletionUserMessage,
    OpenAIChatCompletionChoices,
)
from litellm.types.utils import Choices, ModelResponse, StreamingChoices
from litellm.utils import (
    supports_function_calling,
    supports_parallel_function_calling,
    supports_response_schema,
    supports_system_messages,
)

from .gpt_transformation import OpenAIGPTConfig


class OpenAIOSeriesConfig(OpenAIGPTConfig):
    """
    Reference: https://platform.openai.com/docs/guides/reasoning
    
    Handles O-series model quirks including reverse mapping thinking -> reasoning_effort.
    """

    @classmethod
    def get_config(cls):
        return super().get_config()

    def translate_developer_role_to_system_role(
        self, messages: List[AllMessageValues]
    ) -> List[AllMessageValues]:
        """
        O-series models support `developer` role.
        """
        return messages

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
        
        When routing Claude → O-series, we want the response to maintain Claude's format
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
        """
        Get the supported OpenAI params for the given model

        """

        all_openai_params = super().get_supported_openai_params(model=model)
        non_supported_params = [
            "logprobs",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "top_logprobs",
        ]

        o_series_only_param = ["reasoning_effort", "thinking"]

        all_openai_params.extend(o_series_only_param)

        try:
            model, custom_llm_provider, api_base, api_key = get_llm_provider(
                model=model
            )
        except Exception:
            verbose_logger.debug(
                f"Unable to infer model provider for model={model}, defaulting to openai for o1 supported param check"
            )
            custom_llm_provider = "openai"

        _supports_function_calling = supports_function_calling(
            model, custom_llm_provider
        )
        _supports_response_schema = supports_response_schema(model, custom_llm_provider)
        _supports_parallel_tool_calls = supports_parallel_function_calling(
            model, custom_llm_provider
        )

        if not _supports_function_calling:
            non_supported_params.append("tools")
            non_supported_params.append("tool_choice")
            non_supported_params.append("function_call")
            non_supported_params.append("functions")

        if not _supports_parallel_tool_calls:
            non_supported_params.append("parallel_tool_calls")

        if not _supports_response_schema:
            non_supported_params.append("response_format")

        return [
            param for param in all_openai_params if param not in non_supported_params
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ):
        if "max_tokens" in non_default_params:
            optional_params["max_completion_tokens"] = non_default_params.pop(
                "max_tokens"
            )
            
        # Reverse map thinking -> reasoning_effort for Claude compatibility
        # When routing Claude calls to O-series models, convert the thinking parameter
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
                else:
                    ## UNSUPPORTED TOOL CHOICE VALUE
                    if litellm.drop_params is True or drop_params is True:
                        pass
                    else:
                        raise litellm.utils.UnsupportedParamsError(
                            message="O-series models don't support temperature={}. Only temperature=1 is supported. To drop unsupported openai params from the call, set `litellm.drop_params = True`".format(
                                temperature_value
                            ),
                            status_code=400,
                        )

        return super()._map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: Any,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Override to add thinking_blocks conversion after response parsing.
        
        This ensures reasoning_content from O-series is converted to thinking_blocks
        for Claude API compatibility.
        """
        # Get base transformation
        model_response = super().transform_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )
        
        # Add thinking_blocks conversion for each choice (non-streaming only)
        for choice in model_response.choices:
            # Type check: ensure this is a non-streaming Choices object
            if isinstance(choice, Choices) and choice.message:
                if hasattr(choice.message, 'reasoning_content') and choice.message.reasoning_content:
                    # Convert OpenAI's reasoning_content to Claude's thinking_blocks
                    choice.message.thinking_blocks = self._convert_reasoning_to_thinking_blocks(
                        choice.message.reasoning_content
                    )
        
        return model_response

    def is_model_o_series_model(self, model: str) -> bool:
        model = model.split("/")[-1]  # could be "openai/o3" or "o3"
        return model in litellm.open_ai_chat_completion_models and any(
            model.startswith(pfx) for pfx in ("o1", "o3", "o4")
        )

    @overload
    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[Any, Any, List[AllMessageValues]]:
        ...

    @overload
    def _transform_messages(
        self,
        messages: List[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> List[AllMessageValues]:
        ...

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: bool = False
    ) -> Union[List[AllMessageValues], Coroutine[Any, Any, List[AllMessageValues]]]:
        """
        Handles limitations of O-1 model family.
        - modalities: image => drop param (if user opts in to dropping param)
        - role: system ==> translate to role 'user'
        """
        _supports_system_messages = supports_system_messages(model, "openai")
        for i, message in enumerate(messages):
            if message["role"] == "system" and not _supports_system_messages:
                new_message = ChatCompletionUserMessage(
                    content=message["content"], role="user"
                )
                messages[i] = new_message  # Replace the old message with the new one

        if is_async:
            return super()._transform_messages(
                messages, model, is_async=cast(Literal[True], True)
            )
        else:
            return super()._transform_messages(
                messages, model, is_async=cast(Literal[False], False)
            )
