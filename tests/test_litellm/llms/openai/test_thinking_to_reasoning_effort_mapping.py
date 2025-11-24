"""
Test reverse mapping from Claude's 'thinking' parameter to OpenAI's 'reasoning_effort'

This ensures that when routing Claude calls to OpenAI GPT-5/O-series models,
the thinking parameter is correctly transformed to reasoning_effort.
"""

import pytest

from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET,
)
from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config
from litellm.llms.openai.chat.o_series_transformation import OpenAIOSeriesConfig


class TestThinkingToReasoningEffortMapping:
    """Test that thinking parameter is correctly mapped to reasoning_effort"""

    def test_gpt5_map_thinking_minimal_to_reasoning_effort(self):
        """Test mapping minimal thinking budget to reasoning_effort='minimal'"""
        config = OpenAIGPT5Config()
        thinking = {
            "type": "enabled",
            "budget_tokens": DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET,
        }
        result = config._map_thinking_to_reasoning_effort(thinking)
        assert result == "minimal"

    def test_gpt5_map_thinking_low_to_reasoning_effort(self):
        """Test mapping low thinking budget to reasoning_effort='low'"""
        config = OpenAIGPT5Config()
        thinking = {
            "type": "enabled",
            "budget_tokens": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
        }
        result = config._map_thinking_to_reasoning_effort(thinking)
        assert result == "low"

    def test_gpt5_map_thinking_medium_to_reasoning_effort(self):
        """Test mapping medium thinking budget to reasoning_effort='medium'"""
        config = OpenAIGPT5Config()
        thinking = {
            "type": "enabled",
            "budget_tokens": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
        }
        result = config._map_thinking_to_reasoning_effort(thinking)
        assert result == "medium"

    def test_gpt5_map_thinking_high_to_reasoning_effort(self):
        """Test mapping high thinking budget to reasoning_effort='high'"""
        config = OpenAIGPT5Config()
        thinking = {
            "type": "enabled",
            "budget_tokens": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
        }
        result = config._map_thinking_to_reasoning_effort(thinking)
        assert result == "high"

    def test_gpt5_map_thinking_disabled_returns_none(self):
        """Test that disabled thinking returns None"""
        config = OpenAIGPT5Config()
        thinking = {"type": "disabled", "budget_tokens": 1024}
        result = config._map_thinking_to_reasoning_effort(thinking)
        assert result is None

    def test_gpt5_map_thinking_none_returns_none(self):
        """Test that None thinking returns None"""
        config = OpenAIGPT5Config()
        result = config._map_thinking_to_reasoning_effort(None)
        assert result is None

    def test_gpt5_map_thinking_no_budget_returns_none(self):
        """Test that thinking without budget_tokens returns None"""
        config = OpenAIGPT5Config()
        thinking = {"type": "enabled"}
        result = config._map_thinking_to_reasoning_effort(thinking)
        assert result is None

    def test_gpt5_map_openai_params_converts_thinking(self):
        """Test that map_openai_params converts thinking to reasoning_effort"""
        config = OpenAIGPT5Config()
        non_default_params = {
            "thinking": {
                "type": "enabled",
                "budget_tokens": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
            }
        }
        optional_params = {}
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gpt-5.1-2025-11-13",
            drop_params=False,
        )
        
        # thinking should be removed from non_default_params
        assert "thinking" not in non_default_params
        # reasoning_effort should be added
        assert "reasoning_effort" in result
        assert result["reasoning_effort"] == "medium"

    def test_oseries_map_thinking_to_reasoning_effort(self):
        """Test O-series models also support thinking -> reasoning_effort mapping"""
        config = OpenAIOSeriesConfig()
        thinking = {
            "type": "enabled",
            "budget_tokens": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
        }
        result = config._map_thinking_to_reasoning_effort(thinking)
        assert result == "high"

    def test_oseries_map_openai_params_converts_thinking(self):
        """Test that O-series map_openai_params converts thinking to reasoning_effort"""
        config = OpenAIOSeriesConfig()
        non_default_params = {
            "thinking": {
                "type": "enabled",
                "budget_tokens": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
            }
        }
        optional_params = {}
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="o3-mini",
            drop_params=False,
        )
        
        # thinking should be removed
        assert "thinking" not in non_default_params
        # reasoning_effort should be in result
        assert "reasoning_effort" in result
        assert result["reasoning_effort"] == "low"

    def test_gpt5_boundary_values(self):
        """Test boundary values for thinking budget mapping"""
        config = OpenAIGPT5Config()
        
        # Just below minimal threshold
        thinking = {"type": "enabled", "budget_tokens": 100}
        assert config._map_thinking_to_reasoning_effort(thinking) == "minimal"
        
        # Between minimal and low
        thinking = {"type": "enabled", "budget_tokens": 512}
        assert config._map_thinking_to_reasoning_effort(thinking) == "low"
        
        # Between low and medium
        thinking = {"type": "enabled", "budget_tokens": 1500}
        assert config._map_thinking_to_reasoning_effort(thinking) == "medium"
        
        # Between medium and high
        thinking = {"type": "enabled", "budget_tokens": 3000}
        assert config._map_thinking_to_reasoning_effort(thinking) == "medium"
        
        # Above high threshold
        thinking = {"type": "enabled", "budget_tokens": 8192}
        assert config._map_thinking_to_reasoning_effort(thinking) == "high"

