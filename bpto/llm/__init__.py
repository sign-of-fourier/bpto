from .base import Budget, BudgetExceeded, Completion, ModelClient, ModelConfig, Usage
from .cache import CompletionCache
from .mock import MockClient

__all__ = ["Budget", "BudgetExceeded", "Completion", "ModelClient", "ModelConfig", "Usage",
           "CompletionCache", "MockClient", "AnthropicClient", "OpenAICompatibleClient", "BedrockClient"]


def __getattr__(name):  # lazy: keep `anthropic` import off the hot path for tests
    if name == "AnthropicClient":
        from .anthropic_client import AnthropicClient
        return AnthropicClient
    if name == "BedrockClient":
        from .bedrock import BedrockClient
        return BedrockClient
    if name == "OpenAICompatibleClient":
        from .openai_compat import OpenAICompatibleClient
        return OpenAICompatibleClient
    raise AttributeError(name)
