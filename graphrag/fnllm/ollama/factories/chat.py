# Copyright (c) 2024 Microsoft Corporation.

"""Factory functions for creating OpenAI LLMs."""

from fnllm.caching.base import Cache
from fnllm.events.base import LLMEvents
from fnllm.limiting.base import Limiter
from fnllm.openai.config import OpenAIConfig
from fnllm.openai.llm.chat import OpenAIChatLLMImpl
from fnllm.openai.llm.chat_streaming import OpenAIStreamingChatLLMImpl
from fnllm.openai.llm.features.tools_parsing import OpenAIParseToolsLLM
from fnllm.openai.types.client import (
    OpenAIChatLLM,
    OpenAIClient,
    OpenAIStreamingChatLLM,
    OpenAITextChatLLM,
)
from fnllm.services.cache_interactor import CacheInteractor
from fnllm.services.variable_injector import VariableInjector

from graphrag.fnllm.ollama.config import OllamaConfig
from graphrag.fnllm.ollama.factories.client import create_ollama_client
from graphrag.fnllm.ollama.llm.chat import OpenAIChatLLM
from graphrag.fnllm.ollama.llm.chat_text import OllamaTextChatLLMImpl
from graphrag.fnllm.ollama.types.client import OllamaClient

from .utils import create_rate_limiter


def create_ollama_chat_llm(
    config: OllamaConfig,
    *,
    client: OllamaClient | None = None,
    cache: Cache | None = None,
    cache_interactor: CacheInteractor | None = None,
    events: LLMEvents | None = None,
) -> OpenAIChatLLM:
    """Create an OpenAI chat LLM."""
    if client is None:
        client = create_ollama_client(config)

    text_chat_llm = _create_ollama_text_chat_llm(
        client=client,
        config=config,
        cache=cache,
        cache_interactor=cache_interactor,
        events=events,
    )
    return OpenAIChatLLMImpl(
        text_chat_llm=text_chat_llm,
        streaming_chat_llm=None,  # type: ignore
    )


def _create_ollama_text_chat_llm(
    *,
    client: OllamaClient,
    config: OllamaConfig,
    cache: Cache | None,
    cache_interactor: CacheInteractor | None,
    events: LLMEvents | None,
) -> OpenAITextChatLLM:
    result = OllamaTextChatLLMImpl(
        client=client,
        model=config.model,
        model_parameters=config.chat_parameters,
        # json_handler=create_json_handler(config.json_strategy, config.max_json_retries),
    )

    return OpenAIParseToolsLLM(result)


def _create_openai_streaming_chat_llm(
    *,
    client: OpenAIClient,
    config: OpenAIConfig,
    limiter: Limiter,
    events: LLMEvents | None,
) -> OpenAIStreamingChatLLM:
    """Create an OpenAI streaming chat LLM."""
    return OpenAIStreamingChatLLMImpl(
        client,
        model=config.model,
        model_parameters=config.chat_parameters,
        events=events,
        emit_usage=config.track_stream_usage,
        variable_injector=VariableInjector(),
        rate_limiter=create_rate_limiter(limiter=limiter, config=config, events=events),
    )
