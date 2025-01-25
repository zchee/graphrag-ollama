# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Factory functions for creating Ollama LLMs."""

import asyncio

from graphrag.llm.base import CachingLLM, RateLimitingLLM
from graphrag.llm.limiting import LLMLimiter
from graphrag.llm.types import (
    LLM,
    CompletionLLM,
    EmbeddingLLM,
    ErrorHandlerFn,
    LLMCache,
    LLMInvocationFn,
    OnCacheActionFn,
)
from graphrag.llm.utils import (
    RATE_LIMIT_ERRORS,
    RETRYABLE_ERRORS,
    get_sleep_time_from_error,
    get_token_counter,
)
from graphrag.llm.openai.openai_history_tracking_llm import OpenAIHistoryTrackingLLM
from graphrag.llm.openai.openai_token_replacing_llm import OpenAITokenReplacingLLM

from .json_parsing_llm import JsonParsingLLM
from .ollama_chat_llm import OllamaChatLLM
from .ollama_completion_llm import OllamaCompletionLLM
from .config import OllamaConfig
from .ollama_embeddings_llm import OllamaEmbeddingsLLM
from .types import OllamaClient

# from fnllm.caching.base import Cache
# from fnllm.events.base import LLMEvents
# from fnllm.limiting.base import Limiter
# from fnllm.types import (
#     LLM,
#     # CompletionLLM,
#     # EmbeddingLLM,
#     # ErrorHandlerFn,
#     # LLMCache,
#     # LLMInvocationFn,
#     # OnCacheActionFn,
# )
#
# from graphrag.llm.ollama.config import OllamaConfig
# from graphrag.llm.ollama.types import (
#     OllamaChatLLM,
#     OllamaClient,
#     OllamaStreamingChatLLM,
#     OllamaTextChatLLM,
# )
# from fnllm.services.cache_interactor import CacheInteractor
#
# from graphrag.llm.ollama.create_ollama_client import create_ollama_client
# from fnllm.openai.factories.utils import create_limiter

# from fnllm.openai.factories import create_openai_chat_llm
#
# from graphrag.llm.base import CachingLLM, RateLimitingLLM
# from fnllm.limiting import Limiter
# from graphrag.llm.limiting import LLMLimiter
# from graphrag.llm.types import (
#     LLM,
#     CompletionLLM,
#     EmbeddingLLM,
#     ErrorHandlerFn,
#     LLMCache,
#     LLMInvocationFn,
#     OnCacheActionFn,
# )
# from graphrag.llm.utils import (
#     RATE_LIMIT_ERRORS,
#     RETRYABLE_ERRORS,
#     get_sleep_time_from_error,
#     get_token_counter,
# )
# from graphrag.llm.openai.openai_history_tracking_llm import OpenAIHistoryTrackingLLM
# from graphrag.llm.openai.openai_token_replacing_llm import OpenAITokenReplacingLLM
#
# from .json_parsing_llm import JsonParsingLLM
# from .ollama_chat_llm import OllamaChatLLM
# from .ollama_completion_llm import OllamaCompletionLLM
# from .ollama_configuration import OllamaConfiguration
# from .ollama_embeddings_llm import OllamaEmbeddingsLLM
# from .types import OllamaClientType

# def create__ollama_chat_llm(
#     config: OllamaConfig,
#     *,
#     client: OllamaClient | None = None,
#     cache: Cache | None = None,
#     cache_interactor: CacheInteractor | None = None,
#     events: LLMEvents | None = None,
# ) -> OllamaChatLLM:
#     """Create an OpenAI chat LLM."""
#     if client is None:
#         client = create_ollama_client(config)
#
#     limiter = create_limiter(config)
#
#     text_chat_llm = _create_openai_text_chat_llm(
#         client=client,
#         config=config,
#         cache=cache,
#         cache_interactor=cache_interactor,
#         events=events,
#         limiter=limiter,
#     )
#     streaming_chat_llm = _create_openai_streaming_chat_llm(
#         client=client,
#         config=config,
#         events=events,
#         limiter=limiter,
#     )
#     return OpenAIChatLLMImpl(
#         text_chat_llm=text_chat_llm,
#         streaming_chat_llm=streaming_chat_llm,
#     )
#
#
# def _create_openai_text_chat_llm(
#     *,
#     client: OllamaClient,
#     config: OllamaConfig,
#     limiter: Limiter,
#     cache: Cache | None,
#     cache_interactor: CacheInteractor | None,
#     events: LLMEvents | None,
# ) -> OllamaTextChatLLM:
#     operation = "chat"
#     result = OpenAITextChatLLMImpl(
#         client,
#         model=config.model,
#         model_parameters=config.chat_parameters,
#         cache=cache_interactor or CacheInteractor(events, cache),
#         events=events,
#         json_handler=create_json_handler(config.json_strategy, config.max_json_retries),
#         usage_extractor=OpenAIUsageExtractor(),
#         history_extractor=OpenAIHistoryExtractor(),
#         variable_injector=VariableInjector(),
#         retryer=create_retryer(config=config, operation=operation, events=events),
#         rate_limiter=create_rate_limiter(config=config, limiter=limiter, events=events),
#     )
#
#     return OpenAIParseToolsLLM(result)
#
#
# def _create_openai_streaming_chat_llm(
#     *,
#     client: OpenAIClient,
#     config: OpenAIConfig,
#     limiter: Limiter,
#     events: LLMEvents | None,
# ) -> OpenAIStreamingChatLLM:
#     """Create an OpenAI streaming chat LLM."""
#     return OpenAIStreamingChatLLMImpl(
#         client,
#         model=config.model,
#         model_parameters=config.chat_parameters,
#         events=events,
#         emit_usage=config.track_stream_usage,
#         variable_injector=VariableInjector(),
#         rate_limiter=create_rate_limiter(limiter=limiter, config=config, events=events),
#     )


def create_ollama_chat_llm(
    client: OllamaClient,
    config: OllamaConfig,
    cache: LLMCache | None = None,
    limiter: LLMLimiter | None = None,
    semaphore: asyncio.Semaphore | None = None,
    on_invoke: LLMInvocationFn | None = None,
    on_error: ErrorHandlerFn | None = None,
    on_cache_hit: OnCacheActionFn | None = None,
    on_cache_miss: OnCacheActionFn | None = None,
) -> CompletionLLM:
    """Create an OpenAI chat LLM."""
    operation = "chat"
    result = OllamaChatLLM(client, config)
    result.on_error(on_error)
    if limiter is not None or semaphore is not None:
        result = _rate_limited(result, config, operation, limiter, semaphore, on_invoke)
    if cache is not None:
        result = _cached(result, config, operation, cache, on_cache_hit, on_cache_miss)
    result = OpenAIHistoryTrackingLLM(result)
    result = OpenAITokenReplacingLLM(result)
    return JsonParsingLLM(result)


def create_ollama_completion_llm(
    client: OllamaClient,
    config: OllamaConfig,
    cache: LLMCache | None = None,
    limiter: LLMLimiter | None = None,
    semaphore: asyncio.Semaphore | None = None,
    on_invoke: LLMInvocationFn | None = None,
    on_error: ErrorHandlerFn | None = None,
    on_cache_hit: OnCacheActionFn | None = None,
    on_cache_miss: OnCacheActionFn | None = None,
) -> CompletionLLM:
    """Create an OpenAI completion LLM."""
    operation = "completion"
    result = OllamaCompletionLLM(client, config)
    result.on_error(on_error)
    if limiter is not None or semaphore is not None:
        result = _rate_limited(result, config, operation, limiter, semaphore, on_invoke)
    if cache is not None:
        result = _cached(result, config, operation, cache, on_cache_hit, on_cache_miss)
    return OpenAITokenReplacingLLM(result)


def create_ollama_embedding_llm(
    client: OllamaClient,
    config: OllamaConfig,
    cache: LLMCache | None = None,
    limiter: LLMLimiter | None = None,
    semaphore: asyncio.Semaphore | None = None,
    on_invoke: LLMInvocationFn | None = None,
    on_error: ErrorHandlerFn | None = None,
    on_cache_hit: OnCacheActionFn | None = None,
    on_cache_miss: OnCacheActionFn | None = None,
) -> EmbeddingLLM:
    """Create an OpenAI embeddings LLM."""
    operation = "embedding"
    result = OllamaEmbeddingsLLM(client, config)
    result.on_error(on_error)
    if limiter is not None or semaphore is not None:
        result = _rate_limited(result, config, operation, limiter, semaphore, on_invoke)
    if cache is not None:
        result = _cached(result, config, operation, cache, on_cache_hit, on_cache_miss)
    return result


def _rate_limited(
    delegate: LLM,
    config: OllamaConfig,
    operation: str,
    limiter: Limiter | None,
    semaphore: asyncio.Semaphore | None,
    on_invoke: LLMInvocationFn | None,
):
    result = RateLimitingLLM(
        delegate,
        config,
        operation,
        RETRYABLE_ERRORS,
        RATE_LIMIT_ERRORS,
        limiter,
        semaphore,
        get_token_counter(config),
        get_sleep_time_from_error,
    )
    result.on_invoke(on_invoke)
    return result


def _cached(
    delegate: LLM,
    config: OllamaConfiguration,
    operation: str,
    cache: LLMCache,
    on_cache_hit: OnCacheActionFn | None,
    on_cache_miss: OnCacheActionFn | None,
):
    cache_args = config.get_completion_cache_args()
    result = CachingLLM(delegate, cache_args, operation, cache)
    result.on_cache_hit(on_cache_hit)
    result.on_cache_miss(on_cache_miss)
    return result

