# Copyright (c) 2024 Microsoft Corporation.

"""OpenAI input/output types."""

from collections.abc import AsyncIterable, Awaitable, Callable
from typing import ClassVar, TypeAlias

from fnllm.openai.types.aliases import (
    OpenAIChatCompletionMessageParam,
)
from fnllm.types.generalized import ChatLLMOutput
from fnllm.types.metrics import LLMUsageMetrics
from pydantic import BaseModel, ConfigDict, Field

OpenAIChatMessageInput: TypeAlias = OpenAIChatCompletionMessageParam
"""OpenAI chat message input."""

# OpenAIChatHistoryEntry: TypeAlias = OpenAIChatCompletionMessageParam
"""OpenAI chat history entry."""

# OpenAIChatCompletionInput: TypeAlias = str | OpenAIChatMessageInput | None
OllamaChatCompletionInput: TypeAlias = str | OpenAIChatMessageInput | None
"""Main input type for OpenAI completions."""


OllamaChatOutput: TypeAlias = ChatLLMOutput

OllamaChatInput: TypeAlias = str


OllamaChatHistoryEntry: TypeAlias = OpenAIChatCompletionMessageParam


class OpenAIStreamingChatOutput(BaseModel, arbitrary_types_allowed=True):
    """Async iterable chat content."""

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    raw_input: OpenAIChatMessageInput | None = Field(
        default=None, description="Raw input that resulted in this output."
    )

    usage: LLMUsageMetrics | None = Field(
        default=None,
        description="Usage statistics for the completion request.\nThis will only be available after the stream is complete, if the LLM has been configured to emit usage.",
    )

    content: AsyncIterable[str | None] = Field(exclude=True)

    close: Callable[[], Awaitable[None]] = Field(
        description="Close the underlying iterator", exclude=True
    )
