# Copyright (c) 2024 Microsoft Corporation.

"""OpenAI chat parameters types."""

from fnllm.openai.types.aliases import (
    OpenAIChatModel,
)
from typing_extensions import NotRequired, TypedDict


#
# Note: streaming options have been removed from this class to avoid downstream tying issues.
# OpenAI streaming should be handled with a StreamingLLM, not additional client-side parameters.
#
class OllamaChatParameters(TypedDict):
    """OpenAI allowed chat parameters."""

    model: NotRequired[str | OpenAIChatModel]
