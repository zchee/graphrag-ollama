# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A text-completion based LLM."""

import logging

from typing_extensions import Unpack

from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    CompletionInput,
    CompletionOutput,
    LLMInput,
)
from graphrag.llm.utils import get_completion_llm_args

from .config import OllamaConfig
from .types import OllamaClient


log = logging.getLogger(__name__)


class OllamaCompletionLLM(BaseLLM[CompletionInput, CompletionOutput]):
    """A text-completion based LLM."""

    _client: OllamaClient
    _config: OllamaConfig

    def __init__(self, client: OllamaClient, config: OllamaConfig):
        self.client = client
        self.config = config

    async def _execute_llm(
        self,
        input: CompletionInput,
        **kwargs: Unpack[LLMInput],
    ) -> CompletionOutput | None:
        args = get_completion_llm_args(
            kwargs.get("model_parameters"), self.config
        )
        completion = await self.client.generate(prompt=input, **args)
        return completion["response"]
