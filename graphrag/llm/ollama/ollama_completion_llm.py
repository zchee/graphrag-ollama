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
<<<<<<<< HEAD:graphrag/llm/ollama/ollama_completion_llm.py

from .ollama_configuration import OllamaConfiguration
from .types import OllamaClientType
|||||||| parent of b2736a9 (ollama support.):graphrag/llm/openai/openai_completion_llm.py
========
>>>>>>>> b2736a9 (ollama support.):graphrag/llm/openai/openai_completion_llm.py

<<<<<<<< HEAD:graphrag/llm/ollama/ollama_completion_llm.py
|||||||| parent of b2736a9 (ollama support.):graphrag/llm/openai/openai_completion_llm.py
from .openai_configuration import OpenAIConfiguration
from .types import OpenAIClientTypes
from .utils import get_completion_llm_args
========
from .openai_configuration import OpenAIConfiguration
from .types import OpenAIClientTypes
>>>>>>>> b2736a9 (ollama support.):graphrag/llm/openai/openai_completion_llm.py

log = logging.getLogger(__name__)


class OllamaCompletionLLM(BaseLLM[CompletionInput, CompletionOutput]):
    """A text-completion based LLM."""

    _client: OllamaClientType
    _configuration: OllamaConfiguration

    def __init__(self, client: OllamaClientType, configuration: OllamaConfiguration):
        self.client = client
        self.configuration = configuration

    async def _execute_llm(
        self,
        input: CompletionInput,
        **kwargs: Unpack[LLMInput],
    ) -> CompletionOutput | None:
        args = get_completion_llm_args(
            kwargs.get("model_parameters"), self.configuration
        )
        completion = await self.client.generate(prompt=input, **args)
        return completion["response"]
