# Copyright (c) 2024 Microsoft Corporation.

"""The chat-based LLM implementation."""

from typing import Any

from fnllm.base.base import BaseLLM
from fnllm.services.json import JsonHandler
from fnllm.types.generics import TJsonModel
from fnllm.types.io import LLMInput
from typing_extensions import Unpack

from graphrag.fnllm.ollama.types.chat.io import (
    OllamaChatHistoryEntry,
    OllamaChatInput,
    OllamaChatOutput,
)
from graphrag.fnllm.ollama.types.chat.parameters import OllamaChatParameters
from graphrag.fnllm.ollama.types.client import OllamaClient


class OllamaTextChatLLMImpl(
    BaseLLM[
        OllamaChatInput, OllamaChatOutput, OllamaChatHistoryEntry, OllamaChatParameters
    ]
):
    """A chat-based LLM."""

    def __init__(
        self,
        client: OllamaClient,
        model: str,
        *,
        model_parameters: OllamaChatParameters | None = None,
        json_handler: JsonHandler[OllamaChatOutput, OllamaChatHistoryEntry]
        | None = None,
    ):
        """Create a new OpenAIChatLLM."""
        super().__init__()

        self._client = client
        self._model = model
        self._global_model_parameters = model_parameters or {}

    def child(self, name: str) -> Any:
        """Create a child LLM."""
        return OllamaTextChatLLMImpl(
            self._client,
            self._model,
            model_parameters=self._global_model_parameters,
            json_handler=self._json_handler,
        )

    def _build_completion_parameters(
        self, local_parameters: OllamaChatParameters | None
    ) -> OllamaChatParameters:
        params: OllamaChatParameters = {
            "model": self._model,
            **self._global_model_parameters,
            **(local_parameters or {}),
        }

        return params

    async def _execute_llm(
        self,
        prompt: OllamaChatInput,
        **kwargs: Unpack[
            LLMInput[TJsonModel, OllamaChatHistoryEntry, OllamaChatParameters]
        ],
    ) -> OllamaChatOutput:
        history = kwargs.get("history") or []
        messages = [
            *history,
            {"role": "user", "content": prompt},
        ]

        args = self._build_completion_parameters(kwargs.get("model_parameters"))
        completion = await self._client.chat(messages=messages, **args)

        return OllamaChatOutput(content=completion["message"]["content"])
