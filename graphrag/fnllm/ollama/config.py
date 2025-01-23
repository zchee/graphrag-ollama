# Copyright (c) 2024 Microsoft Corporation.


"""OpenAI Configuration class definition."""

from pydantic import Field

from graphrag.fnllm.ollama.types.chat.parameters import OllamaChatParameters

"""Ollama Configuration class definition."""


from fnllm.config import Config


class PublicOllamaConfig(Config, frozen=True, extra="allow", protected_namespaces=()):
    """Common configuration parameters between Azure OpenAI and Public OpenAI."""

    base_url: str | None = Field(default=None, description="The OpenAI API base URL.")

    api_key: str | None = Field(default=None, description="The OpenAI API key.")

    model: str = Field(default="", description="The OpenAI model to use.")

    encoding: str = Field(default="cl100k_base", description="The encoding model.")

    timeout: float | None = Field(default=None, description="The request timeout.")

    chat_parameters: OllamaChatParameters = Field(
        default_factory=lambda: OllamaChatParameters(),
        description="Global chat parameters to be used across calls.",
    )


OllamaConfig = PublicOllamaConfig
