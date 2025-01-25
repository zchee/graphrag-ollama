# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Create OpenAI client instance."""

import logging

from ollama import AsyncClient, Client

from graphrag.llm.ollama.config import OllamaConfig
from graphrag.llm.ollama.types import OllamaClient

log = logging.getLogger(__name__)


def create_ollama_client(config: OllamaConfig, sync: bool = False,) -> OllamaClient:
    """Create a new Ollama client instance."""

    log.info("Creating OpenAI client base_url=%s", config.base_url)
    if sync:
        return Client(
            host=config.base_url,
            timeout=config.timeout or 180.0,
        )
    return AsyncClient(
        host=config.base_url,
        # Timeout/Retry Configuration - Use Tenacity for Retries, so disable them here
        timeout=config.timeout or 180.0,
    )
