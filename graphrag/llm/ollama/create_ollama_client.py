# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Create OpenAI client instance."""

import logging
from functools import cache

from ollama import AsyncClient, Client

from .ollama_configuration import OllamaConfig
from .types import OllamaClientType

log = logging.getLogger(__name__)

API_BASE_REQUIRED_FOR_AZURE = "api_base is required for Azure OpenAI client"


@cache
def create_ollama_client(
    configuration: OllamaConfig,
    sync: bool = False,
) -> OllamaClientType:
    """Create a new Ollama client instance."""

    log.info("Creating OpenAI client base_url=%s", configuration.base_url)
    if sync:
        return Client(
            host=configuration.base_url,
            timeout=configuration.timeout or 180.0,
        )
    return AsyncClient(
        host=configuration.base_url,
        # Timeout/Retry Configuration - Use Tenacity for Retries, so disable them here
        timeout=configuration.timeout or 180.0,
    )
