# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""LLM Configuration Protocol definition."""

from typing import Protocol


class LLMConfig(Protocol):
    """LLM Configuration Protocol definition."""

    @property
    def max_retries(self) -> int | None:
        """Get the maximum number of retries."""
        ...

    @property
    def max_retry_wait(self) -> float | None:
        """Get the maximum retry wait time."""
        ...

    @property
    def sleep_on_rate_limit_recommendation(self) -> bool | None:
        """Get whether to sleep on rate limit recommendation."""
        ...

    @property
    def tokens_per_minute(self) -> int | None:
        """Get the number of tokens per minute."""
        ...

    @property
    def requests_per_minute(self) -> int | None:
        """Get the number of requests per minute."""
        ...

    def get_completion_cache_args(self) -> dict:
        """Get the cache arguments for a completion LLM."""
        ...

    @property
    def model(self) -> str | None:
        """Model property definition."""

    @property
    def deployment_name(self) -> str | None:
        """Deployment name property definition."""

    @property
    def concurrent_requests(self) -> int | None:
        """Concurrent requests property definition."""
