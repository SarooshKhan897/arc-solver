"""Shared API client for LLM calls."""

import asyncio
import time
from typing import Any

from openai import AsyncOpenAI

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    REQUEST_TIMEOUT,
)

# =============================================================================
# Shared Client
# =============================================================================

_shared_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    """Get shared AsyncOpenAI client with connection pooling."""
    global _shared_client
    if _shared_client is None:
        _shared_client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            timeout=REQUEST_TIMEOUT,
        )
    return _shared_client


# =============================================================================
# Retry Logic
# =============================================================================

RETRYABLE_ERRORS = (
    "rate_limit",
    "timeout",
    "overloaded",
    "502",
    "503",
    "504",
    "connection",
)


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable."""
    error_str = str(error).lower()
    return any(keyword in error_str for keyword in RETRYABLE_ERRORS)


# =============================================================================
# LLM Call Function
# =============================================================================

async def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    extra_body: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.7,
    max_retries: int = 5,
    base_delay: float = 5.0,
) -> tuple[str, float]:
    """
    Call an LLM with retry logic.

    Args:
        model: Model identifier (e.g., "openai/gpt-5.2")
        system_prompt: System message
        user_prompt: User message
        extra_body: Additional body parameters (e.g., reasoning effort)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        max_retries: Number of retry attempts
        base_delay: Base delay for exponential backoff

    Returns:
        (response_content, elapsed_seconds)
    """
    client = get_client()

    for attempt in range(max_retries):
        start_time = time.time()
        try:
            # Build request kwargs
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
            }

            if extra_body:
                kwargs["extra_body"] = extra_body
            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            response = await client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            elapsed = time.time() - start_time

            return content, elapsed

        except Exception as e:
            elapsed = time.time() - start_time
            if is_retryable_error(e) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
                continue
            raise

    raise RuntimeError(f"Max retries ({max_retries}) exceeded")


async def call_llm_with_history(
    model: str,
    messages: list[dict[str, str]],
    extra_body: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.7,
    max_retries: int = 5,
    base_delay: float = 5.0,
) -> tuple[str, float]:
    """
    Call an LLM with a full message history.

    Args:
        model: Model identifier
        messages: List of message dicts with 'role' and 'content'
        extra_body: Additional body parameters
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        max_retries: Number of retry attempts
        base_delay: Base delay for exponential backoff

    Returns:
        (response_content, elapsed_seconds)
    """
    client = get_client()

    for attempt in range(max_retries):
        start_time = time.time()
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }

            if extra_body:
                kwargs["extra_body"] = extra_body
            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            response = await client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            elapsed = time.time() - start_time

            return content, elapsed

        except Exception as e:
            elapsed = time.time() - start_time
            if is_retryable_error(e) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
                continue
            raise

    raise RuntimeError(f"Max retries ({max_retries}) exceeded")

