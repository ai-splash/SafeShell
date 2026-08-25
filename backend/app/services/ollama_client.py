"""
Ollama client.

Thin wrapper around the local Ollama HTTP API (https://ollama.com), used to
run an open-weight model such as Qwen2.5 or Llama3.1 fully locally - no
data leaves the machine.

Kept deliberately dumb: this module knows nothing about intents, prompts,
or conversation history. It only knows how to send a list of chat messages
to Ollama and get a raw text reply back, raising a clear, catchable error
if Ollama isn't installed/running/reachable.
"""

import httpx

from app.config import get_settings
from app.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class OllamaUnavailableError(Exception):
    """Raised when Ollama cannot be reached or returns an error."""


async def chat(messages: list[dict], model: str | None = None) -> str:
    """Send a chat-style request to Ollama and return the raw text reply.

    Args:
        messages: list of {"role": "system"|"user"|"assistant", "content": str}
        model: overrides the configured default model if provided.

    Raises:
        OllamaUnavailableError: if Ollama is unreachable, times out, or
            returns a non-2xx / malformed response.
    """
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model or settings.ollama_model,
        "messages": messages,
        "stream": False,
        # "think" must be a top-level field (not under "options") - this is
        # what actually turns off the long <think>...</think> preamble that
        # reasoning-capable models (Qwen3, DeepSeek-R1, etc.) emit before
        # every reply. It's a no-op on non-reasoning models, so it's always
        # safe to send.
        "think": settings.ollama_think,
        # Constrain decoding to strictly valid JSON. This is Ollama's native
        # structured-output mode (grammar-constrained generation) - it makes
        # the model return a syntactically valid JSON value on essentially
        # every call, instead of relying solely on prompt instructions that
        # can be ignored or wrapped in markdown/<think> preambles. Downstream
        # parsing in ai_assistant.py still defends against edge cases (e.g.
        # models that don't honor this option), but this removes the vast
        # majority of malformed replies at the source.
        "format": "json",
        # Keep the model resident in memory between requests. Without this,
        # Ollama can unload the model after each call and pay the (multi-
        # second, CPU-bound) load cost again on the very next question.
        "keep_alive": settings.ollama_keep_alive,
        "options": {
            "temperature": settings.ollama_temperature,
            # Bounds how much the model can generate. This is the single
            # biggest lever for CPU latency: an unbounded reply can wander
            # on for hundreds of tokens even after the useful JSON is done.
            "num_predict": settings.ollama_num_predict,
            # Smaller context window -> less prompt to process per token on
            # CPU. 2048 is comfortable for the small, intent-scoped prompts
            # this app sends (see prompts.py / context_builder.py).
            "num_ctx": settings.ollama_num_ctx,
        },
    }

    timeout = httpx.Timeout(
        connect=settings.ollama_connect_timeout_seconds,
        read=settings.ollama_timeout_seconds,
        write=settings.ollama_connect_timeout_seconds,
        pool=settings.ollama_connect_timeout_seconds,
    )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
    except httpx.ConnectError as exc:
        raise OllamaUnavailableError(
            f"Could not connect to Ollama at {settings.ollama_base_url}. "
            "Is Ollama installed and running? (`ollama serve`)"
        ) from exc
    except httpx.TimeoutException as exc:
        raise OllamaUnavailableError(
            f"Ollama did not respond within {settings.ollama_timeout_seconds}s."
        ) from exc
    except httpx.HTTPError as exc:
        raise OllamaUnavailableError(f"Unexpected error contacting Ollama: {exc}") from exc

    if response.status_code == 404:
        raise OllamaUnavailableError(
            f"Model '{payload['model']}' not found in Ollama. "
            f"Pull it first with: ollama pull {payload['model']}"
        )
    if response.status_code != 200:
        raise OllamaUnavailableError(
            f"Ollama returned HTTP {response.status_code}: {response.text[:300]}"
        )

    try:
        data = response.json()
        content = data["message"]["content"]
    except (ValueError, KeyError, TypeError) as exc:
        raise OllamaUnavailableError(f"Malformed response from Ollama: {exc}") from exc

    if not content or not content.strip():
        raise OllamaUnavailableError("Ollama returned an empty response.")

    return content


async def check_health() -> dict:
    """Check whether Ollama is reachable and whether the configured model is pulled."""
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(url)
        response.raise_for_status()
        models = [m.get("name") for m in response.json().get("models", [])]
        model_available = any(
            m == settings.ollama_model or (m or "").startswith(settings.ollama_model.split(":")[0])
            for m in models
        )
        return {
            "reachable": True,
            "configured_model": settings.ollama_model,
            "model_available": model_available,
            "installed_models": models,
        }
    except httpx.HTTPError as exc:
        return {
            "reachable": False,
            "configured_model": settings.ollama_model,
            "model_available": False,
            "installed_models": [],
            "error": str(exc),
        }