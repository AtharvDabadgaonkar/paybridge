"""Anthropic client wrapper with graceful degradation.

The application must work with no API key, no network and no SDK installed.
Every failure mode returns a structured result instead of raising.
"""
from __future__ import annotations

from dataclasses import dataclass

from paybridge import config

ERROR_NO_SDK = "sdk_missing"
ERROR_NO_KEY = "no_api_key"
ERROR_RATE_LIMIT = "rate_limited"
ERROR_CONNECTION = "connection_error"
ERROR_API = "api_error"
ERROR_MALFORMED = "malformed_response"
ERROR_TIMEOUT = "timeout"


@dataclass(frozen=True)
class LLMResult:
    ok: bool
    text: str = ""
    error_type: str = ""
    error_message: str = ""
    model: str = ""

    @property
    def user_message(self) -> str:
        return {
            ERROR_NO_SDK: "The Anthropic SDK is not installed, so the LLM layer is disabled.",
            ERROR_NO_KEY: "No Anthropic API key is configured, so the LLM layer is disabled.",
            ERROR_RATE_LIMIT: "The Claude API is rate limiting requests right now.",
            ERROR_CONNECTION: "The Claude API could not be reached.",
            ERROR_API: "The Claude API returned an error.",
            ERROR_MALFORMED: "The Claude API returned a response that could not be read.",
            ERROR_TIMEOUT: "The Claude API request timed out.",
        }.get(self.error_type, self.error_message or "The LLM layer is unavailable.")


def sdk_available() -> bool:
    try:
        import anthropic  # noqa: F401

        return True
    except Exception:
        return False


def api_key_available() -> bool:
    return config.anthropic_api_key() is not None


def llm_available() -> bool:
    return sdk_available() and api_key_available()


def unavailable_reason() -> str | None:
    if not sdk_available():
        return ERROR_NO_SDK
    if not api_key_available():
        return ERROR_NO_KEY
    return None


def complete(
    system_prompt: str,
    user_message: str,
    model: str | None = None,
    max_tokens: int | None = None,
    timeout: float = 30.0,
) -> LLMResult:
    """Single-turn completion. Never raises; always returns an LLMResult."""
    reason = unavailable_reason()
    if reason:
        return LLMResult(ok=False, error_type=reason)

    import anthropic

    model_name = model or config.ANTHROPIC_MODEL
    try:
        client = anthropic.Anthropic(api_key=config.anthropic_api_key(), timeout=timeout)
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens or config.ANTHROPIC_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.RateLimitError as exc:
        return LLMResult(False, error_type=ERROR_RATE_LIMIT, error_message=str(exc), model=model_name)
    except anthropic.APITimeoutError as exc:
        return LLMResult(False, error_type=ERROR_TIMEOUT, error_message=str(exc), model=model_name)
    except anthropic.APIConnectionError as exc:
        return LLMResult(False, error_type=ERROR_CONNECTION, error_message=str(exc), model=model_name)
    except anthropic.APIStatusError as exc:
        return LLMResult(False, error_type=ERROR_API, error_message=str(exc), model=model_name)
    except Exception as exc:  # pragma: no cover - unexpected SDK failure
        return LLMResult(False, error_type=ERROR_API, error_message=f"{type(exc).__name__}: {exc}", model=model_name)

    try:
        blocks = [b.text for b in response.content if getattr(b, "type", "") == "text"]
        text = "\n".join(blocks).strip()
    except Exception:
        return LLMResult(False, error_type=ERROR_MALFORMED, model=model_name)
    if not text:
        return LLMResult(False, error_type=ERROR_MALFORMED, model=model_name)
    return LLMResult(True, text=text, model=model_name)
