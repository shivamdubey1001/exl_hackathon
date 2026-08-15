"""
=============================================================================
LLM ADAPTER
=============================================================================

One entry point for every model call in the app. Works with either provider:

    ANTHROPIC_API_KEY=sk-ant-...          -> Anthropic SDK, direct
    OPENROUTER_API_KEY=sk-or-...          -> OpenAI SDK against OpenRouter

WHY TWO SDKs RATHER THAN ONE BASE-URL SWAP
------------------------------------------
OpenRouter's API is OpenAI-compatible, not Anthropic-compatible. Pointing the
Anthropic SDK at openrouter.ai is an undocumented trick that may or may not
work on any given day. The supported path is the OpenAI SDK with base_url
overridden, so that is what this module does.

The two SDKs differ in three ways, all handled here:

    request      messages=[...] + system=str   vs  messages including a
                                                   {"role": "system"} entry
    response     resp.content[].text           vs  resp.choices[0].message.content
    usage        input_tokens / output_tokens  vs  prompt_tokens / completion_tokens

Callers see none of that. They call llm.call(...) and get a string back.

IF BOTH KEYS ARE SET
--------------------
OpenRouter wins, because someone who has deliberately configured OpenRouter
is trying to use it. Set LAUNCHGUARD_PROVIDER=anthropic to force the other way.
=============================================================================
"""

import os
from typing import Optional, Tuple

# Default model per provider. OpenRouter needs a vendor-prefixed slug.
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_OPENROUTER_MODEL = "anthropic/claude-sonnet-4.5"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# =============================================================================
# PROVIDER DETECTION
# =============================================================================

def active_provider() -> str:
    """Returns 'openrouter', 'anthropic', or 'none'."""
    forced = (os.getenv("LAUNCHGUARD_PROVIDER") or "").strip().lower()
    if forced in ("anthropic", "openrouter"):
        # honour the override only if that provider's key is actually present
        if forced == "openrouter" and os.getenv("OPENROUTER_API_KEY"):
            return "openrouter"
        if forced == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"

    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "none"


def has_key() -> bool:
    return active_provider() != "none"


def normalise_model(model: Optional[str]) -> str:
    """
    Correct a model name for whichever provider is active.

    Slugs are provider-specific: OpenRouter wants 'anthropic/claude-sonnet-4.5',
    Anthropic wants 'claude-sonnet-5'. The wrong shape is a 404 on every call.

    Applied to EVERY model string passed into call(), not just the configured
    default. That means existing call sites can keep passing whatever they
    already pass - including a hardcoded Anthropic name - and it still works
    after switching provider. One less thing to remember to change.
    """
    provider = active_provider()
    model = (model or "").strip()

    if not model:
        return (DEFAULT_OPENROUTER_MODEL if provider == "openrouter"
                else DEFAULT_ANTHROPIC_MODEL)

    if provider == "openrouter" and "/" not in model:
        return f"anthropic/{model}"
    if provider == "anthropic" and "/" in model:
        return model.split("/", 1)[1]
    return model


def default_model() -> str:
    """The model to use when a caller does not specify one."""
    return normalise_model(os.getenv("LAUNCHGUARD_MODEL"))


def status() -> dict:
    """For the health endpoint and the launcher."""
    p = active_provider()
    return {
        "provider": p,
        "has_key": p != "none",
        "model": default_model() if p != "none" else None,
        "base_url": OPENROUTER_BASE_URL if p == "openrouter" else "api.anthropic.com",
    }


# =============================================================================
# THE CALL
# =============================================================================

def _call_anthropic(system: str, user: str, model: str,
                    max_tokens: int) -> Tuple[str, int, int]:
    from anthropic import Anthropic
    client = Anthropic()
    r = client.messages.create(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in r.content if b.type == "text")
    u = getattr(r, "usage", None)
    return (text,
            int(getattr(u, "input_tokens", 0) or 0),
            int(getattr(u, "output_tokens", 0) or 0))


def _call_openrouter(system: str, user: str, model: str,
                     max_tokens: int) -> Tuple[str, int, int]:
    from openai import OpenAI
    client = OpenAI(base_url=OPENROUTER_BASE_URL,
                    api_key=os.getenv("OPENROUTER_API_KEY"))
    r = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        # optional attribution headers; harmless if OpenRouter ignores them
        extra_headers={"X-Title": "LaunchGuard AI"},
    )
    text = (r.choices[0].message.content or "") if r.choices else ""
    u = getattr(r, "usage", None)
    return (text,
            int(getattr(u, "prompt_tokens", 0) or 0),
            int(getattr(u, "completion_tokens", 0) or 0))


def call(system: str, user: str, model: Optional[str] = None,
         max_tokens: int = 4000, step: Optional[str] = None,
         label: str = "") -> str:
    """
    One model call. Returns the text.

    `step` and `label` are passed to usage tracking; omit step to skip it.
    Raises RuntimeError if no provider is configured, so the caller gets a
    readable message rather than an SDK authentication error.
    """
    provider = active_provider()
    if provider == "none":
        raise RuntimeError(
            "No API key configured. Add ANTHROPIC_API_KEY or "
            "OPENROUTER_API_KEY to your .env file and restart.")

    # normalise whatever was passed, so a stale hardcoded name
    # from a call site still resolves correctly
    model = normalise_model(model)

    if provider == "openrouter":
        text, inp, out = _call_openrouter(system, user, model, max_tokens)
    else:
        text, inp, out = _call_anthropic(system, user, model, max_tokens)

    if step:
        try:
            import usage
            usage.record_tokens(step, model, inp, out, label)
        except Exception:
            pass          # cost tracking must never break the actual work

    return text