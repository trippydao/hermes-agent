"""Real tokenization for the local DGX Spark DeepSeek endpoint (issue #55).

The generic chars/4 heuristic that underlies Hermes' token estimates
(``estimate_tokens_rough`` and friends) undercounts structured chat-
completions payloads — tool schemas, JSON, code — because the DeepSeek-V4
tokenizer emits more tokens per character than the ~4 chars/token English-
prose rule the heuristic assumes. Observed ~29%% light on tool-laden requests
(issue #55): Hermes estimated ~46.6K input tokens while the Spark vLLM
counted ~65.6K.

For requests that target the local Spark vLLM, count with the endpoint's own
OpenAI-compatible ``/tokenize`` so the phrase

    max_tokens = context_length - actual_input - SAFETY_BUFFER

is exact (issue #55 Layer 3 depends on Layer 2 here).

Hot-path safety
---------------
* Activation is endpoint-gated — ``is_spark_endpoint()`` only matches the
  DGX Spark vLLM base_url (or a shell serving ``deepseek-v4-flash-dspark``).
  Every other provider keeps the fast chars/4 heuristic untouched.
* Results are cached per unique text string in a bounded LRU. Unchanged
  payloads (system prompt, tool schemas, stable history) never hit the
  network twice; only brand-new text does, and one request is bounded by the
  cache cap.
* Any failure (endpoint down, timeout, connection reset, malformed reply)
  falls back to the chars/4 estimate, so a tokenizer outage can never wedge
  the request path or the compression decision.
"""
from __future__ import annotations

import json
import urllib.request
from collections import OrderedDict

__all__ = [
    "is_spark_endpoint",
    "count_tokens",
    "make_counter",
    "safety_budget",
    "SAFETY_BUFFER",
]

# The local DGX Spark vLLM (tailscale node serving DeepSeek-V4 flash).
# base_url is the OpenAI-compatible wire root (…/v1). vLLM serves its
# OpenAI-compatible /tokenize from the ROOT, not under /v1 (404 under /v1).
SPARK_BASE_URL = "http://100.122.146.36:8888/v1"
SPARK_TOKENIZE_ENDPOINT = "http://100.122.146.36:8888/tokenize"
SPARK_TOKENIZE_MODEL = "deepseek-v4-flash-dspark"

# Reserve this many output tokens at the context ceiling so a request never
# requests more output than the model window has room for (issue #55 Layer 3).
SAFETY_BUFFER = 2048

_TOKENIZE_TIMEOUT_S = 2.0
_TOKENS_CACHE: "OrderedDict[str, int]" = OrderedDict()
_TOKENS_CACHE_MAX = 4096


def _fallback(text: str) -> int:
    """chars/4 ceiling estimate — the safe fallback when /tokenize fails."""
    return (len(text) + 3) // 4


def is_spark_endpoint(base_url: str | None, model: str | None = "") -> bool:
    """True when base_url targets the DGX Spark DeepSeek vLLM.

    Gated on the Spark tailscale address/port, or on a model that only the
    Spark endpoint serves (falls back cleanly if the IP ever changes; the
    tokenizer uses the hardcoded Spark base_url regardless of the passed
    one, so an empty/unknown base_url with a Spark model still counts real).
    """
    if base_url and "100.122.146.36:8888" in str(base_url):
        return True
    # The model id is specific enough to the Spark deployment that serving it
    # is effectively Spark; covers a DHCP/rebuild IP move.
    return "deepseek-v4-flash-dspark" in (model or "")


def _tokenize(text: str) -> int | None:
    """Ask the Spark vLLM /tokenize for an exact count; None on any failure."""
    payload = json.dumps(
        {"model": SPARK_TOKENIZE_MODEL, "prompt": text}, ensure_ascii=False
    ).encode("utf-8")
    req = urllib.request.Request(
        SPARK_TOKENIZE_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_TOKENIZE_TIMEOUT_S) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    count = body.get("count") if isinstance(body, dict) else None
    if isinstance(count, int) and count >= 0:
        return count
    return None


def count_tokens(text: str, base_url: str | None = None) -> int:
    """Real token count against the Spark endpoint, LRU-cached.

    ``base_url`` is accepted for interface symmetry but not used for the
    count — activation is decided once by ``is_spark_endpoint`` at the call
    site, and there is a single Spark endpoint per process.
    """
    text = str(text)
    cached = _TOKENS_CACHE.get(text)
    if cached is not None:
        _TOKENS_CACHE.move_to_end(text)
        return cached
    count = _fallback(text)
    exact = _tokenize(text)
    if exact is not None:
        count = exact
    _TOKENS_CACHE[text] = count
    if len(_TOKENS_CACHE) > _TOKENS_CACHE_MAX:
        _TOKENS_CACHE.popitem(last=False)
    return count


def make_counter(base_url: str | None = None):
    """Return a ``count_fn(text) -> int`` for the Spark endpoint.

    Pass this as ``count_fn=`` to ``estimate_request_tokens_rough`` and the
    other estimate functions so only the Spark path pays for real counts.
    """
    return lambda text: count_tokens(text, base_url)


def safety_budget(
    context_length: int,
    count_fn,
    api_messages: list,
    tools: list | None = None,
    system_prompt: str = "",
) -> tuple[int | None, int]:
    """Compute the issue #55 Layer 3 output budget.

    Returns ``(max_output_budget, actual_input_tokens)``.
    ``budget`` is ``context_length - actual_input - SAFETY_BUFFER``, or None
    when ``context_length`` is unknown/zero. When budget <= 0 the caller
    should NOT send an over-budget request (the reactive path compresses).
    """
    from agent.model_metadata import estimate_request_tokens_rough

    real_input = int(
        estimate_request_tokens_rough(
            api_messages, system_prompt=system_prompt, tools=tools, count_fn=count_fn
        )
    )
    if not context_length or context_length <= 0:
        return None, real_input
    return context_length - real_input - SAFETY_BUFFER, real_input