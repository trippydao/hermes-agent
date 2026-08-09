"""Real tokenization for the local DGX Spark DeepSeek endpoint (issue #55).

The generic chars/4 heuristic that underlies Hermes' token estimates
(``estimate_tokens_rough`` and friends) undercounts structured chat-
completions payloads — tool schemas, JSON, code — because the DeepSeek-V4
tokenizer emits more tokens per character than the ~4 chars/token English-
prose rule the heuristic assumes. Observed ~29% light on tool-laden requests
(issue #55): Hermes estimated ~46.6K input tokens while the Spark vLLM
counted ~65.6K.

For requests that target the local Spark vLLM, count with the endpoint's own
OpenAI-compatible ``/tokenize`` so the phrase

    max_tokens = context_length - actual_input - SAFETY_BUFFER

is exact (issue #55 Layer 3 depends on Layer 2 here).

Counting primitive
------------------
The accurate count is produced by tokenizing the FULL request in ONE
``/tokenize`` call — ``messages`` + ``tools`` + ``add_generation_prompt`` —
exactly as the real chat-completion request would be rendered. This is the
only shape that matches vLLM's ``usage.prompt_tokens``: the chat template
wraps the whole conversation (BOS, per-turn role markers) and renders tool
schemas into the prompt, so a per-string/per-message sum cannot reproduce it.

Validated against the live endpoint: a representative request measured
``usage.prompt_tokens`` = 343; the full-payload ``/tokenize`` messages+tools+
``add_generation_prompt`` form returned exactly 343, while the previous
per-message repr-sum counted 154 (~55% undercount, worse than the heuristic).

Hot-path safety
---------------
* Activation is endpoint-gated — ``is_spark_endpoint()`` only matches the
  DGX Spark vLLM base_url (or a shell serving ``deepseek-v4-flash-dspark``).
  Every other provider keeps the fast chars/4 heuristic untouched.
* Results are cached per canonical serialisation of the request payload in a
  bounded LRU. Unchanged payloads (system prompt, tool schemas, stable
  history) never hit the network twice; only brand-new content does.
* Any failure (endpoint down, timeout, connection reset, malformed reply,
  cache miss + unreachable) falls back to the existing chars/4 estimate —
  identical to non-Spark behaviour — so a tokenizer outage can never wedge
  the request path or the compression decision, and never sends a request
  sized differently than today.
"""
from __future__ import annotations

import json
import urllib.request
from collections import OrderedDict
from typing import Any, Dict, List, Optional

__all__ = [
    "is_spark_endpoint",
    "count_tokens",
    "count_request",
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


def is_spark_endpoint(base_url: Optional[str], model: Optional[str] = "") -> bool:
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


def _tokenize_payload(payload: Dict[str, Any]) -> Optional[int]:
    """POST a JSON body to the Spark /tokenize; None on any failure."""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        SPARK_TOKENIZE_ENDPOINT,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_TOKENIZE_TIMEOUT_S) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    count = body.get("count") if isinstance(body, dict) else None
    if isinstance(count, int) and count >= 0:
        return count
    return None


# Request-payload serialisation cache key. ``messages`` can contain big tool
# results, so reuse vLLM's own JSON shape (separation guaranteed to collapse
# onto the wire body) rather than hashing the object graph ourselves.
def _payload_key(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]],
                 add_generation_prompt: bool) -> str:
    body: Dict[str, Any] = {
        "model": SPARK_TOKENIZE_MODEL,
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools:
        body["tools"] = tools
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"))


def count_request(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    *,
    add_generation_prompt: bool = True,
) -> int:
    """Exact input token count for a real chat-completions request to Spark.

    Tokenizes the whole request (``messages`` + ``tools`` +
    ``add_generation_prompt``) in ONE ``/tokenize`` call, matching vLLM's
    ``usage.prompt_tokens``. This is the authoritative input count for
    ``context_length - actual_input - SAFETY_BUFFER``.

    Falls back to the existing per-string chars/4 estimate (via
    ``estimate_request_tokens_rough``) on any failure so a tokenizer outage
    behaves exactly like a non-Spark provider.
    """
    add_generation_prompt = bool(add_generation_prompt)
    key = _payload_key(messages, tools, add_generation_prompt)
    cached = _TOKENS_CACHE.get(key)
    if cached is not None:
        _TOKENS_CACHE.move_to_end(key)
        return cached

    exact = _tokenize_payload({
        "model": SPARK_TOKENIZE_MODEL,
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
    } | ({"tools": tools} if tools else {}))
    if exact is None:
        from agent.model_metadata import estimate_request_tokens_rough
        exact = int(estimate_request_tokens_rough(messages, tools=tools))

    _TOKENS_CACHE[key] = exact
    if len(_TOKENS_CACHE) > _TOKENS_CACHE_MAX:
        _TOKENS_CACHE.popitem(last=False)
    return exact


def count_tokens(text: str, base_url: Optional[str] = None) -> int:
    """Real token count of a single raw string against Spark, LRU-cached.

    ``base_url`` is accepted for interface symmetry but not used for the
    count — activation is decided once by ``is_spark_endpoint`` at the call
    site, and there is a single Spark endpoint per process.

    NOTE: for a full request use ``count_request`` instead — a bare string
    omits the chat template and tool rendering, so it undercounts relative
    to what vLLM reports for a real request. This remains as the narrow
    single-string primitive for callers that genuinely only need a text
    fragment counted.
    """
    text = str(text)
    cached = _TOKENS_CACHE.get(text)
    if cached is not None:
        _TOKENS_CACHE.move_to_end(text)
        return cached
    count = _fallback(text)
    exact = _tokenize_payload({"model": SPARK_TOKENIZE_MODEL, "prompt": text})
    if exact is not None:
        count = exact
    _TOKENS_CACHE[text] = count
    if len(_TOKENS_CACHE) > _TOKENS_CACHE_MAX:
        _TOKENS_CACHE.popitem(last=False)
    return count


def make_counter(base_url: Optional[str] = None):
    """Return a ``count_fn(text) -> int`` counting single strings vs Spark.

    Provided for callers that pass a ``count_fn`` to the estimate helpers.
    Prefer ``count_request`` for request-level accuracy.
    """
    return lambda text: count_tokens(text, base_url)


def safety_budget(
    context_length: int,
    api_messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = True,
) -> "tuple[int | None, int]":
    """Compute the issue #55 Layer 3 output budget.

    Returns ``(max_output_budget, actual_input_tokens)``.
    ``budget`` is ``context_length - actual_input - SAFETY_BUFFER``, or None
    when ``context_length`` is unknown/zero. When budget <= 0 the caller
    should NOT send an over-budget request (the reactive path compresses).
    """
    real_input = count_request(
        api_messages, tools=tools, add_generation_prompt=add_generation_prompt
    )
    if not context_length or context_length <= 0:
        return None, real_input
    return context_length - real_input - SAFETY_BUFFER, real_input