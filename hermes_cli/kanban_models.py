"""Live model enumeration for kanban decomposition.

Probes each custom-provider ``base_url``'s ``/v1/models`` endpoint so the
decomposer routes only to models that dispatch/runtime actually serve —
not a hardcoded list that drifts from deployment reality.

Built on the exact primitives dispatch/runtime use, so the inventory we
report is the endpoint table in effect, not a reimplementation of it:

* ``hermes_cli.config.get_compatible_custom_providers`` — the deduplicated
  custom-provider view (legacy ``custom_providers`` + v12 ``providers``)
  that ACP/server, runtime, and the picker all resolve against.
* ``hermes_cli.models.probe_api_models`` — the runtime's ``/models`` probe
  (handles the ``/v1`` vs bare-base heuristic, anthropic/x-api-key auth,
  per-provider ``extra_headers``, TLS, and the 5s timeout).

Scope note: this module is deliberately free of kanban config reading.
The "ignore liveness" opt-out knob (``kanban.decompose_ignore_liveness``)
and the wiring into ``decompose_task`` live with the caller, so this stays
a standalone, importable utility.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Sentinel used to mark an endpoint that could not be enumerated (down,
# timeout, auth failure). Consumers treat it as "unknown" and fail soft.
UNKNOWN = "unknown"


def _resolve_entry_api_key(entry: Dict[str, Any]) -> str:
    """Resolve a custom_provider entry's api_key (inline or ``key_env`` ref).

    Mirrors the resolution in ``acp_adapter/server.py``: prefer the literal
    ``api_key`` value, else expand a ``key_env`` name from the environment.
    """
    raw = str(entry.get("api_key", "") or "").strip()
    if raw:
        return raw
    key_env = str(entry.get("key_env", "") or "").strip()
    if key_env:
        return os.environ.get(key_env, "").strip()
    return ""


def _entry_endpoint(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract a normalized ``{base_url, api_key, api_mode, headers}`` blob.

    Returns ``None`` when the entry carries no usable base_url.
    """
    if not isinstance(entry, dict):
        return None
    base_url = str(entry.get("base_url", "") or "").strip()
    if not base_url:
        return None
    headers = entry.get("extra_headers")
    if not isinstance(headers, dict):
        headers = None
    return {
        "base_url": base_url,
        "api_key": _resolve_entry_api_key(entry),
        "api_mode": str(entry.get("api_mode", "") or "").strip() or None,
        "headers": headers or None,
    }


def collect_endpoint_table(
    *,
    include_default: bool = True,
) -> dict[str, Dict[str, Any]]:
    """Return ``{canonical_base_url: {base_url, api_key, api_mode, headers}}``.

    Gathers the union of endpoints across the default config and every
    named profile's ``custom_providers``/``providers`` — the same endpoint
    table dispatch/runtime resolve against, keyed by a canonical (trimmed,
    trailing-slash-stripped) base_url. Where several profiles declare the
    same endpoint, the first non-empty credential wins.

    ``include_default`` collects the active default config. Named profiles
    are read from their own ``config.yaml`` on disk (via the raw primitive
    so we never mutate global config state to visit them — a long-lived
    gateway must not switch HERMES_HOME just to enumerate).
    """
    from hermes_cli.config import (
        get_compatible_custom_providers,
        load_config,
    )
    from hermes_cli.profiles import list_profiles

    table: dict[str, Dict[str, Any]] = {}

    def _ingest(cfg: Any) -> None:
        try:
            entries = get_compatible_custom_providers(cfg)
        except Exception as exc:  # defensive: never break enumeration
            logger.warning("kanban_models: get_compatible_custom_providers failed: %s", exc)
            return
        for entry in entries or []:
            ep = _entry_endpoint(entry)
            if ep is None:
                continue
            key = ep["base_url"].rstrip("/")
            existing = table.get(key)
            if existing is None:
                table[key] = ep
            elif not existing["api_key"] and ep["api_key"]:
                existing["api_key"] = ep["api_key"]

    if include_default:
        try:
            _ingest(load_config())
        except Exception as exc:
            logger.warning("kanban_models: default config load failed: %s", exc)

    # Named profiles: read raw so we don't touch active config state.
    try:
        from hermes_cli.config import read_user_config_raw

        profiles = list_profiles()
    except Exception as exc:
        logger.warning("kanban_models: profile enumeration failed: %s", exc)
        return table

    for pinfo in profiles:
        if pinfo.is_default:
            continue  # already covered by include_default
        cfg_path = pinfo.path / "config.yaml"
        try:
            if not cfg_path.is_file():
                continue
            _ingest(read_user_config_raw(cfg_path))
        except Exception as exc:  # unreadable profile file — fail soft
            logger.warning("kanban_models: read %s failed: %s", cfg_path, exc)

    return table


def probe_endpoint(
    endpoint: Dict[str, Any],
    *,
    timeout: float = 5.0,
) -> Optional[set[str]]:
    """Probe one endpoint's live ``/v1/models``.

    Returns the set of model ids, or ``None`` on any failure (network,
    timeout, auth) — the fail-soft contract callers rely on.
    """
    from hermes_cli.models import probe_api_models

    try:
        result = probe_api_models(
            endpoint.get("api_key"),
            endpoint.get("base_url"),
            timeout=timeout,
            api_mode=endpoint.get("api_mode"),
            request_headers=endpoint.get("headers"),
        )
    except Exception as exc:
        logger.debug(
            "kanban_models: probe %s raised: %s",
            endpoint.get("base_url"), exc,
        )
        return None
    models = result.get("models") if isinstance(result, dict) else None
    if not models:
        return None
    return {str(m) for m in models if m}


def enumerate_live_models(
    base_urls: Optional[list[str]] = None,
    *,
    timeout: float = 5.0,
    endpoint_table: Optional[dict[str, Dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Probe live models for a set of base_urls.

    Returns ``{base_url: set[str] | UNKNOWN}``.

    ``base_urls`` — the endpoints to probe. When ``None``, gathers the
    union across all profiles' custom_providers + default config via
    :func:`collect_endpoint_table`.

    ``endpoint_table`` — precomputed table (useful when a caller already
    built it once and wants to reuse the same view). Takes precedence; when
    given, ``base_urls`` must be a subset of its keys to carry credentials.

    Fail-soft: an endpoint that can't be enumerated maps to ``UNKNOWN``
    rather than raising, so a down backend never aborts the whole pass.
    """
    if endpoint_table is None:
        endpoint_table = collect_endpoint_table()

    if base_urls is None:
        base_urls = list(endpoint_table.keys())

    out: dict[str, Any] = {}
    for raw_url in base_urls:
        base_url = str(raw_url or "").strip()
        if not base_url:
            continue
        key = base_url.rstrip("/")
        ep = endpoint_table.get(key) or {"base_url": base_url}
        models = probe_endpoint(ep, timeout=timeout)
        out[base_url] = models if models is not None else UNKNOWN
    return out