"""Tests for ``hermes_cli.kanban_models`` — live model enumeration utility.

Covers the three things the deliverable promises:
  * gathering the endpoint table across default + named profiles'
    ``custom_providers``/``providers`` (reusing the same table dispatch
    resolves against — mocked ``get_compatible_custom_providers``),
  * probing each base_url's live ``/v1/models`` and returning a set of
    model ids,
  * fail-soft ``UNKNOWN`` on unreachable endpoints.

The network probe (``probe_api_models``), provider normalization
(``get_compatible_custom_providers``), and profile listing are all mocked —
no live HTTP, no real profiles needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli import kanban_models as km


def _entry(base_url, *, api_key="", key_env="", api_mode="", headers=None):
    ep = {"base_url": base_url}
    if api_key:
        ep["api_key"] = api_key
    if key_env:
        ep["key_env"] = key_env
    if api_mode:
        ep["api_mode"] = api_mode
    if headers:
        ep["extra_headers"] = headers
    return ep


def _cp_entries(*entries):
    return [dict(e) for e in entries]


def _profile(tmp_path, name, is_default=False):
    """Build a named-profile stub, writing a real config.yaml on disk so
    ``collect_endpoint_table``'s ``cfg_path.is_file()`` guard passes."""
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text("model: m\n")
    return SimpleNamespace(
        name=name, is_default=is_default, path=d,
        description="", description_auto=False, model="m", provider="p",
        skill_count=1, gateway_running=False,
    )


class TestEntryEndpoint:
    def test_resolves_key_env_from_environ(self, monkeypatch):
        monkeypatch.setenv("MY_ENDPOINT_KEY", "sekret")
        ep = km._entry_endpoint(_entry("http://a", key_env="MY_ENDPOINT_KEY"))
        assert ep["api_key"] == "sekret"

    def test_prefers_literal_api_key(self):
        ep = km._entry_endpoint(_entry("http://a", api_key="lit", key_env="OTHER"))
        assert ep["api_key"] == "lit"

    def test_returns_none_for_empty_base_url(self):
        assert km._entry_endpoint({"base_url": "  "}) is None
        assert km._entry_endpoint({}) is None


class TestCollectEndpointTable:
    def test_union_across_default_and_named_profiles(self, tmp_path):
        default_cfg = {"custom_providers": _cp_entries(
            _entry("http://default", api_key="dk"),
        )}
        profile_cfg = {"custom_providers": _cp_entries(
            _entry("http://profile", api_key="pk"),
        )}

        with patch(
            "hermes_cli.config.load_config", return_value=default_cfg,
        ), patch(
            "hermes_cli.config.get_compatible_custom_providers",
            side_effect=lambda cfg: cfg.get("custom_providers") or [],
        ), patch(
            "hermes_cli.profiles.list_profiles",
            return_value=[_profile(tmp_path, "worker", is_default=False)],
        ), patch(
            "hermes_cli.config.read_user_config_raw",
            return_value=profile_cfg,
        ):
            table = km.collect_endpoint_table()

        assert set(table) == {"http://default", "http://profile"}
        assert table["http://profile"]["api_key"] == "pk"
        assert table["http://default"]["api_key"] == "dk"

    def test_first_non_empty_credential_wins_for_same_endpoint(self, tmp_path):
        default_cfg = {"providers": {"h1": _entry("http://h", api_key="")}}
        # Default declares h with no key; a named profile supplies it.
        profile_cfg = {"custom_providers": _cp_entries(_entry("http://h", api_key="real"))}

        with patch(
            "hermes_cli.config.load_config", return_value=default_cfg,
        ), patch(
            "hermes_cli.config.get_compatible_custom_providers",
            side_effect=lambda cfg: (
                list((cfg.get("providers") or {}).values())
                if isinstance(cfg.get("providers"), dict)
                else cfg.get("custom_providers") or []
            ),
        ), patch(
            "hermes_cli.profiles.list_profiles",
            return_value=[_profile(tmp_path, "worker", is_default=False)],
        ), patch(
            "hermes_cli.config.read_user_config_raw", return_value=profile_cfg,
        ):
            table = km.collect_endpoint_table()

        assert "http://h" in table
        assert table["http://h"]["api_key"] == "real"

    def test_skips_profiles_if_disabled_flag(self, tmp_path):
        default_cfg = {"custom_providers": _cp_entries(_entry("http://default"))}
        with patch(
            "hermes_cli.config.load_config", return_value=default_cfg,
        ), patch(
            "hermes_cli.config.get_compatible_custom_providers",
            side_effect=lambda cfg: cfg.get("custom_providers") or [],
        ), patch(
            "hermes_cli.profiles.list_profiles",
            return_value=[_profile(tmp_path, "worker", is_default=False)],
        ):
            # include_default=False should skip the default config entirely.
            table = km.collect_endpoint_table(include_default=False)
            assert "http://default" not in table


class TestProbeAndEnumerate:
    def _patch_probe(self, mapping):
        """probe_api_models(api_key, base_url, timeout, api_mode, request_headers)
        returns a dict; our function reads .get('models')."""
        def _fake_probe(api_key, base_url, timeout=5.0, api_mode=None, request_headers=None):
            return {"models": mapping.get(base_url)}
        return patch("hermes_cli.models.probe_api_models", side_effect=_fake_probe)

    def test_returns_set_of_models_per_healthy_endpoint(self):
        table = {
            "http://a": {"base_url": "http://a", "api_key": "", "api_mode": None, "headers": None},
            "http://b": {"base_url": "http://b", "api_key": "", "api_mode": None, "headers": None},
        }
        with self._patch_probe({
            "http://a": ["model-1", "model-2", "model-1"],
            "http://b": ["other-model"],
        }):
            out = km.enumerate_live_models(endpoint_table=table)

        assert out["http://a"] == {"model-1", "model-2"}
        assert out["http://b"] == {"other-model"}

    def test_fail_soft_unknown_on_unreachable(self):
        table = {
            "http://down": {"base_url": "http://down", "api_key": "", "api_mode": None, "headers": None},
        }
        # probe returns None (endpoint down) — maps to UNKNOWN, no raise.
        with self._patch_probe({"http://down": None}):
            out = km.enumerate_live_models(endpoint_table=table)
        assert out["http://down"] == km.UNKNOWN

    def test_fail_soft_when_probe_raises(self):
        table = {
            "http://boom": {"base_url": "http://boom", "api_key": "", "api_mode": None, "headers": None},
        }
        def _raise(*a, **k):
            raise TimeoutError("timeout")
        with patch("hermes_cli.models.probe_api_models", side_effect=_raise):
            out = km.enumerate_live_models(endpoint_table=table)
        assert out["http://boom"] == km.UNKNOWN

    def test_explicit_base_urls_with_precomputed_table(self):
        table = {
            "http://a": {"base_url": "http://a", "api_key": "", "api_mode": None, "headers": None},
            "http://b": {"base_url": "http://b", "api_key": "", "api_mode": None, "headers": None},
        }
        with self._patch_probe({"http://a": ["x"], "http://b": ["y"]}):
            out = km.enumerate_live_models(["http://b"], endpoint_table=table)
        # Only the requested subset is probed.
        assert set(out) == {"http://b"}
        assert out["http://b"] == {"y"}

    def test_auto_gathers_urls_when_none_given(self):
        default_cfg = {"custom_providers": _cp_entries(_entry("http://auto"))}
        with patch(
            "hermes_cli.config.load_config", return_value=default_cfg,
        ), patch(
            "hermes_cli.config.get_compatible_custom_providers",
            side_effect=lambda cfg: cfg.get("custom_providers") or [],
        ), patch(
            "hermes_cli.profiles.list_profiles", return_value=[],
        ) as lp, patch(
            "hermes_cli.models.probe_api_models",
            return_value={"models": ["auto-model"]},
        ):
            out = km.enumerate_live_models()
        lp.assert_called_once()
        assert out == {"http://auto": {"auto-model"}}


def test_unknown_is_a_stable_sentinel():
    # Consumers pattern-match on this exact string; changing it is breaking.
    assert km.UNKNOWN == "unknown"