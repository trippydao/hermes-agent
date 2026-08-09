"""Tests for live auto-decompose settings resolution (issue #49638).

The gateway dispatcher used to capture ``kanban.auto_decompose`` once at boot,
so a user who flipped it to ``false`` to STOP runaway auto-decompose (which had
created and launched tasks they didn't intend) found the flag had no effect
without a full gateway restart. ``_resolve_auto_decompose_settings`` is now
called every tick, reading the current config.

Issue #76 adds a third tuple element: ``require_approval``
(``kanban.auto_decompose_require_approval``). When True, auto-decompose holds
decomposed children in ``todo`` (auto_promote=False) so no worker spawns until
a human ``hermes kanban approve`` releases them.
"""

from __future__ import annotations

import pytest

from gateway.kanban_watchers import _resolve_auto_decompose_settings


def test_enabled_by_default_when_key_absent():
    enabled, per_tick, require_approval = _resolve_auto_decompose_settings(
        lambda: {"kanban": {}}
    )
    assert enabled is True
    assert per_tick == 3
    # default is off — approval gate must be opt-in, behaviour unchanged
    assert require_approval is False


def test_disabled_when_flag_false():
    enabled, per_tick, require_approval = _resolve_auto_decompose_settings(
        lambda: {"kanban": {"auto_decompose": False}}
    )
    assert enabled is False


def test_approval_off_by_default():
    _, _, require_approval = _resolve_auto_decompose_settings(
        lambda: {"kanban": {"auto_decompose": True}}
    )
    assert require_approval is False


def test_approval_on_when_flag_true():
    enabled, per_tick, require_approval = _resolve_auto_decompose_settings(
        lambda: {"kanban": {"auto_decompose": True, "auto_decompose_require_approval": True}}
    )
    assert enabled is True
    assert require_approval is True


def test_approval_still_reported_when_disabled():
    # Even when auto-decompose is off, the approval intent is preserved so a
    # re-enable doesn't silently lose the gate.
    _, _, require_approval = _resolve_auto_decompose_settings(
        lambda: {"kanban": {"auto_decompose": False, "auto_decompose_require_approval": True}}
    )
    assert require_approval is True


def test_transient_config_error_fails_safe_with_approval_off():
    def boom():
        raise RuntimeError("config unreadable")

    enabled, per_tick, require_approval = _resolve_auto_decompose_settings(boom)
    assert enabled is False
    assert per_tick == 3
    assert require_approval is False