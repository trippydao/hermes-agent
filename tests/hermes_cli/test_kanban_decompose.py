"""Tests for the decomposer module + `hermes kanban decompose` CLI surface.

The auxiliary LLM client is mocked — no network calls. Tests exercise the
prompt plumbing, response parsing, DB writes (via the real DB helper),
and the assignee-fallback logic.
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_decompose as decomp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_aux_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _mock_client_returning(content: str):
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=_fake_aux_response(content))
    return client


def _patch_aux_client(content: str, *, model: str = "test-model"):
    # decompose_task now routes through call_llm (see #35566) — mock it at
    # the source module so task config, extra_body, and retries stay out of
    # unit-test scope.
    return patch(
        "agent.auxiliary_client.call_llm",
        return_value=_fake_aux_response(content),
    )


def _patch_extra_body():
    # No-op shim retained for call-site compatibility: extra_body plumbing
    # now lives inside call_llm, which _patch_aux_client already mocks.
    return patch("agent.auxiliary_client.get_auxiliary_extra_body", return_value={})


def _patch_list_profiles(names: list[str]):
    """Pretend the named profiles exist. The decomposer uses
    profiles_mod.list_profiles() to build the roster + valid-set, and
    profiles_mod.profile_exists() to resolve orchestrator/default."""
    from types import SimpleNamespace
    fake_profiles = [
        SimpleNamespace(
            name=n, is_default=(i == 0), description=f"desc for {n}",
            description_auto=False, model="m", provider="p", skill_count=1,
        )
        for i, n in enumerate(names)
    ]
    return [
        patch("hermes_cli.profiles.list_profiles", return_value=fake_profiles),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names),
        patch("hermes_cli.profiles.get_active_profile_name", return_value=names[0] if names else "default"),
    ]


def test_decompose_with_fanout_creates_children(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "research", "body": "look it up", "assignee": "researcher", "parents": []},
            {"title": "build", "body": "code it", "assignee": "engineer", "parents": [0]},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "researcher", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is True
    assert outcome.child_ids and len(outcome.child_ids) == 2

    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    assert root.status == "todo"
    assert c0.status == "ready"
    assert c1.status == "todo"
    assert c0.assignee == "researcher"
    assert c1.assignee == "engineer"


def test_decompose_auto_promote_false_holds_children_in_todo(kanban_home):
    # Human-in-the-loop gate (#76): with auto_promote=False the decomposer
    # builds the child graph but leaves parent-free children in 'todo' so no
    # worker spawns until a human 'hermes kanban approve' releases them.
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="approval-gated fanout", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "approval-gated test",
        "tasks": [
            {"title": "first", "body": "a", "assignee": "researcher", "parents": []},
            {"title": "second", "body": "b", "assignee": "engineer", "parents": [0]},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "researcher", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value={"kanban": {"auto_promote_children": True}},
        ):
            # Explicit auto_promote=False overrides the config default True.
            outcome = decomp.decompose_task(tid, author="auto-decomposer", auto_promote=False)
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is True

    with kb.connect() as conn:
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    assert c0 is not None and c1 is not None
    # Parent-free child would normally be 'ready'; approval gate holds it.
    assert c0.status == "todo", c0.status
    assert c1.status == "todo", c1.status


def test_decompose_auto_promote_none_uses_config(kanban_home):
    # Default path (auto_promote not passed) should honour auto_promote_children.
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="config-default fanout", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "config-default test",
        "tasks": [
            {"title": "only", "body": "a", "assignee": "researcher", "parents": []},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "researcher"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value={"kanban": {"auto_promote_children": False}},
        ):
            # No override -> honours config False -> still held in todo.
            outcome = decomp.decompose_task(tid, author="auto-decomposer")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kb.connect() as conn:
        c0 = kb.get_task(conn, outcome.child_ids[0])
    assert c0 is not None
    assert c0.status == "todo", c0.status


def test_decompose_fanout_false_invalid_llm_assignee_uses_default(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="route me safely", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened title",
        "body": "Route to fallback.",
        "assignee": "made_up",
    })

    patches = _patch_list_profiles(["orchestrator", "fallback"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value={"kanban": {"default_assignee": "fallback"}},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.assignee == "fallback"


def test_decompose_returns_false_when_task_not_triage(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x")  # ready, not triage

    patches = _patch_list_profiles(["orchestrator"])
    for p in patches:
        p.start()
    try:
        outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok is False
    assert "not in triage" in outcome.reason


# ---------------------------------------------------------------------------
# Liveness-ignore config knob (gh-78)
# ---------------------------------------------------------------------------

def test_resolve_liveness_ignore_defaults_probe_everything():
    """No knob set -> probe every endpoint (ignore_all False, no opt-outs)."""
    li = decomp._resolve_liveness_ignore({})
    assert li.ignore_all is False
    assert li.ignored_base_urls == frozenset()
    assert li.ignores("http://100.122.146.36:8888/v1") is False


def test_resolve_liveness_ignore_global_opt_out():
    """decompose_ignore_liveness: true disables the probe for every endpoint."""
    li = decomp._resolve_liveness_ignore(
        {"kanban": {"decompose_ignore_liveness": True}}
    )
    assert li.ignore_all is True
    assert li.ignores("http://anything:8001/v1") is True


def test_resolve_liveness_ignore_per_endpoint_opt_out():
    """Only listed base_urls are ignored; trailing slashes are normalized."""
    li = decomp._resolve_liveness_ignore(
        {
            "kanban": {
                "decompose_ignore_liveness_base_urls": [
                    "http://100.122.146.36:8888/v1/",
                    "  http://100.80.86.3:8001/v1  ",
                ]
            }
        }
    )
    assert li.ignore_all is False
    assert li.ignores("http://100.122.146.36:8888/v1") is True
    assert li.ignores("http://100.80.86.3:8001/v1") is True
    # Not listed -> still probed.
    assert li.ignores("http://other:8001/v1") is False


def test_resolve_liveness_ignore_malformed_config_degrades_safely():
    """Non-list base_urls and non-dict cfg never raise; degrade to defaults."""
    li = decomp._resolve_liveness_ignore({"kanban": {"decompose_ignore_liveness_base_urls": "nope"}})
    assert li.ignore_all is False
    assert li.ignored_base_urls == frozenset()
    li2 = decomp._resolve_liveness_ignore("not-a-dict")
    assert li2.ignore_all is False
    assert li2.ignored_base_urls == frozenset()


# ---------------------------------------------------------------------------
# Liveness gate verdicts + wiring in decompose_task (gh-78)
# ---------------------------------------------------------------------------

ENTER = decomp.LivenessIgnore(ignore_all=False, ignored_base_urls=frozenset())
SKIP_ALL = decomp.LivenessIgnore(ignore_all=True, ignored_base_urls=frozenset())


def test_liveness_verdict_pass_when_model_is_live():
    """Model served by an assignee endpoint -> keep the leg."""
    action, reason = decomp._liveness_verdict(
        "deepseek-v4", ["http://ep-a"], {"http://ep-a": {"deepseek-v4", "other"}}, ENTER,
    )
    assert action == "pass"


def test_liveness_verdict_pass_when_endpoint_opted_out():
    """decompose_ignore_liveness_base_urls treats the endpoint as live w/o probe."""
    li = decomp.LivenessIgnore(False, frozenset({"http://ep-a"}))
    action, _ = decomp._liveness_verdict(
        "whatever-model", ["http://ep-a"], {}, li,
    )
    assert action == "pass"


def test_liveness_verdict_ignore_all_passes_every_leg():
    """decompose_ignore_liveness: true skips probing entirely."""
    action, _ = decomp._liveness_verdict("m", ["http://ep"], {}, SKIP_ALL)
    assert action == "pass"


def test_liveness_verdict_skip_when_model_absent_from_all_known_endpoints():
    """All endpoints known but none serve the model -> drop the leg."""
    action, _ = decomp._liveness_verdict(
        "gone-model", ["http://ep-a", "http://ep-b"],
        {"http://ep-a": {"deepseek-v4"}, "http://ep-b": {"other"}}, ENTER,
    )
    assert action == "skip"


def test_liveness_verdict_flag_when_endpoint_unknown():
    """UNKNOWN endpoint (down/fail-soft) -> keep but flag, not skip."""
    action, _ = decomp._liveness_verdict("m", ["http://ep-a"], {}, ENTER)
    assert action == "flag"


def test_liveness_verdict_flag_when_no_endpoint_declared():
    """No custom-provider endpoint -> cannot verify -> flag."""
    action, _ = decomp._liveness_verdict("m", [], {"x": {"m"}}, ENTER)
    assert action == "flag"


def test_liveness_verdict_flag_when_no_model_configured():
    """No configured model -> cannot verify -> flag."""
    action, _ = decomp._liveness_verdict(None, ["http://ep-a"], {"http://ep-a": {"m"}}, ENTER)
    assert action == "flag"


def test_profile_model_and_base_urls_reads_own_config(kanban_home, monkeypatch):
    """Reads model + custom_providers base_urls from the profile's config.yaml."""
    profile_dir = kanban_home / "profiles" / "researcher"
    profile_dir.mkdir(parents=True)
    (profile_dir / "config.yaml").write_text(
        'model:\n  default: deepseek-v4-flash-dspark\n'
        'custom_providers:\n'
        '  - name: dspark\n'
        '    base_url: http://100.122.146.36:8888/v1/\n'
        '    api_key: sk-test\n'
        '  - name: backup\n'
        '    base_url: http://100.80.86.3:8001/v1\n'
    )
    monkeypatch.setattr(decomp.profiles_mod, "get_profile_dir", lambda n: profile_dir if n == "researcher" else kanban_home)
    model, base_urls = decomp._profile_model_and_base_urls("researcher")
    assert model == "deepseek-v4-flash-dspark"
    assert base_urls == ["http://100.122.146.36:8888/v1", "http://100.80.86.3:8001/v1"]


def _patch_liveness(live_map, profile_ctx):
    """Stub the liveness IO so decompose tests don't touch real FS/network.

    ``profile_ctx`` maps assignee -> (model, [base_urls]); anything else
    degrades to the unverified case (None, []).
    """
    return [
        patch("hermes_cli.kanban_decompose._enumerate_live_models", return_value=live_map),
        patch(
            "hermes_cli.kanban_decompose._profile_model_and_base_urls",
            side_effect=lambda a: profile_ctx.get(a, (None, [])),
        ),
    ]


def test_decompose_drops_not_live_leg_and_remaps_parent(kanban_home):
    """A leg whose model isn't served is not emitted; dependents are remapped."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="fan with a dead leg", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True, "rationale": "gate test", "tasks": [
            {"title": "a-live", "assignee": "researcher", "parents": []},
            {"title": "b-dead", "assignee": "engineer", "parents": [0]},  # depends on a-live
            {"title": "c-live", "assignee": "researcher", "parents": []},
        ],
    })

    live_map = {"http://ep-a": {"deepseek-v4"}}
    profile_ctx = {
        "researcher": ("deepseek-v4", ["http://ep-a"]),
        "engineer": ("missing-model", ["http://ep-a"]),  # absent from live set -> skip
    }

    patches = _patch_list_profiles(["orchestrator", "researcher", "engineer"]) + _patch_liveness(live_map, profile_ctx)
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.child_ids and len(outcome.child_ids) == 2  # b-dead dropped

    with kb.connect() as conn:
        names = [kb.get_task(conn, cid).title for cid in outcome.child_ids]
        comments = [c.body for c in kb.list_comments(conn, tid)]
    assert "b-dead" not in names
    assert "a-live" in names and "c-live" in names
    # The parent was annotated about the withheld leg.
    assert any("Skipped by decompose liveness gate" in b for b in comments)


def test_decompose_ignore_all_skips_gate_entirely(kanban_home):
    """decompose_ignore_liveness: true -> all legs created, no probe runs."""
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  decompose_ignore_liveness: true\n"
    )
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ignored gate", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True, "rationale": "gate off", "tasks": [
            {"title": "x", "assignee": "researcher", "parents": []},
            {"title": "y", "assignee": "engineer", "parents": [0]},
        ],
    })

    probe = patch(
        "hermes_cli.kanban_decompose._enumerate_live_models",
        side_effect=AssertionError("probe must not run when ignore_liveness is true"),
    )
    patches = _patch_list_profiles(["orchestrator", "researcher", "engineer"]) + [probe]
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.child_ids and len(outcome.child_ids) == 2
    with kb.connect() as conn:
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    assert c1.title == "y"
    assert c1.status == "todo" and c0.status == "ready"


def test_decompose_unknown_endpoint_flagged_but_emitted(kanban_home):
    """UNKNOWN endpoint (fail-soft) -> leg is kept but annotated as unverified."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="unverifiable leg", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True, "rationale": "fail-soft", "tasks": [
            {"title": "unverifiable", "assignee": "researcher", "parents": []},
        ],
    })

    # live_map empty -> the only endpoint resolves to UNKNOWN.
    patches = _patch_list_profiles(["orchestrator", "researcher"]) + _patch_liveness(
        {}, {"researcher": ("any-model", ["http://ep-down"])},
    )
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.child_ids and len(outcome.child_ids) == 1  # fail-soft: kept
    with kb.connect() as conn:
        comments = [c.body for c in kb.list_comments(conn, tid)]
    assert any("UNVERIFIED" in b for b in comments)


def test_decompose_all_legs_skipped_leaves_root_in_triage(kanban_home):
    """Every leg skipped -> root stays in triage (nothing emitted)."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="all dead", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True, "rationale": "all dead", "tasks": [
            {"title": "dead", "assignee": "engineer", "parents": []},
        ],
    })
    patches = _patch_list_profiles(["orchestrator", "engineer"]) + _patch_liveness(
        {"http://ep-a": {"deepseek-v4"}}, {"engineer": ("missing", ["http://ep-a"])},
    )
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    assert outcome.child_ids is None
    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
    assert root.status == "triage"  # surfaced for a human / retry
