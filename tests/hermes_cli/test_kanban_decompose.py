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
    # Generic manual-review hold: auto_promote=False (without the
    # approval_hold flag, the approval gate is off) leaves children in
    # 'todo' — no worker spawns until a human promotes them. Distinct
    # from the approval-gated 'needs_approval' hold which has its own
    # state; see test_decompose_approval_hold_* below.
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="manual-review fanout", triage=True)

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


def test_decompose_approval_hold_lands_children_in_needs_approval(kanban_home):
    # Approval gate (#76): with approval_hold=True the decomposer builds
    # the child graph but lands EVERY child in a real 'needs_approval'
    # state — distinct from 'todo' — so no worker spawns until a human
    # 'hermes kanban approve' releases the task.
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
            # approval_hold forces auto_promote off regardless of config.
            outcome = decomp.decompose_task(
                tid, author="auto-decomposer", approval_hold=True,
            )
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is True

    with kb.connect() as conn:
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    assert c0 is not None and c1 is not None
    # EVERY child holds in needs_approval — parent-free ones would
    # normally be 'ready'; the approval gate overrides that.
    assert c0.status == "needs_approval", c0.status
    assert c1.status == "needs_approval", c1.status


def test_decompose_approval_hold_releases_via_promote(kanban_home):
    # Accept: hermes kanban approve is an alias of promote, so releasing
    # a 'needs_approval' task must put it in 'ready' and let workers run.
    # Set up state without the LLM: create a triage task, decompose it
    # by hand through the DB helper with child_status='needs_approval'.
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="approval-gated single", triage=True)
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=[
                {"title": "research", "body": "look", "assignee": "researcher", "parents": []},
            ],
            author="auto-decomposer",
            auto_promote=False,
            child_status="needs_approval",
        )
    assert child_ids is not None
    child_id = child_ids[0]

    with kb.connect() as conn:
        assert kb.get_task(conn, child_id).status == "needs_approval"
        ok, err = kb.promote_task(conn, child_id, actor="je")
        assert ok, err
        assert kb.get_task(conn, child_id).status == "ready"


def test_decompose_approval_hold_single_task_no_fanout(kanban_home):
    # Approval gate on the single-task (fanout=false) fallback: the
    # tightened task must hold in 'needs_approval', not auto-run.
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="single unit", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened title",
        "body": "One concrete worker spec.",
        "assignee": "researcher",
    })

    patches = _patch_list_profiles(["orchestrator", "researcher"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value={"kanban": {"auto_promote_children": True}},
        ):
            outcome = decomp.decompose_task(
                tid, author="auto-decomposer", approval_hold=True,
            )
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is False
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "needs_approval", task.status


