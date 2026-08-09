"""Kanban decomposer — fan a triage task out into a graph of child tasks.

Invoked by ``hermes kanban decompose [task_id | --all]`` and the
auto-decompose path in the gateway dispatcher loop. Reads the user's
profile roster (with descriptions) and asks the auxiliary LLM to
return a task graph in JSON. Then atomically creates the children,
links them under the root, and flips the root ``triage -> todo``.

The root task stays alive and becomes the parent of every leaf child,
so when the whole graph completes the root wakes back up — its
assignee (the orchestrator profile) gets a chance to judge completion
and add more tasks if the work isn't done yet.

Design notes
------------

* Mirrors the shape of ``hermes_cli/kanban_specify.py``: lazy aux
  client import inside the function, lenient response parse, never
  raises on expected failure modes.

* The system prompt sees the *configured* profile roster — names plus
  descriptions plus the default fallback. Profiles without a
  description are still listed (with a note) so the decomposer can
  match on name as a fallback, but the user has an obvious incentive
  to describe them.

* ``fanout=false`` collapses to the same effect as ``kanban specify``:
  we tighten the body and flip ``triage -> todo`` as a single task,
  no children created. This makes ``decompose`` a strict superset of
  ``specify`` from the user's perspective.

* If the LLM picks an assignee that doesn't exist as a profile, we
  rewrite it to the configured ``default_assignee`` (or the default
  profile if unset). A child task NEVER ends up with ``assignee=None``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli import profiles as profiles_mod
from hermes_cli.kanban_models import UNKNOWN as _LIVE_UNKNOWN
from hermes_cli.kanban_models import enumerate_live_models as _enumerate_live_models

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are the Kanban decomposer for the Hermes Agent board.

A user dropped a rough idea into the Triage column. Your job is to break it
into a small graph of concrete child tasks and route each one to the best-
matching profile from the available roster.

You will be given:
  - The original task title and body
  - The list of available profiles (each with name + description)
  - The fallback "default_assignee" used when no profile fits

Output a single JSON object with this exact shape:

  {
    "fanout": true,
    "rationale": "<one sentence on why this decomposition>",
    "tasks": [
      {
        "title": "<concrete task title, imperative voice, <= 80 chars>",
        "body":  "<detailed spec for the worker on this child task>",
        "assignee": "<profile name from the roster, or null for default>",
        "parents": [<int>, ...]
      },
      ...
    ]
  }

Rules:
  - "parents" is a list of INDICES (0-based) into this same "tasks" list,
    expressing actual data dependencies. Tasks with no parents run in
    PARALLEL. Tasks with parents wait until every parent completes.
  - Prefer parallelism. If two tasks can be done independently, give
    them no parents so the dispatcher fans them out at once.
  - Use 2-6 tasks for normal work. Don't create 20 tiny tasks. Don't
    cram everything into 1 task.
  - Pick assignees from the roster by matching the task to the profile's
    DESCRIPTION (not just the name). When nothing matches well, use null
    and the system will route to the default_assignee.
  - Each child task body is what a fresh worker will read with no other
    context — be specific about goal, approach, and acceptance criteria.

When the task is genuinely a single unit of work (no useful decomposition),
return:

  {
    "fanout": false,
    "rationale": "<one sentence>",
    "title": "<tightened title>",
    "body":  "<concrete spec for a single worker>",
    "assignee": "<profile name from the roster, or null for default>"
  }

In that case the task stays as one work item, just with a tightened spec and
a concrete assignee. If no profile fits, use null and the system will route to
the default_assignee.

No preamble, no closing remarks, no code fences. Output only the JSON object.
"""


_USER_TEMPLATE = """Task id: {task_id}
Title: {title}
Body:
{body}

Available profiles (assignees you may pick from):
{roster}

Default assignee (used when no profile fits a task): {default_assignee}
"""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class DecomposeOutcome:
    """Result of decomposing a single triage task."""

    task_id: str
    ok: bool
    reason: str = ""
    fanout: bool = False
    child_ids: list[str] | None = None
    new_title: Optional[str] = None
    # Resolved liveness-ignore settings (gh-78). Populated by decompose_task
    # so callers and the child-validation pass can see which endpoints were
    # opted out of the live-model probe.
    liveness_ignore: Optional[LivenessIgnore] = None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _extract_json_blob(raw: str) -> Optional[dict]:
    if not raw:
        return None
    stripped = _FENCE_RE.sub("", raw.strip())
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    candidate = stripped[first : last + 1]
    try:
        val = json.loads(candidate)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(val, dict):
        return None
    return val


def _profile_author() -> str:
    """Mirror of ``hermes_cli.kanban._profile_author``."""
    return (
        os.environ.get("HERMES_PROFILE")
        or os.environ.get("USER")
        or "decomposer"
    )


def _load_config() -> dict:
    try:
        from hermes_cli.config import load_config
        return load_config() or {}
    except Exception:
        return {}


def _resolve_orchestrator_profile(cfg: dict) -> str:
    """Resolve which profile owns the root/orchestration task after fan-out.

    Falls back to the active default profile when ``kanban.orchestrator_profile``
    is unset, so a task is never stranded for lack of an orchestrator.
    """
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    explicit = (kanban_cfg.get("orchestrator_profile") or "").strip()
    if explicit:
        try:
            if profiles_mod.profile_exists(explicit):
                return explicit
        except Exception:
            pass
    # Fall back to the active default profile.
    try:
        return profiles_mod.get_active_profile_name() or "default"
    except Exception:
        return "default"


def _resolve_default_assignee(cfg: dict) -> str:
    """Resolve which profile catches child tasks the orchestrator can't route."""
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    explicit = (kanban_cfg.get("default_assignee") or "").strip()
    if explicit:
        try:
            if profiles_mod.profile_exists(explicit):
                return explicit
        except Exception:
            pass
    try:
        return profiles_mod.get_active_profile_name() or "default"
    except Exception:
        return "default"


@dataclass(frozen=True)
class LivenessIgnore:
    """Config-driven opt-out for the decompose liveness check (gh-78).

    ``ignore_all`` (``kanban.decompose_ignore_liveness``) disables the
    live-model probe entirely — decomposition always proceeds, matching
    pre-feature behavior. ``ignored_base_urls``
    (``kanban.decompose_ignore_liveness_base_urls``) opts specific
    endpoints out: their models are treated as live without probing.
    """

    ignore_all: bool = False
    ignored_base_urls: frozenset[str] = frozenset()

    def ignores(self, base_url: str) -> bool:
        """True when ``base_url`` should skip the liveness probe."""
        if self.ignore_all:
            return True
        return base_url in self.ignored_base_urls


def _resolve_liveness_ignore(cfg: dict) -> LivenessIgnore:
    """Read the decompose liveness-ignore knobs from ``kanban`` config.

    Returns a :class:`LivenessIgnore` describing which endpoints (if any)
    should skip the live-model probe. Never raises: malformed config
    degrades to the default (probe everything).
    """
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    ignore_all = bool(kanban_cfg.get("decompose_ignore_liveness", False))
    raw_urls = kanban_cfg.get("decompose_ignore_liveness_base_urls") or []
    if not isinstance(raw_urls, list):
        raw_urls = []
    urls = frozenset(
        str(u).strip().rstrip("/")
        for u in raw_urls
        if isinstance(u, str) and u.strip()
    )
    return LivenessIgnore(ignore_all=ignore_all, ignored_base_urls=urls)


def _profile_model_and_base_urls(assignee: str) -> tuple[Optional[str], list[str]]:
    """Read an assignee profile's own config for ``(model, base_urls)``.

    ``model`` mirrors :func:`hermes_cli.profiles._read_config_model` (the
    ``model.default`` / ``model.model`` scalar) — the model a worker for this
    profile would actually run. ``base_urls`` are the profile's own
    custom-provider endpoints (canonical, trailing-slash-stripped). Reads the
    profile's ``config.yaml`` raw so a long-lived decomposer never flips
    ``HERMES_HOME`` to visit a named profile.

    Fail-soft: any read/config error degrades to ``(None, [])`` — the caller
    treats that as "cannot verify", which flags rather than drops the leg.
    """
    try:
        pdir = profiles_mod.get_profile_dir(assignee)
        from hermes_cli.config import get_compatible_custom_providers, read_user_config_raw
        cfg = read_user_config_raw(pdir / "config.yaml")
    except Exception:
        return None, []
    if not isinstance(cfg, dict):
        return None, []
    model: Optional[str] = None
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, str):
        model = model_cfg.strip() or None
    elif isinstance(model_cfg, dict):
        raw = model_cfg.get("default") or model_cfg.get("model")
        if isinstance(raw, str):
            model = raw.strip() or None
    base_urls: list[str] = []
    try:
        for entry in get_compatible_custom_providers(cfg) or []:
            if not isinstance(entry, dict):
                continue
            bu = str(entry.get("base_url", "") or "").strip().rstrip("/")
            if bu and bu not in base_urls:
                base_urls.append(bu)
    except Exception:
        pass
    return model, base_urls


def _liveness_verdict(
    model: Optional[str],
    base_urls: list[str],
    live_map: dict,
    liveness_ignore: LivenessIgnore,
) -> tuple[str, str]:
    """Decide ``('pass'|'flag'|'skip', reason)`` for one decomposition leg.

    * ``'pass'`` — the model is served by at least one assignee endpoint, or
      that endpoint is explicitly opted out of probing via
      ``decompose_ignore_liveness_base_urls`` (treated as live).
    * ``'skip'`` — every known assignee endpoint serves models but NOT this
      one: a worker would spawn against a model the runtime doesn't serve, so
      the leg must not be emitted.
    * ``'flag'`` — cannot verify (no endpoint or no model configured, or the
      only endpoints are UNKNOWN / the probe never ran). Fail-soft: keep the
      leg but annotate it as unverified.
    """
    if liveness_ignore.ignore_all:
        return "pass", ""
    if not base_urls:
        return "flag", "assignee declares no custom-provider endpoint to verify against"
    if not model:
        return "flag", "assignee has no configured model to verify"
    served_by_known = False
    unknown_endpoints: list[str] = []
    known_endpoints: list[str] = []
    for bu in base_urls:
        if liveness_ignore.ignores(bu):
            served_by_known = True  # explicit opt-out: treat as live, no probe
            continue
        state = live_map.get(bu, _LIVE_UNKNOWN)
        if state is _LIVE_UNKNOWN:
            unknown_endpoints.append(bu)
            continue
        known_endpoints.append(bu)
        if model in state:
            served_by_known = True
    if served_by_known:
        return "pass", ""
    if unknown_endpoints:
        return "flag", f"endpoint(s) UNKNOWN, liveness unverified (fail-soft): {', '.join(unknown_endpoints)}"
    return "skip", f"model {model!r} not served by {', '.join(known_endpoints) or 'no endpoints'}"


def _annotate_liveness(
    task_id: str,
    skipped: list[tuple[int, str, str]],
    flagged: list[tuple[int, str, str]],
    *,
    author: str,
) -> None:
    """Post one comment on the parent task summarizing the liveness gate.

    Runs best-effort after the fan-out so the root's history records that
    legs were withheld (or left unverified) — a human should not have to infer
    it from the child list being shorter than the LLM intended. Never raises:
    a comment is diagnostic, not load-bearing.
    """
    lines: list[str] = []
    if skipped:
        lines.append("Skipped by decompose liveness gate (model not served by assignee endpoint):")
        for idx, title, reason in skipped:
            lines.append(f"- [{idx}] {title}: {reason}")
    if flagged:
        lines.append("Kept but UNVERIFIED (fail-soft — endpoint/model could not be confirmed):")
        for idx, title, reason in flagged:
            lines.append(f"- [{idx}] {title}: {reason}")
    body = "\n".join(lines)
    if not body:
        return
    try:
        with kb.connect_closing() as conn:
            kb.add_comment(conn, task_id, author, body)
    except Exception as exc:
        logger.warning("decompose: failed to annotate parent %s: %s", task_id, exc)


def _build_roster() -> tuple[list[dict], set[str]]:
    """Return (roster_for_prompt, valid_assignee_names).

    Each roster entry is ``{name, description, has_description}``. The
    valid-set is used after the LLM responds to rewrite invalid
    assignees to the default fallback.
    """
    roster: list[dict] = []
    valid: set[str] = set()
    try:
        all_profiles = profiles_mod.list_profiles()
    except Exception as exc:
        logger.warning("decompose: failed to list profiles: %s", exc)
        return roster, valid
    for p in all_profiles:
        desc = (p.description or "").strip()
        roster.append({
            "name": p.name,
            "description": desc or f"(no description; profile named {p.name!r})",
            "has_description": bool(desc),
        })
        valid.add(p.name)
    return roster, valid


def _format_roster(roster: list[dict]) -> str:
    if not roster:
        return "  (no profiles installed — decomposer cannot route work)"
    lines = []
    for entry in roster:
        tag = "" if entry["has_description"] else " ⚠ undescribed"
        lines.append(f"  - {entry['name']}{tag}: {entry['description']}")
    return "\n".join(lines)


def _normalize_assignee_choice(
    assignee: object,
    *,
    default_assignee: str,
    valid_names: set[str],
) -> str:
    """Return a valid assignee, falling back to ``default_assignee``.

    Fan-out children and the single-task fallback should share the same
    routing guarantee: promoted work must not be left unassigned.
    """
    if not isinstance(assignee, str) or not assignee.strip():
        return default_assignee
    chosen = assignee.strip()
    if chosen not in valid_names:
        return default_assignee
    return chosen


def decompose_task(
    task_id: str,
    *,
    author: Optional[str] = None,
    timeout: Optional[int] = None,
    auto_promote: Optional[bool] = None,
) -> DecomposeOutcome:
    """Decompose a triage task into a graph of child tasks.

    Returns an outcome describing what happened. Never raises for
    expected failure modes (task not in triage, no aux client
    configured, API error, malformed response, decomposer returned
    fanout=true with empty task list) — those surface via ``ok=False``.

    ``auto_promote``:
        - ``None`` — read ``kanban.auto_promote_children`` (default True).
          Children move straight to ``ready`` and the dispatcher runs them.
        - ``False`` — children stay in ``todo`` (held for manual
          review/release). Used by the approval-gated auto-decompose
          path (``kanban.auto_decompose_require_approval``) so an
          unsupervised fan-out never spawns workers until a human
          ``hermes kanban approve`` them.
        - ``True`` — force immediate promotion regardless of config.
    """
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
    if task is None:
        return DecomposeOutcome(task_id, False, "unknown task id")
    if task.status != "triage":
        return DecomposeOutcome(
            task_id, False, f"task is not in triage (status={task.status!r})"
        )

    cfg = _load_config()
    # Liveness-ignore knob (gh-78): read once so the child-validation pass
    # (and any caller inspecting the outcome) can consult it. When
    # decompose_ignore_liveness is true, or a child's endpoint is in
    # decompose_ignore_liveness_base_urls, the live-model probe is skipped
    # for that leg.
    liveness_ignore = _resolve_liveness_ignore(cfg)
    orchestrator = _resolve_orchestrator_profile(cfg)
    default_assignee = _resolve_default_assignee(cfg)
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    if auto_promote is None:
        auto_promote = bool(kanban_cfg.get("auto_promote_children", True))
    roster, valid_names = _build_roster()

    try:
        from agent.auxiliary_client import call_llm  # type: ignore
    except Exception as exc:
        logger.debug("decompose: auxiliary client import failed: %s", exc)
        return DecomposeOutcome(task_id, False, "auxiliary client unavailable")

    user_msg = _USER_TEMPLATE.format(
        task_id=task.id,
        title=_truncate(task.title or "", 400),
        body=_truncate(task.body or "(no body)", 4000),
        roster=_format_roster(roster),
        default_assignee=default_assignee,
    )

    try:
        # Route through call_llm so auxiliary.kanban_decomposer.* config
        # (provider/model/base_url, extra_body, reasoning_effort, retries)
        # all apply — the previous direct client.chat.completions.create()
        # path dropped auxiliary.<task>.extra_body entirely (#35566).
        resp = call_llm(
            task="kanban_decomposer",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=4000,
            timeout=timeout or 180,
        )
    except Exception as exc:
        logger.info(
            "decompose: API call failed for %s (%s)", task_id, exc,
        )
        return DecomposeOutcome(task_id, False, f"LLM error: {type(exc).__name__}")

    try:
        raw = resp.choices[0].message.content or ""
    except Exception:
        raw = ""

    parsed = _extract_json_blob(raw)
    if parsed is None:
        return DecomposeOutcome(task_id, False, "LLM returned malformed JSON")

    fanout = bool(parsed.get("fanout"))
    audit_author = author or _profile_author()

    if not fanout:
        # Fall back to single-task spec promotion (same effect as specify).
        new_title = parsed.get("title")
        new_body = parsed.get("body")
        title_val = new_title.strip() if isinstance(new_title, str) and new_title.strip() else None
        body_val = new_body if isinstance(new_body, str) and new_body.strip() else None
        assignee_val = None
        if not task.assignee:
            assignee_val = _normalize_assignee_choice(
                parsed.get("assignee"),
                default_assignee=default_assignee,
                valid_names=valid_names,
            )
        if title_val is None and body_val is None:
            return DecomposeOutcome(
                task_id, False, "decomposer returned fanout=false with no title/body",
            )
        with kb.connect_closing() as conn:
            ok = kb.specify_triage_task(
                conn,
                task_id,
                title=title_val,
                body=body_val,
                assignee=assignee_val,
                author=audit_author,
            )
        if not ok:
            return DecomposeOutcome(
                task_id, False, "task moved out of triage before promotion",
            )
        return DecomposeOutcome(
            task_id, True, "single task (no fanout)",
            fanout=False, new_title=title_val,
            liveness_ignore=liveness_ignore,
        )

    raw_tasks = parsed.get("tasks") or []
    if not isinstance(raw_tasks, list) or not raw_tasks:
        return DecomposeOutcome(
            task_id, False, "decomposer returned fanout=true with empty tasks list",
        )

    # Rewrite invalid assignees to the default fallback. Never leave a
    # task with assignee=None — the user explicitly does not want that.
    children: list[dict] = []
    for idx, entry in enumerate(raw_tasks):
        if not isinstance(entry, dict):
            return DecomposeOutcome(
                task_id, False, f"tasks[{idx}] is not an object",
            )
        title = entry.get("title")
        if not isinstance(title, str) or not title.strip():
            return DecomposeOutcome(
                task_id, False, f"tasks[{idx}].title is missing or empty",
            )
        body = entry.get("body")
        if not isinstance(body, str):
            body = ""
        assignee = entry.get("assignee")
        chosen = _normalize_assignee_choice(
            assignee,
            default_assignee=default_assignee,
            valid_names=valid_names,
        )
        if (
            isinstance(assignee, str)
            and assignee.strip()
            and assignee.strip() not in valid_names
        ):
            logger.info(
                "decompose: task %s child %d picked unknown assignee %r — "
                "routing to default_assignee %r",
                task_id, idx, assignee, default_assignee,
            )
        parents = entry.get("parents") or []
        if not isinstance(parents, list):
            parents = []
        # Clean parent indices: drop non-int and out-of-range.
        clean_parents = [p for p in parents if isinstance(p, int) and 0 <= p < len(raw_tasks) and p != idx]
        children.append({
            "title": title.strip()[:200],
            "body": body.strip(),
            "assignee": chosen,
            "parents": clean_parents,
        })

    # ------------------------------------------------------------------
    # Liveness gate (gh-78): only emit legs whose worker model is actually
    # served. Probe the live model inventory once, then for each child
    # resolve its assignee profile's endpoint + model and drop legs whose
    # model is definitively absent — a worker would spawn against a model
    # the runtime doesn't serve. Legs we cannot verify (UNKNOWN endpoint,
    # no endpoint, no model) are kept but FLAGGED (fail-soft: a down probe
    # must never abort wiring work). Parent indices are remapped to the
    # surviving legs so a child whose prerequisite was dropped becomes a
    # parallel leaf instead of a dangling reference. The whole pass is
    # skipped when the decompose_ignore_liveness knob disables it.
    # ------------------------------------------------------------------
    skipped: list[tuple[int, str, str]] = []
    flagged: list[tuple[int, str, str]] = []
    if liveness_ignore.ignore_all or not children:
        survivors = children
    else:
        try:
            live_map = _enumerate_live_models()
        except Exception as exc:
            # Probe infrastructure down: fail soft — keep everything, flag.
            logger.warning("decompose: liveness probe unavailable for %s (%s)", task_id, exc)
            live_map = {}
        survivors = []
        keep_orig: list[int] = []
        for orig_idx, child in enumerate(children):
            model, base_urls = _profile_model_and_base_urls(child["assignee"])
            action, reason = _liveness_verdict(model, base_urls, live_map, liveness_ignore)
            title = child["title"]
            if action == "skip":
                skipped.append((orig_idx, title, reason))
                logger.info("decompose: liveness gate drops leg %d (%r): %s", orig_idx, title, reason)
                continue
            survivors.append(child)
            keep_orig.append(orig_idx)
            if action == "flag":
                flagged.append((orig_idx, title, reason))
        old_to_new = {orig: new for new, orig in enumerate(keep_orig)}
        for child in survivors:
            child["parents"] = [old_to_new[p] for p in child.get("parents", []) if p in old_to_new]

    # Record what the gate withheld/left unverified on the parent, so the
    # root's history explains a child list that is shorter than intended.
    if skipped or flagged:
        _annotate_liveness(task_id, skipped, flagged, author=audit_author)

    # Every leg was dropped: nothing to emit and no graph to write. Leave the
    # root in triage so it surfaces for a human (or a fresh decompose after
    # the models are back) rather than silently vanishing.
    if not survivors:
        return DecomposeOutcome(
            task_id, False,
            f"decomposed into 0 children — every leg skipped by liveness gate "
            f"({len(skipped)} skipped, {len(flagged)} unverified)",
        )

    try:
        with kb.connect_closing() as conn:
            child_ids = kb.decompose_triage_task(
                conn,
                task_id,
                root_assignee=orchestrator,
                children=survivors,
                author=audit_author,
                auto_promote=auto_promote,
            )
    except ValueError as exc:
        return DecomposeOutcome(task_id, False, f"DB rejected graph: {exc}")
    except Exception as exc:
        logger.exception("decompose: DB error on task %s", task_id)
        return DecomposeOutcome(task_id, False, f"DB error: {type(exc).__name__}")

    if child_ids is None:
        return DecomposeOutcome(
            task_id, False, "task moved out of triage before decomposition",
        )

    return DecomposeOutcome(
        task_id, True, f"decomposed into {len(child_ids)} children",
        fanout=True, child_ids=child_ids,
        liveness_ignore=liveness_ignore,
    )


def list_triage_ids(*, tenant: Optional[str] = None) -> list[str]:
    """Return task ids currently in the triage column."""
    with kb.connect_closing() as conn:
        rows = kb.list_tasks(
            conn,
            status="triage",
            tenant=tenant,
            limit=1000,
        )
    return [row.id for row in rows]
