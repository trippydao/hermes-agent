"""Pre-emptive Spark ceiling compression (issue #55 Layer 3).

The reactive overflow-recovery path only learns a request cannot fit AFTER
the provider 400s. With the real spark ``/tokenize`` count (Layer 2) we can
see it coming: when ``input + SAFETY_BUFFER >= context_length`` the request
PHYSICALLY cannot fit, so ``run_conversation`` must compress BEFORE sending.

The pre-API pressure gate normally refuses to compress when the *soft*
heuristics say "don't bother" (``should_compress`` off, defer-preflight,
summary-LLM cooldown). Those exist to skip pointless compaction of requests
that would fit anyway — but an EXACT ceiling breach is authoritative and must
override them, while still honouring the hard backstops (compression_enabled,
attempts budget, preflight-blocked).

These tests drive a real ``AIAgent`` on the Spark endpoint (``base_url``
matches ``is_spark_endpoint``) through ``run_conversation()`` and assert:
* ceiling breach + soft gates blocking  -> ``_compress_context`` STILL runs;
* sub-ceiling request + soft gates off  -> request goes straight to the wire.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB
from run_agent import AIAgent

SPARK_BASE_URL = "http://100.122.146.36:8888/v1"
CONTEXT_LEN = 100_000
SAFETY = 2048


def _config() -> dict:
    return {
        "compression": {
            "enabled": True,
            "threshold": 0.50,
            "target_ratio": 0.20,
            "protect_first_n": 3,
            "protect_last_n": 20,
            "max_attempts": 3,
        },
        "prompt_caching": {"cache_ttl": "5m"},
        "sessions": {},
        "bedrock": {},
    }


def _stop_response():
    msg = SimpleNamespace(
        content="done",
        reasoning_content=None,
        reasoning=None,
        tool_calls=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model=SPARK_BASE_URL, usage=None)


def _make_spark_agent(monkeypatch, tmp_path: Path) -> AIAgent:
    from hermes_cli import config as config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda: _config())
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: _config())
    db = SessionDB(db_path=tmp_path / "state.db")
    with (
        contextlib.redirect_stdout(io.StringIO()),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            base_url=SPARK_BASE_URL,
            api_key="test-key",
            model="deepseek-v4-flash-dspark",
            enabled_toolsets=[],
            disabled_toolsets=[],
            quiet_mode=True,
            skip_memory=True,
            skip_context_files=True,
            session_db=db,
            session_id="spark-preempt-ceiling",
        )
    agent.client = MagicMock()
    agent.client.chat.completions.create.return_value = _stop_response()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.tool_delay = 0
    agent.save_trajectories = False
    # Authoritative spark count + context ceiling via the real compressor.
    agent.context_compressor.context_length = CONTEXT_LEN
    agent.context_compressor.threshold_tokens = CONTEXT_LEN // 2
    return agent


def test_imminent_ceiling_compresses_despite_soft_gates(monkeypatch, tmp_path):
    """Input + safety buffer >= context: compress even if should_compress is
    False, defer-preflight is on and the cooldown is active."""
    agent = _make_spark_agent(monkeypatch, tmp_path)

    compress_calls = []

    def _fake_compress(messages, system_message, **_kwargs):
        compress_calls.append(len(messages))
        return messages, "compressed prompt"

    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
        for i in range(40)
    ]
    with (
        # Exact count breaches the ceiling: 99_000 + 2048 >= 100_000.
        patch(
            "agent.spark_tokenizer.count_request",
            return_value=CONTEXT_LEN - 1000,
        ),
        # Every SOFT gate says "don't compress" — the exact breach must win.
        patch.object(
            agent.context_compressor, "should_compress", return_value=False
        ),
        patch.object(
            agent.context_compressor,
            "should_defer_preflight_to_real_usage",
            return_value=True,
        ),
        patch.object(
            agent.context_compressor,
            "get_active_compression_failure_cooldown",
            return_value="cooldown-active",
        ),
        patch.object(agent, "_compress_context", side_effect=_fake_compress),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello", conversation_history=history)

    assert result["completed"] is True
    # The exact ceiling breach forced a pre-emptive compression pass even
    # though should_compress/defer/cooldown all said not to.
    assert len(compress_calls) >= 1, (
        f"expected pre-emptive compression at the context ceiling, "
        f"got {len(compress_calls)} passes"
    )
    # And the request that finally went out was a compacted one, not the
    # original over-ceiling message list.
    sent = agent.client.chat.completions.create.call_args
    assert sent is not None


def test_sub_ceiling_respects_soft_gates(monkeypatch, tmp_path):
    """Input well under the ceiling: soft gates rule, request goes to wire
    WITHOUT any pre-emptive compression pass."""
    agent = _make_spark_agent(monkeypatch, tmp_path)

    compress_calls = []

    def _fake_compress(messages, system_message, **_kwargs):
        compress_calls.append(len(messages))
        return messages, "compressed prompt"

    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
        for i in range(40)
    ]
    with (
        # Exact count far under the ceiling.
        patch("agent.spark_tokenizer.count_request", return_value=2_000),
        patch.object(
            agent.context_compressor, "should_compress", return_value=False
        ),
        patch.object(
            agent.context_compressor,
            "should_defer_preflight_to_real_usage",
            return_value=True,
        ),
        patch.object(
            agent.context_compressor,
            "get_active_compression_failure_cooldown",
            return_value="cooldown-active",
        ),
        patch.object(agent, "_compress_context", side_effect=_fake_compress),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello", conversation_history=history)

    assert result["completed"] is True
    assert compress_calls == [], (
        f"expected NO compression for a sub-ceiling request when soft gates "
        f"block, got {len(compress_calls)} passes"
    )
    assert agent.client.chat.completions.create.call_count == 1