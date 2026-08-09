"""Tests for the Spark real-tokenizer path (issue #55 Layers 2+3).

Covers:
  * ``agent.spark_tokenizer`` endpoint gating, LRU cache, and fallback.
  * the full-request ``count_request`` primitive (the shape that matches
    vLLM's ``usage.prompt_tokens``).
  * the ``context_length - actual_input - SAFETY_BUFFER`` output budget.
"""
from unittest.mock import patch

from agent.model_metadata import estimate_tokens_rough
from agent import spark_tokenizer


# ── spark_tokenizer gating ───────────────────────────────────────────────


def test_is_spark_endpoint_matches_dgx_url_and_model():
    assert spark_tokenizer.is_spark_endpoint("http://100.122.146.36:8888/v1")
    assert spark_tokenizer.is_spark_endpoint(
        "", "deepseek-v4-flash-dspark"
    )
    assert not spark_tokenizer.is_spark_endpoint("")
    assert not spark_tokenizer.is_spark_endpoint(
        "https://api.openai.com/v1", "gpt-5"
    )


# ── count_tokens: single-string primitive, cache + fallback ─────────────


def test_count_tokens_falls_back_when_tokenize_unavailable():
    with patch.object(spark_tokenizer, "_tokenize_payload", return_value=None):
        spark_tokenizer._TOKENS_CACHE.clear()
        # "abcdefgh" -> chars/4 ceiling == 2
        assert spark_tokenizer.count_tokens("abcdefgh") == 2


def test_count_tokens_caches_and_hits():
    with patch.object(spark_tokenizer, "_tokenize_payload", return_value=11) as tok:
        spark_tokenizer._TOKENS_CACHE.clear()
        assert spark_tokenizer.count_tokens("tool-heavy payload") == 11
        assert spark_tokenizer.count_tokens("tool-heavy payload") == 11
        # second call is a cache hit — no extra network
        assert tok.call_count == 1


# ── count_request: full-request primitive (the authoritative count) ─────


def test_count_request_sends_full_payload_with_tools_and_add_gen():
    """The /tokenize body must carry messages + tools + add_generation_prompt,
    the only shape that reproduces vLLM's usage.prompt_tokens."""
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function",
              "function": {"name": "f", "description": "d",
                           "parameters": {"type": "object"}}}]
    captured = {}

    def fake(body):
        captured.update(body)
        return 7

    with patch.object(spark_tokenizer, "_tokenize_payload", side_effect=fake) as tok:
        spark_tokenizer._TOKENS_CACHE.clear()
        assert spark_tokenizer.count_request(messages, tools=tools) == 7
        assert captured["messages"] == messages
        assert captured["tools"] == tools
        assert captured["add_generation_prompt"] is True
        assert tok.call_count == 1


def test_count_request_caches_by_payload():
    messages = [{"role": "user", "content": "same payload"}]
    with patch.object(spark_tokenizer, "_tokenize_payload", return_value=99) as tok:
        spark_tokenizer._TOKENS_CACHE.clear()
        assert spark_tokenizer.count_request(messages) == 99
        assert spark_tokenizer.count_request(messages) == 99
        assert tok.call_count == 1


def test_count_request_different_payload_misses_cache():
    with patch.object(spark_tokenizer, "_tokenize_payload", return_value=5) as tok:
        spark_tokenizer._TOKENS_CACHE.clear()
        spark_tokenizer.count_request([{"role": "user", "content": "a"}])
        spark_tokenizer.count_request([{"role": "user", "content": "bb"}])
        assert tok.call_count == 2


def test_count_request_falls_back_to_rough_estimate_on_tokenize_failure():
    """Endpoint down → count via the chars/4 rough estimate (identical to the
    non-Spark behaviour the caller would otherwise see)."""
    from agent.model_metadata import estimate_request_tokens_rough
    messages = [{"role": "user", "content": "hello world"}]
    with patch.object(spark_tokenizer, "_tokenize_payload", return_value=None):
        spark_tokenizer._TOKENS_CACHE.clear()
        result = spark_tokenizer.count_request(messages)
        expected = int(estimate_request_tokens_rough(messages))
        assert result == expected
        assert result > 0


def test_count_request_without_tools_omits_tools_key():
    with patch.object(spark_tokenizer, "_tokenize_payload", return_value=4) as tok:
        spark_tokenizer._TOKENS_CACHE.clear()
        spark_tokenizer.count_request([{"role": "user", "content": "x"}])
        body = tok.call_args[0][0]
        assert "tools" not in body


# ── Layer 3 output budget ────────────────────────────────────────────────


def test_safety_budget_formula():
    with patch.object(spark_tokenizer, "_tokenize_payload", return_value=100):
        spark_tokenizer._TOKENS_CACHE.clear()
        messages = [{"role": "user", "content": "tool-heavy payload"}]
        budget, real_input = spark_tokenizer.safety_budget(
            131072, messages
        )
        # 131072 - 100 - 2048, real_input is the raw endpoint count
        assert real_input == 100
        assert budget == 131072 - 100 - spark_tokenizer.SAFETY_BUFFER


def test_safety_budget_none_when_context_unknown():
    with patch.object(spark_tokenizer, "_tokenize_payload", return_value=10):
        spark_tokenizer._TOKENS_CACHE.clear()
        budget, real_input = spark_tokenizer.safety_budget(
            0, [{"role": "user", "content": "x"}]
        )
        assert budget is None
        assert real_input == 10


# ── Integration: compute_spark_output_budget caps max_output ────────────


def test_spark_budget_caps_max_output_with_real_count():
    """On the Spark endpoint, the output cap is
    context_length - actual_input - SAFETY_BUFFER using the real count."""
    from types import SimpleNamespace
    from agent.chat_completion_helpers import compute_spark_output_budget

    agent = SimpleNamespace(
        base_url="http://100.122.146.36:8888/v1",
        model="deepseek-v4-flash-dspark",
        max_tokens=8192,
        context_compressor=SimpleNamespace(context_length=131072),
    )
    real_input = 125_000
    with patch.object(spark_tokenizer, "_tokenize_payload", return_value=real_input):
        spark_tokenizer._TOKENS_CACHE.clear()
        is_spark, effective, _ri = compute_spark_output_budget(
            agent, [{"role": "user", "content": "tool-heavy"}], None
        )
    assert is_spark is True
    # Budget = 131072 - 125000 - 2048 = 4024 (< 8192 cap, so it binds)
    expected = 131072 - real_input - spark_tokenizer.SAFETY_BUFFER
    assert expected < 8192
    assert effective == expected
    assert _ri == real_input


def test_spark_budget_never_grows_beyond_configured_cap():
    from types import SimpleNamespace
    from agent.chat_completion_helpers import compute_spark_output_budget

    agent = SimpleNamespace(
        base_url="http://100.122.146.36:8888/v1",
        model="deepseek-v4-flash-dspark",
        max_tokens=8192,
        context_compressor=SimpleNamespace(context_length=131072),
    )
    # Tiny input -> huge budget; must clamp to the configured 8192 cap.
    with patch.object(spark_tokenizer, "_tokenize_payload", return_value=10):
        spark_tokenizer._TOKENS_CACHE.clear()
        _is, effective, _ = compute_spark_output_budget(
            agent, [{"role": "user", "content": "x"}], None
        )
    assert effective == 8192


def test_spark_budget_non_spark_unchanged():
    from types import SimpleNamespace
    from agent.chat_completion_helpers import compute_spark_output_budget

    agent = SimpleNamespace(
        base_url="https://api.openai.com/v1",
        model="gpt-5",
        max_tokens=8192,
        context_compressor=SimpleNamespace(context_length=400000),
    )
    is_spark, effective, _ri = compute_spark_output_budget(
        agent, [{"role": "user", "content": "hi"}], None
    )
    assert is_spark is False
    assert effective == 8192
    assert _ri is None


def test_build_api_kwargs_consults_spark_budget():
    """build_api_kwargs delegates the Spark budget to compute_spark_output_budget
    and stashes the accurate count on the agent for the reactive retry path."""
    import agent.chat_completion_helpers as cch
    from unittest.mock import MagicMock

    real_input = 125_000
    agent = MagicMock()
    agent.base_url = "http://100.122.146.36:8888/v1"
    agent.model = "deepseek-v4-flash-dspark"
    agent.max_tokens = 8192
    agent.api_mode = "openai"
    agent.context_compressor.context_length = 131072

    with patch.object(
        cch, "compute_spark_output_budget", return_value=(True, 4024, real_input)
    ) as budget:
        with patch.object(cch, "_provider_preferences_for_agent", return_value={}):
            with patch("providers.get_provider_profile", return_value=None):
                cch.build_api_kwargs(
                    agent, [{"role": "user", "content": "x"}], tools_for_api=[]
                )
    assert budget.call_count == 1
    # The accurate input count from the helper was stashed for Layer 2.
    assert getattr(agent, "_last_spark_input_tokens", None) == real_input


def test_estimate_tokens_rough_unchanged_without_stargate():
    """The heuristic itself is untouched on non-Spark paths."""
    # "abcdefgh" (8 chars) -> chars/4 ceiling == 2
    assert estimate_tokens_rough("abcdefgh") == 2