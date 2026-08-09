"""Tests for the Spark real-tokenizer path (issue #55 Layers 2+3).

Covers:
  * ``count_fn`` wiring on the estimate functions (leaf + request level).
  * ``agent.spark_tokenizer`` endpoint gating, LRU cache, and fallback.
  * the ``context_length - actual_input - SAFETY_BUFFER`` output budget.
"""
from unittest.mock import patch

from agent.model_metadata import (
    estimate_messages_tokens_rough,
    estimate_request_tokens_rough,
    estimate_tokens_rough,
)
from agent import spark_tokenizer


# ── count_fn wiring on the estimate functions ────────────────────────────


def test_estimate_tokens_rough_uses_count_fn_when_provided():
    states = {"calls": 0}

    def fake_count(text):
        states["calls"] += 1
        return 7

    assert estimate_tokens_rough("hello world", count_fn=fake_count) == 7
    assert states["calls"] == 1


def test_estimate_tokens_rough_falls_back_on_count_fn_error():
    def broken(text):
        raise ConnectionError("tokenizer down")

    # chars/4 ceiling of "abcdefgh" (8 chars) == 2
    assert estimate_tokens_rough("abcdefgh", count_fn=broken) == 2


def test_estimate_messages_rough_forwards_count_fn_and_images_flat():
    calls = []

    def fake_count(text):
        calls.append(text)
        return 5

    messages = [
        {"role": "user", "content": "plain text"},
        {"role": "user", "content": [{"type": "text", "text": "part"},
                                     {"type": "image", "image": "[stripped]"}]},
    ]
    total = estimate_messages_tokens_rough(messages, count_fn=fake_count)
    # two text-bearing messages -> 5 + 5, image adds the flat 1500 cost
    assert total == 5 + 5 + 1500
    assert len(calls) == 2


def test_estimate_request_rough_forwards_count_fn_all_buckets():
    calls = []

    def fake_count(text):
        calls.append(str(text))
        return 3

    messages = [{"role": "user", "content": "prompt"}]
    tools = [{"type": "function",
              "function": {"name": "f", "description": "d",
                           "parameters": {"type": "object"}}}]
    total = estimate_request_tokens_rough(
        messages, system_prompt="sys", tools=tools, count_fn=fake_count
    )
    # system + one message + one tool
    assert total == 3 + 3 + 3
    assert len(calls) == 3


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


# ── count_tokens: cache + fallback ───────────────────────────────────────


def test_count_tokens_falls_back_when_tokenize_unavailable():
    with patch.object(spark_tokenizer, "_tokenize", return_value=None):
        spark_tokenizer._TOKENS_CACHE.clear()
        # "abcdefgh" -> chars/4 ceiling == 2
        assert spark_tokenizer.count_tokens("abcdefgh") == 2


def test_count_tokens_caches_and_hits():
    with patch.object(spark_tokenizer, "_tokenize", return_value=11) as tok:
        spark_tokenizer._TOKENS_CACHE.clear()
        assert spark_tokenizer.count_tokens("tool-heavy payload") == 11
        assert spark_tokenizer.count_tokens("tool-heavy payload") == 11
        # second call is a cache hit — no extra network
        assert tok.call_count == 1


# ── Layer 3 output budget ────────────────────────────────────────────────


def test_safety_budget_formula():
    with patch.object(spark_tokenizer, "_tokenize", return_value=100):
        spark_tokenizer._TOKENS_CACHE.clear()

        def counter(text):
            return spark_tokenizer.count_tokens(text)

        messages = [{"role": "user", "content": "tool-heavy payload"}]
        budget, real_input = spark_tokenizer.safety_budget(
            131072, counter, messages
        )
        # 131072 - 100 - 2048, real_input rounded up to the per-message count
        assert real_input >= 100
        assert budget == 131072 - real_input - spark_tokenizer.SAFETY_BUFFER


def test_safety_budget_none_when_context_unknown():
    with patch.object(spark_tokenizer, "_tokenize", return_value=10):
        spark_tokenizer._TOKENS_CACHE.clear()

        def counter(text):
            return spark_tokenizer.count_tokens(text)

        budget, real_input = spark_tokenizer.safety_budget(
            0, counter, [{"role": "user", "content": "x"}]
        )
        assert budget is None
        assert real_input >= 1