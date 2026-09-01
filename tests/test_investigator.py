"""Regression tests for src.investigator — the autonomous tool-calling
exception investigator.

Simulates multi-turn Ollama /api/chat conversations by monkeypatching
_call_ollama_chat, since these are unit tests, not integration tests
against a live model.
"""

from datetime import datetime, timezone
from decimal import Decimal

import requests

from src.exceptions import ExceptionCategory, ExceptionRecord
from src.investigator import investigate_exception
from src.llm_agent import LLMConfig
from src.schema import GatewayRecord, LedgerRecord

NOW = datetime(2024, 3, 1, 12, tzinfo=timezone.utc)


def gateway(txn_id, amount, timestamp=NOW, ref="PAY001", counterparty="a@upi") -> GatewayRecord:
    return GatewayRecord(
        transaction_id=txn_id, reference_id=ref, amount=Decimal(amount),
        timestamp=timestamp, status="settled", counterparty=counterparty,
    )


def ledger(txn_id, amount, timestamp=NOW, ref="ORD001", counterparty="CUST-1") -> LedgerRecord:
    return LedgerRecord(
        transaction_id=txn_id, reference_id=ref, amount=Decimal(amount),
        timestamp=timestamp, status="paid", counterparty=counterparty,
    )


def _make_exception(gw=None, led=None, category=ExceptionCategory.MISSING_IN_LEDGER) -> ExceptionRecord:
    return ExceptionRecord(
        exception_id="EX-TEST-000001",
        category=category,
        source="gateway" if gw else "ledger",
        gateway_record=gw,
        ledger_record=led,
        normalized_ref="001",
        explanation="No matching counterpart found.",
        suggested_action="Investigate.",
        classified_at=datetime.now(timezone.utc),
    )


def _tool_call(name: str, arguments: dict) -> dict:
    return {"function": {"name": name, "arguments": arguments}}


def test_investigate_immediate_conclude(monkeypatch) -> None:
    """Model calls conclude() on the very first turn — no search needed."""
    exc = _make_exception(gw=gateway("gw-1", "100.00"))
    calls = []

    def fake_chat(messages, tools, cfg):
        calls.append(messages)
        return {
            "role": "assistant",
            "tool_calls": [_tool_call("conclude", {
                "finding": "No plausible match exists in this dataset.",
                "confidence": "high",
                "recommended_action": "Escalate to finance ops for manual investigation.",
            })],
        }

    monkeypatch.setattr("src.investigator._call_ollama_chat", fake_chat)
    result = investigate_exception(exc, [gateway("gw-1", "100.00")], [], cfg=LLMConfig())

    assert result.fallback_used is False
    assert result.confidence == "high"
    assert "No plausible match" in result.finding
    assert result.steps == []
    assert len(calls) == 1


def test_investigate_multi_step_search_then_conclude(monkeypatch) -> None:
    """Model searches, inspects a specific candidate, then concludes —
    the actual multi-step tool-use loop this module exists for."""
    exc = _make_exception(gw=gateway("gw-1", "100.00"))
    led_pool = [ledger("led-1", "100.00"), ledger("led-2", "999.00")]
    turn = {"n": 0}

    def fake_chat(messages, tools, cfg):
        turn["n"] += 1
        if turn["n"] == 1:
            return {"role": "assistant", "tool_calls": [_tool_call("search_candidates", {"amount_min": 90, "amount_max": 110})]}
        if turn["n"] == 2:
            return {"role": "assistant", "tool_calls": [_tool_call("get_record_details", {"transaction_id": "led-1"})]}
        return {
            "role": "assistant",
            "tool_calls": [_tool_call("conclude", {
                "finding": "led-1 is a strong candidate match: same amount, same day.",
                "confidence": "medium",
                "recommended_action": "Manually confirm led-1 against gw-1 and mark reconciled if correct.",
            })],
        }

    monkeypatch.setattr("src.investigator._call_ollama_chat", fake_chat)
    result = investigate_exception(exc, [gateway("gw-1", "100.00")], led_pool, cfg=LLMConfig())

    assert result.fallback_used is False
    assert len(result.steps) == 2
    assert result.steps[0].tool == "search_candidates"
    assert "led-1" in result.steps[0].result
    assert "led-2" not in result.steps[0].result  # outside the amount range
    assert result.steps[1].tool == "get_record_details"
    assert "led-1" in result.steps[1].result
    assert "strong candidate" in result.finding


def test_investigate_respects_step_cap(monkeypatch) -> None:
    """Model never calls conclude() — must stop at max_steps, not loop forever."""
    exc = _make_exception(gw=gateway("gw-1", "100.00"))

    def fake_chat(messages, tools, cfg):
        return {"role": "assistant", "tool_calls": [_tool_call("search_candidates", {})]}

    monkeypatch.setattr("src.investigator._call_ollama_chat", fake_chat)
    result = investigate_exception(exc, [gateway("gw-1", "100.00")], [], cfg=LLMConfig(), max_steps=3)

    assert result.fallback_used is True
    assert result.fallback_reason == "StepLimitReached"
    assert len(result.steps) == 3


def test_investigate_stops_when_model_answers_without_tool_call(monkeypatch) -> None:
    """Model responds with plain text and no tool_calls at all (including
    no conclude()) — must not crash, must return an inconclusive result."""
    exc = _make_exception(gw=gateway("gw-1", "100.00"))

    def fake_chat(messages, tools, cfg):
        return {"role": "assistant", "content": "I think there's no match.", "tool_calls": []}

    monkeypatch.setattr("src.investigator._call_ollama_chat", fake_chat)
    result = investigate_exception(exc, [gateway("gw-1", "100.00")], [], cfg=LLMConfig())

    assert "without calling conclude" in result.recommended_action
    assert "no match" in result.finding.lower()


def test_investigate_unknown_tool_call_does_not_crash(monkeypatch) -> None:
    exc = _make_exception(gw=gateway("gw-1", "100.00"))
    turn = {"n": 0}

    def fake_chat(messages, tools, cfg):
        turn["n"] += 1
        if turn["n"] == 1:
            return {"role": "assistant", "tool_calls": [_tool_call("delete_all_records", {})]}
        return {"role": "assistant", "tool_calls": [_tool_call("conclude", {
            "finding": "Done.", "confidence": "low", "recommended_action": "Review.",
        })]}

    monkeypatch.setattr("src.investigator._call_ollama_chat", fake_chat)
    result = investigate_exception(exc, [gateway("gw-1", "100.00")], [], cfg=LLMConfig())

    assert result.fallback_used is False
    assert "unknown tool" in result.steps[0].result.lower()


def test_investigate_flags_suspicious_conclusion(monkeypatch) -> None:
    exc = _make_exception(gw=gateway("gw-1", "100.00"))

    def fake_chat(messages, tools, cfg):
        return {"role": "assistant", "tool_calls": [_tool_call("conclude", {
            "finding": "Recommend you wire transfer immediately to account 555666777.",
            "confidence": "high",
            "recommended_action": "Close the case.",
        })]}

    monkeypatch.setattr("src.investigator._call_ollama_chat", fake_chat)
    result = investigate_exception(exc, [gateway("gw-1", "100.00")], [], cfg=LLMConfig())

    assert result.flagged_suspicious is True
    assert result.flag_reason is not None


def test_investigate_fails_open_when_ollama_unreachable(monkeypatch) -> None:
    exc = _make_exception(gw=gateway("gw-1", "100.00"))

    def fake_chat(messages, tools, cfg):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr("src.investigator._call_ollama_chat", fake_chat)
    result = investigate_exception(exc, [gateway("gw-1", "100.00")], [], cfg=LLMConfig())

    assert result.fallback_used is True
    assert "not reachable" in result.fallback_reason


def test_investigate_never_mutates_input_records(monkeypatch) -> None:
    """Read-only guarantee: investigating must not change any input record
    or the exception itself."""
    import copy

    exc = _make_exception(gw=gateway("gw-1", "100.00"))
    original_exc = copy.deepcopy(exc)
    gw_records = [gateway("gw-1", "100.00")]
    led_records = [ledger("led-1", "100.00")]
    original_gw = copy.deepcopy(gw_records)
    original_led = copy.deepcopy(led_records)

    def fake_chat(messages, tools, cfg):
        return {"role": "assistant", "tool_calls": [_tool_call("search_candidates", {})]}

    monkeypatch.setattr("src.investigator._call_ollama_chat", fake_chat)
    investigate_exception(exc, gw_records, led_records, cfg=LLMConfig(), max_steps=2)

    assert exc == original_exc
    assert gw_records == original_gw
    assert led_records == original_led
