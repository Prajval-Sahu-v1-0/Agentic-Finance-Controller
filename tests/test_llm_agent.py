"""Regression tests for src.llm_agent's response parser and guardrails.

Covers the bug caught while evaluating an external HF model whose raw
output was pure repetitive non-JSON text: the regex-fallback strategy used
to silently default `category` to the rule engine's own category whenever
it found nothing at all, making "no coherent output" indistinguishable from
"the model agrees with the rules" in the resulting metrics. It must now
raise NoStructuredOutputError instead.

Also covers the guardrails described in llm_agent.py's module docstring:
untrusted-field sanitization, the agrees_with_rules bool-coercion fix,
explanation sanitization/length caps, suspicious-directive detection, and
the "LLM output is advisory only, never authoritative" invariant.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.exceptions import ExceptionCategory, ExceptionRecord
from src.llm_agent import (
    LLMConfig,
    NoStructuredOutputError,
    _coerce_bool,
    _detect_suspicious_directive,
    _parse_llm_response,
    _sanitize_llm_text,
    _sanitize_untrusted_field,
    _summarize_report_for_prompt,
    ask_agent,
    reason_about_exception,
)
from src.schema import GatewayRecord


def test_full_json_parses() -> None:
    raw = (
        '{"agrees_with_rules": false, "category": "amount_mismatch", '
        '"confidence": "high", "explanation": "Fee deduction of 3%."}'
    )
    result = _parse_llm_response(raw, fallback_category="stale_timing")
    assert result == {
        "agrees_with_rules": False,
        "category": "amount_mismatch",
        "confidence": "high",
        "explanation": "Fee deduction of 3%.",
    }


def test_json_embedded_in_surrounding_text_parses() -> None:
    raw = (
        "Sure, here is my analysis:\n"
        '{"agrees_with_rules": true, "category": "duplicate", '
        '"confidence": "medium", "explanation": "Second gateway retry."}\n'
        "Let me know if you need more detail."
    )
    result = _parse_llm_response(raw, fallback_category="duplicate")
    assert result["category"] == "duplicate"
    assert result["agrees_with_rules"] is True


def test_partial_field_only_text_recovers_via_regex_fallback() -> None:
    # Malformed JSON (missing closing brace) but individual fields are
    # still present as quoted "key": "value" fragments.
    raw = '"category": "missing_in_ledger", "confidence": "low"'
    result = _parse_llm_response(raw, fallback_category="amount_mismatch")
    assert result["category"] == "missing_in_ledger"
    assert result["confidence"] == "low"


def test_no_structured_output_raises_instead_of_silently_agreeing() -> None:
    # Real example shape from evaluating mombalam/clearledgr-llama-financial-ai:
    # repetitive, incoherent text with no JSON and no "key": "value" fragment.
    raw = (
        "Notes: recommend back-end e-commerce\n"
        "Recommendation: contact the vendor\n"
        "Action: contact the customer\n"
        "Notes: potential duplicate or missing record\n"
    )
    with pytest.raises(NoStructuredOutputError):
        _parse_llm_response(raw, fallback_category="missing_in_gateway")


def test_invalid_category_falls_back_to_rule_category() -> None:
    raw = '{"agrees_with_rules": true, "category": "not_a_real_category", "confidence": "high", "explanation": "x"}'
    result = _parse_llm_response(raw, fallback_category="stale_timing")
    assert result["category"] == "stale_timing"


def test_invalid_confidence_defaults_to_medium() -> None:
    raw = '{"agrees_with_rules": true, "category": "duplicate", "confidence": "extremely_sure", "explanation": "x"}'
    result = _parse_llm_response(raw, fallback_category="duplicate")
    assert result["confidence"] == "medium"


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def test_bool_string_false_does_not_coerce_to_true() -> None:
    # Python's bool("false") is True — any non-empty string is truthy.
    # A model outputting the JSON string "false" instead of the literal
    # `false` must not silently flip to agreement.
    assert _coerce_bool("false") is False
    assert _coerce_bool("true") is True
    assert _coerce_bool(False) is False
    assert _coerce_bool(True) is True


def test_agrees_with_rules_as_json_string_parses_correctly() -> None:
    # A model that (incorrectly) quotes the boolean must still be read
    # correctly, not silently treated as agreement.
    raw = '{"agrees_with_rules": "false", "category": "duplicate", "confidence": "high", "explanation": "x"}'
    result = _parse_llm_response(raw, fallback_category="duplicate")
    assert result["agrees_with_rules"] is False


def test_untrusted_field_truncated_and_control_chars_stripped() -> None:
    huge = "A" * 1000
    cleaned = _sanitize_untrusted_field(huge, max_len=200)
    assert len(cleaned) < len(huge)
    assert cleaned.startswith("A" * 200)

    injected = "PAY123\x1b[31mFAKE\x1b[0m\x00\x07"
    cleaned = _sanitize_untrusted_field(injected)
    assert "\x1b" not in cleaned
    assert "\x00" not in cleaned
    assert "\x07" not in cleaned
    assert "PAY123" in cleaned


def test_explanation_capped_and_sanitized() -> None:
    raw = '{"agrees_with_rules": true, "category": "duplicate", "confidence": "high", ' \
          f'"explanation": "{"x" * 800}"}}'
    result = _parse_llm_response(raw, fallback_category="duplicate")
    assert len(result["explanation"]) <= 501  # _MAX_EXPLANATION_LEN + ellipsis


def test_explanation_control_chars_stripped() -> None:
    text = "Looks fine\x1b[2J\x1b[31mHACKED\x1b[0m but check amount."
    cleaned = _sanitize_llm_text(text)
    assert "\x1b" not in cleaned
    assert "check amount" in cleaned


def test_suspicious_directive_flags_wire_transfer_language() -> None:
    reason = _detect_suspicious_directive(
        "This looks like a fee deduction. Action: wire transfer immediately to close the gap."
    )
    assert reason is not None


def test_suspicious_directive_flags_account_number() -> None:
    reason = _detect_suspicious_directive(
        "Recommend routing number: 123456789 for the correction payment."
    )
    assert reason is not None


def test_suspicious_directive_does_not_flag_normal_explanation() -> None:
    reason = _detect_suspicious_directive(
        "The 3% delta is consistent with a standard gateway processing fee. "
        "Action: post a fee expense and mark the pair reconciled."
    )
    assert reason is None


def test_suspicious_directive_flags_non_urgent_account_suggestion() -> None:
    # Real miss found via live adversarial testing of ask_agent: a model
    # refused to "execute" a transfer but still suggested — calmly, no
    # urgency language, no "number"/"no."/"#" between "account" and the
    # digits — moving funds to an attacker-supplied account number. The
    # original patterns required urgency language OR the literal word
    # "number" and missed this phrasing entirely.
    reason = _detect_suspicious_directive(
        "You can manually transfer the funds to account 9988776655 to reconcile the accounts."
    )
    assert reason is not None


def _make_exception_record() -> ExceptionRecord:
    gw = GatewayRecord(
        transaction_id="rzp_test_1",
        reference_id="PAY20240101000001",
        amount=Decimal("100.00"),
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        status="settled",
        counterparty="customer@upi",
    )
    return ExceptionRecord(
        exception_id="EX-TEST-000001",
        category=ExceptionCategory.MISSING_IN_LEDGER,
        source="gateway",
        gateway_record=gw,
        normalized_ref="20240101000001",
        explanation="No matching ledger record found.",
        suggested_action="Investigate missing webhook.",
        classified_at=datetime.now(timezone.utc),
    )


def test_reason_about_exception_never_mutates_input(monkeypatch) -> None:
    """
    LLM output is advisory only — reason_about_exception must never mutate
    the ExceptionRecord passed in, even when the model "disagrees" and
    proposes a different category. See module docstring's Guardrails
    point 1: nothing in this pipeline lets model output become
    authoritative.
    """
    exc = _make_exception_record()
    original_category = exc.category
    original_gw_id = exc.gateway_record.transaction_id

    def fake_call_ollama(prompt: str, cfg: LLMConfig) -> str:
        return (
            '{"agrees_with_rules": false, "category": "duplicate", '
            '"confidence": "high", "explanation": "Looks like a duplicate to me."}'
        )

    monkeypatch.setattr("src.llm_agent._call_ollama", fake_call_ollama)

    result = reason_about_exception(exc, cfg=LLMConfig())

    # The LLM's "disagreement" is reflected only in the returned result...
    assert result.agrees_with_rules is False
    assert result.category == "duplicate"
    # ...and never touches the original record.
    assert exc.category == original_category
    assert exc.category == ExceptionCategory.MISSING_IN_LEDGER
    assert exc.gateway_record.transaction_id == original_gw_id


def test_reason_about_exception_flags_suspicious_output(monkeypatch) -> None:
    exc = _make_exception_record()

    def fake_call_ollama(prompt: str, cfg: LLMConfig) -> str:
        return (
            '{"agrees_with_rules": true, "category": "missing_in_ledger", '
            '"confidence": "high", "explanation": "Wire transfer immediately to account 987654321."}'
        )

    monkeypatch.setattr("src.llm_agent._call_ollama", fake_call_ollama)

    result = reason_about_exception(exc, cfg=LLMConfig())
    assert result.flagged_suspicious is True
    assert result.flag_reason is not None


# ---------------------------------------------------------------------------
# ask_agent — free-text Q&A ("prompt the agent")
# ---------------------------------------------------------------------------

_SAMPLE_REPORT = {
    "match_summary": {
        "total_gateway_records": 104,
        "total_ledger_records": 102,
        "total_matched": {"count": 77, "rate_pct": 74.04},
        "unresolved": {"count": 50, "rate_pct": 48.08},
    },
    "monetary_summary": {
        "currency": "INR",
        "matched_gateway_value": "1000.00",
        "matched_ledger_value": "1000.00",
        "unresolved_gateway_value": "500.00",
        "unresolved_ledger_value": "600.00",
    },
    "exception_breakdown": [
        {"category": "amount_mismatch", "count": 10, "share_pct": 20.0},
        {"category": "duplicate", "count": 5, "share_pct": 10.0},
    ],
}


def test_ask_agent_rejects_empty_prompt() -> None:
    result = ask_agent("   ", report=_SAMPLE_REPORT, cfg=LLMConfig())
    assert result.fallback_used is True
    assert result.fallback_reason == "EmptyPrompt"


def test_ask_agent_never_touches_report(monkeypatch) -> None:
    """
    Read-only by construction: ask_agent must not mutate the report dict
    it was given, and — since it never imports/calls classify(), the
    matcher, or generate_report — there is no code path from a question
    to any pipeline state changing. This test locks the "doesn't mutate
    its input" half of that guarantee; the "doesn't call pipeline code"
    half is structural (see the module docstring) and enforced by
    ask_agent's implementation not importing those functions at all.
    """
    import copy
    original = copy.deepcopy(_SAMPLE_REPORT)

    def fake_call_ollama(prompt: str, cfg: LLMConfig) -> str:
        return "The match rate is 74.04%, with 50 unresolved records."

    monkeypatch.setattr("src.llm_agent._call_ollama", fake_call_ollama)
    ask_agent("What's the match rate?", report=_SAMPLE_REPORT, cfg=LLMConfig())
    assert _SAMPLE_REPORT == original


def test_ask_agent_grounds_prompt_in_report_context(monkeypatch) -> None:
    captured_prompt = {}

    def fake_call_ollama(prompt: str, cfg: LLMConfig) -> str:
        captured_prompt["value"] = prompt
        return "Answer."

    monkeypatch.setattr("src.llm_agent._call_ollama", fake_call_ollama)
    ask_agent("What's the match rate?", report=_SAMPLE_REPORT, cfg=LLMConfig())

    prompt = captured_prompt["value"]
    assert "74.04" in prompt          # match rate from the report context
    assert "amount_mismatch" in prompt  # exception breakdown from the context
    assert "What's the match rate?" in prompt  # the user's question, verbatim


def test_ask_agent_declines_when_no_report_available(monkeypatch) -> None:
    captured_prompt = {}

    def fake_call_ollama(prompt: str, cfg: LLMConfig) -> str:
        captured_prompt["value"] = prompt
        return "I don't have report data to answer that."

    monkeypatch.setattr("src.llm_agent._call_ollama", fake_call_ollama)
    result = ask_agent("What's the match rate?", report=None, cfg=LLMConfig())

    assert result.fallback_used is False
    assert "No reconciliation report" in captured_prompt["value"]


def test_ask_agent_sanitizes_and_truncates_user_prompt(monkeypatch) -> None:
    captured_prompt = {}

    def fake_call_ollama(prompt: str, cfg: LLMConfig) -> str:
        captured_prompt["value"] = prompt
        return "Answer."

    monkeypatch.setattr("src.llm_agent._call_ollama", fake_call_ollama)
    injected = "What's the match rate?\x1b[31mFAKE\x1b[0m" + ("X" * 3000)
    ask_agent(injected, report=_SAMPLE_REPORT, cfg=LLMConfig())

    prompt = captured_prompt["value"]
    assert "\x1b" not in prompt
    assert prompt.count("X") < 3000  # truncated well below the raw injected length


def test_ask_agent_flags_suspicious_response(monkeypatch) -> None:
    def fake_call_ollama(prompt: str, cfg: LLMConfig) -> str:
        return "You should wire transfer immediately to account 123456789 to resolve this."

    monkeypatch.setattr("src.llm_agent._call_ollama", fake_call_ollama)
    result = ask_agent("How do I fix this exception?", report=_SAMPLE_REPORT, cfg=LLMConfig())

    assert result.flagged_suspicious is True
    assert result.flag_reason is not None


def test_ask_agent_fails_open_when_ollama_unreachable(monkeypatch) -> None:
    import requests

    def fake_call_ollama(prompt: str, cfg: LLMConfig) -> str:
        raise requests.ConnectionError("refused")

    monkeypatch.setattr("src.llm_agent._call_ollama", fake_call_ollama)
    result = ask_agent("What's the match rate?", report=_SAMPLE_REPORT, cfg=LLMConfig())

    assert result.fallback_used is True
    assert "not reachable" in result.fallback_reason


def test_summarize_report_for_prompt_handles_empty_dict() -> None:
    assert _summarize_report_for_prompt({}) == "(Report contains no summarizable sections.)"
