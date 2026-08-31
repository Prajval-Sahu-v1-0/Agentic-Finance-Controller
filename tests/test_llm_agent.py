"""Regression tests for src.llm_agent's response parser.

Covers the bug caught while evaluating an external HF model whose raw
output was pure repetitive non-JSON text: the regex-fallback strategy used
to silently default `category` to the rule engine's own category whenever
it found nothing at all, making "no coherent output" indistinguishable from
"the model agrees with the rules" in the resulting metrics. It must now
raise NoStructuredOutputError instead.
"""

import pytest

from src.llm_agent import NoStructuredOutputError, _parse_llm_response


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
