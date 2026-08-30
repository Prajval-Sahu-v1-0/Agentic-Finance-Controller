"""Regression tests for reconciliation thresholds and grouped matching."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from src.config import (
    EXCEPTION_SAFETY_MARGIN,
    MATCHER_AMOUNT_TOLERANCE_PCT,
    MATCHER_TIMESTAMP_TOLERANCE_HOURS,
)
from src.exceptions import ExceptionCategory, classify
from src.matcher import ReconciliationEngine
from src.schema import GatewayRecord, LedgerRecord


NOW = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)


def gateway(transaction_id: str, reference_id: str, amount: str, timestamp=NOW) -> GatewayRecord:
    return GatewayRecord(
        transaction_id=transaction_id,
        reference_id=reference_id,
        amount=Decimal(amount),
        timestamp=timestamp,
        status="settled",
        counterparty="customer@upi",
    )


def ledger(transaction_id: str, reference_id: str, amount: str, timestamp=NOW) -> LedgerRecord:
    return LedgerRecord(
        transaction_id=transaction_id,
        reference_id=reference_id,
        amount=Decimal(amount),
        timestamp=timestamp,
        status="paid",
        counterparty="CUST-1",
    )


def test_exact_match() -> None:
    result = ReconciliationEngine().run(
        [gateway("gw-1", "PAY20240101000001", "100.00")],
        [ledger("led-1", "ORD-2024-01-01-000001", "100.00")],
    )
    assert len(result.matched_exact) == 1
    assert not result.unresolved


def test_fuzzy_match_within_tolerance() -> None:
    result = ReconciliationEngine().run(
        [gateway("gw-1", "PAY20240101000001", "98.50", NOW + timedelta(hours=24))],
        [ledger("led-1", "ORD-2024-01-01-000001", "100.00")],
    )
    assert len(result.matched_fuzzy) == 1
    assert not result.unresolved


def test_amount_mismatch_detected() -> None:
    mismatch_pct = MATCHER_AMOUNT_TOLERANCE_PCT * EXCEPTION_SAFETY_MARGIN
    result = ReconciliationEngine().run(
        [gateway("gw-1", "PAY20240101000001", str(Decimal("100") * (Decimal("1") - mismatch_pct / 100)))],
        [ledger("led-1", "ORD-2024-01-01-000001", "100.00")],
    )
    exceptions = classify(result.unresolved)
    assert not result.matched_fuzzy
    assert len(exceptions) == 2
    assert {exception.category for exception in exceptions} == {ExceptionCategory.AMOUNT_MISMATCH}


def test_stale_timing_detected() -> None:
    drift_hours = MATCHER_TIMESTAMP_TOLERANCE_HOURS * float(EXCEPTION_SAFETY_MARGIN)
    result = ReconciliationEngine().run(
        [gateway("gw-1", "PAY20240101000001", "100.00", NOW + timedelta(hours=drift_hours))],
        [ledger("led-1", "ORD-2024-01-01-000001", "100.00")],
    )
    exceptions = classify(result.unresolved)
    assert not result.matched_fuzzy
    assert len(exceptions) == 2
    assert {exception.category for exception in exceptions} == {ExceptionCategory.STALE_TIMING}


def test_duplicate_detected() -> None:
    result = ReconciliationEngine().run(
        [
            gateway("gw-original", "PAY20240101000001", "100.00"),
            gateway("gw-duplicate", "PAY20240101000001", "100.00", NOW + timedelta(minutes=2)),
        ],
        [ledger("led-1", "ORD-2024-01-01-000001", "100.00")],
    )
    exceptions = classify(result.unresolved)
    assert len(result.matched_exact) == 1
    assert [exception.category for exception in exceptions] == [ExceptionCategory.DUPLICATE]


def test_grouped_match_valid_case() -> None:
    result = ReconciliationEngine().run(
        [gateway("gw-batch", "BATCH20240101999999", "100.00")],
        [
            ledger("led-1", "ORD-2024-01-01-000101", "40.00", NOW - timedelta(hours=1)),
            ledger("led-2", "ORD-2024-01-01-000102", "60.00", NOW - timedelta(hours=2)),
        ],
    )
    assert len(result.matched_grouped) == 1
    assert result.matched_grouped[0].match_type == "one_to_many"
    assert not result.unresolved


def test_grouped_match_rejects_false_positive() -> None:
    result = ReconciliationEngine().run(
        [gateway("gw-ordinary", "PAY20240101000099", "100.00")],
        [
            ledger("led-1", "ORD-2024-01-01-000101", "40.00"),
            ledger("led-2", "ORD-2024-01-01-000102", "60.00"),
        ],
    )
    assert not result.matched_grouped
    assert {item.reason for item in result.unresolved} == {"no_counterpart"}
