"""Regression tests for src.ingest — generic account-sheet auto-mapping."""

import io

import pytest

from src.ingest import IngestError, detect_columns, map_to_records, read_table
from src.schema import GatewayRecord, LedgerRecord


def test_detect_columns_typical_gateway_export() -> None:
    columns = ["Transaction ID", "UTR Reference", "Amount", "Settled At", "Status", "VPA"]
    mapping = detect_columns(columns)
    assert mapping["transaction_id"] == "Transaction ID"
    assert mapping["amount"] == "Amount"
    assert mapping["timestamp"] == "Settled At"
    assert mapping["status"] == "Status"
    assert mapping["counterparty"] == "VPA"


def test_detect_columns_falls_back_reference_to_id() -> None:
    # No dedicated reference/description column at all — reference_id
    # should fall back to whatever was detected for transaction_id.
    columns = ["txn_id", "amount", "date"]
    mapping = detect_columns(columns)
    assert mapping["reference_id"] == mapping["transaction_id"] == "txn_id"


def test_pick_column_short_hint_does_not_false_positive_on_substring() -> None:
    # Real bug found via manual testing: "Paid On" contains the literal
    # substring "id" (p-a-ID-on), so a naive substring match on the bare
    # "id" hint wrongly picked it as transaction_id instead of "Txn Ref".
    columns = ["Txn Ref", "Paid On", "Amt", "Status"]
    mapping = detect_columns(columns)
    assert mapping["transaction_id"] == "Txn Ref"
    assert mapping["timestamp"] == "Paid On"


def test_detect_columns_falls_back_id_to_reference() -> None:
    # Real-world case found via manual testing: many exports use
    # "Payment Reference" as THE identifier with no separate "...id"
    # column at all. transaction_id must fall back to it too.
    columns = ["Payment Reference", "Settled At", "Amount", "Status", "VPA"]
    mapping = detect_columns(columns)
    assert mapping["transaction_id"] == "Payment Reference"
    assert mapping["reference_id"] == "Payment Reference"


def test_read_table_csv_round_trip() -> None:
    csv_text = "id,amount,date\nTX1,100.50,2024-03-01\nTX2,200.00,2024-03-02\n"
    rows = read_table(io.BytesIO(csv_text.encode("utf-8")), filename="export.csv")
    assert len(rows) == 2
    assert rows[0]["id"] == "TX1"
    assert rows[0]["amount"] == "100.50"


def test_map_to_records_happy_path_gateway() -> None:
    rows = [
        {"Transaction ID": "PAY001", "Reference": "REF001", "Amount": "1,000.50", "Date": "2024-03-01", "Status": "settled", "VPA": "a@upi"},
        {"Transaction ID": "PAY002", "Reference": "REF002", "Amount": "500.00", "Date": "2024-03-02", "Status": "settled", "VPA": "b@upi"},
    ]
    result = map_to_records(rows, role="gateway")
    assert result.mapped_count == 2
    assert not result.skipped
    assert all(isinstance(r, GatewayRecord) for r in result.records)
    assert result.records[0].amount == 1000.50
    assert result.records[0].reference_id == "REF001"


def test_map_to_records_happy_path_ledger() -> None:
    rows = [{"Order ID": "ORD001", "Amount": "999.99", "Created": "2024-03-01"}]
    result = map_to_records(rows, role="ledger")
    assert result.mapped_count == 1
    assert isinstance(result.records[0], LedgerRecord)
    # No status/counterparty column detected -> placeholders, not a crash.
    assert result.records[0].status == "unknown"
    assert result.records[0].counterparty == "unknown"


def test_map_to_records_skips_bad_rows_with_reasons() -> None:
    rows = [
        {"id": "TX1", "amount": "100.00", "date": "2024-03-01"},   # valid
        {"id": "", "amount": "50.00", "date": "2024-03-01"},        # empty id
        {"id": "TX3", "amount": "not_a_number", "date": "2024-03-01"},  # bad amount
        {"id": "TX4", "amount": "50.00", "date": "not_a_date"},     # bad date
        {"id": "TX5", "amount": "-50.00", "date": "2024-03-01"},    # signed amount, still valid (abs taken)
    ]
    result = map_to_records(rows, role="gateway")
    assert result.mapped_count == 2
    assert [r.transaction_id for r in result.records] == ["TX1", "TX5"]
    assert len(result.skipped) == 3
    reasons = dict(result.skipped)
    assert "empty transaction_id" in reasons[1]
    assert "amount" in reasons[2]
    assert "timestamp" in reasons[3]


def test_map_to_records_missing_required_field_raises() -> None:
    rows = [{"foo": "bar", "baz": "qux"}]
    with pytest.raises(IngestError):
        map_to_records(rows, role="gateway")


def test_map_to_records_column_map_override() -> None:
    rows = [{"weird_col_1": "TX1", "weird_col_2": "100.00", "weird_col_3": "2024-03-01"}]
    result = map_to_records(
        rows, role="gateway",
        column_map={"transaction_id": "weird_col_1", "amount": "weird_col_2", "timestamp": "weird_col_3"},
    )
    assert result.mapped_count == 1
    assert result.records[0].transaction_id == "TX1"


def test_amount_parsing_handles_currency_symbols_and_commas() -> None:
    rows = [{"id": "TX1", "amount": "₹1,23,456.78", "date": "2024-03-01"}]
    result = map_to_records(rows, role="gateway")
    assert result.mapped_count == 1
    assert result.records[0].amount == pytest_approx_decimal("123456.78")


def pytest_approx_decimal(s: str):
    from decimal import Decimal
    return Decimal(s)
