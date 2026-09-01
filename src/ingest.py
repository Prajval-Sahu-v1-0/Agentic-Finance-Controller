"""
ingest.py — Generic account-sheet ingestion (CSV/Excel -> GatewayRecord/LedgerRecord)
=========================================================================================

Answers the question "does the user need to format the account sheets before
uploading?" with: no, not anymore. Before this module, the only way to get a
new real-world source into the pipeline was to write a bespoke mapping script
like ``src/benchrec_map.py`` for that specific source's column layout. This
module auto-detects columns by name (reusing the same hint vocabulary
``inspect_benchrec.py`` already used for read-only inspection) and maps
whatever it finds onto ``GatewayRecord`` / ``LedgerRecord``, so an arbitrary
CSV or Excel export can be uploaded as-is.

This is a heuristic, not a guarantee — it is deliberately transparent about
what it did rather than silently guessing: every mapping result reports
exactly which source column was used for each target field, and every
skipped row reports why, so a human can sanity-check the result before
trusting it (see ``MappingResult`` below).

Column detection
----------------
Column names are matched case-insensitively against hint word-lists (exact
match preferred, substring match as fallback). The same hints
``inspect_benchrec.py`` uses for transaction ID / amount / timestamp /
reference are reused here for consistency, plus new hints for the fields
BenchRec's inspector didn't need: status, counterparty, currency.

Required vs optional fields
----------------------------
``transaction_id``, ``amount``, and ``timestamp`` must be detected (or
explicitly supplied via ``column_map``) — without these there is no record
to build. ``reference_id`` falls back to the transaction ID column if no
better candidate is found (a record always needs SOME reference key for the
matcher's Phase 1/2 to index on, even if it turns out to just be the ID
again). ``status``, ``counterparty``, and ``currency`` default to
placeholder values when absent, exactly like ``benchrec_map.py`` does for
BenchRec (which has no status field at all).

Usage
-----
    from src.ingest import read_table, map_to_records

    rows = read_table(path_or_buffer, filename="settlements.csv")
    result = map_to_records(rows, role="gateway")
    print(result.column_map)          # what was auto-detected
    print(len(result.records))        # successfully mapped
    print(result.skipped)             # [(row_index, reason), ...]
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import ValidationError

from src.schema import GatewayRecord, LedgerRecord

# Reuse the same hint vocabulary inspect_benchrec.py uses for read-only
# dataset inspection, so "what does this column probably mean" is answered
# consistently everywhere in the codebase.
from src.inspect_benchrec import _AMOUNT_HINTS, _ID_HINTS, _REFERENCE_HINTS, _TIME_HINTS

# Common DB-export-style timestamp column names that don't contain "date"
# or "time" as a substring, so _TIME_HINTS alone would miss them.
_TIME_HINTS_EXTRA = (
    "settled_at", "paid_at", "created_at", "updated_at",
    "settled_on", "paid_on", "created_on", "updated_on",
    "settled", "created", "value_date",
)
_STATUS_HINTS = ("status", "state", "txn_status", "payment_status")
_COUNTERPARTY_HINTS = ("counterparty", "customer", "vpa", "payer", "payee", "account_holder", "merchant")
_CURRENCY_HINTS = ("currency", "curr", "ccy")

_REQUIRED_FIELDS = ("transaction_id", "amount", "timestamp")


@dataclass
class MappingResult:
    """
    Result of mapping raw rows onto GatewayRecord/LedgerRecord.

    Attributes
    ----------
    records : list[GatewayRecord | LedgerRecord]
        Successfully mapped and validated records.
    column_map : dict[str, str | None]
        Which source column was used for each target field
        (transaction_id, reference_id, amount, timestamp, status,
        counterparty, currency). None means no column was detected and a
        placeholder/fallback was used instead.
    skipped : list[tuple[int, str]]
        (row_index, reason) for every row that could not be mapped —
        never silently dropped without a reason.
    total_rows : int
        Rows read from the source table, before mapping.
    """
    records: list = field(default_factory=list)
    column_map: dict = field(default_factory=dict)
    skipped: list = field(default_factory=list)
    total_rows: int = 0

    @property
    def mapped_count(self) -> int:
        return len(self.records)


class IngestError(Exception):
    """Raised when the table cannot be mapped at all — e.g. a required
    field has no detected column and no override was supplied. Distinct
    from per-row skips (MappingResult.skipped), which are partial failures
    the caller can inspect and still get a usable partial result from."""


# ---------------------------------------------------------------------------
# Table reading — CSV and Excel, no format-specific caller code needed
# ---------------------------------------------------------------------------

def read_table(source: Union[Path, str, bytes, io.IOBase], filename: str) -> list[dict]:
    """
    Read a CSV or Excel file into a list of row-dicts (column name -> raw
    string/value). `filename` is used only to pick the reader by extension —
    `source` can be a path, raw bytes, or a file-like object (e.g. an
    uploaded file's stream), so this works the same from a CLI script or an
    HTTP upload.
    """
    suffix = Path(filename).suffix.lower()

    if suffix in (".csv", ".tsv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        if isinstance(source, (Path, str)):
            text = Path(source).read_text(encoding="utf-8-sig")
        elif isinstance(source, bytes):
            text = source.decode("utf-8-sig")
        else:
            raw = source.read()
            text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        return [dict(row) for row in reader]

    if suffix in (".xlsx", ".xls"):
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise IngestError(
                "Excel files require pandas (and openpyxl for .xlsx). "
                "CSV does not need either."
            ) from exc
        frame = pd.read_excel(source)
        frame = frame.where(frame.notna(), None)
        return [
            {str(k): ("" if v is None else str(v)) for k, v in row.items()}
            for row in frame.to_dict(orient="records")
        ]

    raise IngestError(f"Unsupported file type: {suffix!r} (expected .csv, .tsv, .xlsx, or .xls)")


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _pick_column(columns: list[str], hints: tuple[str, ...]) -> Optional[str]:
    """
    Best single column match for `hints`, tried in order of decreasing
    strictness:

    1. Exact whole-name match (normalised column name == a hint).
    2. Token-boundary match: an underscore-delimited token of the
       normalised name equals a hint exactly (e.g. "transaction_id" ->
       token "id" matches hint "id").
    3. Substring match, but ONLY for hints of length >= 4 — short bare-word
       hints like "id" or "ref" match as arbitrary substrings far too
       often on real column names ("paid_on" contains "id"; "prefer"
       contains "ref"). Longer hints are specific enough that a substring
       hit is still a reasonable signal even without a token boundary
       (e.g. "reference" inside a concatenated "PaymentReference").

    Returns None if nothing matches at any stage.
    """
    normalized = {c: _normalize(c) for c in columns}
    for col, norm in normalized.items():
        if norm in hints:
            return col
    for col, norm in normalized.items():
        if any(token in hints for token in norm.split("_")):
            return col
    for col, norm in normalized.items():
        if any(len(hint) >= 4 and hint in norm for hint in hints):
            return col
    return None


def detect_columns(columns: list[str]) -> dict:
    """Auto-detect which source column corresponds to each target field.
    Returns a dict with a key per field; value is the source column name
    or None if nothing was detected."""
    # Real-world exports very commonly use "Reference"/"Payment Reference"/
    # "UTR" as THE identifier, with no separate literal "...id" column at
    # all — fall back to a reference-shaped column for transaction_id just
    # as readily as the reverse fallback below (reference_id -> id_col).
    id_col = _pick_column(columns, _ID_HINTS) or _pick_column(columns, _REFERENCE_HINTS)
    return {
        "transaction_id": id_col,
        "reference_id"  : _pick_column(columns, _REFERENCE_HINTS) or id_col,
        "amount"        : _pick_column(columns, _AMOUNT_HINTS),
        "timestamp"     : _pick_column(columns, _TIME_HINTS) or _pick_column(columns, _TIME_HINTS_EXTRA),
        "status"        : _pick_column(columns, _STATUS_HINTS),
        "counterparty"  : _pick_column(columns, _COUNTERPARTY_HINTS),
        "currency"      : _pick_column(columns, _CURRENCY_HINTS),
    }


# ---------------------------------------------------------------------------
# Value parsing — tolerant of common real-world export quirks
# ---------------------------------------------------------------------------

def _parse_amount(raw: str) -> Optional[Decimal]:
    if raw is None:
        return None
    cleaned = str(raw).strip().replace(",", "").replace("₹", "").replace("$", "").replace("INR", "").strip()
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]
    if not cleaned:
        return None
    try:
        return abs(Decimal(cleaned))
    except InvalidOperation:
        return None


def _parse_timestamp(raw: str) -> Optional[datetime]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    # Try the common export formats before giving up. dateutil would be
    # more robust but is an extra dependency for a fallback path that
    # ISO-8601 and a handful of common formats already cover well.
    formats = (
        "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y", "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y",
    )
    for fmt in formats:
        try:
            dt = datetime.strptime(text, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------

def map_to_records(
    rows        : list[dict],
    role        : Literal["gateway", "ledger"],
    column_map  : Optional[dict] = None,
) -> MappingResult:
    """
    Map raw row-dicts onto GatewayRecord (role="gateway") or LedgerRecord
    (role="ledger") objects.

    Parameters
    ----------
    rows : list[dict]
        Output of read_table().
    role : "gateway" | "ledger"
        Which schema to map onto.
    column_map : dict, optional
        Override auto-detection for any field, e.g.
        {"amount": "Settlement Amount"}. Fields not overridden are still
        auto-detected.

    Raises
    ------
    IngestError
        If a REQUIRED field (transaction_id, amount, timestamp) has no
        detected column and no override — there is nothing to map.
    """
    if not rows:
        raise IngestError("No rows to map — the uploaded file appears to be empty.")

    columns = list(rows[0].keys())
    detected = detect_columns(columns)
    if column_map:
        detected.update({k: v for k, v in column_map.items() if v})

    missing_required = [f for f in _REQUIRED_FIELDS if not detected.get(f)]
    if missing_required:
        raise IngestError(
            f"Could not detect a column for required field(s) {missing_required} "
            f"among columns {columns}. Pass column_map to specify them explicitly, "
            f"e.g. column_map={{'amount': 'Your Amount Column'}}."
        )

    model = GatewayRecord if role == "gateway" else LedgerRecord
    result = MappingResult(column_map=detected, total_rows=len(rows))

    for i, row in enumerate(rows):
        txn_id_raw = row.get(detected["transaction_id"])
        amount = _parse_amount(row.get(detected["amount"]))
        timestamp = _parse_timestamp(row.get(detected["timestamp"]))

        if not txn_id_raw or str(txn_id_raw).strip() == "":
            result.skipped.append((i, "empty transaction_id"))
            continue
        if amount is None or amount <= 0:
            result.skipped.append((i, f"unparseable or non-positive amount: {row.get(detected['amount'])!r}"))
            continue
        if timestamp is None:
            result.skipped.append((i, f"unparseable timestamp: {row.get(detected['timestamp'])!r}"))
            continue

        reference_id = (row.get(detected["reference_id"]) or str(txn_id_raw)).strip() if detected["reference_id"] else str(txn_id_raw).strip()
        status = (row.get(detected["status"]) or "unknown").strip() if detected["status"] else "unknown"
        counterparty = (row.get(detected["counterparty"]) or "unknown").strip() if detected["counterparty"] else "unknown"
        currency = (row.get(detected["currency"]) or "INR").strip().upper() if detected["currency"] else "INR"
        if len(currency) != 3:
            currency = "INR"

        try:
            record = model(
                transaction_id=str(txn_id_raw).strip(),
                reference_id=reference_id or str(txn_id_raw).strip(),
                amount=amount,
                currency=currency,
                timestamp=timestamp,
                status=status or "unknown",
                counterparty=counterparty or "unknown",
            )
        except ValidationError as exc:
            result.skipped.append((i, f"validation error: {exc.errors()[0]['msg'] if exc.errors() else exc}"))
            continue

        result.records.append(record)

    return result


# ---------------------------------------------------------------------------
# Persistence — same JSON shape generator.py / benchrec_map.py already write,
# so anything mapped here is immediately loadable via load_gateway_records /
# load_ledger_records and usable by main.py / api.py unmodified.
# ---------------------------------------------------------------------------

def _serialize(record: Union[GatewayRecord, LedgerRecord]) -> dict:
    d = record.model_dump()
    for k, v in d.items():
        if isinstance(v, Decimal):
            d[k] = str(v)
        elif isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


def save_records(records: list, path: Path) -> None:
    """Write mapped records to `path` in the same JSON list-of-dicts shape
    load_gateway_records/load_ledger_records already read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([_serialize(r) for r in records], indent=2), encoding="utf-8")
