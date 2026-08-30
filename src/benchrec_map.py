"""benchrec_map.py — Map the BenchRec cash-reconciliation dataset onto our schema
====================================================================================

Converts ``data/external/raw/BenchRec_cash_v1.0_train.csv`` (ICAIF 2023
BenchRec benchmark) into ``GatewayRecord`` / ``LedgerRecord`` objects plus a
``ground_truth.json`` document in the exact "pairs" shape that
``report.compute_ground_truth_accuracy`` already knows how to score. This lets
the existing reconciliation pipeline run unmodified against a real-world,
non-self-tuned dataset.

BenchRec's own schema (why this mapping looks the way it does)
----------------------------------------------------------------
train.csv is in *long* format: each row is EITHER a Type A (internal ledger)
record OR a Type B (external statement) record, never both. Rows sharing the
same ``matchId`` belong to one match cluster, and a cluster can hold multiple
A rows and multiple B rows — i.e. BenchRec natively encodes many-to-many
groupings, not just 1:1 pairs.

    Type A  (A_* columns) -> LedgerRecord   (internal ledger)
    Type B  (B_* columns) -> GatewayRecord  (external statement)

Group shape (#A rows, #B rows) within one matchId determines how the group
is represented in ground_truth.json's ``pairs`` list:

    (1, 1)   -> expected_match_status="matched"              (1:1 pair)
    (n>1, 1) -> expected_match_status="grouped_one_to_many"   (1 gateway record
                                                                covers n ledger records)
    (1, n>1) -> expected_match_status="grouped_many_to_one"   (n gateway records
                                                                cover 1 ledger record)
    (n, 0)   -> expected_match_status="unmatched_ledger"      (one pair per A row;
                                                                no gateway counterpart)
    (0, n)   -> expected_match_status="unmatched_gateway"     (one pair per B row;
                                                                no ledger counterpart)
    (n>1, m>1) -> EXCLUDED (see "Known limitation" below)

Field mapping
-------------
    transaction_id : A_id / B_id
    reference_id   : A_transactionReferences / B_transactionReferences,
                      independently per side (no leakage of matchId or
                      A_allocation/targetAllocation into either side's
                      reference_id — that field is effectively the answer
                      key). This keeps the validation honest: it tests
                      whether amount/timestamp-based fuzzy and grouped
                      matching can still find the right pairs when the two
                      sides' reference text does NOT obviously agree, which
                      is the realistic case this benchmark is built to test.
    amount         : abs(Decimal(A_amount / B_amount)) — BenchRec carries
                      signed amounts (DR/CR) as a separate column; our schema
                      always stores a positive amount.
    timestamp      : A_valueDate / B_valueDate (date-only, no time-of-day in
                      this dataset), parsed as UTC midnight.
    status         : "unknown" placeholder — BenchRec has no status field.
    counterparty   : A_transactionAttributes / B_transactionAttributes,
                      falling back to A_account / B_account when blank. Kept
                      deliberately separate from reference_id (rather than a
                      duplicate fallback) so the matcher's text-similarity
                      disambiguation phase (Phase 2.75) has two independent
                      text signals per side to compare, not one field
                      compared to itself.

Known limitation
-----------------
``report.compute_ground_truth_accuracy`` only scores 1:1, one_to_many, and
many_to_one groups (matching what matcher.py's grouped-matching phase
produces). Genuine many-to-many groups (>1 record on both sides) have no
representation in that scorer, so this mapper drops them entirely — both
their records and their group are excluded from the exported dataset. This
is a real, documented gap versus the raw benchmark, not a bug: extending
matcher.py to support many-to-many grouping is future work, not something to
paper over here.

Usage
-----
    python -m src.benchrec_map
        Reads data/external/raw/BenchRec_cash_v1.0_train.csv, writes
        data/external/processed/benchrec_gateway_records.json,
        benchrec_ledger_records.json, and benchrec_ground_truth.json.
        Prints a summary of rows skipped and group shapes excluded.

    python -m src.main --run --data-dir data/external/processed --prefix benchrec_
        (see main.py for how to point the existing pipeline at these files)
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from pydantic import ValidationError

from src.schema import GatewayRecord, LedgerRecord

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "external" / "raw"
TRAIN_CSV = RAW_DIR / "BenchRec_cash_v1.0_train.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "external" / "processed"


@dataclass
class _RawGroup:
    match_id: str
    a_rows: list[dict] = field(default_factory=list)
    b_rows: list[dict] = field(default_factory=list)


@dataclass
class MappingStats:
    total_rows: int = 0
    total_groups: int = 0
    groups_excluded_many_to_many: int = 0
    records_excluded_many_to_many: int = 0
    rows_skipped_invalid: int = 0
    ledger_records: int = 0
    gateway_records: int = 0
    pairs_matched: int = 0
    pairs_one_to_many: int = 0
    pairs_many_to_one: int = 0
    pairs_unmatched_ledger: int = 0
    pairs_unmatched_gateway: int = 0


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _to_ledger(row: dict) -> LedgerRecord | None:
    try:
        amount = abs(Decimal(row["A_amount"].strip()))
        reference = row["A_transactionReferences"].strip() or row["A_allocation"].strip() or f"BENCHREC-A-{row['A_id'].strip()}"
        counterparty = row["A_transactionAttributes"].strip() or row["A_account"].strip() or "unknown"
        return LedgerRecord(
            transaction_id=row["A_id"].strip(),
            reference_id=reference,
            amount=amount,
            currency=row["A_currencyCode"].strip() or "USD",
            timestamp=_parse_date(row["A_valueDate"]),
            status="unknown",
            counterparty=counterparty,
        )
    except (InvalidOperation, KeyError, ValueError, ValidationError):
        return None


def _to_gateway(row: dict) -> GatewayRecord | None:
    try:
        amount = abs(Decimal(row["B_amount"].strip()))
        reference = row["B_transactionReferences"].strip() or f"BENCHREC-B-{row['B_id'].strip()}"
        counterparty = row["B_transactionAttributes"].strip() or row["B_account"].strip() or "unknown"
        return GatewayRecord(
            transaction_id=row["B_id"].strip(),
            reference_id=reference,
            amount=amount,
            currency=row["B_currencyCode"].strip() or "USD",
            timestamp=_parse_date(row["B_valueDate"]),
            status="unknown",
            counterparty=counterparty,
        )
    except (InvalidOperation, KeyError, ValueError, ValidationError):
        return None


def _load_groups(path: Path, stats: MappingStats) -> dict[str, _RawGroup]:
    groups: dict[str, _RawGroup] = defaultdict(lambda: _RawGroup(match_id=""))
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stats.total_rows += 1
            match_id = row["matchId"].strip()
            if not match_id:
                continue
            group = groups[match_id]
            group.match_id = match_id
            if row["A_transactionType"].strip() == "A":
                group.a_rows.append(row)
            if row["B_transactionType"].strip() == "B":
                group.b_rows.append(row)
    return groups


def map_benchrec(path: Path = TRAIN_CSV) -> tuple[list[GatewayRecord], list[LedgerRecord], dict, MappingStats]:
    """Map BenchRec train.csv into (gateway_records, ledger_records, ground_truth, stats)."""
    stats = MappingStats()
    groups = _load_groups(path, stats)
    stats.total_groups = len(groups)

    gateway_records: list[GatewayRecord] = []
    ledger_records: list[LedgerRecord] = []
    pairs: list[dict] = []

    for match_id, group in groups.items():
        a_count, b_count = len(group.a_rows), len(group.b_rows)

        if a_count > 1 and b_count > 1:
            stats.groups_excluded_many_to_many += 1
            stats.records_excluded_many_to_many += a_count + b_count
            continue

        led_objs = [_to_ledger(r) for r in group.a_rows]
        gw_objs = [_to_gateway(r) for r in group.b_rows]
        stats.rows_skipped_invalid += sum(1 for o in led_objs if o is None)
        stats.rows_skipped_invalid += sum(1 for o in gw_objs if o is None)
        led_objs = [o for o in led_objs if o is not None]
        gw_objs = [o for o in gw_objs if o is not None]
        if not led_objs and not gw_objs:
            continue

        ledger_records.extend(led_objs)
        gateway_records.extend(gw_objs)

        if len(led_objs) == 1 and len(gw_objs) == 1:
            pairs.append({
                "pair_id": f"BENCHREC-{match_id}",
                "gateway_transaction_id": gw_objs[0].transaction_id,
                "ledger_transaction_id": led_objs[0].transaction_id,
                "discrepancy_type": "benchrec_matched",
                "expected_match_status": "matched",
                "notes": f"BenchRec matchId={match_id}, 1:1 group.",
            })
            stats.pairs_matched += 1
        elif len(led_objs) > 1 and len(gw_objs) == 1:
            pairs.append({
                "pair_id": f"BENCHREC-{match_id}",
                "gateway_transaction_id": gw_objs[0].transaction_id,
                "ledger_transaction_ids": [r.transaction_id for r in led_objs],
                "discrepancy_type": "one_to_many",
                "expected_match_status": "grouped_one_to_many",
                "notes": f"BenchRec matchId={match_id}, 1 gateway record covers {len(led_objs)} ledger records.",
            })
            stats.pairs_one_to_many += 1
        elif len(led_objs) == 1 and len(gw_objs) > 1:
            pairs.append({
                "pair_id": f"BENCHREC-{match_id}",
                "ledger_transaction_id": led_objs[0].transaction_id,
                "gateway_transaction_ids": [r.transaction_id for r in gw_objs],
                "discrepancy_type": "many_to_one",
                "expected_match_status": "grouped_many_to_one",
                "notes": f"BenchRec matchId={match_id}, {len(gw_objs)} gateway records cover 1 ledger record.",
            })
            stats.pairs_many_to_one += 1
        elif led_objs and not gw_objs:
            for r in led_objs:
                pairs.append({
                    "pair_id": f"BENCHREC-{match_id}-{r.transaction_id}",
                    "ledger_transaction_id": r.transaction_id,
                    "discrepancy_type": "missing_gateway",
                    "expected_match_status": "unmatched_ledger",
                    "notes": f"BenchRec matchId={match_id}, no Type B counterpart in this group.",
                })
                stats.pairs_unmatched_ledger += 1
        elif gw_objs and not led_objs:
            for r in gw_objs:
                pairs.append({
                    "pair_id": f"BENCHREC-{match_id}-{r.transaction_id}",
                    "gateway_transaction_id": r.transaction_id,
                    "discrepancy_type": "missing_ledger",
                    "expected_match_status": "unmatched_gateway",
                    "notes": f"BenchRec matchId={match_id}, no Type A counterpart in this group.",
                })
                stats.pairs_unmatched_gateway += 1
        # Remaining shape: (n>1, n>1) already excluded above.

    stats.ledger_records = len(ledger_records)
    stats.gateway_records = len(gateway_records)

    ground_truth = {
        "metadata": {
            "description": "Ground truth labels mapped from BenchRec (ICAIF 2023 cash reconciliation benchmark), train split. External, non-self-tuned validation set — use ONLY to score the matcher, never as matching input.",
            "source": "BenchRec_cash_v1.0_train.csv",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_gateway_records": len(gateway_records),
            "total_ledger_records": len(ledger_records),
            "total_logical_pairs": len(pairs),
            "many_to_many_groups_excluded": stats.groups_excluded_many_to_many,
        },
        "discrepancy_counts": {
            "benchrec_matched": stats.pairs_matched,
            "one_to_many": stats.pairs_one_to_many,
            "many_to_one": stats.pairs_many_to_one,
            "missing_gateway": stats.pairs_unmatched_ledger,
            "missing_ledger": stats.pairs_unmatched_gateway,
        },
        "pairs": pairs,
    }

    return gateway_records, ledger_records, ground_truth, stats


def _serialize(record: GatewayRecord | LedgerRecord) -> dict:
    d = record.model_dump()
    for k, v in d.items():
        if isinstance(v, Decimal):
            d[k] = str(v)
        elif isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


def save_mapped_dataset(
    gateway_records: list[GatewayRecord],
    ledger_records: list[LedgerRecord],
    ground_truth: dict,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchrec_gateway_records.json").write_text(
        json.dumps([_serialize(r) for r in gateway_records], indent=2), encoding="utf-8"
    )
    (output_dir / "benchrec_ledger_records.json").write_text(
        json.dumps([_serialize(r) for r in ledger_records], indent=2), encoding="utf-8"
    )
    (output_dir / "benchrec_ground_truth.json").write_text(
        json.dumps(ground_truth, indent=2), encoding="utf-8"
    )


def _print_summary(stats: MappingStats) -> None:
    print("BENCHREC MAPPING SUMMARY")
    print("=" * 72)
    print(f"Source rows read              : {stats.total_rows:,}")
    print(f"Match groups found             : {stats.total_groups:,}")
    print(f"  many-to-many groups excluded : {stats.groups_excluded_many_to_many:,}"
          f" ({stats.records_excluded_many_to_many:,} records)")
    print(f"Rows skipped (invalid/unparseable): {stats.rows_skipped_invalid:,}")
    print()
    print(f"LedgerRecord objects produced  : {stats.ledger_records:,}")
    print(f"GatewayRecord objects produced : {stats.gateway_records:,}")
    print()
    print("Ground-truth pairs by shape:")
    print(f"  1:1 matched                  : {stats.pairs_matched:,}")
    print(f"  one_to_many (grouped)        : {stats.pairs_one_to_many:,}")
    print(f"  many_to_one (grouped)        : {stats.pairs_many_to_one:,}")
    print(f"  missing_gateway (unmatched ledger): {stats.pairs_unmatched_ledger:,}")
    print(f"  missing_ledger (unmatched gateway): {stats.pairs_unmatched_gateway:,}")


def main() -> None:
    if not TRAIN_CSV.exists():
        print(f"Not found: {TRAIN_CSV}")
        return
    gateway_records, ledger_records, ground_truth, stats = map_benchrec(TRAIN_CSV)
    save_mapped_dataset(gateway_records, ledger_records, ground_truth)
    _print_summary(stats)
    print(f"\nWritten to {PROCESSED_DIR.resolve()}")


if __name__ == "__main__":
    main()
