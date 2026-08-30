"""Explain ground-truth exceptions that the reconciliation engine missed.

Run with::

    python -m src.diagnose

The diagnostic never writes to ``data/``.  It reruns the engine in memory,
then compares every synthetic ground-truth exception with the outcome assigned
to its record(s).  Unlike the legacy aggregate accuracy calculation in
``report.py``, this tool treats all five discrepancy types as exceptions.  In
particular, it makes amount mismatches and timestamp drifts visible when they
were accepted as fuzzy matches.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
import sys
from typing import Iterable, Optional

from src.exceptions import ExceptionCategory, ExceptionRecord, classify
from src.generator import load_gateway_records, load_ground_truth, load_ledger_records
from src.matcher import MatchConfig, ReconciliationEngine, ReconciliationResult


_GT_CATEGORY: dict[str, ExceptionCategory] = {
    "amount_mismatch": ExceptionCategory.AMOUNT_MISMATCH,
    "timestamp_drift": ExceptionCategory.STALE_TIMING,
    "missing_gateway": ExceptionCategory.MISSING_IN_GATEWAY,
    "missing_ledger": ExceptionCategory.MISSING_IN_LEDGER,
    "duplicate": ExceptionCategory.DUPLICATE,
}


@dataclass(frozen=True)
class ExpectedException:
    """One oracle exception and the record that must be surfaced for it."""

    pair_id: str
    expected_category: ExceptionCategory
    target_ids: tuple[str, ...]
    record_ids: tuple[str, ...]
    notes: str
    introduced_delta: Optional[dict]


@dataclass(frozen=True)
class ActualDecision:
    """The engine result that claimed an individual transaction record."""

    label: str
    category: Optional[ExceptionCategory]
    record_ids: tuple[str, ...]
    amount_delta_pct: Optional[Decimal] = None
    timestamp_delta_hours: Optional[float] = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose ground-truth exceptions missed by recon-agent."
    )
    parser.add_argument("--amount-tolerance", type=float, default=2.0, metavar="PCT")
    parser.add_argument("--timestamp-tolerance", type=float, default=72.0, metavar="HOURS")
    parser.add_argument("--exact-timestamp-tolerance", type=float, default=1.0, metavar="HOURS")
    return parser


def _expected_exceptions(ground_truth: dict) -> list[ExpectedException]:
    """Convert oracle pair labels into one expected exception per anomaly."""
    expected: list[ExpectedException] = []

    for pair in ground_truth.get("pairs", []):
        discrepancy_type = pair.get("discrepancy_type")
        category = _GT_CATEGORY.get(discrepancy_type)
        if category is None:
            continue

        gw_id = pair.get("gateway_transaction_id")
        ledger_id = pair.get("ledger_transaction_id")
        duplicate_id = pair.get("gateway_duplicate_transaction_id")
        grouped_gw_ids = pair.get("gateway_transaction_ids", [])
        grouped_ledger_ids = pair.get("ledger_transaction_ids", [])

        # For a duplicate, the injected extra gateway row is the record the
        # current scorer expects to be surfaced.  The original pair is still
        # included in the printed context below.
        if discrepancy_type == "duplicate":
            target_ids = (duplicate_id,) if duplicate_id else ()
        elif gw_id:
            target_ids = (gw_id,)
        elif ledger_id:
            target_ids = (ledger_id,)
        else:
            continue

        record_ids = tuple(
            item
            for item in (
                gw_id,
                duplicate_id,
                ledger_id,
                *grouped_gw_ids,
                *grouped_ledger_ids,
            )
            if item
        )
        expected.append(
            ExpectedException(
                pair_id=pair["pair_id"],
                expected_category=category,
                target_ids=target_ids,
                record_ids=record_ids,
                notes=pair.get("notes", ""),
                introduced_delta=pair.get("introduced_delta"),
            )
        )
    return expected


def _index_actual_output(
    result: ReconciliationResult,
    exceptions: list[ExceptionRecord],
) -> dict[str, ActualDecision]:
    """Map every claimed transaction ID to its current engine outcome."""
    decisions: dict[str, ActualDecision] = {}

    for pair in result.matched_exact + result.matched_fuzzy + result.matched_content + result.matched_text:
        record_ids = (pair.gateway_record.transaction_id, pair.ledger_record.transaction_id)
        decisions_for_pair = ActualDecision(
            label=f"matched_{pair.match_type}",
            category=None,
            record_ids=record_ids,
            amount_delta_pct=(
                abs(pair.amount_delta) / pair.ledger_record.amount * Decimal("100")
                if pair.ledger_record.amount
                else None
            ),
            timestamp_delta_hours=abs(pair.timestamp_delta_seconds) / 3600,
        )
        for record_id in record_ids:
            decisions[record_id] = decisions_for_pair

    for group in result.matched_grouped:
        record_ids = tuple(
            record.transaction_id
            for record in [*group.gateway_records, *group.ledger_records]
        )
        decision = ActualDecision(
            label="matched_grouped",
            category=None,
            record_ids=record_ids,
            amount_delta_pct=group.amount_delta_pct,
        )
        for record_id in record_ids:
            decisions[record_id] = decision

    for exception in exceptions:
        record = exception.gateway_record or exception.ledger_record
        if record is None:  # Defensive: ExceptionRecord always has one source.
            continue
        decisions[record.transaction_id] = ActualDecision(
            label=f"exception:{exception.category.value}",
            category=exception.category,
            record_ids=(record.transaction_id,),
            amount_delta_pct=exception.amount_delta_pct,
            timestamp_delta_hours=exception.timestamp_delta_hours,
        )

    return decisions


def _is_correct(expected: ExpectedException, decisions: dict[str, ActualDecision]) -> bool:
    return all(
        (decision := decisions.get(record_id)) is not None
        and decision.category == expected.expected_category
        for record_id in expected.target_ids
    )


def _record_lookup(records: Iterable[object]) -> dict[str, object]:
    return {record.transaction_id: record for record in records}


def _format_record(record: object) -> str:
    return (
        f"{record.transaction_id} | ref={record.reference_id} | "
        f"{record.currency} {record.amount} | ts={record.timestamp.isoformat()} | "
        f"status={record.status}"
    )


def _decision_text(decision: Optional[ActualDecision]) -> str:
    if decision is None:
        return "not present in engine output"
    details: list[str] = [decision.label]
    if decision.amount_delta_pct is not None:
        details.append(f"amount_delta={decision.amount_delta_pct:.2f}%")
    if decision.timestamp_delta_hours is not None:
        details.append(f"timestamp_delta={decision.timestamp_delta_hours:.2f}h")
    if len(decision.record_ids) > 1:
        details.append("with=" + ", ".join(decision.record_ids))
    return " | ".join(details)


def _tolerance_warning(
    expected: ExpectedException,
    decisions: dict[str, ActualDecision],
    config: MatchConfig,
) -> Optional[str]:
    """Describe a ground-truth exception accepted by a permissive match phase."""
    accepted = [decisions.get(record_id) for record_id in expected.target_ids]
    accepted = [decision for decision in accepted if decision is not None]
    labels = {decision.label for decision in accepted}
    if not (labels & {"matched_fuzzy", "matched_content", "matched_text", "matched_grouped"}):
        return None

    if "matched_fuzzy" in labels:
        decision = next(item for item in accepted if item.label == "matched_fuzzy")
        return (
            "TOLERANCE WARNING: accepted as matched_fuzzy under "
            f"amount <= {config.amount_tolerance_pct}% and timestamp <= "
            f"{config.timestamp_tolerance_hours / 24:.2f} days; "
            f"observed {_decision_text(decision)}"
        )
    if "matched_content" in labels:
        decision = next(item for item in accepted if item.label == "matched_content")
        return (
            "CONTENT-FALLBACK WARNING: accepted as matched_content (no reference "
            f"agreement) under amount <= {config.content_amount_tolerance_pct}% and "
            f"timestamp <= {config.content_timestamp_tolerance_hours / 24:.2f} days; "
            f"observed {_decision_text(decision)}"
        )
    if "matched_text" in labels:
        decision = next(item for item in accepted if item.label == "matched_text")
        return (
            "TEXT-DISAMBIGUATION WARNING: accepted as matched_text (ambiguous "
            "amount/timestamp candidates resolved by reference/counterparty "
            f"text similarity, min_score={config.text_similarity_min_score}, "
            f"min_margin={config.text_similarity_min_margin}); "
            f"observed {_decision_text(decision)}"
        )
    return (
        "GROUPED-MATCH WARNING: a ground-truth exception was absorbed by "
        "matched_grouped. Review Phase 3 grouping constraints and sum-only matching."
    )


def print_diagnostic(
    expected: list[ExpectedException],
    decisions: dict[str, ActualDecision],
    records: dict[str, object],
    config: MatchConfig,
) -> None:
    """Print category totals followed by every missed oracle exception."""
    misses_by_category: dict[ExceptionCategory, list[ExpectedException]] = defaultdict(list)
    detected_by_category: Counter[ExceptionCategory] = Counter()

    # A tolerance failure produces one unresolved record on each side of the
    # same normalized reference.  Treat that pair as one exception *case* so
    # precision is not artificially halved by the engine's record-level model.
    expected_cases = {
        (
            ReconciliationEngine.normalize_ref(records[item.target_ids[0]].reference_id),
            item.expected_category,
        )
        for item in expected
        if item.target_ids[0] in records
    }
    actual_cases = {
        (
            ReconciliationEngine.normalize_ref(records[record_id].reference_id),
            decision.category,
        )
        for record_id, decision in decisions.items()
        if decision.category is not None and record_id in records
    }
    true_positive_cases = expected_cases & actual_cases
    precision = len(true_positive_cases) / len(actual_cases) * 100 if actual_cases else 0.0
    recall = len(true_positive_cases) / len(expected_cases) * 100 if expected_cases else 0.0

    for item in expected:
        if _is_correct(item, decisions):
            detected_by_category[item.expected_category] += 1
        else:
            misses_by_category[item.expected_category].append(item)

    print("GROUND-TRUTH EXCEPTION DIAGNOSTIC")
    print("=" * 72)
    print(
        "Configured tolerances: "
        f"amount={config.amount_tolerance_pct}% | "
        f"fuzzy timestamp={config.timestamp_tolerance_hours / 24:.2f} days | "
        f"exact timestamp={config.exact_timestamp_tolerance_hours:.2f} hours"
    )
    print(
        "Scope: all five oracle discrepancy types are treated as exceptions; "
        "this is broader than report.py's legacy exception score."
    )
    print(
        "Case-level exception metrics: "
        f"precision={precision:.1f}% ({len(true_positive_cases)}/{len(actual_cases)}) | "
        f"recall={recall:.1f}% ({len(true_positive_cases)}/{len(expected_cases)})"
    )
    print("\nCATEGORY SUMMARY")
    for category in ExceptionCategory:
        total = sum(1 for item in expected if item.expected_category == category)
        if not total:
            continue
        detected = detected_by_category[category]
        missed = total - detected
        print(
            f"- {category.value}: missed {missed}/{total} "
            f"({missed / total * 100:.1f}%); correctly surfaced {detected}"
        )

    total_missed = sum(len(items) for items in misses_by_category.values())
    print(f"\nMISSED EXCEPTIONS: {total_missed}/{len(expected)}")
    for category in ExceptionCategory:
        misses = misses_by_category.get(category, [])
        if not misses:
            continue
        print(f"\n[{category.value}] {len(misses)} miss(es)")
        for item in misses:
            print(f"\n  {item.pair_id} | expected={item.expected_category.value}")
            print("  Ground-truth records:")
            for record_id in item.record_ids:
                record = records.get(record_id)
                print(
                    f"    - {_format_record(record)}"
                    if record is not None
                    else f"    - {record_id} (missing from input files)"
                )
            print("  Engine outcome for target record(s):")
            for record_id in item.target_ids:
                print(f"    - {record_id}: {_decision_text(decisions.get(record_id))}")
            if item.introduced_delta:
                print(f"  Oracle delta: {item.introduced_delta}")
            warning = _tolerance_warning(item, decisions, config)
            if warning:
                print(f"  {warning}")
            print(f"  Oracle note: {item.notes}")


def main(argv: Optional[list[str]] = None) -> None:
    # Windows PowerShell may start Python with a legacy code page that cannot
    # print characters contained in oracle notes (for example, the rupee sign).
    # UTF-8 keeps the diagnostic report from aborting partway through a miss.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    config = MatchConfig(
        amount_tolerance_pct=Decimal(str(args.amount_tolerance)),
        timestamp_tolerance_hours=args.timestamp_tolerance,
        exact_timestamp_tolerance_hours=args.exact_timestamp_tolerance,
    )
    gateway_records = load_gateway_records()
    ledger_records = load_ledger_records()
    result = ReconciliationEngine(config).run(gateway_records, ledger_records)
    decisions = _index_actual_output(result, classify(result.unresolved))
    records = _record_lookup([*gateway_records, *ledger_records])
    print_diagnostic(_expected_exceptions(load_ground_truth()), decisions, records, config)


if __name__ == "__main__":
    main()
