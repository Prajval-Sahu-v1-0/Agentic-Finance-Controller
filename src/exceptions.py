"""
exceptions.py — Exception Classification Logic
================================================

Classifies ``UnresolvedRecord`` objects produced by the matching engine into
five human-reviewable exception categories.

Classification rules
--------------------

The ``UnresolvedRecord.reason`` tag from the matcher is the primary signal.
For ``tolerance_exceeded`` records, ``closest_delta`` is used to decide
between ``amount_mismatch`` and ``stale_timing``.

  UnresolvedRecord.reason          -> ExceptionCategory
  -------------------------------------------------------
  no_counterpart  + source=gateway -> MISSING_IN_LEDGER
  no_counterpart  + source=ledger  -> MISSING_IN_GATEWAY
  duplicate_candidate              -> DUPLICATE
  tolerance_exceeded               -> inspect closest_delta:
      amount_delta_pct > ts threshold -> AMOUNT_MISMATCH  (primary driver)
      timestamp_delta_hours > amt thr -> STALE_TIMING     (primary driver)
      both exceeded                   -> AMOUNT_MISMATCH  (amount takes priority;
                                         noted in explanation)

Public API
----------
classify(unresolved)
    Accepts the ``result.unresolved`` list from ``ReconciliationEngine.run()``
    and returns a list of ``ExceptionRecord`` objects ready for human review.

summarise(exceptions)
    Returns a dict[ExceptionCategory, int] and a list of per-category dicts
    suitable for downstream reporting.

Usage
-----
    from src.matcher import ReconciliationEngine
    from src.exceptions import classify, summarise

    result = ReconciliationEngine().run(gateway_records, ledger_records)
    exceptions = classify(result.unresolved)
    summary, breakdown = summarise(exceptions)
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.matcher import UnresolvedRecord
from src.schema import GatewayRecord, LedgerRecord


# ---------------------------------------------------------------------------
# Exception category enum
# ---------------------------------------------------------------------------

class ExceptionCategory(str, Enum):
    """
    Five mutually-exclusive categories that cover all unresolved reconciliation
    exceptions encountered in gateway ↔ ledger matching.

    MISSING_IN_LEDGER
        A GatewayRecord exists (the gateway settled the payment) but there is
        no corresponding LedgerRecord in the merchant's internal system.
        Likely causes: missed webhook, database write failure, abandoned order
        that was nonetheless charged.

    MISSING_IN_GATEWAY
        A LedgerRecord exists (the merchant recorded an order / payment) but
        there is no corresponding GatewayRecord.
        Likely causes: payment abandoned before reaching the gateway, gateway
        webhook never delivered, order recorded optimistically before payment.

    AMOUNT_MISMATCH
        A matching reference_id was found in the other source but the amounts
        differ beyond the configured tolerance (default 2 %).
        Likely causes: gateway fee not deducted in ledger, partial refund
        applied on one side only, FX rounding, or data-entry error.

    STALE_TIMING
        A matching reference_id and near-identical amount were found but the
        timestamp gap exceeds the configured tolerance (default 3 days).
        Likely causes: delayed batch settlement, IST vs UTC offset error,
        ledger timestamp recorded at order creation rather than payment
        confirmation.

    DUPLICATE
        Two or more records in the same source share the same normalised
        reference_id (and therefore competed for the same counterpart).
        The winning record was matched; this record is the extra copy.
        Likely causes: gateway retry storm, double-batch file inclusion,
        duplicate webhook delivery, manual re-entry.
    """

    MISSING_IN_LEDGER  = "missing_in_ledger"
    MISSING_IN_GATEWAY = "missing_in_gateway"
    AMOUNT_MISMATCH    = "amount_mismatch"
    STALE_TIMING       = "stale_timing"
    DUPLICATE          = "duplicate"


# ---------------------------------------------------------------------------
# Thresholds used for AMOUNT_MISMATCH vs STALE_TIMING disambiguation
# ---------------------------------------------------------------------------

# When closest_delta is present for a tolerance_exceeded record, these
# thresholds decide which factor is the *primary* driver.  They are
# intentionally lenient — anything that survived Phase-2 fuzzy filtering
# was already beyond the engine's configured tolerances.
_AMOUNT_MISMATCH_THRESHOLD_PCT   : float = 0.5    # > 0.5 % → amount is primary
_STALE_TIMING_THRESHOLD_HOURS    : float = 2.0    # > 2 h  → timing is primary


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class ExceptionRecord(BaseModel):
    """
    A single classified reconciliation exception ready for human review.

    Attributes
    ----------
    exception_id : str
        Stable identifier in the form ``EX-{category_abbr}-{seq:06d}``
        (e.g. ``EX-AMT-000003``).  Useful for ticketing / audit trail.
    category : ExceptionCategory
        The assigned exception category.
    source : Literal["gateway", "ledger"]
        Which source the primary unresolved record came from.
    gateway_record : Optional[GatewayRecord]
        The gateway-side record, if present.
    ledger_record : Optional[LedgerRecord]
        The ledger-side record, if present.
    normalized_ref : str
        The normalised reference key shared by both sources (or just the
        primary source for no-counterpart cases).
    amount_delta_pct : Optional[Decimal]
        |gateway.amount - ledger.amount| / ledger.amount * 100.
        Populated when a closest candidate was found.
    timestamp_delta_hours : Optional[float]
        |gateway.timestamp - ledger.timestamp| in hours.
        Populated when a closest candidate was found.
    explanation : str
        One-to-three sentence human-readable description of why this record
        could not be auto-matched and what a reviewer should check.
    suggested_action : str
        Short imperative instruction for the operations / finance team.
    classified_at : datetime
        UTC datetime when this exception record was created.
    """

    exception_id          : str
    category              : ExceptionCategory
    source                : Literal["gateway", "ledger"]
    gateway_record        : Optional[GatewayRecord] = None
    ledger_record         : Optional[LedgerRecord]  = None
    normalized_ref        : str
    amount_delta_pct      : Optional[Decimal]       = None
    timestamp_delta_hours : Optional[float]         = None
    explanation           : str
    suggested_action      : str
    classified_at         : datetime


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(unresolved: list[UnresolvedRecord]) -> list[ExceptionRecord]:
    """
    Classify every ``UnresolvedRecord`` into a specific ``ExceptionCategory``
    and attach a human-readable explanation.

    Parameters
    ----------
    unresolved : list[UnresolvedRecord]
        The ``result.unresolved`` list from ``ReconciliationEngine.run()``.

    Returns
    -------
    list[ExceptionRecord]
        One ``ExceptionRecord`` per input record, ordered by category then
        by normalised reference key for easy scanning.

    Notes
    -----
    Classification is deterministic and stateless — calling this function
    twice with the same input always produces the same output.
    """
    # Per-category sequence counters for stable exception IDs
    _seq: dict[ExceptionCategory, int] = {c: 0 for c in ExceptionCategory}

    results: list[ExceptionRecord] = []

    for record in unresolved:
        category    = _determine_category(record)
        _seq[category] += 1
        seq_num     = _seq[category]
        exc_id      = _make_exception_id(category, seq_num)

        amt_pct, ts_hours = _extract_deltas(record)
        explanation, action = _build_narrative(category, record, amt_pct, ts_hours)

        results.append(ExceptionRecord(
            exception_id          = exc_id,
            category              = category,
            source                = record.source,
            gateway_record        = record.gateway_record,
            ledger_record         = record.ledger_record,
            normalized_ref        = record.normalized_ref,
            amount_delta_pct      = amt_pct,
            timestamp_delta_hours = ts_hours,
            explanation           = explanation,
            suggested_action      = action,
            classified_at         = datetime.now(timezone.utc),
        ))

    # Sort: category alphabetically, then ref for stable ordering
    results.sort(key=lambda e: (e.category.value, e.normalized_ref))
    return results


def summarise(
    exceptions: list[ExceptionRecord],
) -> tuple[dict[str, int], list[dict]]:
    """
    Aggregate classified exceptions into counts and per-category breakdowns.

    Parameters
    ----------
    exceptions : list[ExceptionRecord]
        Output of ``classify()``.

    Returns
    -------
    counts : dict[str, int]
        Mapping of ``ExceptionCategory.value`` -> count.
    breakdown : list[dict]
        One dict per category containing count, percentage, and example
        exception_ids (up to 3).  Sorted by count descending.

    Examples
    --------
    >>> counts, breakdown = summarise(exceptions)
    >>> print(counts)
    {'stale_timing': 10, 'missing_in_ledger': 2, ...}
    """
    counts: dict[str, int] = {}
    by_cat: dict[str, list[ExceptionRecord]] = {}

    for exc in exceptions:
        key = exc.category.value
        counts[key]  = counts.get(key, 0) + 1
        by_cat.setdefault(key, []).append(exc)

    total = len(exceptions) or 1   # guard divide-by-zero

    breakdown = [
        {
            "category"     : cat,
            "count"        : cnt,
            "share_pct"    : round(cnt / total * 100, 1),
            "example_ids"  : [e.exception_id for e in by_cat[cat][:3]],
        }
        for cat, cnt in sorted(counts.items(), key=lambda kv: -kv[1])
    ]

    return counts, breakdown


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _determine_category(record: UnresolvedRecord) -> ExceptionCategory:
    """
    Map an ``UnresolvedRecord`` to an ``ExceptionCategory``.

    Decision tree
    -------------
    1. reason == no_counterpart
           source == gateway -> MISSING_IN_LEDGER
           source == ledger  -> MISSING_IN_GATEWAY
    2. reason == duplicate_candidate
           -> DUPLICATE
    3. reason == tolerance_exceeded
           inspect closest_delta:
             amount_delta_pct > _AMOUNT_MISMATCH_THRESHOLD_PCT -> AMOUNT_MISMATCH
             timestamp_delta_hours > _STALE_TIMING_THRESHOLD_HOURS -> STALE_TIMING
             neither (closest_delta absent or both below threshold) -> AMOUNT_MISMATCH
               (conservative default — amount discrepancy is the safer assumption)
    """
    reason = record.reason

    if reason == "no_counterpart":
        return (
            ExceptionCategory.MISSING_IN_LEDGER
            if record.source == "gateway"
            else ExceptionCategory.MISSING_IN_GATEWAY
        )

    if reason == "duplicate_candidate":
        return ExceptionCategory.DUPLICATE

    # tolerance_exceeded — inspect closest_delta to find primary driver
    delta = record.closest_delta or {}
    raw_amt = delta.get("amount_delta_pct", "0")
    raw_ts  = delta.get("timestamp_delta_hours", 0.0)

    try:
        amt_pct  = float(Decimal(str(raw_amt)))
    except Exception:
        amt_pct = 0.0
    ts_hours = float(raw_ts)

    amt_exceeded = amt_pct  > _AMOUNT_MISMATCH_THRESHOLD_PCT
    ts_exceeded  = ts_hours > _STALE_TIMING_THRESHOLD_HOURS

    if amt_exceeded and ts_exceeded:
        # Both exceeded — amount takes priority (more likely actionable first)
        return ExceptionCategory.AMOUNT_MISMATCH
    if amt_exceeded:
        return ExceptionCategory.AMOUNT_MISMATCH
    if ts_exceeded:
        return ExceptionCategory.STALE_TIMING

    # Fallback: no clear signal — treat as amount mismatch for safety
    return ExceptionCategory.AMOUNT_MISMATCH


def _extract_deltas(
    record: UnresolvedRecord,
) -> tuple[Optional[Decimal], Optional[float]]:
    """
    Pull amount_delta_pct and timestamp_delta_hours from closest_delta,
    returning (None, None) if no candidate was found.
    """
    delta = record.closest_delta
    if not delta:
        return None, None
    try:
        amt_pct = Decimal(str(delta["amount_delta_pct"]))
    except (KeyError, Exception):
        amt_pct = None
    try:
        ts_hours = float(delta["timestamp_delta_hours"])
    except (KeyError, Exception):
        ts_hours = None
    return amt_pct, ts_hours


def _make_exception_id(category: ExceptionCategory, seq: int) -> str:
    """
    Build a stable, human-readable exception ID.

    Format: ``EX-{ABBR}-{seq:06d}``

    Abbreviation map
    ----------------
    MISSING_IN_LEDGER  -> MIL
    MISSING_IN_GATEWAY -> MIG
    AMOUNT_MISMATCH    -> AMT
    STALE_TIMING       -> STL
    DUPLICATE          -> DUP
    """
    abbr = {
        ExceptionCategory.MISSING_IN_LEDGER  : "MIL",
        ExceptionCategory.MISSING_IN_GATEWAY : "MIG",
        ExceptionCategory.AMOUNT_MISMATCH    : "AMT",
        ExceptionCategory.STALE_TIMING       : "STL",
        ExceptionCategory.DUPLICATE          : "DUP",
    }[category]
    return f"EX-{abbr}-{seq:06d}"


def _build_narrative(
    category  : ExceptionCategory,
    record    : UnresolvedRecord,
    amt_pct   : Optional[Decimal],
    ts_hours  : Optional[float],
) -> tuple[str, str]:
    """
    Return (explanation, suggested_action) strings for the given category.

    The explanation is written in plain language for a finance operations
    reviewer, not a developer.  It includes concrete values wherever
    closest_delta provides them.
    """
    ref  = record.normalized_ref
    src  = record.source.capitalize()

    # Helper to format the primary record's key fields for context
    if record.source == "gateway" and record.gateway_record:
        r = record.gateway_record
        rec_summary = (
            f"Gateway txn {r.transaction_id} (ref: {r.reference_id}, "
            f"amount: {r.currency} {r.amount}, status: {r.status})"
        )
    elif record.source == "ledger" and record.ledger_record:
        r = record.ledger_record
        rec_summary = (
            f"Ledger txn {r.transaction_id} (ref: {r.reference_id}, "
            f"amount: {r.currency} {r.amount}, status: {r.status})"
        )
    else:
        rec_summary = f"{src} record with normalised ref {ref}"

    # ── MISSING_IN_LEDGER ────────────────────────────────────────────
    if category == ExceptionCategory.MISSING_IN_LEDGER:
        explanation = (
            f"{rec_summary} was settled by the payment gateway but has no "
            f"matching entry in the merchant's ledger (normalised ref: {ref}). "
            "This may indicate a missed webhook, a database write failure on "
            "the merchant's backend, or a charge that was processed after the "
            "corresponding order was closed."
        )
        action = (
            "Search the ledger for this reference in pending/failed states. "
            "If absent, investigate webhook delivery logs and create a manual "
            "ledger entry after confirmation."
        )

    # ── MISSING_IN_GATEWAY ───────────────────────────────────────────
    elif category == ExceptionCategory.MISSING_IN_GATEWAY:
        explanation = (
            f"{rec_summary} exists in the merchant's ledger but has no "
            f"corresponding gateway settlement record (normalised ref: {ref}). "
            "This typically means the customer initiated but did not complete "
            "payment, the gateway rejected the transaction without notifying "
            "the merchant, or the record belongs to a different payment method "
            "not covered by this gateway export."
        )
        action = (
            "Check the gateway dashboard for this reference. "
            "If confirmed unpaid, update the ledger status to 'failed' and "
            "trigger any retry / dunning logic. If paid via alternate method, "
            "ensure that source is included in future reconciliation runs."
        )

    # ── AMOUNT_MISMATCH ──────────────────────────────────────────────
    elif category == ExceptionCategory.AMOUNT_MISMATCH:
        if amt_pct is not None and ts_hours is not None:
            detail = (
                f"The closest candidate differs by {amt_pct} % in amount "
                f"and {ts_hours:.1f} h in timestamp — both beyond auto-match "
                "tolerances."
            )
        elif amt_pct is not None:
            detail = (
                f"The closest candidate differs by {amt_pct} % in amount, "
                "which exceeds the configured tolerance."
            )
        else:
            detail = "No candidate was found within configured tolerances."

        explanation = (
            f"{rec_summary} could not be matched due to an amount discrepancy. "
            f"{detail} "
            "Common causes: gateway processing fee deducted before settlement, "
            "partial refund applied in one source only, FX conversion rounding, "
            "or a data-entry error in the ledger."
        )
        action = (
            "Compare the exact amounts in both the gateway dashboard and the "
            "ledger. Confirm whether a fee, refund, or FX adjustment accounts "
            "for the difference. Adjust the ledger or raise a dispute with the "
            "gateway as appropriate."
        )

    # ── STALE_TIMING ─────────────────────────────────────────────────
    elif category == ExceptionCategory.STALE_TIMING:
        if ts_hours is not None:
            ts_days = round(ts_hours / 24, 1)
            detail  = (
                f"The closest candidate's timestamp is {ts_hours:.1f} h "
                f"({ts_days} days) away from this record's timestamp."
            )
        else:
            detail = "No candidate within the timing window was found."

        explanation = (
            f"{rec_summary} has a matching reference and near-identical amount "
            f"in the other source, but the timestamps are too far apart to "
            f"auto-match. {detail} "
            "Typical causes: gateway T+2 settlement batch delay, IST-to-UTC "
            "timezone conversion error (off by 5 h 30 min), or the ledger "
            "capturing the order-creation time instead of the payment time."
        )
        action = (
            "Verify that both timestamps refer to the same event (settlement "
            "vs. order creation). Check for timezone issues in the data "
            "pipeline. If the records are confirmed to be the same transaction, "
            "widen the timestamp tolerance or reconcile manually."
        )

    # ── DUPLICATE ────────────────────────────────────────────────────
    elif category == ExceptionCategory.DUPLICATE:
        explanation = (
            f"{rec_summary} shares the same normalised reference ({ref}) with "
            "another record in the same source that was already successfully "
            "matched. This record is the extra copy and could not be paired. "
            "Likely causes: gateway retry storm, double-batch file inclusion, "
            "or a duplicate webhook delivery processed twice by the merchant."
        )
        action = (
            f"Search the {src.lower()} system for all records with reference "
            f"{ref}. Confirm which is the canonical record (check timestamps "
            "and transaction IDs) and void or reverse the duplicate. Contact "
            "the gateway if the duplication originated on their side."
        )

    else:
        # Exhaustiveness guard — should never reach here
        explanation  = "Unclassified exception."
        action       = "Manual review required."

    return explanation, action
