"""
generator.py — Synthetic Data Generator
========================================

Generates 100 paired GatewayRecord / LedgerRecord entries that simulate the
messy reality of multi-source financial reconciliation.  All randomness is
seeded for full reproducibility.

Discrepancy breakdown (out of 100 logical pairs)
-------------------------------------------------
  65  exact_match       — amounts identical, timestamps within 30 min,
                          reference IDs differ only in formatting prefix.
  10  amount_mismatch   — gateway settled 2 % less than ledger (fee deduction).
  10  timestamp_drift   — gateway settlement lagged 1-3 days after ledger entry.
   3  missing_gateway   — ledger record exists; gateway record absent.
   2  missing_ledger    — gateway record exists; ledger record absent.
   5  duplicate         — gateway emitted two rows for the same reference_id.
   3  one_to_many       — 1 GW settlement covers 2-3 ledger orders (batch payout).
   2  many_to_one       — 2 GW partial payments sum to 1 ledger order.

Total records produced
----------------------
  gateway_records : 107   (65+10+10+2+10 dup + 3 otm + 4 mto + 3 extra otm-gw)
  ledger_records  : 106   (65+10+10+3+5 dup + 7 otm-led + 2 mto-led)

Output files  (data/)
---------------------
  gateway_records.json   — list of GatewayRecord dicts
  ledger_records.json    — list of LedgerRecord dicts
  ground_truth.json      — metadata + per-pair labels used to score the matcher

Usage
-----
    # From the project root with the virtualenv active:
    python -m src.generator

    # Or programmatically:
    from src.generator import generate_dataset
    gw, led, gt = generate_dataset(seed=42, save=True)
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Optional

from src.config import (
    EXCEPTION_SAFETY_MARGIN,
    MATCHER_AMOUNT_TOLERANCE_PCT,
    MATCHER_TIMESTAMP_TOLERANCE_HOURS,
)
from src.schema import GatewayRecord, LedgerRecord


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

DEFAULT_SEED    = 42
FEE_RATE        = (
    MATCHER_AMOUNT_TOLERANCE_PCT * EXCEPTION_SAFETY_MARGIN / Decimal("100")
)
BASE_DATE       = datetime(2024, 3, 1, tzinfo=timezone.utc)   # window start
DATA_DIR        = Path(__file__).parent.parent / "data"

# Target pair counts — must reflect intended discrepancy distribution.
PAIR_COUNTS: dict[str, int] = {
    "exact_match"      : 65,
    "amount_mismatch"  : 10,
    "timestamp_drift"  : 10,
    "missing_gateway"  :  3,   # ledger-only records
    "missing_ledger"   :  2,   # gateway-only records
    "duplicate"        :  5,   # base pair + one extra gateway row
    "one_to_many"      :  3,   # 1 GW batch settlement covers 2-3 ledger orders
    "many_to_one"      :  2,   # 2 GW partial payments sum to 1 ledger record
}

# Number of child records per grouped group (index = group number)
_OTM_CHILD_COUNTS = [2, 2, 3]   # ledger children per one_to_many group
_MTO_CHILD_COUNTS = [2, 2]      # gateway children per many_to_one group

# ---------------------------------------------------------------------------
# Realistic data pools
# ---------------------------------------------------------------------------

_UPI_HANDLES = [
    "rahul@ybl",      "priya@oksbi",    "amit@paytm",
    "neha@apl",       "suresh@icici",   "pooja@axl",
    "vikas@upi",      "anita@hdfcbank", "raj@okicici",
    "divya@paytm",    "kiran@ybl",      "meera@oksbi",
    "tarun@paytm",    "swati@apl",      "rohit@icici",
]

_CUSTOMER_IDS = [
    "CUST-10021", "CUST-10022", "CUST-10023", "CUST-10024", "CUST-10025",
    "CUST-10026", "CUST-10027", "CUST-10028", "CUST-10029", "CUST-10030",
    "CUST-10031", "CUST-10032", "CUST-10033", "CUST-10034", "CUST-10035",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _random_amount(rng: random.Random) -> Decimal:
    """Return a realistic INR transaction amount between ₹100 and ₹50,000."""
    raw = round(rng.uniform(100.0, 50_000.0), 2)
    return Decimal(str(raw)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _random_ledger_ts(rng: random.Random) -> datetime:
    """Return a random UTC datetime within 30 days of BASE_DATE."""
    offset = rng.randint(0, 30 * 24 * 3_600)
    return BASE_DATE + timedelta(seconds=offset)


def _gateway_ref(seq: int, date: datetime) -> str:
    """
    Gateway reference format: ``PAY{YYYYMMDD}{seq:06d}``

    Example: ``PAY20240315000042``
    Deliberately different from the ledger format to exercise the reference-ID
    normalisation logic in the matcher.
    """
    return f"PAY{date.strftime('%Y%m%d')}{seq:06d}"


def _ledger_ref(seq: int, date: datetime) -> str:
    """
    Ledger reference format: ``ORD-{YYYY-MM-DD}-{seq:06d}``

    Example: ``ORD-2024-03-15-000042``
    Same underlying sequence number as the gateway reference, formatted
    differently — the canonical test of reference-ID fuzzy matching.
    """
    return f"ORD-{date.strftime('%Y-%m-%d')}-{seq:06d}"


def _batch_ref(seq: int, date: datetime) -> str:
    """
    Gateway batch-settlement reference: ``BATCH{YYYYMMDD}{seq:06d}``

    Used for one_to_many cases.  Deliberately uses a prefix (BATCH) whose
    digit-stripped form shares NO overlap with any of the individual ledger
    order references (which have unique seq numbers), so the Phase-1/2 ref
    index never links them — the grouped phase must find the match by sum.
    """
    return f"BATCH{date.strftime('%Y%m%d')}{seq:06d}"


def _split_ref(seq: int, part: int, date: datetime) -> str:
    """
    Gateway partial-payment reference: ``SPLIT{YYYYMMDD}{seq:06d}P{part}``

    Used for many_to_one cases.  Each split GW record has a different
    reference from the ledger's payout reference, so no ref-based match
    fires and the grouped phase must discover the sum relationship.
    """
    return f"SPLIT{date.strftime('%Y%m%d')}{seq:06d}P{part}"


def _payout_ref(seq: int, date: datetime) -> str:
    """Ledger payout reference for many_to_one: ``PAYOUT-{YYYY-MM-DD}-{seq:06d}``"""
    return f"PAYOUT-{date.strftime('%Y-%m-%d')}-{seq:06d}"


def _gw_txn_id(rng: random.Random) -> str:
    """Simulate a Razorpay-style live transaction ID."""
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    suffix = "".join(rng.choices(chars, k=14))
    return f"rzp_live_{suffix}"


def _led_txn_id(seq: int) -> str:
    return f"LED-2024-{seq:06d}"


def _counterparties(rng: random.Random) -> tuple[str, str]:
    """Return a (gateway_upi_handle, ledger_customer_id) pair."""
    idx = rng.randrange(len(_UPI_HANDLES))
    return _UPI_HANDLES[idx], _CUSTOMER_IDS[idx % len(_CUSTOMER_IDS)]


def _serialize(record: GatewayRecord | LedgerRecord) -> dict:
    """Serialize a Pydantic model to a JSON-safe plain dict."""
    d = record.model_dump()
    for k, v in d.items():
        if isinstance(v, Decimal):
            d[k] = str(v)
        elif isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_dataset(
    seed: int = DEFAULT_SEED,
    output_dir: Path = DATA_DIR,
    save: bool = True,
) -> tuple[list[GatewayRecord], list[LedgerRecord], dict]:
    """
    Generate a synthetic, labeled reconciliation dataset and (optionally) save
    it to disk as three JSON files.

    Parameters
    ----------
    seed : int
        RNG seed for full reproducibility.  Same seed always produces the same
        dataset, which is critical for deterministic matcher evaluation.
    output_dir : Path
        Directory where ``gateway_records.json``, ``ledger_records.json``, and
        ``ground_truth.json`` will be written.  Created if it does not exist.
    save : bool
        If ``True`` (default), serialize and write all three files.
        Set to ``False`` to generate in-memory only (useful in unit tests).

    Returns
    -------
    gateway_records : list[GatewayRecord]
        102 gateway-side records (70 exact + 10 amount + 10 ts + 2 missing +
        10 from 5 duplicate pairs).
    ledger_records : list[LedgerRecord]
        98 ledger-side records  (70 exact + 10 amount + 10 ts + 3 missing +
        5 from duplicate pairs — one ledger row per duplicate pair).
    ground_truth : dict
        Structured ground-truth document containing:
          - ``metadata``          : generation parameters and record counts
          - ``discrepancy_counts``: per-type summary
          - ``pairs``             : one entry per logical pair with:
              pair_id, seq, gateway_transaction_id, ledger_transaction_id,
              discrepancy_type, expected_match_status, introduced_delta, notes

    Notes
    -----
    The ground truth is intentionally kept separate from both source files so
    it can serve as an objective oracle during matcher evaluation.  The matcher
    should never read ground_truth.json — only the scoring / reporting layer
    should compare its output against it.
    """
    rng = random.Random(seed)

    gateway_records : list[GatewayRecord] = []
    ledger_records  : list[LedgerRecord]  = []
    gt_pairs        : list[dict]          = []

    seq      = 1    # shared sequence number — the "hidden" link between sources
    pair_num = 1

    # ------------------------------------------------------------------
    # 1. EXACT MATCHES  (~70 %)
    # ------------------------------------------------------------------
    for _ in range(PAIR_COUNTS["exact_match"]):
        led_ts  = _random_ledger_ts(rng)
        gw_ts   = led_ts + timedelta(minutes=rng.randint(0, 30))
        amount  = _random_amount(rng)
        upi, cid = _counterparties(rng)

        gw = GatewayRecord(
            transaction_id = _gw_txn_id(rng),
            reference_id   = _gateway_ref(seq, led_ts),
            amount         = amount,
            currency       = "INR",
            timestamp      = gw_ts,
            status         = "settled",
            counterparty   = upi,
        )
        led = LedgerRecord(
            transaction_id = _led_txn_id(seq),
            reference_id   = _ledger_ref(seq, led_ts),
            amount         = amount,
            currency       = "INR",
            timestamp      = led_ts,
            status         = "paid",
            counterparty   = cid,
        )
        gateway_records.append(gw)
        ledger_records.append(led)

        gt_pairs.append({
            "pair_id"                  : f"PAIR-{pair_num:04d}",
            "seq"                      : seq,
            "gateway_transaction_id"   : gw.transaction_id,
            "ledger_transaction_id"    : led.transaction_id,
            "discrepancy_type"         : "exact_match",
            "expected_match_status"    : "matched",
            "introduced_delta"         : None,
            "notes"                    : (
                "Amounts are identical. Gateway timestamp is 0-30 min after the "
                "ledger timestamp (normal sub-hour settlement lag). "
                "Reference IDs share sequence {seq} but use different formatting "
                "prefixes (PAY... vs ORD-...) to simulate the most common "
                "real-world reference mismatch."
            ).format(seq=seq),
        })
        seq += 1
        pair_num += 1

    # ------------------------------------------------------------------
    # 2. AMOUNT MISMATCHES  (~10 %)
    #    Gateway settled net amount = ledger gross amount − 2 % fee.
    # ------------------------------------------------------------------
    for _ in range(PAIR_COUNTS["amount_mismatch"]):
        led_ts         = _random_ledger_ts(rng)
        gw_ts          = led_ts + timedelta(minutes=rng.randint(0, 30))
        ledger_amount  = _random_amount(rng)
        fee            = (ledger_amount * FEE_RATE).quantize(
                            Decimal("0.01"), rounding=ROUND_HALF_UP)
        gw_amount      = ledger_amount - fee
        upi, cid       = _counterparties(rng)

        gw = GatewayRecord(
            transaction_id = _gw_txn_id(rng),
            reference_id   = _gateway_ref(seq, led_ts),
            amount         = gw_amount,
            currency       = "INR",
            timestamp      = gw_ts,
            status         = "settled",
            counterparty   = upi,
        )
        led = LedgerRecord(
            transaction_id = _led_txn_id(seq),
            reference_id   = _ledger_ref(seq, led_ts),
            amount         = ledger_amount,
            currency       = "INR",
            timestamp      = led_ts,
            status         = "paid",
            counterparty   = cid,
        )
        gateway_records.append(gw)
        ledger_records.append(led)

        gt_pairs.append({
            "pair_id"                : f"PAIR-{pair_num:04d}",
            "seq"                    : seq,
            "gateway_transaction_id" : gw.transaction_id,
            "ledger_transaction_id"  : led.transaction_id,
            "discrepancy_type"       : "amount_mismatch",
            "expected_match_status"  : "partial",
            "introduced_delta"       : {
                "ledger_amount"  : str(ledger_amount),
                "gateway_amount" : str(gw_amount),
                "fee_deducted"   : str(fee),
                "fee_rate_pct"   : str(MATCHER_AMOUNT_TOLERANCE_PCT * EXCEPTION_SAFETY_MARGIN),
            },
            "notes": (
                f"Gateway deducted a 2 % processing fee (₹{fee}). "
                "The ledger holds the gross order value; the gateway settled "
                "the net value. Amount delta is always negative (gateway < ledger)."
            ),
        })
        seq += 1
        pair_num += 1

    # ------------------------------------------------------------------
    # 3. TIMESTAMP DRIFT  (~10 %)
    #    Same amount + reference; gateway settled 1-3 days later.
    # ------------------------------------------------------------------
    for _ in range(PAIR_COUNTS["timestamp_drift"]):
        led_ts     = _random_ledger_ts(rng)
        minimum_drift_hours = (
            MATCHER_TIMESTAMP_TOLERANCE_HOURS * float(EXCEPTION_SAFETY_MARGIN)
        )
        gw_ts = led_ts + timedelta(
            hours=minimum_drift_hours + rng.randint(0, 24)
        )
        amount     = _random_amount(rng)
        upi, cid   = _counterparties(rng)

        gw = GatewayRecord(
            transaction_id = _gw_txn_id(rng),
            reference_id   = _gateway_ref(seq, led_ts),
            amount         = amount,
            currency       = "INR",
            timestamp      = gw_ts,
            status         = "settled",
            counterparty   = upi,
        )
        led = LedgerRecord(
            transaction_id = _led_txn_id(seq),
            reference_id   = _ledger_ref(seq, led_ts),
            amount         = amount,
            currency       = "INR",
            timestamp      = led_ts,
            status         = "paid",
            counterparty   = cid,
        )
        gateway_records.append(gw)
        ledger_records.append(led)

        drift_secs = (gw_ts - led_ts).total_seconds()
        gt_pairs.append({
            "pair_id"                : f"PAIR-{pair_num:04d}",
            "seq"                    : seq,
            "gateway_transaction_id" : gw.transaction_id,
            "ledger_transaction_id"  : led.transaction_id,
            "discrepancy_type"       : "timestamp_drift",
            "expected_match_status"  : "partial",
            "introduced_delta"       : {
                "ledger_timestamp"  : led_ts.isoformat(),
                "gateway_timestamp" : gw_ts.isoformat(),
                "drift_days"        : round(drift_secs / 86_400, 2),
                "drift_seconds"     : drift_secs,
            },
            "notes": (
                f"Gateway settlement was delayed by at least {minimum_drift_hours / 24:.1f} day(s) "
                f"({drift_secs:,.0f} s total). Amounts are identical; only the "
                "timestamp gap exceeds the normal sub-hour tolerance window."
            ),
        })
        seq += 1
        pair_num += 1

    # ------------------------------------------------------------------
    # 4. MISSING — ledger-only  (no gateway)  (~3 records)
    #    Represents abandoned checkouts or failed gateway webhooks.
    # ------------------------------------------------------------------
    for _ in range(PAIR_COUNTS["missing_gateway"]):
        led_ts   = _random_ledger_ts(rng)
        amount   = _random_amount(rng)
        _, cid   = _counterparties(rng)

        led = LedgerRecord(
            transaction_id = _led_txn_id(seq),
            reference_id   = _ledger_ref(seq, led_ts),
            amount         = amount,
            currency       = "INR",
            timestamp      = led_ts,
            status         = "pending",          # never received gateway confirmation
            counterparty   = cid,
        )
        ledger_records.append(led)

        gt_pairs.append({
            "pair_id"                : f"PAIR-{pair_num:04d}",
            "seq"                    : seq,
            "gateway_transaction_id" : None,
            "ledger_transaction_id"  : led.transaction_id,
            "discrepancy_type"       : "missing_gateway",
            "expected_match_status"  : "unmatched_ledger",
            "introduced_delta"       : None,
            "notes"                  : (
                "Ledger recorded a pending payment; the gateway has no "
                "corresponding settlement record. Typical causes: customer "
                "abandoned the checkout before payment, or the gateway's "
                "success webhook was never delivered to the merchant."
            ),
        })
        seq += 1
        pair_num += 1

    # ------------------------------------------------------------------
    # 5. MISSING — gateway-only  (no ledger)  (~2 records)
    #    Represents successful gateway settlements with no internal record.
    # ------------------------------------------------------------------
    for _ in range(PAIR_COUNTS["missing_ledger"]):
        led_ts   = _random_ledger_ts(rng)
        gw_ts    = led_ts + timedelta(hours=rng.randint(1, 6))
        amount   = _random_amount(rng)
        upi, _   = _counterparties(rng)

        gw = GatewayRecord(
            transaction_id = _gw_txn_id(rng),
            reference_id   = _gateway_ref(seq, led_ts),
            amount         = amount,
            currency       = "INR",
            timestamp      = gw_ts,
            status         = "settled",
            counterparty   = upi,
        )
        gateway_records.append(gw)

        gt_pairs.append({
            "pair_id"                : f"PAIR-{pair_num:04d}",
            "seq"                    : seq,
            "gateway_transaction_id" : gw.transaction_id,
            "ledger_transaction_id"  : None,
            "discrepancy_type"       : "missing_ledger",
            "expected_match_status"  : "unmatched_gateway",
            "introduced_delta"       : None,
            "notes"                  : (
                "Gateway processed and settled this payment, but the merchant's "
                "internal ledger has no record of it. Likely caused by a database "
                "write failure on the merchant side after the payment was confirmed."
            ),
        })
        seq += 1
        pair_num += 1

    # ------------------------------------------------------------------
    # 6. DUPLICATES  (~5 %)
    #    Base pair is a valid match; the gateway also contains a second row
    #    with the same reference_id (retry storm / double-batch inclusion).
    #    Both gateway rows are present in gateway_records; only one ledger
    #    row exists — the matcher must flag this as a duplicate.
    # ------------------------------------------------------------------
    for _ in range(PAIR_COUNTS["duplicate"]):
        led_ts        = _random_ledger_ts(rng)
        gw_ts_base    = led_ts + timedelta(minutes=rng.randint(0, 30))
        amount        = _random_amount(rng)
        upi, cid      = _counterparties(rng)
        ref_id        = _gateway_ref(seq, led_ts)

        gw_orig = GatewayRecord(
            transaction_id = _gw_txn_id(rng),
            reference_id   = ref_id,
            amount         = amount,
            currency       = "INR",
            timestamp      = gw_ts_base,
            status         = "settled",
            counterparty   = upi,
        )
        # Duplicate: different transaction_id, same reference_id + amount.
        # Timestamp is 1-5 min later (retry).
        gw_dup = GatewayRecord(
            transaction_id = _gw_txn_id(rng),
            reference_id   = ref_id,                    # ← identical reference_id
            amount         = amount,                    # ← identical amount
            currency       = "INR",
            timestamp      = gw_ts_base + timedelta(minutes=rng.randint(1, 5)),
            status         = "settled",
            counterparty   = upi,
        )
        led = LedgerRecord(
            transaction_id = _led_txn_id(seq),
            reference_id   = _ledger_ref(seq, led_ts),
            amount         = amount,
            currency       = "INR",
            timestamp      = led_ts,
            status         = "paid",
            counterparty   = cid,
        )

        gateway_records.append(gw_orig)
        gateway_records.append(gw_dup)      # ← the injected duplicate
        ledger_records.append(led)

        gt_pairs.append({
            "pair_id"                          : f"PAIR-{pair_num:04d}",
            "seq"                              : seq,
            "gateway_transaction_id"           : gw_orig.transaction_id,
            "gateway_duplicate_transaction_id" : gw_dup.transaction_id,
            "ledger_transaction_id"            : led.transaction_id,
            "discrepancy_type"                 : "duplicate",
            "expected_match_status"            : "duplicate",
            "introduced_delta"                 : {
                "shared_reference_id"          : ref_id,
                "original_gw_txn_id"           : gw_orig.transaction_id,
                "duplicate_gw_txn_id"          : gw_dup.transaction_id,
                "timestamp_gap_seconds"        : (
                    gw_dup.timestamp - gw_orig.timestamp
                ).total_seconds(),
            },
            "notes": (
                f"Gateway emitted two settlement rows for reference {ref_id}. "
                "Both share the same amount and counterparty; only their "
                "transaction_id and timestamp differ. "
                "Caused by a gateway retry storm or double-batch file inclusion. "
                "The matcher should flag both gateway rows and the single "
                "ledger row as a DUPLICATE exception."
            ),
        })
        seq += 1
        pair_num += 1

    # ------------------------------------------------------------------
    # 7.  ONE-TO-MANY  (~3 %)
    #     A single gateway batch settlement covers 2-3 individual ledger
    #     orders.  The GW record uses a BATCH reference that shares no
    #     digits with the individual ledger ORD references, so Phases 1+2
    #     cannot match by reference key — Phase 3 must find the sum.
    # ------------------------------------------------------------------
    for i, n_children in enumerate(_OTM_CHILD_COUNTS):
        led_ts    = _random_ledger_ts(rng)
        gw_ts     = led_ts + timedelta(hours=rng.randint(1, 6))
        upi, _    = _counterparties(rng)
        batch_seq = seq

        # Generate n_children ledger orders with individual amounts
        child_amounts = [_random_amount(rng) for _ in range(n_children)]
        batch_amount  = sum(child_amounts)

        gw_batch = GatewayRecord(
            transaction_id = _gw_txn_id(rng),
            reference_id   = _batch_ref(batch_seq, led_ts),   # BATCH... prefix
            amount         = batch_amount,
            currency       = "INR",
            timestamp      = gw_ts,
            status         = "settled",
            counterparty   = upi,
        )
        gateway_records.append(gw_batch)
        seq += 1  # advance seq so batch ref is unique

        child_led_records = []
        child_led_ids     = []
        child_led_refs    = []
        for j, amt in enumerate(child_amounts):
            _, cid = _counterparties(rng)
            child_ts = led_ts - timedelta(hours=rng.randint(0, 4))
            led_child = LedgerRecord(
                transaction_id = _led_txn_id(seq),
                reference_id   = _ledger_ref(seq, led_ts),    # ORD-... prefix
                amount         = amt,
                currency       = "INR",
                timestamp      = child_ts,
                status         = "paid",
                counterparty   = cid,
            )
            ledger_records.append(led_child)
            child_led_records.append(led_child)
            child_led_ids.append(led_child.transaction_id)
            child_led_refs.append(led_child.reference_id)
            seq += 1

        gt_pairs.append({
            "pair_id"                  : f"PAIR-{pair_num:04d}",
            "seq"                      : batch_seq,
            "gateway_transaction_id"   : gw_batch.transaction_id,
            "ledger_transaction_ids"   : child_led_ids,        # plural — list
            "ledger_transaction_id"    : None,                 # not a 1:1 pair
            "discrepancy_type"         : "one_to_many",
            "expected_match_status"    : "grouped_one_to_many",
            "introduced_delta"         : {
                "batch_gateway_ref"    : gw_batch.reference_id,
                "batch_amount"         : str(batch_amount),
                "child_ledger_refs"    : child_led_refs,
                "child_amounts"        : [str(a) for a in child_amounts],
                "n_children"           : n_children,
            },
            "notes": (
                f"Gateway issued a single batch settlement (ref: {gw_batch.reference_id}) "
                f"covering {n_children} individual ledger orders. "
                f"The batch amount (INR {batch_amount}) equals the exact sum of "
                "the child order amounts. Phases 1+2 cannot match by ref because "
                "BATCH... and ORD-... references share no digit overlap. "
                "Phase 3 must discover the match via sum equality."
            ),
        })
        pair_num += 1

    # ------------------------------------------------------------------
    # 8.  MANY-TO-ONE  (~2 %)
    #     A customer split a large payment across 2 gateway transactions
    #     (e.g. partial COD + UPI).  The ledger has one combined order
    #     record.  The GW records use SPLIT references; the ledger uses a
    #     PAYOUT reference — no shared digit key, so Phase 3 must sum.
    # ------------------------------------------------------------------
    for i, n_parts in enumerate(_MTO_CHILD_COUNTS):
        led_ts     = _random_ledger_ts(rng)
        _, cid     = _counterparties(rng)
        payout_seq = seq

        # Generate n_parts partial GW payments that sum to one ledger amount
        part_amounts = [_random_amount(rng) for _ in range(n_parts - 1)]
        total_amount = sum(part_amounts) + _random_amount(rng)
        # Make last part fill the gap exactly
        last_part = total_amount - sum(part_amounts)
        part_amounts.append(last_part)

        child_gw_records = []
        child_gw_ids     = []
        child_gw_refs    = []
        for part_idx, amt in enumerate(part_amounts):
            upi, _ = _counterparties(rng)
            gw_ts  = led_ts + timedelta(minutes=rng.randint(0, 60))
            gw_part = GatewayRecord(
                transaction_id = _gw_txn_id(rng),
                reference_id   = _split_ref(seq, part_idx + 1, led_ts),
                amount         = amt,
                currency       = "INR",
                timestamp      = gw_ts,
                status         = "settled",
                counterparty   = upi,
            )
            gateway_records.append(gw_part)
            child_gw_records.append(gw_part)
            child_gw_ids.append(gw_part.transaction_id)
            child_gw_refs.append(gw_part.reference_id)
            seq += 1

        led_payout = LedgerRecord(
            transaction_id = _led_txn_id(seq),
            reference_id   = _payout_ref(payout_seq, led_ts),
            amount         = total_amount,
            currency       = "INR",
            timestamp      = led_ts,
            status         = "paid",
            counterparty   = cid,
        )
        ledger_records.append(led_payout)
        seq += 1

        gt_pairs.append({
            "pair_id"                  : f"PAIR-{pair_num:04d}",
            "seq"                      : payout_seq,
            "gateway_transaction_ids"  : child_gw_ids,         # plural — list
            "gateway_transaction_id"   : None,                 # not a 1:1 pair
            "ledger_transaction_id"    : led_payout.transaction_id,
            "discrepancy_type"         : "many_to_one",
            "expected_match_status"    : "grouped_many_to_one",
            "introduced_delta"         : {
                "payout_ledger_ref"    : led_payout.reference_id,
                "total_amount"         : str(total_amount),
                "split_gateway_refs"   : child_gw_refs,
                "split_amounts"        : [str(a) for a in part_amounts],
                "n_parts"              : n_parts,
            },
            "notes": (
                f"Customer split a single order (ledger ref: {led_payout.reference_id}, "
                f"amount: INR {total_amount}) across {n_parts} gateway transactions "
                "(e.g. partial UPI + partial card). Each split GW record uses a "
                "SPLIT... reference; the ledger uses a PAYOUT reference. No shared "
                "digit key exists between them. Phase 3 must find the match by summing "
                "the gateway parts."
            ),
        })
        pair_num += 1

    # ------------------------------------------------------------------
    # Assemble ground truth document
    # ------------------------------------------------------------------
    type_counts = {}
    for p in gt_pairs:
        dt = p["discrepancy_type"]
        type_counts[dt] = type_counts.get(dt, 0) + 1

    ground_truth: dict = {
        "metadata": {
            "description": (
                "Ground truth labels for the synthetic reconciliation dataset. "
                "Use this file ONLY to score the matcher's output — never as "
                "input to the matching engine itself."
            ),
            "generated_at"           : datetime.now(timezone.utc).isoformat(),
            "seed"                   : seed,
            "fee_rate_pct"           : float(FEE_RATE * 100),
            "total_gateway_records"  : len(gateway_records),
            "total_ledger_records"   : len(ledger_records),
            "total_logical_pairs"    : len(gt_pairs),
        },
        "discrepancy_counts": type_counts,
        "pairs": gt_pairs,
    }

    if save:
        _save_to_disk(gateway_records, ledger_records, ground_truth, output_dir)

    return gateway_records, ledger_records, ground_truth


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_to_disk(
    gateway_records : list[GatewayRecord],
    ledger_records  : list[LedgerRecord],
    ground_truth    : dict,
    output_dir      : Path,
) -> None:
    """
    Write the three dataset files to *output_dir*.

    Files written
    -------------
    gateway_records.json   — serialised list of GatewayRecord dicts
    ledger_records.json    — serialised list of LedgerRecord dicts
    ground_truth.json      — metadata + per-pair ground truth labels
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    gw_path  = output_dir / "gateway_records.json"
    led_path = output_dir / "ledger_records.json"
    gt_path  = output_dir / "ground_truth.json"

    gw_path.write_text(
        json.dumps([_serialize(r) for r in gateway_records], indent=2),
        encoding="utf-8",
    )
    led_path.write_text(
        json.dumps([_serialize(r) for r in ledger_records], indent=2),
        encoding="utf-8",
    )
    gt_path.write_text(
        json.dumps(ground_truth, indent=2),
        encoding="utf-8",
    )

    print(f"  gateway_records.json  -> {len(gateway_records):>4} records   {gw_path}")
    print(f"  ledger_records.json   -> {len(ledger_records):>4} records   {led_path}")
    print(f"  ground_truth.json     -> {len(ground_truth['pairs']):>4} pairs     {gt_path}")


def load_gateway_records(path: Path = DATA_DIR / "gateway_records.json") -> list[GatewayRecord]:
    """
    Deserialise gateway_records.json from disk back into GatewayRecord objects.

    Parameters
    ----------
    path : Path
        Path to the JSON file.  Defaults to ``data/gateway_records.json``.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [GatewayRecord(**r) for r in raw]


def load_ledger_records(path: Path = DATA_DIR / "ledger_records.json") -> list[LedgerRecord]:
    """
    Deserialise ledger_records.json from disk back into LedgerRecord objects.

    Parameters
    ----------
    path : Path
        Path to the JSON file.  Defaults to ``data/ledger_records.json``.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [LedgerRecord(**r) for r in raw]


def load_ground_truth(path: Path = DATA_DIR / "ground_truth.json") -> dict:
    """
    Load the ground truth document from disk.

    Parameters
    ----------
    path : Path
        Path to the JSON file.  Defaults to ``data/ground_truth.json``.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _serialize(record: GatewayRecord | LedgerRecord) -> dict:
    """Convert a Pydantic model to a JSON-safe plain dict."""
    d = record.model_dump()
    for k, v in d.items():
        if isinstance(v, Decimal):
            d[k] = str(v)
        elif isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("recon-agent - synthetic dataset generator")
    print(f"Seed         : {DEFAULT_SEED}")
    print(f"Output dir   : {DATA_DIR.resolve()}\n")
    print("Generating ...")

    gw, led, gt = generate_dataset(seed=DEFAULT_SEED, save=True)

    print("\nDiscrepancy breakdown:")
    print(f"  {'Type':<25} {'Count':>5}   {'Share':>6}")
    print(f"  {'-'*25} {'-'*5}   {'-'*6}")
    total_pairs = gt["metadata"]["total_logical_pairs"]
    for dtype, count in gt["discrepancy_counts"].items():
        share = count / total_pairs * 100
        print(f"  {dtype:<25} {count:>5}   {share:>5.1f} %")

    print(f"\n  Total gateway records : {gt['metadata']['total_gateway_records']}")
    print(f"  Total ledger records  : {gt['metadata']['total_ledger_records']}")
    print(f"  Total logical pairs   : {total_pairs}")
    print("\nDone.")
