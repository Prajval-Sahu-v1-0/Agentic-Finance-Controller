"""
matcher.py — Core Matching Engine
===================================

Implements a three-phase reconciliation cascade:

  Phase 1 — Exact Match
      Normalised reference_id equality + identical amount (Decimal exact) +
      timestamp within ``exact_timestamp_tolerance_hours`` (default 1 h).
      Confidence is always 1.0.  O(n) via hash-map lookup.

  Phase 2 — Fuzzy Match
      Applied to records that survive Phase 1 unmatched.
      Normalised reference_id equality + amount within ``amount_tolerance_pct``
      (default 2 %) + timestamp within ``timestamp_tolerance_hours``
      (default 72 h / 3 days).
      Confidence is a weighted composite score in [0.0, 1.0].

  Phase 2.5 — Content Fallback Match
      Applied to records still unmatched after Phase 2, i.e. records whose
      normalised reference keys never agreed at all (no shared digit
      sequence between the two sources — common with real-world bank/gateway
      exports whose reference text is independently generated on each side).
      Falls back to amount + timestamp proximity alone, using tolerances
      much stricter than Phase 2's (see ``content_amount_tolerance_pct`` /
      ``content_timestamp_tolerance_hours``), and additionally requires the
      candidate pair to be *mutually unique*: the ledger record's only
      qualifying gateway candidate must, symmetrically, have this ledger
      record as its only qualifying candidate. This guards against
      coincidental amount/date collisions in larger batches, at the cost of
      leaving genuinely ambiguous cases unresolved rather than guessing.

  Phase 2.75 — Text Disambiguation Match
      Applied to records still unmatched after Phase 2.5 — specifically the
      ambiguous case where a ledger (or gateway) record has MULTIPLE
      amount+timestamp candidates within Phase 2.5's tolerance, so mutual
      uniqueness fails there. Re-ranks those candidates by text similarity
      between ``reference_id`` and ``counterparty`` on each side (independent
      free-text fields carrying whatever narration the source system
      recorded), and accepts the top-ranked candidate only if it clears both
      an absolute similarity floor and a margin over the runner-up, checked
      mutually on both sides. This is deliberately a classic string-metric
      fallback (difflib ratio), not an LLM call — at real-world batch sizes
      the ambiguous pool can be in the tens of thousands, which is only
      tractable with a fast deterministic re-ranker; the LLM layer instead
      spot-audits a sample of accepted Phase 2.5/2.75 matches for precision
      (see ``llm_agent.reason_about_pair``).

  Phase 3 — Grouped Match
      Applied to records still unmatched after Phase 2.75.  Searches for
      multiplicity relationships where ref-based pairing is impossible:

        one_to_many — a single GatewayRecord whose amount equals the SUM
            of 2-N LedgerRecords (e.g. one batch settlement covering multiple
            individual orders).

        many_to_one — multiple GatewayRecords whose combined amount equals
            a single LedgerRecord (e.g. a split payment processed in parts).

      The search is bounded by ``max_group_size`` (default 4) to keep
      combinatorial complexity manageable on typical batch sizes.

      KNOWN LIMITATION: candidate entry into this phase requires an explicit
      BATCH/SPLIT/PAYOUT reference-ID prefix (see ``_group_candidate_pools``),
      a convention specific to our synthetic generator. Real-world data has
      no such marker, so this phase is currently inert outside synthetic
      data — confirmed empirically (0 grouped matches on the BenchRec
      validation run despite ~1,500 genuine grouped pairs in its oracle).
      See ``_group_candidate_pools``'s docstring for the fix plan.

  Unresolved
      Any record still unmatched after both phases.  Carries a ``reason`` tag
      (no_counterpart | tolerance_exceeded | duplicate_candidate) so the
      downstream exception classifier can act on it without re-reading the data.

Reference-ID normalisation
--------------------------
Real-world gateway and ledger systems format the same payment reference
differently.  The engine strips all non-digit characters before indexing:

    GW  "PAY20240307000096"   ->  "20240307000096"
    LED "ORD-2024-03-07-000096" ->  "20240307000096"

This makes the index key identical for both sources while preserving the
ability to detect genuinely different transactions.

Duplicate handling
------------------
When multiple records from the same source share a normalised reference key,
the engine greedily assigns the highest-scoring candidate to the opposite
source and marks all remaining competitors as ``duplicate_candidate``.

Usage
-----
    from src.matcher import ReconciliationEngine, MatchConfig

    config = MatchConfig(amount_tolerance_pct=Decimal("2.0"),
                         timestamp_tolerance_hours=72.0)
    engine = ReconciliationEngine(config)
    result = engine.run(gateway_records, ledger_records)

    print(len(result.matched_exact))   # phase-1 hits
    print(len(result.matched_fuzzy))   # phase-2 hits
    print(len(result.unresolved))      # leftovers for exception classifier
"""

from __future__ import annotations

import bisect
import itertools
import re
import time
from difflib import SequenceMatcher
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from src.config import (
    MATCHER_AMOUNT_TOLERANCE_PCT,
    MATCHER_CONTENT_AMOUNT_TOLERANCE_PCT,
    MATCHER_CONTENT_TIMESTAMP_TOLERANCE_HOURS,
    MATCHER_GROUP_MAX_TIMESTAMP_SPREAD_HOURS,
    MATCHER_GROUP_TOLERANCE_PCT,
    MATCHER_TEXT_SIMILARITY_MIN_MARGIN,
    MATCHER_TEXT_SIMILARITY_MIN_SCORE,
    MATCHER_TIMESTAMP_TOLERANCE_HOURS,
)
from src.schema import GatewayRecord, LedgerRecord


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class MatchConfig(BaseModel):
    """
    Tolerance and weight parameters for the reconciliation engine.

    All tolerances are *inclusive* upper bounds.  Tune these values to trade
    off between recall (catching more real matches) and precision (avoiding
    false positives).

    Attributes
    ----------
    exact_timestamp_tolerance_hours : float
        Maximum timestamp gap (in hours) allowed for a Phase-1 exact match.
        Default 1.0 h covers sub-hour settlement lags without catching
        multi-day drifts.
    amount_tolerance_pct : Decimal
        Maximum percentage difference between gateway and ledger amounts for a
        Phase-2 fuzzy match.  Default 2.0 % covers standard gateway fee rates.
        Formula: |gw.amount - led.amount| / led.amount * 100 <= tolerance.
    timestamp_tolerance_hours : float
        Maximum timestamp gap (in hours) for a Phase-2 fuzzy match.
        Default 72.0 h (3 days) covers T+2 settlement cycles.
    confidence_weight_amount : float
        Weight assigned to the amount score in the confidence formula.
        Must satisfy confidence_weight_amount + confidence_weight_timestamp == 1.0.
    confidence_weight_timestamp : float
        Weight assigned to the timestamp score in the confidence formula.
    min_fuzzy_confidence : float
        Minimum composite confidence score [0.0, 1.0] required to accept a
        fuzzy match.  Set higher to discard borderline candidates.
    """

    exact_timestamp_tolerance_hours : float   = 1.0
    amount_tolerance_pct             : Decimal = MATCHER_AMOUNT_TOLERANCE_PCT
    timestamp_tolerance_hours        : float   = MATCHER_TIMESTAMP_TOLERANCE_HOURS
    group_tolerance_pct              : Decimal = MATCHER_GROUP_TOLERANCE_PCT
    group_max_timestamp_spread_hours : float   = MATCHER_GROUP_MAX_TIMESTAMP_SPREAD_HOURS
    enable_content_fallback          : bool    = True
    content_amount_tolerance_pct     : Decimal = MATCHER_CONTENT_AMOUNT_TOLERANCE_PCT
    content_timestamp_tolerance_hours: float   = MATCHER_CONTENT_TIMESTAMP_TOLERANCE_HOURS
    """Phase 2.5 fallback for records whose normalised reference keys never
    agreed at all. Deliberately stricter than Phase 2 and additionally
    requires mutual uniqueness — see ReconciliationEngine._content_phase."""
    enable_text_fallback             : bool    = True
    text_similarity_min_score        : float   = MATCHER_TEXT_SIMILARITY_MIN_SCORE
    text_similarity_min_margin       : float   = MATCHER_TEXT_SIMILARITY_MIN_MARGIN
    """Phase 2.75: re-ranks Phase 2.5's ambiguous (multi-candidate) clusters
    by reference_id/counterparty text similarity. Accepts the top candidate
    only if its score clears text_similarity_min_score AND beats the
    runner-up by text_similarity_min_margin, checked mutually on both sides
    — see ReconciliationEngine._text_phase."""
    confidence_weight_amount         : float   = 0.5
    confidence_weight_timestamp      : float   = 0.5
    min_fuzzy_confidence             : float   = 0.0
    max_group_size                   : int     = 4
    """Maximum number of records on either side of a grouped match (Phase 3).
    Setting this to 1 disables grouped matching entirely.
    C(n, k) combinations are tried for k in range(2, max_group_size+1), so
    keep this <= 4 for datasets up to ~200 unresolved records."""

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "MatchConfig":
        total = round(self.confidence_weight_amount + self.confidence_weight_timestamp, 6)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"confidence weights must sum to 1.0, got {total}"
            )
        return self


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class MatchedPair(BaseModel):
    """
    A successfully reconciled (GatewayRecord, LedgerRecord) pair.

    Attributes
    ----------
    gateway_record : GatewayRecord
    ledger_record  : LedgerRecord
    match_type     : "exact" for Phase-1 hits, "fuzzy" for Phase-2 hits,
                     "content" for Phase-2.5 (amount+timestamp-only) hits.
    confidence     : 1.0 for exact; weighted composite in (0.0, 1.0] for
                     fuzzy/content.
    normalized_ref : The normalised key used to index this pair.
    amount_delta   : gateway.amount - ledger.amount (negative = fee deducted).
    timestamp_delta_seconds : gateway.timestamp - ledger.timestamp in seconds.
    matched_at     : UTC datetime when the engine produced this record.
    """

    gateway_record          : GatewayRecord
    ledger_record           : LedgerRecord
    match_type              : Literal["exact", "fuzzy", "content", "text"]
    confidence              : float
    normalized_ref          : str
    amount_delta            : Decimal
    timestamp_delta_seconds : float
    matched_at              : datetime


class GroupedMatch(BaseModel):
    """
    A Phase-3 multiplicity match: one-to-many or many-to-one.

    Attributes
    ----------
    match_type : "one_to_many" | "many_to_one"
        one_to_many — 1 GatewayRecord settled multiple LedgerRecords.
        many_to_one — multiple GatewayRecords sum to 1 LedgerRecord.
    gateway_records : list[GatewayRecord]
        The gateway side of the group (1 record for one_to_many,
        N records for many_to_one).
    ledger_records : list[LedgerRecord]
        The ledger side of the group (N records for one_to_many,
        1 record for many_to_one).
    gateway_total : Decimal
        Sum of all gateway record amounts in this group.
    ledger_total : Decimal
        Sum of all ledger record amounts in this group.
    amount_delta : Decimal
        gateway_total - ledger_total.
    amount_delta_pct : Decimal
        |gateway_total - ledger_total| / ledger_total * 100.
    normalized_refs : list[str]
        Normalised reference keys of all records in the group.
    confidence : float
        Amount-based score in (0.0, 1.0]; 1.0 means exact sum equality.
    matched_at : datetime
        UTC datetime when the engine produced this record.
    """

    match_type       : Literal["one_to_many", "many_to_one"]
    gateway_records  : list[GatewayRecord]
    ledger_records   : list[LedgerRecord]
    gateway_total    : Decimal
    ledger_total     : Decimal
    amount_delta     : Decimal
    amount_delta_pct : Decimal
    normalized_refs  : list[str]
    confidence       : float
    matched_at       : datetime


class UnresolvedRecord(BaseModel):
    """
    A record that could not be matched after both phases.

    Exactly one of ``gateway_record`` / ``ledger_record`` is populated,
    determined by ``source``.

    Attributes
    ----------
    source         : Which source this record came from.
    gateway_record : Set when source == "gateway".
    ledger_record  : Set when source == "ledger".
    normalized_ref : The normalised key for this record.
    reason         :
        no_counterpart      — no record in the other source shares this key.
        tolerance_exceeded  — a counterpart was found but amount or timestamp
                              exceeded all configured tolerances.
        duplicate_candidate — this normalised key was already claimed by another
                              record from the same source.
    closest_delta  : Dict with diagnostic info about the nearest candidate
                     (amount_delta_pct, timestamp_delta_hours) if one was found.
    """

    source          : Literal["gateway", "ledger"]
    gateway_record  : Optional[GatewayRecord] = None
    ledger_record   : Optional[LedgerRecord]  = None
    normalized_ref  : str
    reason          : Literal["no_counterpart", "tolerance_exceeded", "duplicate_candidate"]
    closest_delta   : Optional[dict]          = None

    @model_validator(mode="after")
    def exactly_one_record(self) -> "UnresolvedRecord":
        has_gw  = self.gateway_record is not None
        has_led = self.ledger_record  is not None
        if has_gw == has_led:   # both set or both None
            raise ValueError(
                "Exactly one of gateway_record / ledger_record must be set."
            )
        if self.source == "gateway" and not has_gw:
            raise ValueError("source='gateway' but gateway_record is None.")
        if self.source == "ledger" and not has_led:
            raise ValueError("source='ledger' but ledger_record is None.")
        return self


class ReconciliationResult(BaseModel):
    """
    Complete output of a single reconciliation run.

    Attributes
    ----------
    matched_exact   : Phase-1 pairs (confidence == 1.0).
    matched_fuzzy   : Phase-2 pairs (confidence < 1.0, within tolerances).
    matched_content : Phase-2.5 pairs — amount+timestamp only, no reference
                      agreement, mutually-unique candidates only.
    matched_text    : Phase-2.75 pairs — amount+timestamp candidates that
                      were ambiguous in Phase 2.5, disambiguated by text
                      similarity on reference_id/counterparty.
    matched_grouped : Phase-3 groups (one-to-many or many-to-one sum matches).
    unresolved      : Records that could not be matched by any phase.
    run_metadata    : Engine configuration and timing statistics.
    """

    matched_exact   : list[MatchedPair]
    matched_fuzzy   : list[MatchedPair]
    matched_content : list[MatchedPair] = Field(default_factory=list)
    matched_text    : list[MatchedPair] = Field(default_factory=list)
    matched_grouped : list[GroupedMatch]
    unresolved      : list[UnresolvedRecord]
    run_metadata    : dict

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def total_matched(self) -> int:
        """Records claimed across all phases."""
        grouped_gw  = sum(len(g.gateway_records) for g in self.matched_grouped)
        return (
            len(self.matched_exact)
            + len(self.matched_fuzzy)
            + len(self.matched_content)
            + len(self.matched_text)
            + grouped_gw   # count gateway-side grouped records
        )

    @property
    def match_rate(self) -> float:
        """Fraction of gateway records resolved across all phases."""
        total_gw = self.run_metadata.get("total_gateway_records", 0)
        n_exact   = len(self.matched_exact)
        n_fuzzy   = len(self.matched_fuzzy)
        n_content = len(self.matched_content)
        n_text    = len(self.matched_text)
        n_grp_gw  = sum(len(g.gateway_records) for g in self.matched_grouped)
        return (n_exact + n_fuzzy + n_content + n_text + n_grp_gw) / total_gw if total_gw else 0.0

    def summary(self) -> dict:
        """Return a plain-dict summary suitable for logging or reporting."""
        n_grp = len(self.matched_grouped)
        n_grp_gw  = sum(len(g.gateway_records) for g in self.matched_grouped)
        n_grp_led = sum(len(g.ledger_records)  for g in self.matched_grouped)
        return {
            "matched_exact"             : len(self.matched_exact),
            "matched_fuzzy"             : len(self.matched_fuzzy),
            "matched_content"           : len(self.matched_content),
            "matched_text"              : len(self.matched_text),
            "matched_grouped"           : n_grp,
            "matched_grouped_gw_records": n_grp_gw,
            "matched_grouped_led_records": n_grp_led,
            "total_matched"             : self.total_matched,
            "unresolved"                : len(self.unresolved),
            "match_rate_pct"            : round(self.match_rate * 100, 2),
            "unresolved_no_counterpart" : sum(
                1 for u in self.unresolved if u.reason == "no_counterpart"
            ),
            "unresolved_tol_exceeded"   : sum(
                1 for u in self.unresolved if u.reason == "tolerance_exceeded"
            ),
            "unresolved_duplicate"      : sum(
                1 for u in self.unresolved if u.reason == "duplicate_candidate"
            ),
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ReconciliationEngine:
    """
    Two-phase financial reconciliation engine.

    Instantiate once with a ``MatchConfig``, then call ``run()`` for each
    batch of records.  The engine is stateless between ``run()`` calls.

    Parameters
    ----------
    config : MatchConfig
        Tolerance and weight configuration.  Pass a custom instance to tune
        the engine without subclassing.

    Examples
    --------
    >>> engine = ReconciliationEngine()
    >>> result = engine.run(gateway_records, ledger_records)
    >>> print(result.summary())
    """

    def __init__(self, config: Optional[MatchConfig] = None) -> None:
        self.config: MatchConfig = config or MatchConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        gateway_records : list[GatewayRecord],
        ledger_records  : list[LedgerRecord],
    ) -> ReconciliationResult:
        """
        Execute the full two-phase matching cascade.

        Parameters
        ----------
        gateway_records : list[GatewayRecord]
            Records from the payment gateway export.
        ledger_records : list[LedgerRecord]
            Records from the merchant's internal ledger.

        Returns
        -------
        ReconciliationResult
            Contains matched_exact, matched_fuzzy, and unresolved buckets
            plus run_metadata with timing and record counts.
        """
        t_start = time.perf_counter()

        # Build normalised-ref indexes
        gw_index  : dict[str, list[GatewayRecord]] = self._build_index(gateway_records, "gateway")
        led_index : dict[str, list[LedgerRecord]]  = self._build_index(ledger_records,  "ledger")

        matched_exact   : list[MatchedPair]    = []
        matched_fuzzy   : list[MatchedPair]    = []
        matched_content : list[MatchedPair]    = []
        matched_text    : list[MatchedPair]    = []
        matched_grouped : list[GroupedMatch]   = []
        unresolved      : list[UnresolvedRecord] = []

        # Track which records have been claimed so far
        claimed_gw  : set[str] = set()   # transaction_ids
        claimed_led : set[str] = set()   # transaction_ids

        all_keys = set(gw_index) | set(led_index)

        # ── Phase 1 : Exact matching ──────────────────────────────────
        for key in all_keys:
            gw_candidates  = gw_index.get(key, [])
            led_candidates = led_index.get(key, [])

            if not gw_candidates or not led_candidates:
                continue   # handled in unresolved pass below

            for led in led_candidates:
                if led.transaction_id in claimed_led:
                    continue
                best_pair = self._best_exact_candidate(
                    led, gw_candidates, claimed_gw
                )
                if best_pair is not None:
                    gw, pair = best_pair
                    matched_exact.append(pair)
                    claimed_gw.add(gw.transaction_id)
                    claimed_led.add(led.transaction_id)

        # ── Phase 2 : Fuzzy matching ──────────────────────────────────
        for key in all_keys:
            gw_candidates  = gw_index.get(key, [])
            led_candidates = led_index.get(key, [])

            if not gw_candidates or not led_candidates:
                continue

            for led in led_candidates:
                if led.transaction_id in claimed_led:
                    continue
                best_pair = self._best_fuzzy_candidate(
                    led, gw_candidates, claimed_gw
                )
                if best_pair is not None:
                    gw, pair = best_pair
                    matched_fuzzy.append(pair)
                    claimed_gw.add(gw.transaction_id)
                    claimed_led.add(led.transaction_id)

        # ── Phase 2.5 : Content fallback matching ───────────────────────
        if self.config.enable_content_fallback:
            unclaimed_gw  = [
                g for g in gateway_records if g.transaction_id not in claimed_gw
            ]
            unclaimed_led = [
                l for l in ledger_records  if l.transaction_id not in claimed_led
            ]
            matched_content = self._content_phase(unclaimed_gw, unclaimed_led)
            for pair in matched_content:
                claimed_gw.add(pair.gateway_record.transaction_id)
                claimed_led.add(pair.ledger_record.transaction_id)

        # ── Phase 2.75 : Text disambiguation matching ───────────────────
        if self.config.enable_text_fallback:
            unclaimed_gw  = [
                g for g in gateway_records if g.transaction_id not in claimed_gw
            ]
            unclaimed_led = [
                l for l in ledger_records  if l.transaction_id not in claimed_led
            ]
            matched_text = self._text_phase(unclaimed_gw, unclaimed_led)
            for pair in matched_text:
                claimed_gw.add(pair.gateway_record.transaction_id)
                claimed_led.add(pair.ledger_record.transaction_id)

        # ── Phase 3 : Grouped matching ────────────────────────────────
        if self.config.max_group_size >= 2:
            unclaimed_gw  = [
                g for g in gateway_records if g.transaction_id not in claimed_gw
            ]
            unclaimed_led = [
                l for l in ledger_records  if l.transaction_id not in claimed_led
            ]
            # Reserve ordinary unmatched and duplicate rows for exception
            # handling.  Phase 3 is only allowed for records with explicit
            # batch/split/payout corroboration.
            group_gw, group_led = self._group_candidate_pools(
                unclaimed_gw, unclaimed_led
            )
            matched_grouped = self._grouped_phase(group_gw, group_led)
            for gm in matched_grouped:
                for g in gm.gateway_records:
                    claimed_gw.add(g.transaction_id)
                for l in gm.ledger_records:
                    claimed_led.add(l.transaction_id)

        # ── Unresolved pass ───────────────────────────────────────────
        for key in all_keys:
            gw_candidates  = gw_index.get(key, [])
            led_candidates = led_index.get(key, [])

            has_opposite_gw  = bool(gw_candidates)
            has_opposite_led = bool(led_candidates)

            # Gateway records that were not claimed
            first_unclaimed_gw = True
            for gw in gw_candidates:
                if gw.transaction_id in claimed_gw:
                    continue
                if has_opposite_led:
                    # A counterpart exists but tolerances were exceeded,
                    # OR this is a second GW row competing for an already-claimed LED
                    already_claimed_led = all(
                        l.transaction_id in claimed_led for l in led_candidates
                    )
                    reason : Literal["no_counterpart", "tolerance_exceeded", "duplicate_candidate"]
                    if already_claimed_led and not first_unclaimed_gw:
                        reason = "duplicate_candidate"
                    elif already_claimed_led:
                        reason = "duplicate_candidate"
                    else:
                        reason = "tolerance_exceeded"
                    closest = self._closest_delta(gw, led_candidates)
                else:
                    reason  = "no_counterpart"
                    closest = None

                unresolved.append(UnresolvedRecord(
                    source         = "gateway",
                    gateway_record = gw,
                    normalized_ref = key,
                    reason         = reason,
                    closest_delta  = closest,
                ))
                first_unclaimed_gw = False

            # Ledger records that were not claimed
            first_unclaimed_led = True
            for led in led_candidates:
                if led.transaction_id in claimed_led:
                    continue
                if has_opposite_gw:
                    already_claimed_gw = all(
                        g.transaction_id in claimed_gw for g in gw_candidates
                    )
                    if already_claimed_gw and not first_unclaimed_led:
                        reason = "duplicate_candidate"
                    elif already_claimed_gw:
                        reason = "duplicate_candidate"
                    else:
                        reason = "tolerance_exceeded"
                    closest = self._closest_delta_led(led, gw_candidates)
                else:
                    reason  = "no_counterpart"
                    closest = None

                unresolved.append(UnresolvedRecord(
                    source        = "ledger",
                    ledger_record = led,
                    normalized_ref = key,
                    reason         = reason,
                    closest_delta  = closest,
                ))
                first_unclaimed_led = False

        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

        return ReconciliationResult(
            matched_exact   = matched_exact,
            matched_fuzzy   = matched_fuzzy,
            matched_content = matched_content,
            matched_text    = matched_text,
            matched_grouped = matched_grouped,
            unresolved      = unresolved,
            run_metadata    = {
                "total_gateway_records"          : len(gateway_records),
                "total_ledger_records"           : len(ledger_records),
                "elapsed_ms"                     : elapsed_ms,
                "config": {
                    "exact_timestamp_tolerance_hours" : self.config.exact_timestamp_tolerance_hours,
                    "amount_tolerance_pct"            : str(self.config.amount_tolerance_pct),
                    "timestamp_tolerance_hours"       : self.config.timestamp_tolerance_hours,
                    "group_tolerance_pct"             : str(self.config.group_tolerance_pct),
                    "group_max_timestamp_spread_hours": self.config.group_max_timestamp_spread_hours,
                    "confidence_weight_amount"        : self.config.confidence_weight_amount,
                    "confidence_weight_timestamp"     : self.config.confidence_weight_timestamp,
                    "min_fuzzy_confidence"            : self.config.min_fuzzy_confidence,
                    "max_group_size"                  : self.config.max_group_size,
                    "enable_content_fallback"         : self.config.enable_content_fallback,
                    "content_amount_tolerance_pct"    : str(self.config.content_amount_tolerance_pct),
                    "content_timestamp_tolerance_hours": self.config.content_timestamp_tolerance_hours,
                    "enable_text_fallback"            : self.config.enable_text_fallback,
                    "text_similarity_min_score"       : self.config.text_similarity_min_score,
                    "text_similarity_min_margin"      : self.config.text_similarity_min_margin,
                },
            },
        )

    # ------------------------------------------------------------------
    # Reference normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_ref(ref: str) -> str:
        """
        Reduce a raw reference ID to a digits-only canonical key.

        Strips all non-digit characters (prefixes, hyphens, underscores,
        whitespace) so that semantically equivalent references from different
        systems map to the same index key.

        Examples
        --------
        >>> ReconciliationEngine.normalize_ref("PAY20240307000096")
        '20240307000096'
        >>> ReconciliationEngine.normalize_ref("ORD-2024-03-07-000096")
        '20240307000096'
        >>> ReconciliationEngine.normalize_ref("  RAZORPAY20240315123456  ")
        '20240315123456'
        """
        return re.sub(r"\D", "", ref.strip())

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(
        self,
        records : list,
        source  : Literal["gateway", "ledger"],
    ) -> dict[str, list]:
        """
        Build a normalised-ref -> [records] lookup dict.

        Multiple records can share the same key (duplicates); the engine
        handles them in the matching phases.
        """
        index: dict[str, list] = {}
        for record in records:
            key = self.normalize_ref(record.reference_id)
            index.setdefault(key, []).append(record)
        return index

    # ------------------------------------------------------------------
    # Phase 1 helpers
    # ------------------------------------------------------------------

    def _best_exact_candidate(
        self,
        led           : LedgerRecord,
        gw_candidates : list[GatewayRecord],
        claimed_gw    : set[str],
    ) -> Optional[tuple[GatewayRecord, MatchedPair]]:
        """
        Return the best (GatewayRecord, MatchedPair) from *gw_candidates*
        that satisfies exact-match criteria against *led*, or None.

        Exact criteria
        --------------
        - amount     : Decimal-equal (no tolerance).
        - timestamp  : |delta| <= exact_timestamp_tolerance_hours.

        When multiple candidates pass, the one with the smallest timestamp
        gap is chosen (tightest match wins).
        """
        tol_secs = self.config.exact_timestamp_tolerance_hours * 3600
        best_gw    : Optional[GatewayRecord] = None
        best_secs  : float = float("inf")

        for gw in gw_candidates:
            if gw.transaction_id in claimed_gw:
                continue
            if gw.amount != led.amount:
                continue
            delta_secs = abs(_ts_delta(gw.timestamp, led.timestamp))
            if delta_secs <= tol_secs and delta_secs < best_secs:
                best_gw   = gw
                best_secs = delta_secs

        if best_gw is None:
            return None

        signed_delta_secs = _ts_delta(best_gw.timestamp, led.timestamp)
        pair = MatchedPair(
            gateway_record          = best_gw,
            ledger_record           = led,
            match_type              = "exact",
            confidence              = 1.0,
            normalized_ref          = self.normalize_ref(led.reference_id),
            amount_delta            = best_gw.amount - led.amount,
            timestamp_delta_seconds = signed_delta_secs,
            matched_at              = datetime.now(timezone.utc),
        )
        return best_gw, pair

    # ------------------------------------------------------------------
    # Phase 2 helpers
    # ------------------------------------------------------------------

    def _best_fuzzy_candidate(
        self,
        led           : LedgerRecord,
        gw_candidates : list[GatewayRecord],
        claimed_gw    : set[str],
    ) -> Optional[tuple[GatewayRecord, MatchedPair]]:
        """
        Return the best (GatewayRecord, MatchedPair) from *gw_candidates*
        that satisfies fuzzy-match criteria against *led*, or None.

        Fuzzy criteria
        --------------
        - amount     : |delta_pct| <= amount_tolerance_pct.
        - timestamp  : |delta| <= timestamp_tolerance_hours.
        - confidence : weighted composite >= min_fuzzy_confidence.

        The candidate with the highest confidence score wins.
        """
        tol_amount_pct = self.config.amount_tolerance_pct
        tol_ts_secs    = self.config.timestamp_tolerance_hours * 3600

        best_gw    : Optional[GatewayRecord] = None
        best_conf  : float = -1.0
        best_pair  : Optional[MatchedPair] = None

        for gw in gw_candidates:
            if gw.transaction_id in claimed_gw:
                continue

            # Amount check
            delta_pct = _amount_delta_pct(gw.amount, led.amount)
            if delta_pct > tol_amount_pct:
                continue

            # Timestamp check
            delta_secs = abs(_ts_delta(gw.timestamp, led.timestamp))
            if delta_secs > tol_ts_secs:
                continue

            conf = self._confidence(delta_pct, delta_secs)
            if conf < self.config.min_fuzzy_confidence:
                continue

            if conf > best_conf:
                best_conf = conf
                best_gw   = gw
                signed_delta_secs = _ts_delta(gw.timestamp, led.timestamp)
                best_pair = MatchedPair(
                    gateway_record          = gw,
                    ledger_record           = led,
                    match_type              = "fuzzy",
                    confidence              = round(conf, 4),
                    normalized_ref          = self.normalize_ref(led.reference_id),
                    amount_delta            = gw.amount - led.amount,
                    timestamp_delta_seconds = signed_delta_secs,
                    matched_at              = datetime.now(timezone.utc),
                )

        if best_gw is None or best_pair is None:
            return None
        return best_gw, best_pair

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def _confidence(self, amount_delta_pct: Decimal, ts_delta_secs: float) -> float:
        """
        Compute a composite confidence score for a fuzzy candidate pair.

        Both sub-scores are linear: 1.0 at zero delta, 0.0 at the tolerance
        boundary.  They are combined via the configured weights.

        Parameters
        ----------
        amount_delta_pct : Decimal
            Absolute percentage difference between gateway and ledger amounts.
        ts_delta_secs : float
            Absolute timestamp difference in seconds.

        Returns
        -------
        float in [0.0, 1.0]
        """
        tol_pct  = float(self.config.amount_tolerance_pct)
        tol_secs = self.config.timestamp_tolerance_hours * 3600

        amount_score = max(0.0, 1.0 - float(amount_delta_pct) / tol_pct)  if tol_pct  else 1.0
        ts_score     = max(0.0, 1.0 - ts_delta_secs          / tol_secs)  if tol_secs else 1.0

        return (
            self.config.confidence_weight_amount    * amount_score
            + self.config.confidence_weight_timestamp * ts_score
        )

    # ------------------------------------------------------------------
    # Phase 2.5 helpers — content (amount + timestamp only) fallback
    # ------------------------------------------------------------------

    def _content_phase(
        self,
        unclaimed_gw  : list[GatewayRecord],
        unclaimed_led : list[LedgerRecord],
    ) -> list[MatchedPair]:
        """
        Match remaining records purely on amount + timestamp proximity,
        for the case where reference IDs on the two sides never agree at
        all (no shared normalised key — see module docstring).

        This is materially weaker evidence than a reference-anchored match,
        so two safeguards apply beyond the tighter
        ``content_amount_tolerance_pct`` / ``content_timestamp_tolerance_hours``
        thresholds:

        1. Mutual uniqueness — a ledger record is only paired with a gateway
           record if each is the OTHER's single qualifying candidate. If two
           ledger records both fall within tolerance of the same gateway
           record (or vice versa), none of them are matched here; they are
           left for Phase 3 / the exception classifier rather than guessed.
        2. Runs only on records still unclaimed after Phase 1 and Phase 2,
           so it can never override a reference-anchored decision.

        Complexity: candidates are pruned by binary-searching an
        amount-sorted gateway array for the ``[led.amount * (1 -
        tol), led.amount * (1 + tol)]`` window, so cost is roughly
        O((n_gw + n_led) log n_gw + k) where k is the number of
        amount-tolerance hits — tractable at 10^4-10^5 records per side
        given the deliberately tight default tolerance.
        """
        if not unclaimed_gw or not unclaimed_led:
            return []

        tol_amount_pct = self.config.content_amount_tolerance_pct
        tol_ts_secs    = self.config.content_timestamp_tolerance_hours * 3600

        sorted_gw   = sorted(unclaimed_gw, key=lambda g: g.amount)
        gw_amounts  = [g.amount for g in sorted_gw]

        led_candidates: dict[str, list[GatewayRecord]] = {}
        gw_candidates : dict[str, list[LedgerRecord]]  = {}

        for led in unclaimed_led:
            if tol_amount_pct:
                spread = led.amount * tol_amount_pct / Decimal("100")
            else:
                spread = Decimal("0")
            lo = bisect.bisect_left(gw_amounts, led.amount - spread)
            hi = bisect.bisect_right(gw_amounts, led.amount + spread)
            for gw in sorted_gw[lo:hi]:
                delta_pct = _amount_delta_pct(gw.amount, led.amount)
                if delta_pct > tol_amount_pct:
                    continue
                delta_secs = abs(_ts_delta(gw.timestamp, led.timestamp))
                if delta_secs > tol_ts_secs:
                    continue
                led_candidates.setdefault(led.transaction_id, []).append(gw)
                gw_candidates.setdefault(gw.transaction_id, []).append(led)

        results: list[MatchedPair] = []
        for led in unclaimed_led:
            candidates = led_candidates.get(led.transaction_id, [])
            if len(candidates) != 1:
                continue
            gw = candidates[0]
            if len(gw_candidates.get(gw.transaction_id, [])) != 1:
                continue

            delta_pct  = _amount_delta_pct(gw.amount, led.amount)
            delta_secs = abs(_ts_delta(gw.timestamp, led.timestamp))
            amount_score = max(0.0, 1.0 - float(delta_pct) / float(tol_amount_pct)) if tol_amount_pct else 1.0
            ts_score     = max(0.0, 1.0 - delta_secs / tol_ts_secs) if tol_ts_secs else 1.0
            conf = (
                self.config.confidence_weight_amount    * amount_score
                + self.config.confidence_weight_timestamp * ts_score
            )
            signed_delta_secs = _ts_delta(gw.timestamp, led.timestamp)
            results.append(MatchedPair(
                gateway_record          = gw,
                ledger_record           = led,
                match_type              = "content",
                confidence              = round(conf, 4),
                normalized_ref          = self.normalize_ref(led.reference_id),
                amount_delta            = gw.amount - led.amount,
                timestamp_delta_seconds = signed_delta_secs,
                matched_at              = datetime.now(timezone.utc),
            ))
        return results

    # ------------------------------------------------------------------
    # Phase 2.75 helpers — text similarity disambiguation
    # ------------------------------------------------------------------

    @staticmethod
    def _text_blob(record: GatewayRecord | LedgerRecord) -> str:
        return f"{record.reference_id} {record.counterparty}".lower()

    def _text_phase(
        self,
        unclaimed_gw  : list[GatewayRecord],
        unclaimed_led : list[LedgerRecord],
    ) -> list[MatchedPair]:
        """
        Disambiguate Phase 2.5's ambiguous (multi-candidate) clusters using
        text similarity on ``reference_id`` + ``counterparty``.

        For each ledger record with 2+ amount/timestamp candidates (the
        cases Phase 2.5 had to skip because mutual uniqueness failed), rank
        candidates by ``difflib.SequenceMatcher`` ratio against the
        combined reference/counterparty text. The top candidate is accepted
        only if:

        1. its score clears ``text_similarity_min_score``,
        2. it beats the runner-up by at least ``text_similarity_min_margin``
           (an unclear ranking is left unresolved, not guessed), and
        3. the same holds symmetrically from the gateway record's side —
           this ledger record must also be that gateway record's clear
           top-ranked candidate among *its* amount/timestamp candidates.

        Candidates with exactly one amount/timestamp match were already
        claimed or rejected in Phase 2.5 (single candidate but mutual
        uniqueness failed on the *other* side); this phase does not
        revisit those, since text similarity cannot resolve a conflict
        that isn't about ranking multiple options.
        """
        if not unclaimed_gw or not unclaimed_led:
            return []

        tol_amount_pct = self.config.content_amount_tolerance_pct
        tol_ts_secs    = self.config.content_timestamp_tolerance_hours * 3600
        min_score      = self.config.text_similarity_min_score
        min_margin     = self.config.text_similarity_min_margin

        sorted_gw  = sorted(unclaimed_gw, key=lambda g: g.amount)
        gw_amounts = [g.amount for g in sorted_gw]

        led_candidates: dict[str, list[GatewayRecord]] = {}
        gw_candidates : dict[str, list[LedgerRecord]]  = {}

        for led in unclaimed_led:
            spread = led.amount * tol_amount_pct / Decimal("100") if tol_amount_pct else Decimal("0")
            lo = bisect.bisect_left(gw_amounts, led.amount - spread)
            hi = bisect.bisect_right(gw_amounts, led.amount + spread)
            for gw in sorted_gw[lo:hi]:
                if _amount_delta_pct(gw.amount, led.amount) > tol_amount_pct:
                    continue
                if abs(_ts_delta(gw.timestamp, led.timestamp)) > tol_ts_secs:
                    continue
                led_candidates.setdefault(led.transaction_id, []).append(gw)
                gw_candidates.setdefault(gw.transaction_id, []).append(led)

        def _best_and_margin(
            anchor_blob: str, candidates: list,
        ) -> tuple[Optional[object], float, float]:
            scored = sorted(
                (
                    (SequenceMatcher(None, anchor_blob, self._text_blob(c)).ratio(), c)
                    for c in candidates
                ),
                key=lambda pair: pair[0],
                reverse=True,
            )
            if not scored:
                return None, 0.0, 0.0
            best_score, best = scored[0]
            runner_up_score = scored[1][0] if len(scored) > 1 else 0.0
            return best, best_score, best_score - runner_up_score

        results: list[MatchedPair] = []
        for led in unclaimed_led:
            candidates = led_candidates.get(led.transaction_id, [])
            if len(candidates) < 2:
                continue

            gw, score, margin = _best_and_margin(self._text_blob(led), candidates)
            if gw is None or score < min_score or margin < min_margin:
                continue

            gw_side_candidates = gw_candidates.get(gw.transaction_id, [])
            back_led, back_score, back_margin = _best_and_margin(
                self._text_blob(gw), gw_side_candidates
            )
            if back_led is None or back_led.transaction_id != led.transaction_id:
                continue
            if len(gw_side_candidates) >= 2 and (back_score < min_score or back_margin < min_margin):
                continue

            delta_pct  = _amount_delta_pct(gw.amount, led.amount)
            delta_secs = abs(_ts_delta(gw.timestamp, led.timestamp))
            amount_score = max(0.0, 1.0 - float(delta_pct) / float(tol_amount_pct)) if tol_amount_pct else 1.0
            ts_score     = max(0.0, 1.0 - delta_secs / tol_ts_secs) if tol_ts_secs else 1.0
            conf = (
                self.config.confidence_weight_amount    * amount_score
                + self.config.confidence_weight_timestamp * ts_score
            )
            signed_delta_secs = _ts_delta(gw.timestamp, led.timestamp)
            results.append(MatchedPair(
                gateway_record          = gw,
                ledger_record           = led,
                match_type              = "text",
                confidence              = round(conf, 4),
                normalized_ref          = self.normalize_ref(led.reference_id),
                amount_delta            = gw.amount - led.amount,
                timestamp_delta_seconds = signed_delta_secs,
                matched_at              = datetime.now(timezone.utc),
            ))
        return results

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _closest_delta(
        gw             : GatewayRecord,
        led_candidates : list[LedgerRecord],
    ) -> Optional[dict]:
        """Return delta info for the nearest LedgerRecord to a GatewayRecord."""
        if not led_candidates:
            return None
        best = min(
            led_candidates,
            key=lambda l: abs(_ts_delta(gw.timestamp, l.timestamp)),
        )
        return {
            "amount_delta_pct"    : str(_amount_delta_pct(gw.amount, best.amount)),
            "timestamp_delta_hours": round(
                abs(_ts_delta(gw.timestamp, best.timestamp)) / 3600, 2
            ),
        }

    @staticmethod
    def _closest_delta_led(
        led           : LedgerRecord,
        gw_candidates : list[GatewayRecord],
    ) -> Optional[dict]:
        """Return delta info for the nearest GatewayRecord to a LedgerRecord."""
        if not gw_candidates:
            return None
        best = min(
            gw_candidates,
            key=lambda g: abs(_ts_delta(g.timestamp, led.timestamp)),
        )
        return {
            "amount_delta_pct"     : str(_amount_delta_pct(best.amount, led.amount)),
            "timestamp_delta_hours": round(
                abs(_ts_delta(best.timestamp, led.timestamp)) / 3600, 2
            ),
        }

    def _grouped_phase(
        self,
        unclaimed_gw  : list[GatewayRecord],
        unclaimed_led : list[LedgerRecord],
    ) -> list[GroupedMatch]:
        """
        Phase 3 — bounded combinatorial search for multiplicity matches.

        Searches unclaimed_gw and unclaimed_led for:
          one_to_many : 1 GW amount == sum of k LED records (k in 2..max_group_size)
          many_to_one : sum of k GW records == 1 LED amount

        Amount equality is checked within the stricter group_tolerance_pct.
        A group must have an explicit BATCH/PAYOUT anchor and satisfy the
        configured timestamp-spread corroboration check.
        The method is greedy: first valid combination per anchor is accepted.

        Complexity: O(n * C(m, k)) per phase per k.
        For 30 unclaimed records, max_group_size=4 -> ~32k total iterations.
        """
        tol_pct  = self.config.group_tolerance_pct
        max_k    = self.config.max_group_size
        results  : list[GroupedMatch] = []
        used_gw  : set[str] = set()
        used_led : set[str] = set()
        now      = datetime.now(timezone.utc)

        def _make_grouped(
            match_type    : Literal["one_to_many", "many_to_one"],
            gw_group      : list[GatewayRecord],
            led_group     : list[LedgerRecord],
            gw_total      : Decimal,
            led_total     : Decimal,
            delta_pct     : Decimal,
        ) -> GroupedMatch:
            conf = float(max(Decimal("0"), Decimal("1") - delta_pct / tol_pct)) if tol_pct else 1.0
            return GroupedMatch(
                match_type       = match_type,
                gateway_records  = gw_group,
                ledger_records   = led_group,
                gateway_total    = gw_total,
                ledger_total     = led_total,
                amount_delta     = gw_total - led_total,
                amount_delta_pct = delta_pct,
                normalized_refs  = [
                    self.normalize_ref(r.reference_id)
                    for r in gw_group + led_group
                ],
                confidence       = conf,
                matched_at       = now,
            )

        # one_to_many: each unclaimed GW record is the anchor
        for gw in unclaimed_gw:
            if gw.transaction_id in used_gw:
                continue
            if not self._has_prefix(gw, "BATCH"):
                continue
            target   = gw.amount
            pool_led = [l for l in unclaimed_led if l.transaction_id not in used_led]
            found    = False
            for k in range(2, min(max_k + 1, len(pool_led) + 1)):
                if found:
                    break
                for combo in itertools.combinations(pool_led, k):
                    combo_sum = sum(l.amount for l in combo)
                    dp = _amount_delta_pct(combo_sum, target)
                    if dp <= tol_pct and self._timestamps_are_grouped([gw, *combo]):
                        results.append(_make_grouped("one_to_many", [gw], list(combo), target, combo_sum, dp))
                        used_gw.add(gw.transaction_id)
                        for l in combo:
                            used_led.add(l.transaction_id)
                        found = True
                        break

        # many_to_one: each unclaimed LED record is the anchor
        for led in unclaimed_led:
            if led.transaction_id in used_led:
                continue
            if not self._has_prefix(led, "PAYOUT"):
                continue
            target  = led.amount
            pool_gw = [g for g in unclaimed_gw if g.transaction_id not in used_gw]
            found   = False
            for k in range(2, min(max_k + 1, len(pool_gw) + 1)):
                if found:
                    break
                for combo in itertools.combinations(pool_gw, k):
                    combo_sum = sum(g.amount for g in combo)
                    dp = _amount_delta_pct(combo_sum, target)
                    if dp <= tol_pct and self._timestamps_are_grouped([*combo, led]):
                        results.append(_make_grouped("many_to_one", list(combo), [led], combo_sum, target, dp))
                        used_led.add(led.transaction_id)
                        for g in combo:
                            used_gw.add(g.transaction_id)
                        found = True
                        break

        return results

    def _group_candidate_pools(
        self,
        gateway_records: list[GatewayRecord],
        ledger_records: list[LedgerRecord],
    ) -> tuple[list[GatewayRecord], list[LedgerRecord]]:
        """Return only records corroborated by an explicit grouping cue.

        Unmatched ordinary payment rows are intentionally left for the
        unresolved pass.  BATCH anchors may bring nearby ledger children into
        one-to-many matching; PAYOUT anchors may bring nearby gateway splits
        into many-to-one matching.

        KNOWN LIMITATION — inert on real-world data without this convention
        --------------------------------------------------------------------
        The BATCH/SPLIT/PAYOUT prefix requirement is a convention specific
        to our synthetic generator (see generator.py). Real-world exports —
        including BenchRec, the external validation dataset — have no such
        marker in their reference text, so this gate admits nothing and
        Phase 3 never fires. Confirmed empirically: 0 grouped matches on
        the BenchRec run, despite its oracle containing 1,164 genuine
        one-to-many and 334 many-to-one groups (see benchrec_ground_truth.json's
        discrepancy_counts) — those ~1,500 groups fall through to
        missing_in_gateway/missing_in_ledger exceptions undetected.

        This is distinct from the many-to-many exclusion documented in
        benchrec_map.py (which is a scoring-methodology gap in report.py);
        this one is a detection gap in the matcher itself.

        Fix sketch, not yet implemented (deliberately deferred rather than
        rushed — the previous incarnation of this phase had a real
        precision incident from unguarded sum-matching, see EXCEPTION_SAFETY_MARGIN
        in config.py and the module docstring's "Duplicate handling" history):
          1. Add a second corroboration path that doesn't require a prefix:
             restrict candidates to k=2 combinations only (BenchRec's grouped
             shapes are dominated by size-2 groups) and require the SAME
             mutual-uniqueness safeguard Phase 2.5/2.75 already use — accept
             a pairing only if it's the anchor's single best sum-match AND
             that pair's best anchor is symmetrically this one.
          2. This is a real combinatorial-cost problem at scale: naively
             pairing every unclaimed record against every other is
             O(n^2) per anchor. Bucket candidates by (rounded) date first
             (group_max_timestamp_spread_hours is only 24h, so same-day
             buckets collapse the search space by orders of magnitude on a
             multi-year dataset like BenchRec), then run a two-sum search
             within each bucket rather than brute-force combinations.
          3. Validate on BOTH datasets before trusting it: synthetic must
             stay at 100%/100% precision/recall (regression tests already
             cover this), and BenchRec's grouped-match precision must be
             checked directly, not assumed — sum-matching without a
             corroborating signal is exactly what caused the original
             false-positive incident this phase was rewritten to fix.
        """
        batch_gw = [g for g in gateway_records if self._has_prefix(g, "BATCH")]
        group_gw = [
            g for g in gateway_records
            if self._has_prefix(g, "BATCH")
            or self._has_prefix(g, "SPLIT")
        ]
        group_led = [
            l for l in ledger_records
            if self._has_prefix(l, "PAYOUT")
            or any(self._timestamps_are_grouped([batch, l]) for batch in batch_gw)
        ]
        return group_gw, group_led

    @staticmethod
    def _has_prefix(record: GatewayRecord | LedgerRecord, prefix: str) -> bool:
        return record.reference_id.upper().startswith(prefix)

    def _timestamps_are_grouped(
        self, records: list[GatewayRecord | LedgerRecord]
    ) -> bool:
        """Require all grouped records to lie in one settlement window."""
        timestamps = [record.timestamp for record in records]
        spread_seconds = (max(timestamps) - min(timestamps)).total_seconds()
        return abs(spread_seconds) <= self.config.group_max_timestamp_spread_hours * 3600


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def _ts_delta(gw_ts: datetime, led_ts: datetime) -> float:
    """
    Signed timestamp difference in seconds: gw_ts - led_ts.

    Both datetimes are coerced to UTC-aware before subtraction to handle
    naive/aware mismatches from inconsistent source systems.
    """
    def _to_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    return (_to_utc(gw_ts) - _to_utc(led_ts)).total_seconds()


def _amount_delta_pct(gw_amount: Decimal, led_amount: Decimal) -> Decimal:
    """
    Absolute percentage difference: |gw - led| / led * 100.

    Returns Decimal("0") if led_amount is zero to avoid division by zero.
    """
    if led_amount == 0:
        return Decimal("0")
    raw = abs(gw_amount - led_amount) / led_amount * Decimal("100")
    return raw.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
