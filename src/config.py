"""Shared reconciliation thresholds used by both the matcher and generator."""

from decimal import Decimal


# Phase-2 fuzzy matching thresholds.
MATCHER_AMOUNT_TOLERANCE_PCT = Decimal("2.0")
MATCHER_TIMESTAMP_TOLERANCE_HOURS = 72.0

# Synthetic exception cases must be this far beyond the acceptance threshold.
EXCEPTION_SAFETY_MARGIN = Decimal("1.5")

# Phase-3 sum matching is deliberately stricter because coincidental sums are
# much more likely than a direct reference match.
MATCHER_GROUP_TOLERANCE_PCT = Decimal("0.5")
MATCHER_GROUP_MAX_TIMESTAMP_SPREAD_HOURS = 24.0

# Phase-2.5 content fallback (amount + timestamp only, no reference-id
# agreement) is deliberately much stricter than Phase 2's reference-anchored
# fuzzy tolerance, since amount/date proximity alone is weaker evidence.
# It is also required to be a *mutually unique* candidate pair — see
# matcher.py's _content_phase docstring.
MATCHER_CONTENT_AMOUNT_TOLERANCE_PCT = Decimal("0.1")
MATCHER_CONTENT_TIMESTAMP_TOLERANCE_HOURS = 48.0

# Phase-2.75 text disambiguation re-ranks Phase 2.5's ambiguous (multi-
# candidate) clusters by reference_id/counterparty text similarity
# (difflib.SequenceMatcher ratio, 0..1). Calibrated against BenchRec: true
# matched pairs average ~0.13 similarity vs ~0.09 for random pairs, so both
# thresholds are set low relative to a "confident" text match elsewhere —
# the amount+timestamp tolerance from Phase 2.5 has already done most of the
# narrowing; text similarity here only has to out-rank the other candidates
# within that already-narrow window, not identify a match from scratch.
MATCHER_TEXT_SIMILARITY_MIN_SCORE = 0.10
MATCHER_TEXT_SIMILARITY_MIN_MARGIN = 0.03

# Phase-3b (unmarked pair grouping — see matcher.py's
# _unmarked_pair_grouped_phase) has no BATCH/SPLIT/PAYOUT marker to lean on
# at all, unlike Phase 3's group_tolerance_pct=0.5% band, so it needs a
# stronger signal than "close enough": exact-to-the-cent sum equality.
# Empirically necessary, not a guess — an earlier version of this phase used
# group_tolerance_pct (0.5%) and produced a real false positive on the
# synthetic oracle: three UNRELATED exception pairs (two amount_mismatch,
# one stale_timing) coincidentally summed to within 0.19% of each other and
# got wrongly grouped, dropping synthetic precision/recall from 100%/100%
# to 90%/90%. Exact equality closed that specific hole; mutual uniqueness
# (see the phase's docstring) is the other half of the guard.
MATCHER_UNMARKED_GROUP_AMOUNT_TOLERANCE_PCT = Decimal("0")
