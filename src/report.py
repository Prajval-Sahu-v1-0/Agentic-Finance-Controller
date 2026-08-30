"""
report.py — Match Rate & Exception Reporting
=============================================

Produces the final, judge-facing reconciliation report in two forms:

  Console  — Rich-formatted tables printed to stdout, designed to be readable
             at a glance without any tooling.
  JSON     — report.json saved to data/, containing every metric as structured
             data suitable for downstream dashboards or CI assertions.

Report sections
---------------
1. Match Summary
   Total records per source, matched_exact count & rate, matched_fuzzy count
   & rate, combined match rate, unresolved count.  Match rate is computed
   against gateway_records as the authoritative denominator (the gateway is
   the source of truth for what was actually settled).

2. Monetary Summary
   Total INR value matched (exact + fuzzy), total unresolved value, and the
   combined amount_delta across all matched pairs (negative = fees deducted by
   gateway vs. ledger gross values).

3. Exception Breakdown
   Per-category count and share, sorted by count descending.  Low match rates
   and high exception counts are NOT hidden — the report uses plain language
   and avoids euphemisms.

4. Ground Truth Accuracy  (shown only when ground_truth dict is supplied)
   Evaluates the engine against the synthetic dataset's oracle labels.

   Matching metrics (against all 100 logical pairs):
     True  Positive (TP)  — engine matched a pair GT says should match
     False Positive (FP)  — engine matched a pair GT says should NOT match
     False Negative (FN)  — GT says should match, engine left unresolved
     Precision  = TP / (TP + FP)
     Recall     = TP / (TP + FN)
     F1         = 2 * P * R / (P + R)

   Exception metrics (against all exception-labelled GT pairs):
     TP  — correctly flagged as exception (any category)
     FP  — flagged as exception but GT says it should have been matched
     FN  — GT says exception, but engine matched it instead
     Category Accuracy — among TP exceptions, fraction with correct category

Public API
----------
    generate_report(result, exceptions, ground_truth=None, output_dir=DATA_DIR)
        Full pipeline: compute metrics -> print console report -> save JSON.
        Returns the report dict.

    compute_match_summary(result)      -> dict
    compute_monetary_summary(result)   -> dict
    compute_exception_breakdown(exceptions) -> list[dict]
    compute_ground_truth_accuracy(result, exceptions, ground_truth) -> dict
    print_console_report(report)
    save_json_report(report, path)

Usage
-----
    from src.generator import load_gateway_records, load_ledger_records, load_ground_truth
    from src.matcher   import ReconciliationEngine
    from src.exceptions import classify
    from src.report    import generate_report

    gw, led = load_gateway_records(), load_ledger_records()
    gt       = load_ground_truth()          # optional; omit for production data
    result   = ReconciliationEngine().run(gw, led)
    excs     = classify(result.unresolved)
    report   = generate_report(result, excs, ground_truth=gt)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.exceptions import ExceptionCategory, ExceptionRecord
from src.matcher import ReconciliationResult

DATA_DIR = Path(__file__).parent.parent / "data"

# Ground-truth discrepancy_type -> ExceptionCategory
_GT_TYPE_TO_CATEGORY = {
    "amount_mismatch"  : ExceptionCategory.AMOUNT_MISMATCH,
    "timestamp_drift"  : ExceptionCategory.STALE_TIMING,
    "missing_gateway"  : ExceptionCategory.MISSING_IN_GATEWAY,
    "missing_ledger"   : ExceptionCategory.MISSING_IN_LEDGER,
    "duplicate"        : ExceptionCategory.DUPLICATE,
}
# Ground truth expected_match_status values that mean "should be matched"
_SHOULD_MATCH = {"matched", "partial"}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_report(
    result       : ReconciliationResult,
    exceptions   : list[ExceptionRecord],
    ground_truth : Optional[dict] = None,
    output_dir   : Path = DATA_DIR,
    llm_results  : Optional[list] = None,
    pair_audit_pairs   : Optional[list] = None,
    pair_audit_results : Optional[list] = None,
) -> dict:
    """
    Compute all metrics, print the console report, and save report.json.

    Parameters
    ----------
    result : ReconciliationResult
        Output of ``ReconciliationEngine.run()``.
    exceptions : list[ExceptionRecord]
        Output of ``classify(result.unresolved)``.
    ground_truth : dict, optional
        The ground_truth.json dict from the synthetic generator.  When
        supplied, a precision/recall section is added to the report.
        Omit for production data where the oracle is unavailable.
    output_dir : Path
        Directory where ``report.json`` is written.
    llm_results : list[LLMReasoningResult], optional
        Output of ``reason_about_exceptions_batch(exceptions)``, same length
        and order as ``exceptions``. When supplied, an LLM Review section is
        added showing agreement rate and any category overrides.
    pair_audit_pairs : list[MatchedPair], optional
        The sampled Phase 2.5/2.75 matches passed to
        ``reason_about_pairs_batch`` — must line up 1:1 with
        ``pair_audit_results``.
    pair_audit_results : list[LLMReasoningResult], optional
        Output of ``reason_about_pairs_batch(pair_audit_pairs)``. When
        supplied (together with ``pair_audit_pairs``), a Pair Audit section
        shows how often the LLM agrees these no-reference-agreement matches
        are genuine.

    Returns
    -------
    dict
        The full report as a plain Python dict (also serialised to JSON).
    """
    report = _assemble_report(
        result, exceptions, ground_truth, llm_results,
        pair_audit_pairs, pair_audit_results,
    )
    print_console_report(report)
    save_json_report(report, output_dir / "report.json")
    return report


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_match_summary(result: ReconciliationResult) -> dict:
    """
    High-level matching counts and rates.

    The denominator for all rates is ``total_gateway_records`` — the gateway
    export is treated as the authoritative settlement ledger.
    """
    total_gw  = result.run_metadata["total_gateway_records"]
    total_led = result.run_metadata["total_ledger_records"]
    n_exact   = len(result.matched_exact)
    n_fuzzy   = len(result.matched_fuzzy)
    n_content = len(result.matched_content)
    n_text    = len(result.matched_text)
    n_grp_gw  = sum(len(g.gateway_records) for g in result.matched_grouped)
    n_grp_led = sum(len(g.ledger_records)  for g in result.matched_grouped)
    n_grp     = len(result.matched_grouped)
    n_matched = n_exact + n_fuzzy + n_content + n_text + n_grp_gw
    n_unres   = len(result.unresolved)

    def _pct(n: int) -> float:
        return round(n / total_gw * 100, 2) if total_gw else 0.0

    return {
        "total_gateway_records" : total_gw,
        "total_ledger_records"  : total_led,
        "matched_exact"   : {"count": n_exact,   "rate_pct": _pct(n_exact)},
        "matched_fuzzy"   : {"count": n_fuzzy,   "rate_pct": _pct(n_fuzzy)},
        "matched_content" : {"count": n_content, "rate_pct": _pct(n_content)},
        "matched_text"    : {"count": n_text,    "rate_pct": _pct(n_text)},
        "matched_grouped" : {
            "groups"           : n_grp,
            "gateway_records"  : n_grp_gw,
            "ledger_records"   : n_grp_led,
            "rate_pct"         : _pct(n_grp_gw),
        },
        "total_matched"   : {"count": n_matched, "rate_pct": _pct(n_matched)},
        "unresolved"      : {"count": n_unres,   "rate_pct": _pct(n_unres)},
        "elapsed_ms"      : result.run_metadata.get("elapsed_ms", 0),
    }


def compute_monetary_summary(result: ReconciliationResult) -> dict:
    """
    Aggregate INR values across matched and unresolved buckets.

    amount_delta_total is the sum of (gateway.amount - ledger.amount) across
    all matched pairs.  A negative value means the gateway settled less than
    the ledger expected in aggregate — the typical fee-deduction signature.
    """
    def _sum_gw(pairs):
        return sum(p.gateway_record.amount for p in pairs)

    def _sum_led(pairs):
        return sum(p.ledger_record.amount for p in pairs)

    def _sum_delta(pairs):
        return sum(p.amount_delta for p in pairs)

    all_matched = result.matched_exact + result.matched_fuzzy + result.matched_content + result.matched_text

    matched_gw_value  = _sum_gw(all_matched)
    matched_led_value = _sum_led(all_matched)
    delta_total       = _sum_delta(all_matched)

    unres_gw_value  = sum(
        u.gateway_record.amount for u in result.unresolved
        if u.gateway_record is not None
    )
    unres_led_value = sum(
        u.ledger_record.amount for u in result.unresolved
        if u.ledger_record is not None
    )

    return {
        "currency"                    : "INR",
        "matched_gateway_value"       : str(matched_gw_value),
        "matched_ledger_value"        : str(matched_led_value),
        "matched_amount_delta_total"  : str(delta_total),
        "unresolved_gateway_value"    : str(unres_gw_value),
        "unresolved_ledger_value"     : str(unres_led_value),
    }


def compute_exception_breakdown(exceptions: list[ExceptionRecord]) -> list[dict]:
    """
    Per-category exception counts and share percentages, sorted by count desc.
    """
    total = len(exceptions) or 1
    counts: dict[str, int] = {}
    for exc in exceptions:
        key = exc.category.value
        counts[key] = counts.get(key, 0) + 1

    return [
        {
            "category"  : cat,
            "count"     : cnt,
            "share_pct" : round(cnt / total * 100, 1),
        }
        for cat, cnt in sorted(counts.items(), key=lambda kv: -kv[1])
    ]


def compute_llm_review(
    exceptions  : list[ExceptionRecord],
    llm_results : list,
) -> dict:
    """
    Summarise a batch of ``LLMReasoningResult`` objects paired 1:1 with
    ``exceptions`` (same order, same length — see
    ``reason_about_exceptions_batch``).

    Returns
    -------
    dict with:
      consulted        — count where the LLM actually responded (not fallback)
      fallback_count    — count where Ollama was unreachable / capped / unparseable
      agreement_rate_pct — % of consulted results where agrees_with_rules=True
      overrides         — list of {exception_id, rule_category, llm_category,
                           confidence, explanation} for consulted disagreements
      samples           — up to 5 consulted explanations, for a quick read
    """
    consulted  = [r for r in llm_results if not r.fallback_used]
    fallback_n = len(llm_results) - len(consulted)
    agreeing   = sum(1 for r in consulted if r.agrees_with_rules)

    overrides = [
        {
            "exception_id" : exc.exception_id,
            "rule_category": exc.category.value,
            "llm_category" : r.category,
            "confidence"   : r.confidence,
            "explanation"  : r.explanation,
        }
        for exc, r in zip(exceptions, llm_results)
        if not r.fallback_used and not r.agrees_with_rules
    ]

    samples = [
        {
            "exception_id": exc.exception_id,
            "category"    : r.category,
            "confidence"  : r.confidence,
            "explanation" : r.explanation,
        }
        for exc, r in zip(exceptions, llm_results)
        if not r.fallback_used
    ][:5]

    return {
        "total_exceptions"    : len(llm_results),
        "consulted"           : len(consulted),
        "fallback_count"      : fallback_n,
        "agreement_rate_pct"  : round(agreeing / len(consulted) * 100, 1) if consulted else None,
        "override_count"      : len(overrides),
        "overrides"           : overrides,
        "samples"             : samples,
    }


def compute_pair_audit(
    pairs   : list,
    results : list,
) -> dict:
    """
    Summarise an LLM audit of a sample of Phase 2.5 ("content") / Phase 2.75
    ("text") matches — pairs the engine matched WITHOUT any reference-ID
    agreement, so worth a second opinion on precision. ``pairs`` and
    ``results`` must be 1:1, same order (see ``reason_about_pairs_batch``).
    """
    consulted  = [r for r in results if not r.fallback_used]
    fallback_n = len(results) - len(consulted)
    agreeing   = sum(1 for r in consulted if r.agrees_with_rules)

    flagged = [
        {
            "gateway_transaction_id": p.gateway_record.transaction_id,
            "ledger_transaction_id" : p.ledger_record.transaction_id,
            "match_type"            : p.match_type,
            "engine_confidence"     : round(p.confidence, 3),
            "llm_confidence"        : r.confidence,
            "explanation"           : r.explanation,
        }
        for p, r in zip(pairs, results)
        if not r.fallback_used and not r.agrees_with_rules
    ]

    return {
        "sample_size"       : len(results),
        "consulted"         : len(consulted),
        "fallback_count"    : fallback_n,
        "agreement_rate_pct": round(agreeing / len(consulted) * 100, 1) if consulted else None,
        "flagged_as_coincidental": flagged,
    }


def compute_ground_truth_accuracy(
    result       : ReconciliationResult,
    exceptions   : list[ExceptionRecord],
    ground_truth : dict,
) -> dict:
    """
    Evaluate the engine against the synthetic ground-truth oracle.

    Returns a dict with:
      match_metrics     — TP / FP / FN / precision / recall / F1 for matching
      exception_metrics — TP / FP / FN / precision / recall / F1 + category accuracy
      per_category      — per ExceptionCategory recall on exception records
    """
    pairs = ground_truth.get("pairs", [])

    # ── Build oracle sets ────────────────────────────────────────────

    # 1:1 pairs where both sides should be matched
    should_match_gw : dict[str, str] = {}   # gw_txn_id -> led_txn_id
    should_match_led: dict[str, str] = {}   # led_txn_id -> gw_txn_id

    # Grouped pairs: gw_txn_id -> frozenset of led_txn_ids  (one_to_many)
    should_group_otm: dict[str, frozenset] = {}
    # Grouped pairs: led_txn_id -> frozenset of gw_txn_ids  (many_to_one)
    should_group_mto: dict[str, frozenset] = {}

    # Records that should surface as exceptions
    should_except_gw : dict[str, ExceptionCategory] = {}
    should_except_led: dict[str, ExceptionCategory] = {}

    for p in pairs:
        gw_id       = p.get("gateway_transaction_id")
        led_id      = p.get("ledger_transaction_id")
        exp_status  = p.get("expected_match_status", "")
        disc_type   = p.get("discrepancy_type", "")
        dup_gw_id   = p.get("gateway_duplicate_transaction_id")

        if exp_status in _SHOULD_MATCH and gw_id and led_id:
            should_match_gw[gw_id]  = led_id
            should_match_led[led_id] = gw_id

        if exp_status == "duplicate" and gw_id and led_id:
            should_match_gw[gw_id]  = led_id
            should_match_led[led_id] = gw_id
            if dup_gw_id:
                should_except_gw[dup_gw_id] = ExceptionCategory.DUPLICATE

        if exp_status == "unmatched_gateway" and gw_id:
            cat = _GT_TYPE_TO_CATEGORY.get(disc_type, ExceptionCategory.MISSING_IN_LEDGER)
            should_except_gw[gw_id] = cat

        if exp_status == "unmatched_ledger" and led_id:
            cat = _GT_TYPE_TO_CATEGORY.get(disc_type, ExceptionCategory.MISSING_IN_GATEWAY)
            should_except_led[led_id] = cat

        if exp_status == "partial" and gw_id and led_id:
            should_match_gw[gw_id]  = led_id
            should_match_led[led_id] = gw_id

        # Grouped GT entries
        if exp_status == "grouped_one_to_many" and gw_id:
            led_ids = frozenset(p.get("ledger_transaction_ids", []))
            should_group_otm[gw_id] = led_ids

        if exp_status == "grouped_many_to_one" and led_id:
            gw_ids = frozenset(p.get("gateway_transaction_ids", []))
            should_group_mto[led_id] = gw_ids

    # ── Evaluate 1:1 matched pairs ───────────────────────────────────

    all_matched = result.matched_exact + result.matched_fuzzy + result.matched_content + result.matched_text
    match_tp = match_fp = 0
    for pair in all_matched:
        gw_id  = pair.gateway_record.transaction_id
        led_id = pair.ledger_record.transaction_id
        if gw_id in should_match_gw and should_match_gw[gw_id] == led_id:
            match_tp += 1
        else:
            match_fp += 1

    # ── Evaluate grouped matches ─────────────────────────────────────

    grouped_tp = grouped_fp = 0
    for gm in result.matched_grouped:
        if gm.match_type == "one_to_many":
            gw_id  = gm.gateway_records[0].transaction_id
            led_set = frozenset(l.transaction_id for l in gm.ledger_records)
            if gw_id in should_group_otm and should_group_otm[gw_id] == led_set:
                grouped_tp += 1
            else:
                grouped_fp += 1
        else:  # many_to_one
            led_id = gm.ledger_records[0].transaction_id
            gw_set = frozenset(g.transaction_id for g in gm.gateway_records)
            if led_id in should_group_mto and should_group_mto[led_id] == gw_set:
                grouped_tp += 1
            else:
                grouped_fp += 1

    # FN: GT grouped pairs not found by engine
    matched_gw_ids  = {p.gateway_record.transaction_id for p in all_matched}
    # Also collect all GW IDs claimed by grouped matches
    for gm in result.matched_grouped:
        for g in gm.gateway_records:
            matched_gw_ids.add(g.transaction_id)

    match_fn  = sum(1 for gid in should_match_gw   if gid not in matched_gw_ids)
    match_fn += sum(1 for gid in should_group_otm  if gid not in matched_gw_ids)
    # many_to_one FN: led not found in any grouped match
    matched_led_ids_grp = set()
    for gm in result.matched_grouped:
        for l in gm.ledger_records:
            matched_led_ids_grp.add(l.transaction_id)
    match_fn += sum(1 for lid in should_group_mto if lid not in matched_led_ids_grp)

    # ── Evaluate exceptions ──────────────────────────────────────────

    exc_tp = exc_fp = exc_cat_correct = 0
    for exc in exceptions:
        if exc.gateway_record:
            tid      = exc.gateway_record.transaction_id
            expected = should_except_gw.get(tid)
        else:
            tid      = exc.ledger_record.transaction_id
            expected = should_except_led.get(tid)

        if expected is not None:
            exc_tp += 1
            if exc.category == expected:
                exc_cat_correct += 1
        else:
            exc_fp += 1

    # FN: things that should be exceptions but ended up matched
    all_except_gw_ids  = {
        e.gateway_record.transaction_id
        for e in exceptions if e.gateway_record
    }
    all_except_led_ids = {
        e.ledger_record.transaction_id
        for e in exceptions if e.ledger_record
    }
    exc_fn  = sum(1 for gid in should_except_gw  if gid not in all_except_gw_ids)
    exc_fn += sum(1 for lid in should_except_led  if lid not in all_except_led_ids)

    # Per-category recall
    per_cat_should: dict[str, int] = {}
    per_cat_correct: dict[str, int] = {}
    for cat in should_except_gw.values():
        per_cat_should[cat.value] = per_cat_should.get(cat.value, 0) + 1
    for cat in should_except_led.values():
        per_cat_should[cat.value] = per_cat_should.get(cat.value, 0) + 1
    for exc in exceptions:
        if exc.gateway_record:
            tid = exc.gateway_record.transaction_id
            expected = should_except_gw.get(tid)
        else:
            tid = exc.ledger_record.transaction_id
            expected = should_except_led.get(tid)
        if expected and exc.category == expected:
            per_cat_correct[expected.value] = per_cat_correct.get(expected.value, 0) + 1

    per_category = {
        cat: {
            "should_flag"    : per_cat_should.get(cat, 0),
            "correctly_flagged": per_cat_correct.get(cat, 0),
            "recall_pct"     : round(
                per_cat_correct.get(cat, 0) / per_cat_should[cat] * 100, 1
            ) if per_cat_should.get(cat) else None,
        }
        for cat in per_cat_should
    }

    def _prf(tp, fp, fn):
        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return round(p * 100, 2), round(r * 100, 2), round(f1 * 100, 2)

    mp, mr, mf = _prf(match_tp + grouped_tp, match_fp + grouped_fp, match_fn)
    ep, er, ef = _prf(exc_tp,   exc_fp,   exc_fn)

    cat_acc = round(exc_cat_correct / exc_tp * 100, 2) if exc_tp else 0.0

    return {
        "oracle_pairs_evaluated"  : len(pairs),
        "match_metrics": {
            "true_positives"         : match_tp + grouped_tp,
            "false_positives"        : match_fp + grouped_fp,
            "false_negatives"        : match_fn,
            "precision_pct"          : mp,
            "recall_pct"             : mr,
            "f1_pct"                 : mf,
            "grouped_tp"             : grouped_tp,
            "grouped_fp"             : grouped_fp,
        },
        "exception_metrics": {
            "true_positives"         : exc_tp,
            "false_positives"        : exc_fp,
            "false_negatives"        : exc_fn,
            "precision_pct"          : ep,
            "recall_pct"             : er,
            "f1_pct"                 : ef,
            "category_accuracy_pct"  : cat_acc,
        },
        "per_category_recall": per_category,
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _assemble_report(
    result       : ReconciliationResult,
    exceptions   : list[ExceptionRecord],
    ground_truth : Optional[dict],
    llm_results  : Optional[list] = None,
    pair_audit_pairs   : Optional[list] = None,
    pair_audit_results : Optional[list] = None,
) -> dict:
    report: dict = {
        "generated_at"        : datetime.now(timezone.utc).isoformat(),
        "engine_config"       : result.run_metadata.get("config", {}),
        "match_summary"       : compute_match_summary(result),
        "monetary_summary"    : compute_monetary_summary(result),
        "exception_breakdown" : compute_exception_breakdown(exceptions),
    }
    if llm_results is not None:
        report["llm_review"] = compute_llm_review(exceptions, llm_results)
    if pair_audit_pairs is not None and pair_audit_results is not None:
        report["llm_pair_audit"] = compute_pair_audit(pair_audit_pairs, pair_audit_results)
    if ground_truth is not None:
        report["ground_truth_accuracy"] = compute_ground_truth_accuracy(
            result, exceptions, ground_truth
        )
    return report


# ---------------------------------------------------------------------------
# Console rendering
# ---------------------------------------------------------------------------

def print_console_report(report: dict) -> None:
    """
    Render the full report as Rich-formatted tables to stdout.

    Colour conventions
    ------------------
    green  — good outcomes (high match rates, high precision/recall)
    yellow — borderline / needs attention
    red    — poor outcomes that require immediate attention
    """
    console = Console()
    ms  = report["match_summary"]
    mon = report["monetary_summary"]
    exc = report["exception_breakdown"]

    console.print()
    console.print(Panel(
        f"[bold white]RECON-AGENT  -  Reconciliation Report[/bold white]\n"
        f"[dim]Generated: {report['generated_at']}[/dim]",
        box=box.DOUBLE_EDGE,
        style="bold blue",
        padding=(0, 2),
    ))

    # ── Match summary ────────────────────────────────────────────────
    t = Table(title="Match Summary", box=box.SIMPLE_HEAVY, show_lines=False)
    t.add_column("Category",              style="bold",  min_width=28)
    t.add_column("Count",  justify="right", min_width=8)
    t.add_column("Rate",   justify="right", min_width=9)

    total_gw  = ms["total_gateway_records"]
    total_led = ms["total_ledger_records"]
    t.add_row("Gateway records (input)",  str(total_gw),  "—")
    t.add_row("Ledger records (input)",   str(total_led), "—")
    t.add_section()
    t.add_row(
        "Matched - Exact (Phase 1)",
        str(ms["matched_exact"]["count"]),
        _rate_cell(ms["matched_exact"]["rate_pct"]),
    )
    t.add_row(
        "Matched - Fuzzy (Phase 2)",
        str(ms["matched_fuzzy"]["count"]),
        _rate_cell(ms["matched_fuzzy"]["rate_pct"]),
    )
    if ms.get("matched_content", {}).get("count", 0):
        t.add_row(
            "Matched - Content (Phase 2.5)",
            str(ms["matched_content"]["count"]),
            _rate_cell(ms["matched_content"]["rate_pct"]),
        )
    if ms.get("matched_text", {}).get("count", 0):
        t.add_row(
            "Matched - Text (Phase 2.75)",
            str(ms["matched_text"]["count"]),
            _rate_cell(ms["matched_text"]["rate_pct"]),
        )
    grp = ms.get("matched_grouped", {})
    if grp.get("groups", 0):
        t.add_row(
            f"Matched - Grouped (Phase 3)  [{grp['groups']} groups]",
            str(grp["gateway_records"]),
            _rate_cell(grp["rate_pct"]),
        )
    t.add_section()
    t.add_row(
        "[bold]Total Matched[/bold]",
        f"[bold]{ms['total_matched']['count']}[/bold]",
        f"[bold]{_rate_cell(ms['total_matched']['rate_pct'])}[/bold]",
    )
    t.add_row(
        "[yellow]Unresolved[/yellow]",
        f"[yellow]{ms['unresolved']['count']}[/yellow]",
        f"[yellow]{ms['unresolved']['rate_pct']} %[/yellow]",
    )
    t.caption = f"Rate denominator = gateway records ({total_gw})   |   Elapsed: {ms['elapsed_ms']} ms"
    console.print(t)

    # ── Monetary summary ─────────────────────────────────────────────
    t2 = Table(title="Monetary Summary", box=box.SIMPLE_HEAVY, show_lines=False)
    t2.add_column("Item",   style="bold", min_width=32)
    t2.add_column("Amount", justify="right", min_width=16)
    t2.add_row("Matched gateway value",       f"INR {_fmt_inr(mon['matched_gateway_value'])}")
    t2.add_row("Matched ledger value",        f"INR {_fmt_inr(mon['matched_ledger_value'])}")
    delta = Decimal(mon["matched_amount_delta_total"])
    delta_style = "red" if delta < 0 else "green"
    t2.add_row(
        "Net amount delta (GW - Ledger)",
        f"[{delta_style}]INR {_fmt_inr(str(delta))}[/{delta_style}]",
    )
    t2.add_section()
    t2.add_row("Unresolved gateway value",    f"[yellow]INR {_fmt_inr(mon['unresolved_gateway_value'])}[/yellow]")
    t2.add_row("Unresolved ledger value",     f"[yellow]INR {_fmt_inr(mon['unresolved_ledger_value'])}[/yellow]")
    console.print(t2)

    # ── Exception breakdown ──────────────────────────────────────────
    t3 = Table(title=f"Exception Breakdown  ({ms['unresolved']['count']} records)", box=box.SIMPLE_HEAVY)
    t3.add_column("Category",   style="bold", min_width=22)
    t3.add_column("Count",  justify="right", min_width=7)
    t3.add_column("Share",  justify="right", min_width=8)
    for row in exc:
        t3.add_row(row["category"], str(row["count"]), f"{row['share_pct']} %")
    console.print(t3)

    # ── LLM review ────────────────────────────────────────────────────
    if "llm_review" in report:
        lr = report["llm_review"]
        console.print()
        console.print(Panel(
            "[bold white]LLM Review[/bold white]  "
            "[dim](Ollama second opinion on flagged exceptions)[/dim]",
            style="cyan", box=box.ROUNDED, padding=(0, 2),
        ))
        t_llm = Table(box=box.SIMPLE_HEAVY, show_lines=False)
        t_llm.add_column("Metric", style="bold", min_width=28)
        t_llm.add_column("Value", justify="right", min_width=10)
        t_llm.add_row("Exceptions consulted", f"{lr['consulted']} / {lr['total_exceptions']}")
        t_llm.add_row("Fallback (unavailable/capped)", str(lr["fallback_count"]))
        agree = lr["agreement_rate_pct"]
        agree_str = f"{agree} %" if agree is not None else "—"
        agree_style = "green" if agree is not None and agree >= 80 else ("yellow" if agree is not None and agree >= 50 else "red")
        t_llm.add_row("Agreement with rule engine", f"[{agree_style}]{agree_str}[/{agree_style}]")
        t_llm.add_row("Category overrides proposed", str(lr["override_count"]))
        console.print(t_llm)

        if lr["overrides"]:
            t_ov = Table(title="Proposed Overrides", box=box.SIMPLE_HEAVY, show_lines=True)
            t_ov.add_column("Exception", min_width=14)
            t_ov.add_column("Rule Category", min_width=16)
            t_ov.add_column("LLM Category", min_width=16)
            t_ov.add_column("Explanation")
            for ov in lr["overrides"][:5]:
                t_ov.add_row(ov["exception_id"], ov["rule_category"], ov["llm_category"], ov["explanation"])
            console.print(t_ov)

    # ── LLM pair audit ───────────────────────────────────────────────
    if "llm_pair_audit" in report:
        pa = report["llm_pair_audit"]
        console.print()
        console.print(Panel(
            "[bold white]LLM Pair Audit[/bold white]  "
            "[dim](sample of no-reference-agreement matches, Phase 2.5/2.75)[/dim]",
            style="cyan", box=box.ROUNDED, padding=(0, 2),
        ))
        t_pa = Table(box=box.SIMPLE_HEAVY, show_lines=False)
        t_pa.add_column("Metric", style="bold", min_width=28)
        t_pa.add_column("Value", justify="right", min_width=10)
        t_pa.add_row("Pairs audited", f"{pa['consulted']} / {pa['sample_size']}")
        t_pa.add_row("Fallback (unavailable/capped)", str(pa["fallback_count"]))
        agree = pa["agreement_rate_pct"]
        agree_str = f"{agree} %" if agree is not None else "—"
        agree_style = "green" if agree is not None and agree >= 80 else ("yellow" if agree is not None and agree >= 50 else "red")
        t_pa.add_row("Judged genuine (not coincidental)", f"[{agree_style}]{agree_str}[/{agree_style}]")
        t_pa.add_row("Flagged as coincidental", str(len(pa["flagged_as_coincidental"])))
        console.print(t_pa)

        if pa["flagged_as_coincidental"]:
            t_fl = Table(title="Flagged as Possibly Coincidental", box=box.SIMPLE_HEAVY, show_lines=True)
            t_fl.add_column("Gateway ID", min_width=14)
            t_fl.add_column("Ledger ID", min_width=14)
            t_fl.add_column("Type", min_width=8)
            t_fl.add_column("Explanation")
            for row in pa["flagged_as_coincidental"][:5]:
                t_fl.add_row(
                    row["gateway_transaction_id"], row["ledger_transaction_id"],
                    row["match_type"], row["explanation"],
                )
            console.print(t_fl)

    # ── Ground truth accuracy ────────────────────────────────────────
    if "ground_truth_accuracy" in report:
        gta = report["ground_truth_accuracy"]
        mm  = gta["match_metrics"]
        em  = gta["exception_metrics"]
        pcr = gta["per_category_recall"]

        console.print()
        console.print(Panel(
            "[bold white]Ground Truth Accuracy[/bold white]  "
            "[dim](synthetic oracle evaluation)[/dim]",
            style="magenta", box=box.ROUNDED, padding=(0, 2),
        ))

        t4 = Table(title="Matching Metrics", box=box.SIMPLE_HEAVY, show_lines=False)
        t4.add_column("Metric",    style="bold", min_width=30)
        t4.add_column("Value",     justify="right", min_width=10)
        t4.add_row("True Positives (correct matches)",  str(mm["true_positives"]))
        t4.add_row("False Positives (wrong matches)",   f"[{'red' if mm['false_positives'] else 'green'}]{mm['false_positives']}[/{'red' if mm['false_positives'] else 'green'}]")
        t4.add_row("False Negatives (missed matches)",  f"[{'red' if mm['false_negatives'] else 'green'}]{mm['false_negatives']}[/{'red' if mm['false_negatives'] else 'green'}]")
        t4.add_section()
        t4.add_row("Precision",   f"[bold]{_pct_cell(mm['precision_pct'])}[/bold]")
        t4.add_row("Recall",      f"[bold]{_pct_cell(mm['recall_pct'])}[/bold]")
        t4.add_row("F1 Score",    f"[bold]{_pct_cell(mm['f1_pct'])}[/bold]")
        console.print(t4)

        t5 = Table(title="Exception Classification Metrics", box=box.SIMPLE_HEAVY, show_lines=False)
        t5.add_column("Metric",    style="bold", min_width=34)
        t5.add_column("Value",     justify="right", min_width=10)
        t5.add_row("True Positives (correctly flagged)",   str(em["true_positives"]))
        t5.add_row("False Positives (incorrectly flagged)",f"[{'red' if em['false_positives'] else 'green'}]{em['false_positives']}[/{'red' if em['false_positives'] else 'green'}]")
        t5.add_row("False Negatives (missed exceptions)",  f"[{'red' if em['false_negatives'] else 'green'}]{em['false_negatives']}[/{'red' if em['false_negatives'] else 'green'}]")
        t5.add_section()
        t5.add_row("Precision",           f"[bold]{_pct_cell(em['precision_pct'])}[/bold]")
        t5.add_row("Recall",              f"[bold]{_pct_cell(em['recall_pct'])}[/bold]")
        t5.add_row("F1 Score",            f"[bold]{_pct_cell(em['f1_pct'])}[/bold]")
        t5.add_row("Category Accuracy",   f"[bold]{_pct_cell(em['category_accuracy_pct'])}[/bold]")
        console.print(t5)

        t6 = Table(title="Per-Category Exception Recall", box=box.SIMPLE_HEAVY)
        t6.add_column("Exception Category", style="bold", min_width=22)
        t6.add_column("Should Flag", justify="right", min_width=11)
        t6.add_column("Flagged OK",  justify="right", min_width=10)
        t6.add_column("Recall",      justify="right", min_width=9)
        for cat, stats in sorted(pcr.items()):
            recall = stats["recall_pct"]
            recall_str = f"{recall} %" if recall is not None else "—"
            style = "green" if recall == 100.0 else ("yellow" if recall and recall >= 50 else "red")
            t6.add_row(
                cat,
                str(stats["should_flag"]),
                str(stats["correctly_flagged"]),
                f"[{style}]{recall_str}[/{style}]",
            )
        console.print(t6)

    console.print()


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------

def save_json_report(report: dict, path: Path) -> None:
    """
    Serialise the report dict to JSON and write to *path*.

    Decimal values are coerced to strings to preserve precision.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, default=_json_default),
        encoding="utf-8",
    )
    Console().print(f"[dim]Report saved -> {path}[/dim]")


def _json_default(obj):
    if isinstance(obj, Decimal):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _rate_cell(pct: float) -> str:
    """Colour-code a match-rate percentage."""
    color = "green" if pct >= 80 else ("yellow" if pct >= 50 else "red")
    return f"[{color}]{pct} %[/{color}]"


def _pct_cell(pct: float) -> str:
    """Colour-code a precision/recall/F1 percentage."""
    color = "green" if pct >= 90 else ("yellow" if pct >= 70 else "red")
    return f"[{color}]{pct} %[/{color}]"


def _fmt_inr(value: str) -> str:
    """Format an INR Decimal string with comma thousands separator."""
    try:
        d = Decimal(value)
        # Simple thousands formatting without locale dependency
        parts = f"{abs(d):,.2f}"
        return ("-" if d < 0 else "") + parts
    except Exception:
        return value
