"""
llm_eval.py — LLM Prompting Consistency Evaluation
=====================================================

The project's stated plan (see README / project notes) is: prompt a local
Ollama model with few-shot examples first, and only consider fine-tuning as
a stretch goal IF base prompting proves inconsistent. That decision has
never actually been measured — this script measures it.

What it checks, per candidate model
------------------------------------
1. Category agreement — does the LLM's category match the rule engine's
   category? On the synthetic dataset the rule engine is independently
   proven 100% accurate against the ground-truth oracle (see
   ``src/diagnose.py``), so "agrees with rules" here is also "agrees with
   ground truth."
2. Parse reliability — fraction of calls that returned valid, parseable
   JSON (``fallback_used=False``). A model that can't reliably follow the
   output format is unusable regardless of reasoning quality.
3. Latency — wall-clock seconds per call, since this runs synchronously in
   the CLI pipeline today.
4. Determinism — for the winning model, the same exception is sent
   ``--repeats`` times at the configured temperature; category and
   confidence are compared across repeats to check whether output is
   stable enough to trust for repeated/production use.

Usage
-----
    python -m src.llm_eval
    python -m src.llm_eval --models qwen2.5:7b-instruct-q4_K_M qwen2.5:14b
    python -m src.llm_eval --per-category 1 --repeats 2

Output
------
Console summary table + data/llm_eval_report.json with full per-call detail.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from src.exceptions import ExceptionCategory, ExceptionRecord, classify
from src.generator import load_gateway_records, load_ledger_records
from src.llm_agent import LLMConfig, reason_about_exception
from src.matcher import ReconciliationEngine

DATA_DIR = Path(__file__).parent.parent / "data"

DEFAULT_MODELS = [
    "qwen2.5:7b-instruct-q4_K_M",
    "qwen2.5:14b",
    "llama3:latest",
    "phi3:latest",
]


def export_sample(sample: list[ExceptionRecord], path: Path) -> None:
    """
    Dump the stratified sample to portable JSON for an external testing
    environment (e.g. a separate venv running a HuggingFace model) so the
    exact same inputs can be compared against ``evaluate_model``'s results
    without that environment needing recon-agent's own dependencies.
    """
    def _record_dict(r) -> Optional[dict]:
        if r is None:
            return None
        return {
            "transaction_id": r.transaction_id,
            "reference_id"  : r.reference_id,
            "amount"        : str(r.amount),
            "currency"      : r.currency,
            "status"        : r.status,
            "timestamp"     : r.timestamp.isoformat(),
            "counterparty"  : r.counterparty,
        }

    payload = [
        {
            "exception_id"         : exc.exception_id,
            "category"             : exc.category.value,
            "source"                : exc.source,
            "gateway_record"        : _record_dict(exc.gateway_record),
            "ledger_record"         : _record_dict(exc.ledger_record),
            "amount_delta_pct"      : str(exc.amount_delta_pct) if exc.amount_delta_pct is not None else None,
            "timestamp_delta_hours" : exc.timestamp_delta_hours,
            "rule_explanation"      : exc.explanation,
        }
        for exc in sample
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate LLM prompting consistency across local Ollama models.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, metavar="TAG")
    parser.add_argument("--per-category", type=int, default=2, metavar="N",
                         help="Exceptions sampled per category for the main comparison (default: 2).")
    parser.add_argument("--repeats", type=int, default=3, metavar="N",
                         help="Repeats per exception for the winning model's determinism check (default: 3).")
    parser.add_argument("--determinism-sample", type=int, default=4, metavar="N",
                         help="Exceptions used for the determinism check (default: 4).")
    parser.add_argument("--timeout", type=float, default=120.0, metavar="SECONDS")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, metavar="DIR",
                         help="Directory to load {prefix}gateway_records.json etc. from "
                              "(default: data/ — the synthetic dataset with a proven-accurate "
                              "rule engine). Use with --prefix to evaluate against a mapped "
                              "external dataset, e.g. --data-dir data/external/processed "
                              "--prefix benchrec_.")
    parser.add_argument("--prefix", type=str, default="", metavar="STR",
                         help="Filename prefix for --data-dir's input files (default: '').")
    parser.add_argument("--skip-determinism", action="store_true",
                         help="Skip the repeated-call determinism check (e.g. when it was already "
                              "established on another dataset and you just want the model comparison).")
    return parser


def _stratified_sample(exceptions: list[ExceptionRecord], per_category: int) -> list[ExceptionRecord]:
    by_cat: dict[ExceptionCategory, list[ExceptionRecord]] = defaultdict(list)
    for exc in exceptions:
        by_cat[exc.category].append(exc)
    sample: list[ExceptionRecord] = []
    for cat in ExceptionCategory:
        sample.extend(by_cat.get(cat, [])[:per_category])
    return sample


def _load_exceptions(data_dir: Path = DATA_DIR, prefix: str = "") -> list[ExceptionRecord]:
    gw_path  = data_dir / f"{prefix}gateway_records.json"
    led_path = data_dir / f"{prefix}ledger_records.json"
    gw  = load_gateway_records(gw_path)
    led = load_ledger_records(led_path)
    result = ReconciliationEngine().run(gw, led)
    return classify(result.unresolved)


def evaluate_model(
    model      : str,
    exceptions : list[ExceptionRecord],
    timeout    : float,
) -> dict:
    cfg = LLMConfig(model=model, timeout_seconds=timeout)
    calls = []
    for exc in exceptions:
        t0 = time.perf_counter()
        result = reason_about_exception(exc, cfg=cfg)
        elapsed = round(time.perf_counter() - t0, 2)
        calls.append({
            "exception_id"     : exc.exception_id,
            "rule_category"    : exc.category.value,
            "llm_category"     : result.category,
            "agrees_with_rules": result.agrees_with_rules,
            "category_matches_rule": result.category == exc.category.value,
            "confidence"       : result.confidence,
            "fallback_used"    : result.fallback_used,
            "fallback_reason"  : result.fallback_reason,
            "elapsed_seconds"  : elapsed,
        })

    consulted = [c for c in calls if not c["fallback_used"]]
    n_fallback = len(calls) - len(consulted)
    n_category_ok = sum(1 for c in consulted if c["category_matches_rule"])

    return {
        "model"                     : model,
        "total_calls"               : len(calls),
        "fallback_count"            : n_fallback,
        "parse_reliability_pct"     : round(len(consulted) / len(calls) * 100, 1) if calls else 0.0,
        "category_agreement_pct"    : round(n_category_ok / len(consulted) * 100, 1) if consulted else None,
        "avg_latency_seconds"       : round(sum(c["elapsed_seconds"] for c in calls) / len(calls), 2) if calls else 0.0,
        "calls"                     : calls,
    }


def evaluate_determinism(
    model      : str,
    exceptions : list[ExceptionRecord],
    repeats    : int,
    timeout    : float,
) -> dict:
    cfg = LLMConfig(model=model, timeout_seconds=timeout)
    per_exception = []
    for exc in exceptions:
        runs = []
        for _ in range(repeats):
            result = reason_about_exception(exc, cfg=cfg)
            runs.append({
                "category"     : result.category,
                "confidence"   : result.confidence,
                "fallback_used": result.fallback_used,
            })
        categories  = {r["category"] for r in runs if not r["fallback_used"]}
        confidences = {r["confidence"] for r in runs if not r["fallback_used"]}
        per_exception.append({
            "exception_id"      : exc.exception_id,
            "runs"              : runs,
            "category_stable"   : len(categories) <= 1,
            "confidence_stable" : len(confidences) <= 1,
        })

    n = len(per_exception)
    cat_stable = sum(1 for e in per_exception if e["category_stable"])
    conf_stable = sum(1 for e in per_exception if e["confidence_stable"])
    return {
        "model"                        : model,
        "repeats_per_exception"        : repeats,
        "exceptions_tested"            : n,
        "category_stable_pct"          : round(cat_stable / n * 100, 1) if n else None,
        "confidence_stable_pct"        : round(conf_stable / n * 100, 1) if n else None,
        "detail"                       : per_exception,
    }


def _print_comparison_table(results: list[dict]) -> None:
    print("MODEL COMPARISON")
    print("=" * 88)
    header = f"{'Model':<28} {'Parse OK':>9} {'Category Agree':>15} {'Avg Latency':>12} {'Fallback':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        cat = f"{r['category_agreement_pct']}%" if r["category_agreement_pct"] is not None else "—"
        print(
            f"{r['model']:<28} {r['parse_reliability_pct']:>8}% {cat:>15} "
            f"{r['avg_latency_seconds']:>10}s {r['fallback_count']:>9}"
        )
    print()


def _pick_winner(results: list[dict]) -> dict:
    def _score(r: dict) -> tuple:
        cat = r["category_agreement_pct"] if r["category_agreement_pct"] is not None else -1
        return (r["parse_reliability_pct"], cat, -r["avg_latency_seconds"])
    return max(results, key=_score)


def main(argv: Optional[list[str]] = None) -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    args = build_parser().parse_args(argv)
    is_synthetic = args.data_dir == DATA_DIR and args.prefix == ""

    print(f"[llm_eval] Loading dataset from {args.data_dir} (prefix={args.prefix!r}) and classifying exceptions ...")
    if not is_synthetic:
        print(
            "[llm_eval] NOTE: on a non-synthetic dataset, 'category agreement' means agreement "
            "with the RULE ENGINE's own category, not necessarily the ground-truth oracle — the "
            "rule engine's exception precision on BenchRec, e.g., is far below 100% (see "
            "report.py's ground_truth_accuracy). This measures LLM<->rule-engine consistency, "
            "not LLM<->ground-truth accuracy, on non-synthetic data."
        )
    exceptions = _load_exceptions(args.data_dir, args.prefix)
    sample = _stratified_sample(exceptions, args.per_category)
    print(f"[llm_eval] {len(exceptions)} total exceptions; evaluating on a stratified sample of {len(sample)}.")
    for cat in ExceptionCategory:
        n = sum(1 for e in sample if e.category == cat)
        print(f"    {cat.value:<20} {n}")
    print()

    sample_path = args.data_dir / f"{args.prefix}llm_eval_sample.json"
    export_sample(sample, sample_path)
    print(f"[llm_eval] Sample exported to {sample_path.resolve()} for use by external testing environments.\n")

    model_results = []
    for model in args.models:
        print(f"[llm_eval] Evaluating {model} on {len(sample)} exceptions ...")
        r = evaluate_model(model, sample, args.timeout)
        model_results.append(r)
        print(
            f"    parse_reliability={r['parse_reliability_pct']}% "
            f"category_agreement={r['category_agreement_pct']}% "
            f"avg_latency={r['avg_latency_seconds']}s"
        )
    print()

    _print_comparison_table(model_results)

    winner = _pick_winner(model_results)
    print(f"[llm_eval] Winner: {winner['model']}")
    print()

    if args.skip_determinism:
        determinism = {"category_stable_pct": None, "confidence_stable_pct": None, "skipped": True}
    else:
        determinism_sample = sample[: args.determinism_sample]
        print(
            f"[llm_eval] Running determinism check on {winner['model']}: "
            f"{len(determinism_sample)} exceptions x {args.repeats} repeats ..."
        )
        determinism = evaluate_determinism(winner["model"], determinism_sample, args.repeats, args.timeout)
        print(
            f"[llm_eval] Category stable across repeats: {determinism['category_stable_pct']}% | "
            f"Confidence stable: {determinism['confidence_stable_pct']}%"
        )
        print()

    threshold_ok = (
        winner["parse_reliability_pct"] >= 95
        and (winner["category_agreement_pct"] or 0) >= 90
        and (determinism["category_stable_pct"] is None or determinism["category_stable_pct"] >= 90)
    )
    print("VERDICT")
    print("=" * 88)
    if threshold_ok:
        print(
            f"Prompted {winner['model']} meets the consistency bar "
            "(parse >=95%, category agreement >=90%, determinism >=90%). "
            "Fine-tuning is NOT warranted — stays a stretch goal, per the original plan."
        )
    else:
        print(
            f"Prompted {winner['model']} did NOT clear the consistency bar. "
            "Consider: more/better few-shot examples in src/prompts.py, a larger local model, "
            "or treating fine-tuning as the next step rather than a stretch goal."
        )

    report = {
        "sample_size"       : len(sample),
        "model_results"     : model_results,
        "winner"            : winner["model"],
        "determinism"       : determinism,
        "meets_consistency_bar": threshold_ok,
    }
    out_path = args.data_dir / f"{args.prefix}llm_eval_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
