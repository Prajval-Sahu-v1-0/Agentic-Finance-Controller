"""
main.py — CLI Entry Point
==========================

Orchestrates the full recon-agent pipeline via a simple argparse interface.

Commands
--------
    python -m src.main --generate
        Generate fresh synthetic gateway + ledger data (+ ground truth) and
        save them to data/.  Then immediately run the full pipeline on the
        new data.

    python -m src.main --run
        Load existing data/gateway_records.json and data/ledger_records.json,
        run the reconciliation engine and exception classifier, and print the
        final report.  Also loads data/ground_truth.json if present so that
        accuracy metrics are included in the report.

    python -m src.main --generate --seed 99
        Generate with a specific random seed (default: 42).

    python -m src.main --run --amount-tolerance 3.0 --timestamp-tolerance 48
        Run with custom engine tolerances.

Pipeline
--------
    generate_dataset()          (if --generate)
         |
         v
    ReconciliationEngine.run()
         |
         v
    classify()
         |
         v
    generate_report()  ->  console + data/report.json
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> None:
    """
    Generate a fresh synthetic dataset, then run the full pipeline on it.

    Synthetic data is written to data/ (gateway_records.json,
    ledger_records.json, ground_truth.json).  The pipeline is run
    immediately so the user sees a report without a second command.
    """
    from src.generator import generate_dataset

    print(f"[generate] Generating synthetic dataset (seed={args.seed}) ...")
    gw, led, gt = generate_dataset(seed=args.seed, save=True)
    print(
        f"[generate] {len(gw)} gateway records, "
        f"{len(led)} ledger records written to {DATA_DIR.resolve()}"
    )
    print()
    _run_pipeline(gw, led, ground_truth=gt, args=args)


def cmd_run(args: argparse.Namespace) -> None:
    """
    Load existing data files from data/ and run the reconciliation pipeline.

    Exits with a clear error message if the data files are missing — run
    ``--generate`` first to create them.
    """
    from src.generator import load_gateway_records, load_ledger_records, load_ground_truth

    data_dir = args.data_dir
    gw_path  = data_dir / f"{args.prefix}gateway_records.json"
    led_path = data_dir / f"{args.prefix}ledger_records.json"
    gt_path  = data_dir / f"{args.prefix}ground_truth.json"

    for p in (gw_path, led_path):
        if not p.exists():
            print(
                f"[error] {p} not found.\n"
                "Run  python -m src.main --generate  to create synthetic data first, "
                "or point --data-dir/--prefix at an existing dataset (e.g. "
                "data/external/processed/ with --prefix benchrec_).",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"[run] Loading data from {data_dir.resolve()} (prefix={args.prefix!r}) ...")
    gw  = load_gateway_records(gw_path)
    led = load_ledger_records(led_path)
    gt  = load_ground_truth(gt_path) if gt_path.exists() else None
    if gt is None:
        print("[run] ground_truth.json not found — accuracy metrics will be skipped.")
    print(f"[run] {len(gw)} gateway records, {len(led)} ledger records loaded.")
    print()
    _run_pipeline(gw, led, ground_truth=gt, args=args)


# ---------------------------------------------------------------------------
# Shared pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(gw, led, ground_truth, args: argparse.Namespace) -> None:
    """
    Run ReconciliationEngine -> classify -> generate_report.

    Shared by both --generate and --run so there is no code duplication.
    """
    from src.matcher    import ReconciliationEngine, MatchConfig
    from src.exceptions import classify
    from src.report     import generate_report

    config = MatchConfig(
        amount_tolerance_pct            = Decimal(str(args.amount_tolerance)),
        timestamp_tolerance_hours       = args.timestamp_tolerance,
        exact_timestamp_tolerance_hours = args.exact_timestamp_tolerance,
    )

    print(
        f"[engine] Tolerances — exact ts: {config.exact_timestamp_tolerance_hours} h | "
        f"fuzzy amount: {config.amount_tolerance_pct} % | "
        f"fuzzy ts: {config.timestamp_tolerance_hours} h"
    )

    engine = ReconciliationEngine(config)
    result = engine.run(gw, led)

    print(
        f"[engine] Done in {result.run_metadata['elapsed_ms']} ms  |  "
        f"matched_exact={len(result.matched_exact)}  "
        f"matched_fuzzy={len(result.matched_fuzzy)}  "
        f"matched_content={len(result.matched_content)}  "
        f"matched_text={len(result.matched_text)}  "
        f"unresolved={len(result.unresolved)}"
    )
    print()

    exceptions = classify(result.unresolved)
    print(f"[classify] {len(exceptions)} exceptions classified.")
    print()

    llm_results      = None
    pair_audit_results = None
    pair_audit_pairs   = None
    if getattr(args, "use_llm", False):
        import random

        from src.llm_agent import LLMConfig, reason_about_exceptions_batch, reason_about_pairs_batch

        llm_cfg = LLMConfig(model=args.llm_model)
        print(
            f"[llm] Consulting Ollama ({llm_cfg.model} @ {llm_cfg.base_url}) on up to "
            f"{args.llm_max_calls} of {len(exceptions)} exceptions ..."
        )
        llm_results = reason_about_exceptions_batch(
            exceptions, cfg=llm_cfg, max_calls=args.llm_max_calls
        )
        n_fallback = sum(1 for r in llm_results if r.fallback_used)
        print(f"[llm] Done. {len(llm_results) - n_fallback} consulted, {n_fallback} fallback.")
        print()

        weak_matches = result.matched_content + result.matched_text
        if weak_matches:
            sample_size = min(args.llm_max_calls, len(weak_matches))
            pair_audit_pairs = random.Random(42).sample(weak_matches, sample_size)
            print(
                f"[llm] Auditing a sample of {sample_size} content/text matches "
                f"(no reference-ID agreement) for precision ..."
            )
            pair_audit_results = reason_about_pairs_batch(
                pair_audit_pairs, cfg=llm_cfg, max_calls=args.llm_max_calls
            )
            n_fallback = sum(1 for r in pair_audit_results if r.fallback_used)
            print(f"[llm] Done. {len(pair_audit_results) - n_fallback} consulted, {n_fallback} fallback.")
            print()

    generate_report(
        result,
        exceptions,
        ground_truth      = ground_truth,
        output_dir        = getattr(args, "data_dir", DATA_DIR),
        llm_results        = llm_results,
        pair_audit_pairs   = pair_audit_pairs,
        pair_audit_results = pair_audit_results,
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "python -m src.main",
        description = "recon-agent — multi-source financial reconciliation CLI",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
examples:
  python -m src.main --generate
  python -m src.main --generate --seed 99
  python -m src.main --run
  python -m src.main --run --amount-tolerance 3.0 --timestamp-tolerance 48
        """,
    )

    # Mutually exclusive mode flags
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--generate", "-g",
        action  = "store_true",
        help    = "Generate fresh synthetic data, then run the pipeline.",
    )
    mode.add_argument(
        "--run", "-r",
        action  = "store_true",
        help    = "Run the pipeline on existing data/ files.",
    )

    # --run data location options (ignored by --generate, which always writes to data/)
    parser.add_argument(
        "--data-dir",
        type    = Path,
        default = DATA_DIR,
        metavar = "DIR",
        help    = "Directory to load {prefix}gateway_records.json etc. from, for --run "
                   "(default: data/). Use with --prefix to load a mapped external "
                   "dataset, e.g. --data-dir data/external/processed --prefix benchrec_.",
    )
    parser.add_argument(
        "--prefix",
        type    = str,
        default = "",
        metavar = "STR",
        help    = "Filename prefix for --run's input files (default: '').",
    )

    # Generator options
    parser.add_argument(
        "--seed",
        type    = int,
        default = 42,
        metavar = "INT",
        help    = "Random seed for synthetic data generation (default: 42).",
    )

    # Engine tolerance options
    parser.add_argument(
        "--amount-tolerance",
        type    = float,
        default = 2.0,
        metavar = "PCT",
        help    = "Fuzzy amount tolerance in %% (default: 2.0).",
    )
    parser.add_argument(
        "--timestamp-tolerance",
        type    = float,
        default = 72.0,
        metavar = "HOURS",
        help    = "Fuzzy timestamp tolerance in hours (default: 72.0 = 3 days).",
    )
    parser.add_argument(
        "--exact-timestamp-tolerance",
        type    = float,
        default = 1.0,
        metavar = "HOURS",
        help    = "Exact-match timestamp tolerance in hours (default: 1.0).",
    )

    # LLM reasoning layer options
    parser.add_argument(
        "--use-llm",
        action  = "store_true",
        help    = "Consult a local Ollama model for a second opinion on each "
                   "classified exception (see src/llm_agent.py). Requires "
                   "Ollama running locally; falls back to rule-based output "
                   "only if unreachable.",
    )
    parser.add_argument(
        "--llm-model",
        type    = str,
        default = "phi3:latest",
        metavar = "TAG",
        help    = "Ollama model tag to use with --use-llm "
                   "(default: phi3:latest, selected by src/llm_eval.py — "
                   "fastest of the evaluated local models at 100%% category "
                   "agreement with the rule engine).",
    )
    parser.add_argument(
        "--llm-max-calls",
        type    = int,
        default = 20,
        metavar = "N",
        help    = "Maximum number of exceptions to send to the LLM with "
                   "--use-llm; remaining exceptions get a fallback result "
                   "(default: 20).",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args   = parser.parse_args(argv)

    if args.generate:
        cmd_generate(args)
    elif args.run:
        cmd_run(args)


if __name__ == "__main__":
    main()
