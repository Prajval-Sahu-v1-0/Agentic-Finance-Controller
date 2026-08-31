"""
api.py — Demo REST API for the reconciliation pipeline
==========================================================

Thin FastAPI wrapper around the existing pipeline, built for the Razorpay
Buildathon submission demo. Deliberately does not reimplement any pipeline
logic — every endpoint calls the same functions main.py's CLI does
(ReconciliationEngine, classify, generate_report, reason_about_*). This is
a demo/presentation layer, not a new source of truth.

Run
---
    uvicorn src.api:app --reload
    # then open http://127.0.0.1:8000/ for the dashboard

Endpoints
---------
    GET  /health              liveness check
    GET  /datasets            which mapped datasets are available on disk
    POST /reconcile           run the pipeline against a chosen dataset,
                               returns the full report dict (same shape as
                               report.json)
    GET  /report?dataset=...  the last report generated THIS SESSION for
                               that dataset (in-memory only — a fresh
                               server has nothing until /reconcile runs)
    GET  /                    the dashboard (static/index.html)

State
-----
Reports live in an in-memory dict keyed by dataset name, not a shared
report.json file — synthetic/benchrec_train/benchrec_eval all write to
different report.json paths already except the two BenchRec splits, which
share data/external/processed/ and would otherwise clobber each other's
report.json. Keeping state in the running process sidesteps that and is
the right lifetime for a live demo anyway: restart the server, state resets.
"""

from __future__ import annotations

import random
from decimal import Decimal
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.exceptions import classify
from src.generator import load_gateway_records, load_ground_truth, load_ledger_records
from src.llm_agent import LLMConfig, reason_about_exceptions_batch, reason_about_pairs_batch
from src.matcher import MatchConfig, ReconciliationEngine
from src.report import generate_report

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "external" / "processed"
STATIC_DIR = PROJECT_ROOT / "static"

app = FastAPI(
    title="recon-agent",
    description="AI Finance Controller — multi-source reconciliation demo API",
)

# dataset name -> (directory holding {prefix}gateway_records.json etc., prefix)
_DATASET_MAP: dict[str, tuple[Path, str]] = {
    "synthetic"      : (DATA_DIR, ""),
    "benchrec_train" : (PROCESSED_DIR, "benchrec_"),
    "benchrec_eval"  : (PROCESSED_DIR, "benchrec_eval_"),
}

# In-memory report cache — see module docstring's "State" section.
_last_reports: dict[str, dict] = {}


class ReconcileRequest(BaseModel):
    dataset                  : str   = "synthetic"
    use_llm                  : bool  = False
    llm_model                : str   = "phi3:latest"
    llm_max_calls            : int   = 10
    amount_tolerance_pct     : float = 2.0
    timestamp_tolerance_hours: float = 72.0


def _dataset_paths(dataset: str) -> tuple[Path, Path, Path]:
    if dataset not in _DATASET_MAP:
        raise HTTPException(400, f"Unknown dataset {dataset!r}. Valid: {list(_DATASET_MAP)}")
    data_dir, prefix = _DATASET_MAP[dataset]
    return (
        data_dir / f"{prefix}gateway_records.json",
        data_dir / f"{prefix}ledger_records.json",
        data_dir / f"{prefix}ground_truth.json",
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/datasets")
def list_datasets() -> dict:
    """Which mapped datasets are actually present on disk right now."""
    out = []
    for name in _DATASET_MAP:
        gw_path, _, _ = _dataset_paths(name)
        out.append({"name": name, "available": gw_path.exists()})
    return {"datasets": out}


@app.post("/reconcile")
def reconcile(req: ReconcileRequest) -> dict:
    """
    Run the full pipeline (match -> classify -> report) against a mapped
    dataset, optionally with LLM exception review and pair auditing. Same
    engine, same guardrails, same output shape as `python -m src.main --run`.
    """
    gw_path, led_path, gt_path = _dataset_paths(req.dataset)
    if not gw_path.exists():
        raise HTTPException(
            404,
            f"Dataset {req.dataset!r} not found at {gw_path}. "
            "Generate it first (python -m src.main --generate) or map it "
            "(python -m src.benchrec_map --split train|eval).",
        )

    gw  = load_gateway_records(gw_path)
    led = load_ledger_records(led_path)
    gt  = load_ground_truth(gt_path) if gt_path.exists() else None

    config = MatchConfig(
        amount_tolerance_pct      = Decimal(str(req.amount_tolerance_pct)),
        timestamp_tolerance_hours = req.timestamp_tolerance_hours,
    )
    result = ReconciliationEngine(config).run(gw, led)
    exceptions = classify(result.unresolved)

    llm_results: Optional[list] = None
    pair_audit_pairs: Optional[list] = None
    pair_audit_results: Optional[list] = None
    if req.use_llm:
        cfg = LLMConfig(model=req.llm_model)
        llm_results = reason_about_exceptions_batch(exceptions, cfg=cfg, max_calls=req.llm_max_calls)
        weak_matches = result.matched_content + result.matched_text
        if weak_matches:
            sample_size = min(req.llm_max_calls, len(weak_matches))
            pair_audit_pairs = random.Random(42).sample(weak_matches, sample_size)
            pair_audit_results = reason_about_pairs_batch(pair_audit_pairs, cfg=cfg, max_calls=req.llm_max_calls)

    data_dir, _ = _DATASET_MAP[req.dataset]
    report = generate_report(
        result, exceptions,
        ground_truth       = gt,
        output_dir         = data_dir,
        llm_results         = llm_results,
        pair_audit_pairs    = pair_audit_pairs,
        pair_audit_results  = pair_audit_results,
    )
    _last_reports[req.dataset] = report
    return report


@app.get("/report")
def get_report(dataset: str = "synthetic") -> dict:
    """The last report generated THIS SESSION for `dataset` — POST /reconcile first."""
    if dataset not in _DATASET_MAP:
        raise HTTPException(400, f"Unknown dataset {dataset!r}. Valid: {list(_DATASET_MAP)}")
    if dataset not in _last_reports:
        raise HTTPException(404, f"No report yet for {dataset!r} this session — POST /reconcile first.")
    return _last_reports[dataset]


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
