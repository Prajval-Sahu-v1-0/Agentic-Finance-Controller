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
    GET  /datasets            which mapped datasets are available on disk,
                               static + uploaded
    POST /upload              upload a raw CSV/Excel account sheet, get it
                               auto-mapped onto GatewayRecord/LedgerRecord
                               and saved as (part of) a named dataset — see
                               src/ingest.py for the "does it need to be
                               pre-formatted" answer (no)
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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.exceptions import classify
from src.generator import load_gateway_records, load_ground_truth, load_ledger_records
from src.ingest import IngestError, map_to_records, read_table, save_records
from src.llm_agent import LLMConfig, reason_about_exceptions_batch, reason_about_pairs_batch
from src.matcher import MatchConfig, ReconciliationEngine
from src.report import generate_report

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "external" / "processed"
UPLOADED_DIR = DATA_DIR / "uploaded"
STATIC_DIR = PROJECT_ROOT / "static"

app = FastAPI(
    title="recon-agent",
    description="AI Finance Controller — multi-source reconciliation demo API",
)

# dataset name -> (directory holding {prefix}gateway_records.json etc., prefix)
# for the built-in, pre-mapped datasets. Anything else is looked up in
# UPLOADED_DIR instead — see _dataset_paths.
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
    """
    Resolve a dataset name to its (gateway, ledger, ground_truth) file
    paths. Built-in names use _DATASET_MAP; anything else is treated as a
    user-uploaded dataset id and looked up in UPLOADED_DIR — POST /upload
    creates these, one call per side (gateway/ledger), under the same
    `dataset` name. Existence is checked by callers (e.g. /reconcile
    already 404s cleanly if the gateway file isn't there yet), not here,
    since an uploaded dataset legitimately might have only one side so far.
    """
    if dataset in _DATASET_MAP:
        data_dir, prefix = _DATASET_MAP[dataset]
    else:
        data_dir, prefix = UPLOADED_DIR, f"{dataset}_"
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
    """Which datasets are actually present on disk right now — the
    built-in ones plus anything uploaded via POST /upload."""
    out = []
    for name in _DATASET_MAP:
        gw_path, led_path, _ = _dataset_paths(name)
        out.append({"name": name, "available": gw_path.exists() and led_path.exists()})

    if UPLOADED_DIR.is_dir():
        uploaded_names = sorted({
            p.name[: -len("_gateway_records.json")]
            for p in UPLOADED_DIR.glob("*_gateway_records.json")
        })
        for name in uploaded_names:
            gw_path, led_path, _ = _dataset_paths(name)
            out.append({
                "name": name,
                "available": gw_path.exists() and led_path.exists(),
                "uploaded": True,
                "has_gateway": gw_path.exists(),
                "has_ledger": led_path.exists(),
            })

    return {"datasets": out}


@app.post("/upload")
async def upload(
    file    : UploadFile = File(...),
    role    : str        = Form(...),
    dataset : str        = Form("uploaded"),
) -> dict:
    """
    Upload a raw CSV or Excel account sheet — no pre-formatting required.
    Columns are auto-detected by name (see src/ingest.py); the result is
    saved as the `role` side ("gateway" or "ledger") of `dataset`. Upload
    both sides under the same `dataset` name, then POST /reconcile with
    that dataset name once both are present (check via GET /datasets).

    Returns a mapping summary — which source column was used for each
    target field, how many rows mapped successfully, and why any row was
    skipped — so the auto-detection is never a silent guess.
    """
    if role not in ("gateway", "ledger"):
        raise HTTPException(400, f"role must be 'gateway' or 'ledger', got {role!r}")
    if not dataset or not dataset.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(400, "dataset must be a non-empty alphanumeric name (- and _ allowed)")

    content = await file.read()
    try:
        rows = read_table(content, filename=file.filename or "upload.csv")
        result = map_to_records(rows, role=role)
    except IngestError as exc:
        raise HTTPException(422, str(exc))

    if not result.records:
        raise HTTPException(
            422,
            f"0 of {result.total_rows} rows mapped successfully. "
            f"Detected columns: {result.column_map}. First few skip reasons: {result.skipped[:5]}",
        )

    target_path = UPLOADED_DIR / f"{dataset}_{role}_records.json"
    save_records(result.records, target_path)

    return {
        "dataset"       : dataset,
        "role"          : role,
        "saved_to"      : str(target_path),
        "total_rows"    : result.total_rows,
        "mapped_count"  : result.mapped_count,
        "skipped_count" : len(result.skipped),
        "skipped_sample": [{"row": i, "reason": reason} for i, reason in result.skipped[:20]],
        "column_map"    : result.column_map,
    }


@app.post("/reconcile")
def reconcile(req: ReconcileRequest) -> dict:
    """
    Run the full pipeline (match -> classify -> report) against a mapped
    dataset, optionally with LLM exception review and pair auditing. Same
    engine, same guardrails, same output shape as `python -m src.main --run`.
    """
    gw_path, led_path, gt_path = _dataset_paths(req.dataset)
    if not gw_path.exists() or not led_path.exists():
        missing_side = "gateway" if not gw_path.exists() else "ledger"
        raise HTTPException(
            404,
            f"Dataset {req.dataset!r} is missing its {missing_side} side "
            f"(expected at {gw_path if missing_side == 'gateway' else led_path}). "
            "Generate a built-in dataset (python -m src.main --generate), map "
            "BenchRec (python -m src.benchrec_map --split train|eval), or "
            "POST /upload for a custom dataset (upload both gateway and ledger sides).",
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

    data_dir = gw_path.parent
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
    if dataset not in _last_reports:
        raise HTTPException(404, f"No report yet for {dataset!r} this session — POST /reconcile first.")
    return _last_reports[dataset]


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
