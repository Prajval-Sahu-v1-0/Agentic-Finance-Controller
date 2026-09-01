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
    POST /ask                 free-text Q&A ("prompt the agent") grounded
                               in the last report generated for a dataset —
                               read-only, advisory, same guardrails as the
                               rest of the LLM layer; see
                               src/llm_agent.py's ask_agent docstring
    POST /investigate         autonomous multi-step tool-calling
                               investigation of ONE exception — the LLM
                               decides what to check next itself; see
                               src/investigator.py's module docstring
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
from src.investigator import investigate_exception
from src.llm_agent import LLMConfig, ask_agent, reason_about_exceptions_batch, reason_about_pairs_batch
from src.matcher import MatchConfig, ReconciliationEngine
from src.paths import DATA_DIR, PROCESSED_DIR, STATIC_DIR, UPLOADED_DIR
from src.report import generate_report

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
# Same lifetime/rationale, so POST /investigate can look up a specific
# exception by ID and search within the same record set without re-running
# the whole pipeline just to investigate one exception.
_last_exceptions: dict[str, list] = {}
_last_records: dict[str, tuple[list, list]] = {}


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
    _last_exceptions[req.dataset] = exceptions
    _last_records[req.dataset] = (gw, led)
    return report


@app.get("/report")
def get_report(dataset: str = "synthetic") -> dict:
    """The last report generated THIS SESSION for `dataset` — POST /reconcile first."""
    if dataset not in _last_reports:
        raise HTTPException(404, f"No report yet for {dataset!r} this session — POST /reconcile first.")
    return _last_reports[dataset]


class AskRequest(BaseModel):
    dataset  : str = "synthetic"
    prompt   : str
    llm_model: str = "phi3:latest"


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    """
    Free-text Q&A ("prompt the agent") grounded in the last report
    generated for `req.dataset` this session — POST /reconcile first for a
    data-grounded answer; without one, only general questions get answered.

    Read-only and advisory only: this never re-runs reconciliation,
    classification, or anything else that could change pipeline state — it
    can only read an already-generated report and talk about it. See
    src/llm_agent.py's ask_agent docstring (and its module docstring's
    Guardrails section) for the full detail on how a user's free text is
    handled safely: length-capped and sanitized before entering the
    prompt, the model is told it cannot execute actions or invent report
    data, and the response is scanned for directive-like financial
    language before being returned — check `flagged_suspicious` before
    surfacing an answer a user might act on.
    """
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(400, "prompt must not be empty")

    report = _last_reports.get(req.dataset)
    cfg = LLMConfig(model=req.llm_model)
    result = ask_agent(req.prompt, report=report, cfg=cfg)
    return result.to_dict()


class InvestigateRequest(BaseModel):
    dataset       : str   = "synthetic"
    exception_id  : str
    llm_model     : str   = "qwen2.5:7b-instruct-q4_K_M"
    max_steps     : int   = 6
    # Higher than LLMConfig's normal 60s default: each step is a full
    # /api/chat round-trip with the tool schema attached, and the first
    # call in particular pays a model-load cost — measured at up to ~130s
    # for a 4-step investigation on this project's dev hardware. A per-call
    # cap that's too tight fails the FIRST call before the loop even gets
    # going, which is a worse failure mode than just waiting.
    timeout_seconds: float = 150.0


@app.post("/investigate")
def investigate(req: InvestigateRequest) -> dict:
    """
    Autonomous multi-step investigation of one exception ("agent, go figure
    out why this didn't match"). Unlike /ask, the LLM here decides for
    itself what to check next — it calls read-only tools (search for a
    candidate counterpart, inspect a specific record) across up to
    `max_steps` turns, then concludes with a finding, a confidence level,
    and a recommended next step. See src/investigator.py's module
    docstring for why this needs a tool-calling-capable model (the
    project's tuned phi3:latest default has no tool-calling capability at
    all) and the guardrails that apply — same philosophy as /ask
    (read-only, advisory, suspicious-directive scanning) adapted for a
    multi-step tool-using loop instead of one prompt.

    POST /reconcile first — this looks up `exception_id` in the exceptions
    classified by the last run for `dataset`.
    """
    if req.dataset not in _last_exceptions:
        raise HTTPException(404, f"No exceptions available for {req.dataset!r} this session — POST /reconcile first.")

    exceptions = _last_exceptions[req.dataset]
    exc = next((e for e in exceptions if e.exception_id == req.exception_id), None)
    if exc is None:
        raise HTTPException(404, f"No exception {req.exception_id!r} found in the last run for {req.dataset!r}.")

    gw, led = _last_records[req.dataset]
    cfg = LLMConfig(model=req.llm_model, timeout_seconds=req.timeout_seconds)
    result = investigate_exception(exc, gw, led, cfg=cfg, max_steps=req.max_steps)
    return result.to_dict()


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
