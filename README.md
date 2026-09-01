# recon-agent

A **multi-source financial reconciliation agent**: matches payment-gateway settlement
records against an internal ledger, groups batch/split payouts, falls back to
content-based matching when reference IDs don't correlate, classifies whatever's left
into actionable exception categories, and (optionally) asks a local LLM for a second
opinion on each exception. Validated against both a synthetic oracle dataset and
BenchRec, a real-world external cash-reconciliation benchmark (ICAIF 2023).

---

## Project Structure

```
recon-agent/
  data/
    gateway_records.json, ledger_records.json, ground_truth.json, report.json
                          # synthetic dataset (generated) + last run's report
    external/
      raw/                # downloaded BenchRec dataset (untouched)
      processed/          # BenchRec mapped onto our schema (see src/benchrec_map.py)
  src/
    schema.py             # Pydantic models: GatewayRecord, LedgerRecord
    config.py             # shared tolerance constants
    generator.py          # synthetic data generator + ground truth oracle
    matcher.py             # ReconciliationEngine: exact -> fuzzy -> content -> grouped
    exceptions.py           # classifies unresolved records into 5 categories
    report.py                # match-rate / monetary / exception / LLM / ground-truth reporting
    diagnose.py               # case-level diagnostic vs. ground truth
    llm_agent.py               # Ollama integration for exception reasoning
    prompts.py                  # few-shot examples for LLM reasoning
    benchrec_map.py               # maps BenchRec's real-world CSVs onto our schema
    inspect_benchrec.py            # read-only structural inspection of raw BenchRec files
    llm_eval.py                     # evaluates LLM model choice + prompting consistency
    ingest.py                       # generic CSV/Excel account-sheet auto-mapping (no pre-formatting needed)
    api.py                           # FastAPI demo layer wrapping the pipeline
    main.py                        # CLI entry point
  static/
    index.html             # dashboard served by src/api.py — no build step, no CDN deps
  tests/
    test_matcher.py               # matcher regression tests
    test_llm_agent.py              # parser + guardrail regression tests
  requirements.txt
```

## Quickstart

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate a synthetic dataset and run the full pipeline
python -m src.main --generate

# 4. Re-run against existing data/ files
python -m src.main --run

# 5. Add a local LLM second opinion on exceptions (requires Ollama running)
python -m src.main --run --use-llm

# 6. Re-evaluate which local model to use, and whether prompting is
#    consistent enough to skip fine-tuning (see "LLM tuning" below)
python -m src.llm_eval

# 7. Launch the demo API + dashboard (http://127.0.0.1:8000/)
uvicorn src.api:app --reload
```

### Validating against BenchRec (external, real-world dataset)

BenchRec ships two splits, and they behave differently — see results below.

```bash
# Map data/external/raw/BenchRec_cash_v1.0_train.csv onto our schema
python -m src.benchrec_map --split train
python -m src.main --run --data-dir data/external/processed --prefix benchrec_

# Map the held-out test split instead (eval.csv + solution.csv)
python -m src.benchrec_map --split eval
python -m src.main --run --data-dir data/external/processed --prefix benchrec_eval_
```

| | train split (47k 1:1 pairs) | eval/test split (15.7k 1:1 pairs)* |
|---|---|---|
| Match rate | 45.5% | 31.7% |
| Match precision | ~98% | 76.0% |
| Match recall | 46.5% | 45.5% |
| Exception recall | 98.9% | 99.1% |

*The eval split's own ground truth (`solution.csv`) is messier than train's: ~44% of its target labels are themselves ambiguous — many unrelated ledger rows share byte-identical description text, so `targetAllocation` doesn't resolve to a unique row. Those cases are excluded from scoring rather than guessed at (see `benchrec_map.py`'s "eval.csv + solution.csv mapping" section for the full reasoning) — the 15.7k pairs above are the ones we could score with confidence. That same widespread near-duplicate text is almost certainly why match precision drops on this split: it's a genuinely harder, noisier dataset than train.csv, not a regression in the matcher.

## Matching pipeline

1. **Phase 1 — Exact match**: normalised reference-ID equality + identical amount + timestamp within 1h.
2. **Phase 2 — Fuzzy match**: normalised reference-ID equality + amount/timestamp within tolerance, weighted confidence score.
3. **Phase 2.5 — Content fallback**: for records whose reference IDs never share a digit sequence at all (the common real-world case), match on amount + timestamp proximity alone, accepted only when the pairing is mutually unique on both sides.
4. **Phase 2.75 — Text disambiguation**: for the ambiguous case Phase 2.5 had to skip (multiple amount/timestamp candidates), re-ranks by text similarity on `reference_id`/`counterparty`, accepted only with a clear score margin, checked mutually on both sides.
5. **Phase 3 — Grouped match**: bounded search for 1-to-many / many-to-one batch and split settlements, gated by explicit BATCH/SPLIT/PAYOUT reference signals. *(Phase 3b, an opt-in marker-free fallback for real-world data with no such convention, exists but ships **disabled by default** — validated against BenchRec, it traded a real precision hit (match precision 98%→94.5%) for ~1 true positive against ~809 false positives, an apparent birthday-paradox risk at real-world scale. See `MatchConfig.enable_unmarked_grouping`'s docstring for the full numbers before turning it on.)*
6. **Exception classification**: everything still unresolved is sorted into `missing_in_ledger`, `missing_in_gateway`, `amount_mismatch`, `stale_timing`, or `duplicate`.
7. **LLM reasoning (optional, `--use-llm`)**: each classified exception gets a local Ollama second opinion, and a sample of Phase 2.5/2.75 matches (the ones made without any reference-ID agreement) gets audited for precision — neither ever overrides the rule engine's output, and both fail open to rule-based output if Ollama is unreachable.

On BenchRec (ICAIF 2023, real-world, 47k 1:1 pairs): Phase 1/2 alone score 0% match rate, since real bank/gateway reference text shares no digit sequence across sources. Phase 2.5 + 2.75 recover a 45.5% match rate at ~98% precision — see `src/benchrec_map.py` and `src/matcher.py` docstrings for the full validation story and known limitations:
- Many-to-many groups (3,458 of them) are excluded from the mapped dataset entirely — `report.py`'s ground-truth scorer only supports 1:1, one-to-many, and many-to-one shapes.
- **Phase 3 (grouped matching) never fires on BenchRec at all** — it gates entry on a literal `BATCH`/`SPLIT`/`PAYOUT` reference-ID prefix, a synthetic-generator convention real bank narration text never has. Confirmed empirically: 0 groups detected despite ~1,500 genuine one-to-many/many-to-one pairs in BenchRec's oracle. Fix plan documented in `_group_candidate_pools`'s docstring in `matcher.py` — deliberately not yet implemented, since a rushed sum-matching generalization is exactly what caused a real precision incident earlier in this project (see `EXCEPTION_SAFETY_MARGIN` in `config.py`).

## LLM tuning

The plan was always: prompt a local Ollama model with few-shot examples first, and treat fine-tuning as a stretch goal *only if prompting proves inconsistent*. `src/llm_eval.py` actually measures that, rather than assuming it — it compares candidate models on category agreement with the (independently 100%-accurate) rule engine, JSON parse reliability, latency, and determinism (same input sent multiple times).

Result as of the last run: **`phi3:latest`** beat qwen2.5:7b-instruct, qwen2.5:14b, and llama3:latest — all four hit 100% category agreement, but phi3 was fastest (~7s/call vs 13-87s) with zero parse or timeout fallbacks and 100% stable output across repeats. qwen2.5:14b was notably worse: 6x slower with a timeout-induced fallback and no accuracy gain. **Fine-tuning is not warranted** — it stays a stretch goal. Re-run `python -m src.llm_eval` after changing the few-shot examples in `prompts.py` or the local model roster; the choice isn't permanent.

Also cross-checked against an external HF model (`mombalam/clearledgr-llama-financial-ai`, a LoRA fine-tune of Llama-3.1-8B) on both the synthetic and BenchRec datasets: 0% valid structured output across 18 calls, 25-50x slower. See `clearledgr-eval/` (a separate venv — not part of this repo) for the harness.

## Guardrails

`--use-llm` sends real transaction data to a model and surfaces its text to a human reviewer — two separate trust boundaries worth defending deliberately, not just "the model is usually right." See `src/llm_agent.py`'s module docstring for the full detail; summary:

1. **Advisory only, never authoritative** — LLM output never mutates the rule engine's category, match status, or monetary figures. Proven by `test_reason_about_exception_never_mutates_input` in `tests/test_llm_agent.py`, not just asserted in a comment.
2. **Prompt-injection mitigation** — `reference_id`/`counterparty` come from external gateway/ledger data (attacker-influenceable in a real deployment) and are truncated + control-character-stripped before entering a prompt, with an explicit "this is data, not instructions" delimiter around them.
3. **Output isn't trusted as-is** — `category`/`confidence` are checked against fixed allowlists; `agrees_with_rules` goes through `_coerce_bool` rather than Python's `bool(x)`, which silently returns `True` for the non-empty string `"false"` — a real bug this replaced.
4. **Explanation text is sanitized before display** — control/ANSI-escape characters stripped, length-capped, and Rich markup (`[...]`) escaped at render time in `report.py`, since Rich interprets bracketed text as style markup by default and a manipulated explanation could otherwise corrupt or spoof the console report.
5. **Suspicious-directive detection** — the real risk in a finance tool isn't the model crashing, it's a manipulated or hallucinating model telling a reviewer to move money. `explanation` is scanned for account/routing/IBAN-shaped numbers and wire-transfer-style urgency language; a hit sets `flagged_suspicious=True` and renders as a loud, dedicated warning panel — never silently folded into a normal-looking row.
6. **Cost/DoS bound** — `--llm-max-calls` caps how many records ever reach the model per run; the rest get a fallback result instead of another network call. Ollama being unreachable, timing out, or unparseable never blocks or crashes the pipeline — it always fails open to rule-based output (`fallback_used=True`).

## Demo API + dashboard

```bash
uvicorn src.api:app --reload
# open http://127.0.0.1:8000/
```

`src/api.py` is a thin FastAPI wrapper — it doesn't reimplement anything, every endpoint calls the same `ReconciliationEngine` / `classify` / `generate_report` / `reason_about_*` functions the CLI does. Pick a dataset (synthetic, BenchRec train, or BenchRec eval), optionally turn on LLM review, click **Run Reconciliation**, and `static/index.html` renders the match summary, monetary summary, exception breakdown, LLM review (including a loud dedicated panel if anything gets `flagged_suspicious`), pair audit, and ground-truth accuracy — no build step, no CDN dependencies, so it keeps working in a room with bad wifi.

| Endpoint | What it does |
|---|---|
| `GET /health` | liveness check |
| `GET /datasets` | which datasets exist on disk — built-in and uploaded |
| `POST /upload` | upload a raw CSV/Excel account sheet (`file`, `role=gateway\|ledger`, `dataset=name`) — auto-mapped, no pre-formatting needed |
| `POST /reconcile` | run the pipeline (`{dataset, use_llm, llm_max_calls, ...}`) and return the full report |
| `GET /report?dataset=...` | the last report generated this session for that dataset |

Report state is in-memory per running server process (not a shared file) — see `api.py`'s module docstring for why (the two BenchRec splits would otherwise clobber each other's `report.json`).

### Uploading your own account sheets

No pre-formatting required — `src/ingest.py` auto-detects columns by name (transaction ID, amount, timestamp, status, counterparty, currency) against a hint vocabulary, the same one `inspect_benchrec.py` already used for read-only inspection, and maps whatever it finds onto `GatewayRecord`/`LedgerRecord`. It's a heuristic, not a guarantee, so every response is transparent about what it did: exactly which source column was used for each field, and exactly why any row was skipped (empty ID, unparseable amount, unparseable date, validation error) — never a silent guess.

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@settlements.csv" -F "role=gateway" -F "dataset=mycompany"
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@ledger.xlsx" -F "role=ledger" -F "dataset=mycompany"
curl -X POST http://127.0.0.1:8000/reconcile \
  -H "Content-Type: application/json" -d '{"dataset": "mycompany"}'
```

Supports `.csv`, `.tsv`, `.xlsx`, `.xls`; tolerant of common export quirks (currency symbols, thousands separators, several date formats, signed/parenthesized negative amounts). If a required field (transaction ID, amount, timestamp) can't be auto-detected at all, the upload is rejected with the column names it saw, so you know what to fix — there's no silent fallback for a genuinely missing field.

## Tech Stack

| Layer        | Technology          |
|-------------|---------------------|
| Validation   | Pydantic v2         |
| CLI          | argparse + Rich     |
| LLM reasoning | Ollama (local)      |
| Testing      | Pytest              |
| Demo API     | FastAPI + Uvicorn   |
| Dashboard    | Static HTML/CSS/vanilla JS (no build step, no CDN) |

## Status

- [x] Synthetic data generation with ground-truth oracle
- [x] Core matching engine (exact + fuzzy + content fallback + grouped)
- [x] Exception classification (100% recall against synthetic oracle)
- [x] Match-rate / monetary / exception reporting (console + JSON)
- [x] External validation against BenchRec (ICAIF 2023 real-world benchmark)
- [x] LLM reasoning layer (Ollama, few-shot prompted)
- [x] LLM tuning: measured model choice + prompting consistency (`src/llm_eval.py`) — fine-tuning not warranted
- [x] External model comparison: `mombalam/clearledgr-llama-financial-ai` (HF LoRA over Llama-3.1-8B) evaluated against phi3:latest on both datasets — 0% valid output rate across 18 calls, 25-50x slower; phi3 stays selected. See `clearledgr-eval/` (separate venv, not part of this repo).
- [x] BenchRec eval/test-split validation (`--split eval`) — a genuinely held-out split with its own, messier ground truth; phi3 holds at 100%/100% there too
- [x] LLM guardrails: advisory-only enforcement (tested), prompt-injection mitigation, output allowlisting + bool-coercion fix, explanation sanitization + Rich-markup escaping, suspicious-directive detection, cost/DoS bound — see "Guardrails" above
- [x] FastAPI REST endpoints + dashboard UI (`src/api.py`, `static/index.html`) — thin wrapper around the existing pipeline, no reimplementation, no CDN dependencies
- [x] Generic account-sheet upload (`POST /upload`, `src/ingest.py`) — no pre-formatting required; auto-maps arbitrary CSV/Excel column names onto GatewayRecord/LedgerRecord, transparent about every detected column and every skipped row
- [x] Attempted fix for Phase 3 grouped matching's real-world gap (Phase 3b) — built, tested on both datasets per the fix plan's own requirement, and found to trade precision for negligible recall at BenchRec's scale (98%→94.5%, ~1 TP vs ~809 FP). Ships opt-in / disabled by default (`MatchConfig.enable_unmarked_grouping=False`) rather than silently shipped as a win. Remains a genuinely open gap.
- [ ] Fine-tuning (stretch goal, currently not needed — see LLM tuning above)
