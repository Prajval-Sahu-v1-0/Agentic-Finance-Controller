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
    main.py                        # CLI entry point
  tests/
    test_matcher.py               # matcher regression tests
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
```

### Validating against BenchRec (external, real-world dataset)

```bash
# Map data/external/raw/BenchRec_cash_v1.0_train.csv onto our schema
python -m src.benchrec_map

# Run the reconciliation pipeline against the mapped BenchRec data
python -m src.main --run --data-dir data/external/processed --prefix benchrec_
```

## Matching pipeline

1. **Phase 1 — Exact match**: normalised reference-ID equality + identical amount + timestamp within 1h.
2. **Phase 2 — Fuzzy match**: normalised reference-ID equality + amount/timestamp within tolerance, weighted confidence score.
3. **Phase 2.5 — Content fallback**: for records whose reference IDs never share a digit sequence at all (the common real-world case), match on amount + timestamp proximity alone, accepted only when the pairing is mutually unique on both sides.
4. **Phase 2.75 — Text disambiguation**: for the ambiguous case Phase 2.5 had to skip (multiple amount/timestamp candidates), re-ranks by text similarity on `reference_id`/`counterparty`, accepted only with a clear score margin, checked mutually on both sides.
5. **Phase 3 — Grouped match**: bounded search for 1-to-many / many-to-one batch and split settlements, gated by explicit BATCH/SPLIT/PAYOUT reference signals.
6. **Exception classification**: everything still unresolved is sorted into `missing_in_ledger`, `missing_in_gateway`, `amount_mismatch`, `stale_timing`, or `duplicate`.
7. **LLM reasoning (optional, `--use-llm`)**: each classified exception gets a local Ollama second opinion, and a sample of Phase 2.5/2.75 matches (the ones made without any reference-ID agreement) gets audited for precision — neither ever overrides the rule engine's output, and both fail open to rule-based output if Ollama is unreachable.

On BenchRec (ICAIF 2023, real-world, 47k 1:1 pairs): Phase 1/2 alone score 0% match rate, since real bank/gateway reference text shares no digit sequence across sources. Phase 2.5 + 2.75 recover a 45.5% match rate at ~98% precision — see `src/benchrec_map.py` and `src/matcher.py` docstrings for the full validation story and known limitations (many-to-many groups excluded from scoring; recall is capped by how often amount+date+text alone can uniquely identify a transaction).

## LLM tuning

The plan was always: prompt a local Ollama model with few-shot examples first, and treat fine-tuning as a stretch goal *only if prompting proves inconsistent*. `src/llm_eval.py` actually measures that, rather than assuming it — it compares candidate models on category agreement with the (independently 100%-accurate) rule engine, JSON parse reliability, latency, and determinism (same input sent multiple times).

Result as of the last run: **`phi3:latest`** beat qwen2.5:7b-instruct, qwen2.5:14b, and llama3:latest — all four hit 100% category agreement, but phi3 was fastest (~7s/call vs 13-87s) with zero parse or timeout fallbacks and 100% stable output across repeats. qwen2.5:14b was notably worse: 6x slower with a timeout-induced fallback and no accuracy gain. **Fine-tuning is not warranted** — it stays a stretch goal. Re-run `python -m src.llm_eval` after changing the few-shot examples in `prompts.py` or the local model roster; the choice isn't permanent.

## Tech Stack

| Layer        | Technology          |
|-------------|---------------------|
| Validation   | Pydantic v2         |
| CLI          | argparse + Rich     |
| LLM reasoning | Ollama (local)      |
| Testing      | Pytest              |
| API / DB (future) | FastAPI + SQLite (not yet built) |

## Status

- [x] Synthetic data generation with ground-truth oracle
- [x] Core matching engine (exact + fuzzy + content fallback + grouped)
- [x] Exception classification (100% recall against synthetic oracle)
- [x] Match-rate / monetary / exception reporting (console + JSON)
- [x] External validation against BenchRec (ICAIF 2023 real-world benchmark)
- [x] LLM reasoning layer (Ollama, few-shot prompted)
- [x] LLM tuning: measured model choice + prompting consistency (`src/llm_eval.py`) — fine-tuning not warranted
- [ ] FastAPI REST endpoints
- [ ] Dashboard UI
- [ ] Fine-tuning (stretch goal, currently not needed — see LLM tuning above)
