"""
recon-agent
===========
Multi-source financial reconciliation tool.

Sub-modules
-----------
- schema           : Pydantic data models for transactions and reconciliation records.
- config           : Shared tolerance constants used by the generator and matcher.
- generator        : Synthetic dataset generator for development and testing.
- matcher          : Core matching / reconciliation engine.
- exceptions       : Exception classification and categorisation logic.
- report           : Match-rate computation and exception/LLM reporting.
- diagnose         : Case-level diagnostic vs. the synthetic ground-truth oracle.
- benchrec_map     : Maps the BenchRec external benchmark onto our schema.
- inspect_benchrec : Read-only structural inspection of a downloaded BenchRec dataset.
- llm_agent        : Ollama integration for LLM exception/match reasoning.
- llm_eval         : Evaluates LLM model choice and prompting consistency.
- prompts          : Few-shot examples for LLM reasoning.
- main             : CLI entry point (argparse).
"""
