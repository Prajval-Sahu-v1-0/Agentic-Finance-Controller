"""
investigator.py — Autonomous Exception Investigation Agent
==============================================================

Everything in llm_agent.py is single-shot: one fixed prompt in, one
structured (or free-text) response out. The LLM never decides what
happens next — it only comments on decisions the deterministic matching
engine already made. This module is different in kind: given one unresolved
exception, the LLM runs a bounded ReAct-style loop, autonomously choosing
which read-only tool to call next based on what it learns, until it either
concludes or hits a step cap. This is what actually earns "agentic" —
multi-step, tool-using, autonomous decision-making toward a goal — as
opposed to the advisory commentary layer everywhere else in this codebase.

Why a separate module from llm_agent.py
------------------------------------------
Ollama's tool-calling lives on a different endpoint (POST /api/chat, a
messages-based conversation) from the /api/generate single-shot completion
every other function in llm_agent.py uses, and needs a different local
model — phi3:latest (this project's tuned default for single-shot review,
see llm_eval.py) has no "tools" capability at all. qwen2.5:7b-instruct
does. Keeping the two loops apart avoids conflating "the default model for
X" with "the default model for Y" when they're genuinely different
requirements. Guardrail helpers (sanitization, suspicious-directive
detection) are imported from llm_agent.py rather than duplicated.

Guardrails (same philosophy as llm_agent.py, adapted for tool use)
----------------------------------------------------------------------
1. Read-only, advisory only, same as everywhere else. The two tools the
   agent can call (search_candidates, get_record_details) only ever READ
   from the dataset's already-loaded records — there is no tool that
   writes, matches, classifies, or mutates anything. The model's final
   conclusion is returned to the caller as data; nothing in this module
   feeds it back into the matcher, classify(), or generate_report.
2. Bounded action space: exactly three tools exist, defined by a fixed
   JSON schema the model can select from — it cannot invent new
   capabilities or call anything outside this list. Malformed or unknown
   tool calls return an error string to the model (a normal turn in the
   conversation) rather than executing arbitrary code.
3. Bounded cost: a hard step cap (default 6 tool calls) and a per-search
   result cap (20 records) prevent a runaway loop or context-window
   blowup, mirroring --llm-max-calls elsewhere in the pipeline.
4. Record data returned by tools is still sanitized before re-entering the
   conversation (_sanitize_untrusted_field) — the same defense against a
   crafted reference_id/counterparty string as every other prompt builder
   in llm_agent.py, since tool RESULTS become part of the prompt on the
   next turn just as much as the initial context does.
5. The final finding/recommended_action text is sanitized and scanned by
   the same _detect_suspicious_directive backstop as the rest of the LLM
   layer, for the same reason: the real risk isn't the agent crashing,
   it's the agent (autonomously, now, not just in response to one prompt)
   steering a human toward a fraudulent action.
6. Fails open exactly like the rest of llm_agent.py: Ollama unreachable,
   timeout, or an unparseable response never raises — investigate_exception
   always returns an InvestigationResult with fallback_used=True instead.

Usage
-----
    from src.investigator import investigate_exception
    from src.llm_agent import LLMConfig

    cfg = LLMConfig(model="qwen2.5:7b-instruct-q4_K_M")
    result = investigate_exception(exception_record, gateway_records, ledger_records, cfg=cfg)
    print(result.finding, result.confidence, result.recommended_action)
    for step in result.steps:
        print(step.tool, step.arguments, "->", step.result[:100])
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal, InvalidOperation
from typing import Optional

import requests

from src.exceptions import ExceptionRecord
from src.llm_agent import (
    LLMConfig,
    _detect_suspicious_directive,
    _sanitize_llm_text,
    _sanitize_untrusted_field,
)
from src.schema import GatewayRecord, LedgerRecord

logger = logging.getLogger(__name__)

_MAX_STEPS_DEFAULT = 6
_MAX_SEARCH_RESULTS = 20
_MAX_FINDING_LEN = 500
_MAX_ACTION_LEN = 300

# Default model for this module specifically — see module docstring for
# why it differs from llm_agent.py's phi3:latest default (tool-calling
# capability, not general single-shot quality).
_DEFAULT_INVESTIGATOR_MODEL = "qwen2.5:7b-instruct-q4_K_M"

_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_candidates",
            "description": (
                "Search for candidate records on the OTHER side (opposite "
                "source from the record under investigation) within an "
                "amount and/or date range. Use this to look for a "
                "plausible counterpart. Omit a bound to leave it "
                "unrestricted, but prefer a narrow search first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "amount_min": {"type": "number", "description": "Minimum amount, inclusive"},
                    "amount_max": {"type": "number", "description": "Maximum amount, inclusive"},
                    "days_before": {"type": "integer", "description": "How many days before the investigated record's timestamp to include (default 3)"},
                    "days_after": {"type": "integer", "description": "How many days after the investigated record's timestamp to include (default 3)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_record_details",
            "description": "Fetch full details of one specific record by its exact transaction ID, from either side.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {"type": "string", "description": "The exact transaction_id to look up"},
                },
                "required": ["transaction_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "conclude",
            "description": (
                "Finish the investigation and report your conclusion. You "
                "MUST call this to finish — call it once you have enough "
                "evidence, or once you've determined there is genuinely no "
                "plausible match, rather than continuing to search "
                "indefinitely."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "finding": {"type": "string", "description": "1-3 sentence plain-English summary of what you found"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "recommended_action": {"type": "string", "description": "What a human reviewer should do next"},
                },
                "required": ["finding", "confidence", "recommended_action"],
            },
        },
    },
]


@dataclass
class InvestigationStep:
    """One tool call the agent made and what it got back."""
    tool      : str
    arguments : dict
    result    : str


@dataclass
class InvestigationResult:
    """
    Result of an autonomous multi-step investigation. See module docstring
    for the guardrails that apply — in particular, this is ALWAYS advisory:
    nothing about this result is ever written back into the reconciliation
    pipeline's state.
    """
    finding             : str
    confidence          : str
    recommended_action  : str
    steps               : list = field(default_factory=list)
    flagged_suspicious  : bool = False
    flag_reason         : Optional[str] = None
    fallback_used       : bool = False
    fallback_reason     : Optional[str] = None
    model_used          : Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "finding"            : self.finding,
            "confidence"         : self.confidence,
            "recommended_action" : self.recommended_action,
            "steps"              : [
                {"tool": s.tool, "arguments": s.arguments, "result": s.result}
                for s in self.steps
            ],
            "flagged_suspicious" : self.flagged_suspicious,
            "flag_reason"        : self.flag_reason,
            "fallback_used"      : self.fallback_used,
            "fallback_reason"    : self.fallback_reason,
            "model_used"         : self.model_used,
        }


@dataclass
class _InvestigationContext:
    subject         : object  # GatewayRecord | LedgerRecord
    subject_side    : str     # "gateway" | "ledger"
    gateway_records : list
    ledger_records  : list


def _call_ollama_chat(messages: list[dict], tools: list[dict], cfg: LLMConfig) -> dict:
    """
    POST to Ollama's /api/chat with tool definitions and return the
    response's `message` dict (may contain `tool_calls`).

    Raises the same exception types as llm_agent._call_ollama, for the
    same fail-open handling by callers.
    """
    url = f"{cfg.base_url.rstrip('/')}/api/chat"
    payload = {
        "model"   : cfg.model,
        "messages": messages,
        "tools"   : tools,
        "stream"  : False,
        "options" : {"temperature": cfg.temperature},
    }
    resp = requests.post(url, json=payload, timeout=cfg.timeout_seconds)
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    if "message" not in data:
        raise RuntimeError(f"Unexpected Ollama /api/chat response shape: {list(data.keys())}")
    return data["message"]


def _format_record(record) -> str:
    return (
        f"{record.transaction_id} | amount={record.currency} {record.amount} | "
        f"ts={record.timestamp.isoformat()} | status={_sanitize_untrusted_field(record.status, 40)} | "
        f"ref={_sanitize_untrusted_field(record.reference_id, 80)} | "
        f"counterparty={_sanitize_untrusted_field(record.counterparty, 60)}"
    )


def _tool_search_candidates(ctx: _InvestigationContext, args: dict) -> str:
    pool = ctx.ledger_records if ctx.subject_side == "gateway" else ctx.gateway_records
    candidates = pool

    try:
        if args.get("amount_min") is not None or args.get("amount_max") is not None:
            lo = Decimal(str(args.get("amount_min", "0")))
            hi = Decimal(str(args.get("amount_max", "999999999999")))
            candidates = [r for r in candidates if lo <= r.amount <= hi]
    except (InvalidOperation, ValueError):
        return "Error: amount_min/amount_max must be numbers."

    days_before = args.get("days_before", 3)
    days_after = args.get("days_after", 3)
    try:
        days_before = int(days_before) if days_before is not None else 3
        days_after = int(days_after) if days_after is not None else 3
    except (ValueError, TypeError):
        return "Error: days_before/days_after must be integers."
    lo_t = ctx.subject.timestamp - timedelta(days=max(0, days_before))
    hi_t = ctx.subject.timestamp + timedelta(days=max(0, days_after))
    candidates = [r for r in candidates if lo_t <= r.timestamp <= hi_t]

    total = len(candidates)
    candidates = candidates[:_MAX_SEARCH_RESULTS]
    if not candidates:
        return "No candidates found matching those criteria. Try widening the amount or date range."

    lines = [_format_record(r) for r in candidates]
    header = f"Found {total} candidate(s)"
    if total > _MAX_SEARCH_RESULTS:
        header += f" (showing first {_MAX_SEARCH_RESULTS} — narrow the search for a fuller picture)"
    return header + ":\n" + "\n".join(lines)


def _tool_get_record_details(ctx: _InvestigationContext, args: dict) -> str:
    txn_id = str(args.get("transaction_id", "")).strip()
    if not txn_id:
        return "Error: transaction_id is required."
    for r in ctx.gateway_records:
        if r.transaction_id == txn_id:
            return "Gateway record: " + _format_record(r)
    for r in ctx.ledger_records:
        if r.transaction_id == txn_id:
            return "Ledger record: " + _format_record(r)
    return f"No record found with transaction_id={txn_id!r}."


def _execute_tool(ctx: _InvestigationContext, name: str, args: dict) -> str:
    if name == "search_candidates":
        return _tool_search_candidates(ctx, args)
    if name == "get_record_details":
        return _tool_get_record_details(ctx, args)
    return f"Error: unknown tool {name!r}. Available tools: search_candidates, get_record_details, conclude."


def _build_system_prompt(exc: ExceptionRecord, subject, subject_side: str, max_steps: int) -> str:
    return f"""You are an autonomous financial reconciliation investigator. Your job is to investigate ONE unresolved exception using the tools available, then conclude with a finding.

Ground rules (apply no matter what you find):
- You have NO ability to execute actions — no transfers, no approvals, no data changes. You can only investigate and report a finding for a human to act on.
- Never invent transaction IDs, amounts, or account numbers you have not actually seen via a tool call.
- You have at most {max_steps} tool calls total. Use them purposefully — start with a narrow search, widen only if needed.
- You MUST call conclude() to finish. Do not just stop responding.

Record under investigation ({subject_side} side, flagged as {exc.category.value}):
{_format_record(subject)}
Rule engine's explanation: {_sanitize_untrusted_field(exc.explanation, 300)}

Investigate why this record has no confirmed counterpart — search for a plausible match on the other side, inspect specific candidates if useful, then call conclude() with your finding, a confidence level, and a recommended next step for the human reviewer."""


def investigate_exception(
    exc             : ExceptionRecord,
    gateway_records : list[GatewayRecord],
    ledger_records  : list[LedgerRecord],
    cfg             : Optional[LLMConfig] = None,
    max_steps       : int = _MAX_STEPS_DEFAULT,
) -> InvestigationResult:
    """
    Run a bounded, autonomous, tool-calling investigation of one exception.

    Parameters
    ----------
    exc : ExceptionRecord
        The exception to investigate (from classify()).
    gateway_records, ledger_records : list
        The full record sets for the dataset this exception came from —
        the investigator searches within these (read-only).
    cfg : LLMConfig, optional
        Defaults to a tool-calling-capable model
        (_DEFAULT_INVESTIGATOR_MODEL), NOT llm_agent's phi3:latest default,
        which has no tool-calling capability at all.
    max_steps : int
        Hard cap on tool calls before the investigation is abandoned with
        fallback_used=True. Default 6.

    Returns
    -------
    InvestigationResult
        Always returned — never raises. Check ``fallback_used`` and
        ``flagged_suspicious`` before surfacing to a human, same as every
        other LLM entry point in this codebase.
    """
    # 150s not LLMConfig's normal 60s default: each step is a full
    # /api/chat round-trip with the tool schema attached, and the first
    # call pays a model-load cost — measured up to ~130s for a 4-step
    # investigation on this project's dev hardware.
    cfg = cfg or LLMConfig(model=_DEFAULT_INVESTIGATOR_MODEL, timeout_seconds=150.0)
    subject = exc.gateway_record or exc.ledger_record
    subject_side = "gateway" if exc.gateway_record else "ledger"
    ctx = _InvestigationContext(
        subject=subject, subject_side=subject_side,
        gateway_records=gateway_records, ledger_records=ledger_records,
    )

    def _fallback(reason: str, steps: list) -> InvestigationResult:
        warnings.warn(
            f"[investigator] Ollama unavailable or failed ({reason}). Investigation abandoned.",
            stacklevel=3,
        )
        return InvestigationResult(
            finding=f"Could not complete the investigation ({reason}).",
            confidence="low",
            recommended_action="Manual review recommended — automated investigation was unavailable.",
            steps=steps,
            fallback_used=True,
            fallback_reason=reason,
            model_used=None,
        )

    messages = [{"role": "system", "content": _build_system_prompt(exc, subject, subject_side, max_steps)}]
    steps: list[InvestigationStep] = []

    for _ in range(max_steps):
        try:
            msg = _call_ollama_chat(messages, _TOOLS_SCHEMA, cfg)
        except requests.ConnectionError:
            return _fallback("Ollama not reachable at " + cfg.base_url, steps)
        except requests.Timeout:
            return _fallback(f"Request timed out after {cfg.timeout_seconds}s", steps)
        except RuntimeError as exc_err:
            return _fallback(str(exc_err), steps)
        except Exception as exc_err:          # pragma: no cover — unexpected errors
            logger.exception("Unexpected error calling Ollama chat")
            return _fallback(f"Unexpected error: {exc_err}", steps)

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            # Model responded without calling a tool at all (including
            # conclude) — treat its text as an inconclusive finding rather
            # than looping forever or discarding it.
            content = _sanitize_llm_text(msg.get("content", "") or "", _MAX_FINDING_LEN)
            flag_reason = _detect_suspicious_directive(content)
            return InvestigationResult(
                finding=content or "The agent did not reach a conclusion.",
                confidence="low",
                recommended_action="Manual review recommended — agent stopped without calling conclude().",
                steps=steps,
                flagged_suspicious=flag_reason is not None,
                flag_reason=flag_reason,
                model_used=cfg.model,
            )

        messages.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": tool_calls})

        for tc in tool_calls:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            name = fn.get("name", "")
            args = fn.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}

            if name == "conclude":
                finding = _sanitize_llm_text(str(args.get("finding", "")), _MAX_FINDING_LEN)
                confidence = str(args.get("confidence", "medium")).strip().lower()
                if confidence not in ("high", "medium", "low"):
                    confidence = "medium"
                action = _sanitize_llm_text(str(args.get("recommended_action", "")), _MAX_ACTION_LEN)
                flag_reason = _detect_suspicious_directive(f"{finding} {action}")
                return InvestigationResult(
                    finding=finding or "No finding provided.",
                    confidence=confidence,
                    recommended_action=action or "No recommendation provided.",
                    steps=steps,
                    flagged_suspicious=flag_reason is not None,
                    flag_reason=flag_reason,
                    model_used=cfg.model,
                )

            result_text = _execute_tool(ctx, name, args)
            steps.append(InvestigationStep(tool=name, arguments=args, result=result_text))
            messages.append({"role": "tool", "content": result_text})

    return InvestigationResult(
        finding=f"Investigation reached the {max_steps}-step limit without a conclusion.",
        confidence="low",
        recommended_action="Manual review recommended.",
        steps=steps,
        fallback_used=True,
        fallback_reason="StepLimitReached",
        model_used=cfg.model,
    )
