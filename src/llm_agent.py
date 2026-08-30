"""
llm_agent.py — LLM-Powered Exception Reasoning
================================================

Wraps a locally running Ollama instance to provide a second opinion on each
exception flagged by the rule-based classifier in src/exceptions.py.

Design goals
------------
1. Lightweight — uses ``requests`` directly against Ollama's REST API
   (POST /api/generate).  No heavy frameworks.
2. Resilient — if Ollama is unreachable or returns a malformed response the
   function returns a ``LLMReasoningResult`` with ``fallback_used=True`` and
   the original rule-based output unchanged.  It never raises or crashes the
   pipeline.
3. Structured output — the model is asked to respond in JSON; the parser
   tries json.loads() first, then regex extraction as a fallback.
4. Auditable — the raw model response is stored on every result so reviewers
   can trace exactly what the LLM said.

Few-shot examples
-----------------
Curated examples live in ``src/prompts.py`` as ``FewShotExample`` dataclass
instances.  Edit, reorder, or replace them there — the prompt builder here
calls ``build_few_shot_block()`` at render time so changes take effect
immediately without touching this file.

Public API
----------
    from src.llm_agent import LLMConfig, reason_about_exception, LLMReasoningResult

    cfg    = LLMConfig(model="llama3.1")
    result = reason_about_exception(exception_record, cfg=cfg)

    print(result.agrees_with_rules)   # bool
    print(result.category)            # confirmed or overridden category
    print(result.confidence)          # "high" | "medium" | "low"
    print(result.explanation)         # plain-English for a human reviewer

Usage with a GroupedMatch
--------------------------
    from src.llm_agent import reason_about_grouped
    result = reason_about_grouped(grouped_match, cfg=cfg)
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal, Optional, Union

import requests

from src.exceptions import ExceptionCategory, ExceptionRecord
from src.matcher import GroupedMatch, MatchedPair
from src.prompts import build_few_shot_block

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """
    Configuration for the Ollama LLM backend.

    Attributes
    ----------
    model : str
        Ollama model tag to use.  Default "phi3:latest" was selected by
        ``src/llm_eval.py`` over qwen2.5:7b-instruct, qwen2.5:14b, and
        llama3:latest on a stratified sample of synthetic exceptions: all
        four hit 100% category agreement with the rule engine, but phi3 was
        the fastest (~7s/call vs 13-87s) and the only one with zero parse
        fallbacks AND zero timeout fallbacks. qwen2.5:14b in particular was
        6x slower with no accuracy benefit and one timeout. Re-run
        ``python -m src.llm_eval`` if you add/remove local models or change
        the few-shot examples in prompts.py — this is not a one-time choice.
    base_url : str
        Base URL of the running Ollama instance.
    timeout_seconds : float
        HTTP request timeout.  Set higher for larger models on slower hardware.
    temperature : float
        Sampling temperature.  Low values (0.1) give deterministic JSON output;
        raise to 0.7+ for more creative explanations.
    max_tokens : int
        Maximum tokens the model may generate per call.
    """
    model          : str   = "phi3:latest"
    base_url       : str   = "http://localhost:11434"
    timeout_seconds: float = 60.0
    temperature    : float = 0.1
    max_tokens     : int   = 600


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

@dataclass
class LLMReasoningResult:
    """
    Structured output from ``reason_about_exception``.

    Attributes
    ----------
    category : str
        The exception category.  May differ from the rule-based category if
        the LLM challenges the classification (``agrees_with_rules=False``).
    explanation : str
        Plain-English description a finance operations reviewer can act on.
    confidence : "high" | "medium" | "low"
        The LLM's self-reported confidence in its analysis.
    agrees_with_rules : bool
        True  — LLM confirms the rule-based category.
        False — LLM proposes a different category or interpretation.
    raw_response : str | None
        The verbatim text the model returned (for audit / debugging).
    fallback_used : bool
        True when the LLM was unavailable or returned an unparseable response.
        In that case all fields reflect the rule-based output only.
    fallback_reason : str | None
        Why the fallback was triggered (e.g. "ConnectionError", "ParseError").
    model_used : str | None
        The Ollama model tag that was actually called.
    """
    category        : str
    explanation     : str
    confidence      : Literal["high", "medium", "low"]
    agrees_with_rules: bool
    raw_response    : Optional[str]  = None
    fallback_used   : bool           = False
    fallback_reason : Optional[str]  = None
    model_used      : Optional[str]  = None

    def to_dict(self) -> dict:
        return {
            "category"         : self.category,
            "explanation"      : self.explanation,
            "confidence"       : self.confidence,
            "agrees_with_rules": self.agrees_with_rules,
            "fallback_used"    : self.fallback_used,
            "fallback_reason"  : self.fallback_reason,
            "model_used"       : self.model_used,
        }


# (Few-shot examples are loaded dynamically from src/prompts.py
#  via build_few_shot_block() — edit them there.)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_exception_prompt(record: ExceptionRecord, cfg: LLMConfig) -> str:
    """
    Construct the full prompt for a single ExceptionRecord.

    The prompt includes:
      - Role description
      - Three few-shot examples showing the expected JSON format
      - The record's details serialised as plain text
      - Strict output format instructions
    """
    # Build a compact transaction summary from whichever side is populated
    if record.gateway_record:
        r = record.gateway_record
        rec_lines = (
            f"  Source             : GATEWAY\n"
            f"  Transaction ID     : {r.transaction_id}\n"
            f"  Reference ID       : {r.reference_id}\n"
            f"  Amount             : {r.currency} {r.amount}\n"
            f"  Status             : {r.status}\n"
            f"  Timestamp          : {r.timestamp.isoformat()}\n"
            f"  Counterparty       : {r.counterparty}"
        )
    else:
        r = record.ledger_record
        rec_lines = (
            f"  Source             : LEDGER\n"
            f"  Transaction ID     : {r.transaction_id}\n"
            f"  Reference ID       : {r.reference_id}\n"
            f"  Amount             : {r.currency} {r.amount}\n"
            f"  Status             : {r.status}\n"
            f"  Timestamp          : {r.timestamp.isoformat()}\n"
            f"  Counterparty       : {r.counterparty}"
        )

    delta_lines = "  (No nearest counterpart found — record is entirely absent from the other source)"
    if record.amount_delta_pct is not None or record.timestamp_delta_hours is not None:
        parts = []
        if record.amount_delta_pct is not None:
            parts.append(f"amount_delta_pct={record.amount_delta_pct}%")
        if record.timestamp_delta_hours is not None:
            parts.append(f"timestamp_delta_hours={record.timestamp_delta_hours}h")
        delta_lines = "  " + ", ".join(parts)

    rule_cat     = record.category.value
    rule_explain  = record.explanation[:300]   # truncate for prompt economy
    few_shot_block = build_few_shot_block()

    return f"""You are a senior financial reconciliation analyst reviewing payment exceptions flagged by an automated rule engine.

Your task:
1. Review the flagged record and the rule engine's category.
2. Confirm or challenge the category based on your analysis.
3. Write a plain-English explanation a non-technical finance reviewer can act on immediately.
4. Assign a confidence level: "high", "medium", or "low".

Output ONLY valid JSON in exactly this format — no markdown, no extra text:
{{
  "agrees_with_rules": <true or false>,
  "category": "<category_name>",
  "confidence": "<high|medium|low>",
  "explanation": "<one to three sentences>"
}}

Valid category names: missing_in_ledger, missing_in_gateway, amount_mismatch, stale_timing, duplicate

--- FEW-SHOT EXAMPLES ---
{few_shot_block}

--- RECORD TO ANALYSE ---
Exception ID       : {record.exception_id}
Rule-based category: {rule_cat}
Rule explanation   : {rule_explain}

Transaction details:
{rec_lines}

Nearest counterpart delta:
{delta_lines}

Now output your JSON analysis:"""


def _build_grouped_prompt(group: GroupedMatch, cfg: LLMConfig) -> str:
    """Construct a prompt for a GroupedMatch (Phase 3 multiplicity result)."""
    if group.match_type == "one_to_many":
        gw    = group.gateway_records[0]
        side  = f"1 gateway record (ref {gw.reference_id}, INR {group.gateway_total}) matched against {len(group.ledger_records)} ledger records whose amounts sum to INR {group.ledger_total}."
        delta = f"Amount delta: INR {group.amount_delta} ({group.amount_delta_pct}%)"
    else:
        led   = group.ledger_records[0]
        side  = f"{len(group.gateway_records)} gateway records whose amounts sum to INR {group.gateway_total} matched against 1 ledger record (ref {led.reference_id}, INR {group.ledger_total})."
        delta = f"Amount delta: INR {group.amount_delta} ({group.amount_delta_pct}%)"

    return f"""You are a senior financial reconciliation analyst reviewing a grouped payment match.

The rule engine found a {group.match_type.replace('_',' ')} match: {side}
{delta}
Confidence score from engine: {round(group.confidence, 3)}

Your task:
1. Assess whether this grouped match is likely correct or a false positive.
2. Write a plain-English explanation a finance reviewer can act on.
3. Assign a confidence level: "high", "medium", or "low".
4. Set agrees_with_rules=true if the grouping looks correct, false if it looks like a coincidental sum.

Output ONLY valid JSON — no markdown, no extra text:
{{
  "agrees_with_rules": <true or false>,
  "category": "grouped_{group.match_type}",
  "confidence": "<high|medium|low>",
  "explanation": "<one to three sentences>"
}}

Now output your JSON analysis:"""


def _build_pair_audit_prompt(pair: MatchedPair, cfg: LLMConfig) -> str:
    """
    Construct a prompt asking the LLM to sanity-check a Phase-2.5 (content)
    or Phase-2.75 (text) match — i.e. a pair the engine matched WITHOUT the
    two sides' reference IDs agreeing at all, purely on amount/timestamp
    proximity (and, for Phase 2.75, secondary text similarity). This is a
    genuinely weaker signal than reference-anchored matching, so it is worth
    a second opinion on a sample of these, distinct from
    ``reason_about_exception``'s review of *unmatched* records.
    """
    gw, led = pair.gateway_record, pair.ledger_record
    basis = (
        "amount + timestamp proximity only (no shared reference at all)"
        if pair.match_type == "content"
        else "amount + timestamp proximity, disambiguated among multiple "
             "candidates by reference/counterparty text similarity"
    )
    return f"""You are a senior financial reconciliation analyst auditing a match the rule engine made WITHOUT reference-ID agreement between the two sides — matched on: {basis}.

Gateway (external) record:
  Transaction ID : {gw.transaction_id}
  Reference ID   : {gw.reference_id}
  Counterparty   : {gw.counterparty}
  Amount         : {gw.currency} {gw.amount}
  Timestamp      : {gw.timestamp.isoformat()}

Ledger (internal) record:
  Transaction ID : {led.transaction_id}
  Reference ID   : {led.reference_id}
  Counterparty   : {led.counterparty}
  Amount         : {led.currency} {led.amount}
  Timestamp      : {led.timestamp.isoformat()}

Engine confidence: {round(pair.confidence, 3)}
Amount delta: {pair.amount_delta} | Timestamp delta: {round(pair.timestamp_delta_seconds / 3600, 2)}h

Your task:
1. Judge whether this is plausibly the SAME real-world transaction, or a coincidental amount/date collision between two unrelated transactions.
2. Assign a confidence level: "high", "medium", or "low".
3. Write a one-to-two sentence explanation a reviewer can act on.

Output ONLY valid JSON — no markdown, no extra text:
{{
  "agrees_with_rules": <true if this looks like a genuine match, false if it looks coincidental>,
  "category": "{pair.match_type}_match",
  "confidence": "<high|medium|low>",
  "explanation": "<one to two sentences>"
}}

Now output your JSON analysis:"""


# ---------------------------------------------------------------------------
# Ollama REST call
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, cfg: LLMConfig) -> str:
    """
    POST a prompt to Ollama's /api/generate endpoint and return the response text.

    Raises
    ------
    requests.ConnectionError
        When Ollama is not reachable.
    requests.Timeout
        When the model takes longer than cfg.timeout_seconds.
    RuntimeError
        When Ollama returns a non-200 status or a response without the
        expected ``response`` key.
    """
    url     = f"{cfg.base_url.rstrip('/')}/api/generate"
    payload = {
        "model"  : cfg.model,
        "prompt" : prompt,
        "stream" : False,
        "options": {
            "temperature" : cfg.temperature,
            "num_predict" : cfg.max_tokens,
        },
    }
    resp = requests.post(url, json=payload, timeout=cfg.timeout_seconds)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}"
        )
    data = resp.json()
    if "response" not in data:
        raise RuntimeError(f"Unexpected Ollama response shape: {list(data.keys())}")
    return data["response"]


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

_VALID_CATEGORIES = {c.value for c in ExceptionCategory} | {
    "grouped_one_to_many", "grouped_many_to_one",
    "content_match", "text_match",
}
_VALID_CONFIDENCE = {"high", "medium", "low"}


class NoStructuredOutputError(Exception):
    """
    Raised when the raw text contains no JSON AND no individually
    regex-matchable "key": "value" field at all — i.e. the model produced
    no usable structured signal, as distinct from almost-valid-but-malformed
    JSON that strategy 3 can still partially recover.

    This must NOT be silently absorbed by defaulting ``category`` to the
    rule engine's own category, because that makes "the model said nothing
    coherent" indistinguishable from "the model agreed with the rule
    engine" in the resulting agreement-rate metrics. Callers should catch
    this alongside other parse failures and set ``fallback_used=True``
    (which ``reason_about_exception`` / ``reason_about_pair`` already do
    for any exception raised here).
    """


def _parse_llm_response(raw: str, fallback_category: str) -> dict:
    """
    Parse the model's text response into a clean dict.

    Strategy
    --------
    1. Try json.loads() on the full response (ideal path).
    2. Try to extract a JSON object with regex (handles leading text / markdown).
    3. Fall back to regex field-by-field extraction from plain text.
    4. Raise NoStructuredOutputError if nothing in 1-3 found anything at
       all — never silently default to "agrees with rules."
    """
    # --- Strategy 1: full JSON ---
    try:
        parsed = json.loads(raw.strip())
        return _validate_parsed(parsed, fallback_category)
    except json.JSONDecodeError:
        pass

    # --- Strategy 2: JSON block in the middle of text ---
    json_match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            return _validate_parsed(parsed, fallback_category)
        except json.JSONDecodeError:
            pass

    # --- Strategy 3: regex field extraction ---
    agrees_m = re.search(r'"agrees_with_rules"\s*:\s*(true|false)', raw, re.IGNORECASE)
    cat_m    = re.search(r'"category"\s*:\s*"([a-z_]+)"', raw, re.IGNORECASE)
    conf_m   = re.search(r'"confidence"\s*:\s*"(high|medium|low)"', raw, re.IGNORECASE)
    expl_m   = re.search(r'"explanation"\s*:\s*"(.*?)"', raw, re.DOTALL)

    if not any((agrees_m, cat_m, conf_m, expl_m)):
        # No JSON, and not even one recognisable "key": "value" fragment —
        # this is an absence of structured output, not a formatting slip.
        raise NoStructuredOutputError(
            f"No JSON object and no individual field found in raw output "
            f"(first 120 chars: {raw[:120]!r})"
        )

    agrees = agrees_m.group(1).lower() == "true" if agrees_m else True

    cat = fallback_category
    if cat_m and cat_m.group(1) in _VALID_CATEGORIES:
        cat = cat_m.group(1)

    conf = conf_m.group(1).lower() if conf_m else "medium"

    # Extract explanation: grab text between "explanation": " ... "
    expl = expl_m.group(1).replace("\\n", " ").strip() if expl_m else ""
    if not expl:
        # Use first sentence of raw response as explanation
        expl = raw.split(".")[0].strip()[:300] or "See raw response."

    return {
        "agrees_with_rules": agrees,
        "category"         : cat,
        "confidence"       : conf,
        "explanation"      : expl,
    }


def _validate_parsed(parsed: dict, fallback_category: str) -> dict:
    """Normalise and validate a successfully parsed dict."""
    cat  = str(parsed.get("category", fallback_category)).lower().strip()
    if cat not in _VALID_CATEGORIES:
        cat = fallback_category
    conf = str(parsed.get("confidence", "medium")).lower().strip()
    if conf not in _VALID_CONFIDENCE:
        conf = "medium"
    agrees = bool(parsed.get("agrees_with_rules", True))
    expl   = str(parsed.get("explanation", "")).strip()
    if not expl:
        expl = "No explanation provided by model."
    return {
        "agrees_with_rules": agrees,
        "category"         : cat,
        "confidence"       : conf,
        "explanation"      : expl,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reason_about_exception(
    record : ExceptionRecord,
    cfg    : Optional[LLMConfig] = None,
) -> LLMReasoningResult:
    """
    Ask the LLM to reason about a single classified exception record.

    Parameters
    ----------
    record : ExceptionRecord
        Output from ``src.exceptions.classify()``.
    cfg : LLMConfig, optional
        Ollama configuration.  Defaults to ``LLMConfig()`` (llama3.1, localhost).

    Returns
    -------
    LLMReasoningResult
        Always returns a result — never raises.  Check ``fallback_used`` to
        know whether the LLM was actually consulted.
    """
    cfg = cfg or LLMConfig()
    fallback_cat = record.category.value

    def _fallback(reason: str) -> LLMReasoningResult:
        warnings.warn(
            f"[llm_agent] Ollama unavailable or failed ({reason}). "
            "Returning rule-based output only.",
            stacklevel=3,
        )
        return LLMReasoningResult(
            category         = fallback_cat,
            explanation      = record.explanation,
            confidence       = "medium",
            agrees_with_rules= True,
            raw_response     = None,
            fallback_used    = True,
            fallback_reason  = reason,
            model_used       = None,
        )

    try:
        prompt  = _build_exception_prompt(record, cfg)
        raw     = _call_ollama(prompt, cfg)
    except requests.ConnectionError:
        return _fallback("Ollama not reachable at " + cfg.base_url)
    except requests.Timeout:
        return _fallback(f"Request timed out after {cfg.timeout_seconds}s")
    except RuntimeError as exc:
        return _fallback(str(exc))
    except Exception as exc:          # pragma: no cover — unexpected errors
        logger.exception("Unexpected error calling Ollama")
        return _fallback(f"Unexpected error: {exc}")

    try:
        parsed = _parse_llm_response(raw, fallback_cat)
    except Exception as exc:
        return LLMReasoningResult(
            category         = fallback_cat,
            explanation      = record.explanation,
            confidence       = "medium",
            agrees_with_rules= True,
            raw_response     = raw,
            fallback_used    = True,
            fallback_reason  = f"ParseError: {exc}",
            model_used       = cfg.model,
        )

    return LLMReasoningResult(
        category         = parsed["category"],
        explanation      = parsed["explanation"],
        confidence       = parsed["confidence"],
        agrees_with_rules= parsed["agrees_with_rules"],
        raw_response     = raw,
        fallback_used    = False,
        fallback_reason  = None,
        model_used       = cfg.model,
    )


def reason_about_grouped(
    group : GroupedMatch,
    cfg   : Optional[LLMConfig] = None,
) -> LLMReasoningResult:
    """
    Ask the LLM to validate a Phase-3 grouped match.

    Parameters
    ----------
    group : GroupedMatch
        A one-to-many or many-to-one match from ``ReconciliationEngine``.
    cfg : LLMConfig, optional
        Ollama configuration.

    Returns
    -------
    LLMReasoningResult
        ``agrees_with_rules=True`` means the LLM believes the group is a
        genuine match.  ``False`` suggests it may be a coincidental sum.
    """
    cfg = cfg or LLMConfig()
    fallback_cat = f"grouped_{group.match_type}"

    def _fallback(reason: str) -> LLMReasoningResult:
        warnings.warn(
            f"[llm_agent] Ollama unavailable or failed ({reason}). "
            "Returning engine-confidence as fallback.",
            stacklevel=3,
        )
        conf_str = (
            "high" if group.confidence >= 0.8
            else "medium" if group.confidence >= 0.5
            else "low"
        )
        return LLMReasoningResult(
            category         = fallback_cat,
            explanation      = (
                f"Phase-3 engine matched {len(group.gateway_records)} gateway "
                f"record(s) (total INR {group.gateway_total}) to "
                f"{len(group.ledger_records)} ledger record(s) "
                f"(total INR {group.ledger_total}). "
                f"Amount delta: {group.amount_delta_pct}%. "
                "LLM validation was unavailable."
            ),
            confidence       = conf_str,
            agrees_with_rules= True,
            raw_response     = None,
            fallback_used    = True,
            fallback_reason  = reason,
            model_used       = None,
        )

    try:
        prompt = _build_grouped_prompt(group, cfg)
        raw    = _call_ollama(prompt, cfg)
    except requests.ConnectionError:
        return _fallback("Ollama not reachable at " + cfg.base_url)
    except requests.Timeout:
        return _fallback(f"Request timed out after {cfg.timeout_seconds}s")
    except RuntimeError as exc:
        return _fallback(str(exc))
    except Exception as exc:
        logger.exception("Unexpected error calling Ollama")
        return _fallback(f"Unexpected error: {exc}")

    try:
        parsed = _parse_llm_response(raw, fallback_cat)
    except Exception as exc:
        conf_str = "high" if group.confidence >= 0.8 else "medium" if group.confidence >= 0.5 else "low"
        return LLMReasoningResult(
            category=fallback_cat, explanation=raw[:300],
            confidence=conf_str, agrees_with_rules=True,
            raw_response=raw, fallback_used=True,
            fallback_reason=f"ParseError: {exc}", model_used=cfg.model,
        )

    return LLMReasoningResult(
        category         = parsed["category"],
        explanation      = parsed["explanation"],
        confidence       = parsed["confidence"],
        agrees_with_rules= parsed["agrees_with_rules"],
        raw_response     = raw,
        fallback_used    = False,
        fallback_reason  = None,
        model_used       = cfg.model,
    )


def reason_about_pair(
    pair : MatchedPair,
    cfg  : Optional[LLMConfig] = None,
) -> LLMReasoningResult:
    """
    Ask the LLM to audit a Phase-2.5 ("content") or Phase-2.75 ("text")
    match — one made without any reference-ID agreement between the two
    sides. Intended for spot-auditing a sample of these matches for
    precision, not for bulk matching: at real-world batch sizes the
    candidate pool for this kind of match can be in the tens of thousands,
    which only a fast deterministic phase (see matcher.py) can process;
    the LLM's role here is a qualitative second opinion on a sample.

    Parameters
    ----------
    pair : MatchedPair
        A pair with ``match_type in {"content", "text"}`` from
        ``ReconciliationResult.matched_content`` / ``matched_text``.
    cfg : LLMConfig, optional

    Returns
    -------
    LLMReasoningResult
        ``agrees_with_rules=True`` means the LLM believes this is a genuine
        match. ``False`` suggests it may be a coincidental collision.
    """
    cfg = cfg or LLMConfig()
    fallback_cat = f"{pair.match_type}_match"

    def _fallback(reason: str) -> LLMReasoningResult:
        warnings.warn(
            f"[llm_agent] Ollama unavailable or failed ({reason}). "
            "Returning engine-confidence as fallback.",
            stacklevel=3,
        )
        conf_str = (
            "high" if pair.confidence >= 0.8
            else "medium" if pair.confidence >= 0.5
            else "low"
        )
        return LLMReasoningResult(
            category         = fallback_cat,
            explanation      = (
                f"Phase {'2.5' if pair.match_type == 'content' else '2.75'} matched "
                f"{pair.gateway_record.transaction_id} to {pair.ledger_record.transaction_id} "
                f"on amount/timestamp proximity alone (confidence {round(pair.confidence, 3)}). "
                "LLM validation was unavailable."
            ),
            confidence       = conf_str,
            agrees_with_rules= True,
            raw_response     = None,
            fallback_used    = True,
            fallback_reason  = reason,
            model_used       = None,
        )

    try:
        prompt = _build_pair_audit_prompt(pair, cfg)
        raw    = _call_ollama(prompt, cfg)
    except requests.ConnectionError:
        return _fallback("Ollama not reachable at " + cfg.base_url)
    except requests.Timeout:
        return _fallback(f"Request timed out after {cfg.timeout_seconds}s")
    except RuntimeError as exc:
        return _fallback(str(exc))
    except Exception as exc:
        logger.exception("Unexpected error calling Ollama")
        return _fallback(f"Unexpected error: {exc}")

    try:
        parsed = _parse_llm_response(raw, fallback_cat)
    except Exception as exc:
        conf_str = "high" if pair.confidence >= 0.8 else "medium" if pair.confidence >= 0.5 else "low"
        return LLMReasoningResult(
            category=fallback_cat, explanation=raw[:300],
            confidence=conf_str, agrees_with_rules=True,
            raw_response=raw, fallback_used=True,
            fallback_reason=f"ParseError: {exc}", model_used=cfg.model,
        )

    return LLMReasoningResult(
        category         = parsed["category"],
        explanation      = parsed["explanation"],
        confidence       = parsed["confidence"],
        agrees_with_rules= parsed["agrees_with_rules"],
        raw_response     = raw,
        fallback_used    = False,
        fallback_reason  = None,
        model_used       = cfg.model,
    )


def reason_about_pairs_batch(
    pairs     : list[MatchedPair],
    cfg       : Optional[LLMConfig] = None,
    max_calls : int = 20,
) -> list[LLMReasoningResult]:
    """Process a list of MatchedPairs sequentially, capped at ``max_calls``."""
    cfg     = cfg or LLMConfig()
    results = []

    for i, pair in enumerate(pairs):
        if i >= max_calls:
            conf_str = "high" if pair.confidence >= 0.8 else "medium" if pair.confidence >= 0.5 else "low"
            results.append(LLMReasoningResult(
                category         = f"{pair.match_type}_match",
                explanation      = f"Engine confidence {round(pair.confidence, 3)}.",
                confidence       = conf_str,
                agrees_with_rules= True,
                fallback_used    = True,
                fallback_reason  = "batch_cap_exceeded",
                model_used       = None,
            ))
        else:
            results.append(reason_about_pair(pair, cfg=cfg))

    return results


def reason_about_exceptions_batch(
    exceptions : list[ExceptionRecord],
    cfg        : Optional[LLMConfig] = None,
    max_calls  : int = 20,
) -> list[LLMReasoningResult]:
    """
    Process a list of ExceptionRecords sequentially, capped at ``max_calls``.

    Records beyond the cap receive a fallback result to avoid runaway LLM costs
    when datasets are large.

    Parameters
    ----------
    exceptions : list[ExceptionRecord]
    cfg        : LLMConfig, optional
    max_calls  : int
        Maximum number of Ollama calls.  Records beyond this limit get
        ``fallback_used=True`` with reason "batch_cap_exceeded".

    Returns
    -------
    list[LLMReasoningResult]
        Same length as ``exceptions``.
    """
    cfg     = cfg or LLMConfig()
    results = []

    for i, exc in enumerate(exceptions):
        if i >= max_calls:
            results.append(LLMReasoningResult(
                category         = exc.category.value,
                explanation      = exc.explanation,
                confidence       = "medium",
                agrees_with_rules= True,
                fallback_used    = True,
                fallback_reason  = "batch_cap_exceeded",
                model_used       = None,
            ))
        else:
            results.append(reason_about_exception(exc, cfg=cfg))

    return results
