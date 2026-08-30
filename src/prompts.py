"""
prompts.py — Few-Shot Example Library for LLM Exception Reasoning
==================================================================

This module contains the hand-written few-shot examples that are injected
into every LLM prompt built by src/llm_agent.py.

HOW TO EDIT
-----------
Each example is a ``FewShotExample`` dataclass instance.  The three fields
you will most often edit are:

  - ``input_block``    : the raw-text description of the transaction(s) shown
                         to the model.  Use the same field order as the live
                         prompts so the model generalises correctly.
  - ``expected_output``: the ideal JSON the model should produce.  Edit the
                         ``explanation`` string to match the reasoning style
                         you want in production outputs.
  - ``notes``          : developer commentary explaining why this example was
                         chosen.  NOT included in the prompt — editorial only.

ADDING / REMOVING EXAMPLES
---------------------------
Add new ``FewShotExample`` instances to ``ALL_EXAMPLES``.  The
``build_few_shot_block()`` function renders them all in order.  You can also
call ``build_few_shot_block(examples=[EXAMPLE_AMOUNT_MISMATCH])`` to use a
custom subset.

STYLE GUIDE FOR EXPLANATIONS
-----------------------------
Good explanations:
  ✓ Cite the actual INR amounts and percentage deltas.
  ✓ Name the specific likely cause first ("This is consistent with a 2%
    Razorpay processing fee...").
  ✓ End with a concrete "Action:" sentence that tells the reviewer exactly
    what to do next.
  ✓ Are 2-3 sentences.  Longer is rarely more useful.

Bad explanations (avoid):
  ✗ "There may be a discrepancy between the two sources."
  ✗ "Please investigate further."
  ✗ Repeating the rule-based category with no added insight.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FewShotExample:
    """
    A single few-shot example to be injected into the LLM prompt.

    Attributes
    ----------
    title : str
        Short human-readable label shown as the section header.
        Example: "EXAMPLE 2 — amount_mismatch (LLM agrees)"
    category : str
        The rule-based category name, e.g. "amount_mismatch".
    record_type : str
        One-line description of the record shape, e.g.
        "gateway-only (tolerance exceeded)".
    input_block : str
        Multi-line text describing the transaction(s) exactly as they would
        appear in the live prompt.  Keep the same field labels.
    expected_output : dict
        The ideal JSON response the model should produce.
        Keys: agrees_with_rules (bool), category (str),
              confidence (str), explanation (str).
    notes : str
        Developer notes on why this example was designed this way.
        Never included in the rendered prompt.
    """
    title           : str
    category        : str
    record_type     : str
    input_block     : str
    expected_output : dict
    notes           : str = ""


# ---------------------------------------------------------------------------
# Example 1 — AMOUNT_MISMATCH  (LLM agrees, medium confidence)
#
# Design rationale
# ----------------
# The 2% delta is the canonical Razorpay processing fee scenario.  We use
# "medium" rather than "high" confidence deliberately: the model should
# acknowledge that a partial refund is a plausible alternative.  This teaches
# the model to hedge appropriately when two explanations fit the same numbers.
#
# The amounts are chosen to make the arithmetic obvious:
#   INR 49,000 × 0.02 = INR 980  →  gateway settles INR 48,020.
# A reviewer can verify this in seconds.
# ---------------------------------------------------------------------------

EXAMPLE_AMOUNT_MISMATCH = FewShotExample(
    title       = "EXAMPLE 1 — amount_mismatch (LLM agrees, medium confidence)",
    category    = "amount_mismatch",
    record_type = "gateway-only (amount tolerance exceeded)",
    input_block = textwrap.dedent("""\
        Source             : GATEWAY
        Transaction ID     : rzp_live_Hk7mPqR2sTwX9n
        Reference ID       : PAY20240314000059
        Amount             : INR 48020.00
        Status             : settled
        Timestamp          : 2024-03-14T09:12:44+00:00
        Counterparty       : amit@paytm

        Nearest counterpart delta:
          amount_delta_pct=2.00%, timestamp_delta_hours=0.2h

        Nearest ledger record:
          Reference ID : ORD-2024-03-14-000059
          Amount       : INR 49000.00
          Status       : paid
          Timestamp    : 2024-03-14T09:04:11+00:00"""),
    expected_output = {
        "agrees_with_rules": True,
        "category"         : "amount_mismatch",
        "confidence"       : "medium",
        "explanation"      : (
            "The gateway settled INR 980.00 less than the ledger amount "
            "(INR 48,020 vs INR 49,000 — exactly 2.0%). "
            "This matches the standard Razorpay processing fee rate and is "
            "almost certainly a fee deduction netted before remittance, not "
            "a payment error. "
            "A partial refund applied only on the gateway side is a less "
            "likely but possible alternative. "
            "Action: if 2% is the contracted gateway fee, post a "
            "'gateway processing fee' expense of INR 980 to match this pair "
            "and mark it reconciled. "
            "If no fee should apply, raise a dispute with Razorpay citing "
            "reference PAY20240314000059."
        ),
    },
    notes = (
        "Medium confidence is intentional — teaches the model to acknowledge "
        "the partial-refund alternative rather than being overconfident. "
        "The 'Action:' line branches on the fee-vs-refund ambiguity, giving "
        "the reviewer a clear decision tree."
    ),
)


# ---------------------------------------------------------------------------
# Example 2 — STALE_TIMING  (LLM challenges the category, high confidence)
#
# Design rationale
# ----------------
# This is the most important example for preventing false positives.
# A 5h30m timestamp gap at exactly IST-UTC offset is a classic ETL bug, not a
# settlement delay.  The model must learn to challenge the rule-based category
# when the data pattern is recognisable.
#
# "agrees_with_rules: false" + "category: stale_timing" means:
#   → the category label is unchanged but the interpretation is different
#     (timezone bug, not genuine staleness).
# This teaches the model that challenging the *interpretation* is valid even
# when the category string stays the same.
#
# The amounts are identical (0% delta) to isolate the timing signal.
# ---------------------------------------------------------------------------

EXAMPLE_STALE_TIMING = FewShotExample(
    title       = "EXAMPLE 2 — stale_timing (LLM challenges, high confidence)",
    category    = "stale_timing",
    record_type = "gateway-only (timestamp tolerance exceeded)",
    input_block = textwrap.dedent("""\
        Source             : GATEWAY
        Transaction ID     : rzp_live_Lp3nQvT5uYaZ8k
        Reference ID       : PAY20240308000033
        Amount             : INR 22750.00
        Status             : settled
        Timestamp          : 2024-03-08T18:42:00+00:00
        Counterparty       : neha@apl

        Nearest counterpart delta:
          amount_delta_pct=0.00%, timestamp_delta_hours=5.5h

        Nearest ledger record:
          Reference ID : ORD-2024-03-08-000033
          Amount       : INR 22750.00
          Status       : paid
          Timestamp    : 2024-03-08T13:12:00+00:00"""),
    expected_output = {
        "agrees_with_rules": False,
        "category"         : "stale_timing",
        "confidence"       : "high",
        "explanation"      : (
            "The amounts match exactly (INR 22,750.00) and the references are "
            "identical, but the rule engine flagged this because the timestamps "
            "differ by 5h 30min — which is precisely the IST-to-UTC offset "
            "(UTC+5:30). "
            "This is not a genuine settlement delay; the ledger recorded the "
            "timestamp in IST while the gateway used UTC, creating a "
            "systematic timezone mismatch across all records ingested from "
            "this source. "
            "Action: normalise all ledger timestamps to UTC in the ETL "
            "pipeline (add 5h30m to all IST values). Once fixed, this pair "
            "will auto-match as exact. Do NOT raise a dispute or create a "
            "manual entry — this is a data pipeline bug, not a payment error."
        ),
    },
    notes = (
        "agrees_with_rules=False with the same category name signals that the "
        "model accepts the label but rejects the implied cause. "
        "High confidence is justified because 5h30m = IST offset is a "
        "deterministic signal, not a probabilistic one. "
        "The 'Action' explicitly says NOT to raise a dispute — this "
        "prevents reviewers from taking harmful action on a false positive."
    ),
)


# ---------------------------------------------------------------------------
# Example 3 — GROUPED (one_to_many)  (LLM agrees, high confidence)
#
# Design rationale
# ----------------
# Grouped matches are unfamiliar to most LLMs because they are not the
# standard 1:1 reconciliation case.  This example teaches the model:
#   1. What a batch-settlement looks like (one GW settles multiple orders).
#   2. That an exact sum with zero delta is strong positive evidence.
#   3. To look for whether a separate fee invoice might account for the
#      batch rather than assuming the amount is net of fees.
#
# The three child amounts (INR 18,200 + INR 24,890 + INR 24,250 = 67,340)
# are chosen to sum cleanly with no rounding, so the model can verify the
# arithmetic in the explanation without confusion.
# ---------------------------------------------------------------------------

EXAMPLE_GROUPED_ONE_TO_MANY = FewShotExample(
    title       = "EXAMPLE 3 — grouped_one_to_many (LLM agrees, high confidence)",
    category    = "grouped_one_to_many",
    record_type = "batch settlement: 1 gateway record covering 3 ledger orders",
    input_block = textwrap.dedent("""\
        Match type         : one_to_many
        Engine confidence  : 1.000

        GATEWAY SIDE (1 record):
          Transaction ID   : rzp_live_Wq8nDcK1vRmY6j
          Reference ID     : BATCH20240320000096
          Amount           : INR 67340.00
          Status           : settled
          Timestamp        : 2024-03-20T14:05:00+00:00
          Counterparty     : vikas@upi

        LEDGER SIDE (3 records, sum = INR 67340.00):
          [1] Transaction : LED-2024-000087 | ref ORD-2024-03-18-000087
              Amount      : INR 18200.00   | status: paid
          [2] Transaction : LED-2024-000094 | ref ORD-2024-03-19-000094
              Amount      : INR 24890.00   | status: paid
          [3] Transaction : LED-2024-000102 | ref ORD-2024-03-20-000102
              Amount      : INR 24250.00   | status: paid

        Amount delta       : INR 0.00 (0.00%)"""),
    expected_output = {
        "agrees_with_rules": True,
        "category"         : "grouped_one_to_many",
        "confidence"       : "high",
        "explanation"      : (
            "The batch settlement BATCH20240320000096 (INR 67,340.00) matches "
            "the exact sum of three ledger orders — INR 18,200 + INR 24,890 "
            "+ INR 24,250 = INR 67,340 — with zero amount delta. "
            "This is a standard netting settlement where the gateway batched "
            "multiple individual orders into a single payout; the lack of a "
            "fee deduction suggests gateway fees are billed separately via "
            "a monthly invoice rather than netted per transaction. "
            "Action: link ledger records LED-2024-000087, LED-2024-000094, "
            "and LED-2024-000102 to gateway record rzp_live_Wq8nDcK1vRmY6j "
            "and mark all four as reconciled. "
            "Verify the fee billing arrangement with your gateway contract."
        ),
    },
    notes = (
        "High confidence is appropriate here because the sum is exact (0% "
        "delta) and all three ledger records have status=paid, which is "
        "consistent with the gateway having settled them. "
        "The explanation names each ledger record explicitly — this is the "
        "'citing actual numbers' requirement the user asked for. "
        "The fee-invoice observation is a useful insight the rule engine "
        "cannot provide, demonstrating the LLM adds real value beyond labelling."
    ),
)


# ---------------------------------------------------------------------------
# Ordered list — controls the sequence in the rendered prompt block.
# You can reorder, add, or remove examples here.
# ---------------------------------------------------------------------------

ALL_EXAMPLES: list[FewShotExample] = [
    EXAMPLE_AMOUNT_MISMATCH,
    EXAMPLE_STALE_TIMING,
    EXAMPLE_GROUPED_ONE_TO_MANY,
]


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render_example(example: FewShotExample, index: int) -> str:
    """
    Render a single ``FewShotExample`` into the plain-text block that is
    injected into the LLM prompt.

    The output format matches what the prompt template in llm_agent.py
    expects:

        === EXAMPLE N — <title> ===
        RECORD TYPE: <record_type>
        RULE CATEGORY: <category>
        <input_block lines>

        EXPECTED OUTPUT:
        {
          "agrees_with_rules": ...,
          ...
        }

    Parameters
    ----------
    example : FewShotExample
    index   : int   — 1-based example number shown in the header.

    Returns
    -------
    str
    """
    json_str = json.dumps(example.expected_output, indent=2, ensure_ascii=False)
    return (
        f"=== EXAMPLE {index} — {example.title} ===\n"
        f"RECORD TYPE   : {example.record_type}\n"
        f"RULE CATEGORY : {example.category}\n\n"
        f"{example.input_block}\n\n"
        f"EXPECTED OUTPUT:\n"
        f"{json_str}"
    )


def build_few_shot_block(
    examples: Optional[list[FewShotExample]] = None,
    separator: str = "\n\n",
) -> str:
    """
    Render a list of ``FewShotExample`` instances into the full block that
    is embedded in every LLM prompt.

    Parameters
    ----------
    examples : list[FewShotExample], optional
        Defaults to ``ALL_EXAMPLES`` (all three examples in order).
        Pass a custom list to use a subset or different ordering.
    separator : str
        Text placed between rendered examples.  Default: two newlines.

    Returns
    -------
    str
        The fully rendered few-shot block, ready to drop into a prompt
        f-string at the ``{few_shot_block}`` placeholder.

    Example
    -------
    >>> from src.prompts import build_few_shot_block, EXAMPLE_STALE_TIMING
    >>> block = build_few_shot_block(examples=[EXAMPLE_STALE_TIMING])
    >>> print(block[:80])
    """
    if examples is None:
        examples = ALL_EXAMPLES
    rendered = [render_example(ex, i + 1) for i, ex in enumerate(examples)]
    return separator.join(rendered)


# ---------------------------------------------------------------------------
# Quick smoke-test  (python -m src.prompts)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    block = build_few_shot_block()
    print(block)
    print()
    print(f"--- Total examples: {len(ALL_EXAMPLES)} ---")
    print(f"--- Total chars in block: {len(block)} ---")
