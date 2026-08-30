"""
schema.py — Data Models for Transactions
=========================================

Defines Pydantic v2 models representing financial transaction records from two
distinct, real-world data sources that must be reconciled against each other:

  1. GatewayRecord  — A settlement record exported by a payment gateway
                      (e.g. Razorpay, Stripe, PayU). These records reflect
                      what actually moved through the payment network.

  2. LedgerRecord   — An internal order / accounting record maintained by the
                      merchant's own system (e.g. an ERP, OMS, or accounting DB).
                      These records reflect what the merchant *believes* happened.

Why these two sources diverge in practice
------------------------------------------
Even for the same underlying transaction, GatewayRecord and LedgerRecord will
routinely differ in the following ways:

  reference_id formatting
      The gateway typically emits a raw UTR / RRN (e.g. "RAZORPAY20240315123456"),
      while the merchant's system may store only the order portion
      (e.g. "ORD-20240315-123456") or apply its own prefix scheme.
      Case differences and leading-zero stripping are also common.

  amount mismatches
      Gateways deduct processing fees, GST, or interchange before settling.
      The ledger records the gross order value; the gateway settles the net value.
      Partial refunds processed on the gateway side may not yet be reflected
      in the ledger, or vice-versa.

  timestamp skew
      The gateway timestamp marks *settlement* (T+1 or T+2 in India).
      The ledger timestamp marks the moment the *order was placed* or *payment
      was confirmed* by the customer, which can be hours or days earlier.
      Time-zone normalisation errors (IST vs UTC) add a further 5h 30m offset.

  duplicate records
      Gateway retry storms or webhook re-deliveries can produce multiple
      GatewayRecords with the same reference_id but slightly different amounts
      or timestamps. The ledger may also double-book in edge cases.

  missing records
      A GatewayRecord with status "failed" may have no corresponding
      LedgerRecord if the merchant's system never received the webhook.
      Conversely, a LedgerRecord with status "pending" may have no matching
      GatewayRecord if the payment was abandoned before reaching the gateway.

  status label inconsistency
      Gateways use their own status vocabularies ("captured", "settled",
      "refunded") while internal ledgers use business-domain terms
      ("paid", "closed", "returned").  StatusNormaliser (in matcher.py)
      maps these to a canonical set.

Also defined here
-----------------
  TransactionStatus   — Canonical status enum used internally after normalisation.
  MatchStatus         — Result enum for a reconciliation pair.
  ReconciliationRecord — Pairs a GatewayRecord + LedgerRecord and records the
                         outcome of a reconciliation attempt.

Usage
-----
    from src.schema import GatewayRecord, LedgerRecord, ReconciliationRecord
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TransactionStatus(str, Enum):
    """
    Canonical status values used *internally* by the reconciliation engine
    after raw status strings from each source have been normalised.

    The matcher maps gateway-specific labels (e.g. "captured", "settled")
    and ledger-specific labels (e.g. "paid", "closed") onto these values
    before attempting any comparison.
    """
    SETTLED  = "settled"
    PENDING  = "pending"
    FAILED   = "failed"
    REFUNDED = "refunded"
    REVERSED = "reversed"
    UNKNOWN  = "unknown"


class MatchStatus(str, Enum):
    """
    Result of attempting to reconcile one GatewayRecord against one LedgerRecord.

    MATCHED
        Both records exist and all key fields agree within configured tolerances.
    PARTIAL
        Both records exist but one or more fields (amount, timestamp, status)
        fall outside tolerance — requires manual review.
    UNMATCHED_GATEWAY
        A GatewayRecord has no corresponding LedgerRecord.
    UNMATCHED_LEDGER
        A LedgerRecord has no corresponding GatewayRecord.
    DUPLICATE
        More than one record from the same source maps to the same counterpart.
    """
    MATCHED            = "matched"
    PARTIAL            = "partial"
    UNMATCHED_GATEWAY  = "unmatched_gateway"
    UNMATCHED_LEDGER   = "unmatched_ledger"
    DUPLICATE          = "duplicate"


# ---------------------------------------------------------------------------
# Source Models
# ---------------------------------------------------------------------------

class GatewayRecord(BaseModel):
    """
    Represents a single settlement record exported by a payment gateway.

    This model captures what the payment network actually processed and settled.
    Gateway exports are typically received as CSV or JSON files on a T+1 or T+2
    settlement cycle and may cover thousands of transactions in a single batch.

    Real-world quirks to be aware of
    ----------------------------------
    - ``reference_id`` follows the gateway's own format (e.g. ``"PAY_XXXXXXXX"``
      for Razorpay, ``"ch_XXXXXXXX"`` for Stripe) and rarely matches the
      merchant's internal order reference directly.
    - ``amount`` is the *net settled* amount after deduction of gateway fees and
      taxes; it is almost never equal to the gross order amount in LedgerRecord.
    - ``timestamp`` is the *settlement* timestamp, not the authorisation time.
      For Indian payment rails (UPI, NEFT, IMPS) this can lag the ledger
      timestamp by anywhere from a few minutes to 48 hours.
    - ``status`` uses gateway-native vocabulary:
        "captured"  → money authorised and captured
        "settled"   → funds transferred to merchant's bank account
        "failed"    → payment declined or timed out
        "refunded"  → full or partial refund initiated by gateway
    - Duplicate rows (same ``reference_id``, slightly different ``amount``) appear
      when a gateway re-sends settlement batches after corrections.

    Attributes
    ----------
    transaction_id : str
        Gateway-assigned unique identifier for this settlement line item.
        Example: ``"rzp_live_AbCdEfGhIjKl"``
    reference_id : str
        Shared payment reference that *should* link this record to a LedgerRecord
        (e.g. UTR number, RRN, or order ID echoed back by the gateway).
        Formatting often differs from the ledger's copy — see module docstring.
    amount : Decimal
        Net amount settled by the gateway in the stated currency.
        Always positive; refund amounts are recorded separately with
        ``status="refunded"``.
    currency : str
        ISO-4217 currency code. Defaults to ``"INR"``.
    timestamp : datetime
        UTC datetime when the gateway settled (credited) the funds.
    status : str
        Gateway-native status string. Will be normalised to ``TransactionStatus``
        by the reconciliation engine before comparison.
    counterparty : str
        Customer identifier as known to the gateway — typically the customer's
        masked phone, email, or VPA (UPI handle).
        Example: ``"user@upi"``, ``"+91XXXXXX7890"``
    """

    transaction_id : str          = Field(..., description="Gateway-assigned unique settlement ID")
    reference_id   : str          = Field(..., description="Shared payment reference (UTR / RRN / echoed order ID)")
    amount         : Decimal      = Field(..., gt=0, decimal_places=2, description="Net settled amount (after fees)")
    currency       : str          = Field(default="INR", min_length=3, max_length=3, description="ISO-4217 currency code")
    timestamp      : datetime     = Field(..., description="UTC datetime of gateway settlement")
    status         : str          = Field(..., description="Gateway-native status string")
    counterparty   : str          = Field(..., description="Customer identifier as known to the gateway")

    @field_validator("currency")
    @classmethod
    def currency_uppercase(cls, v: str) -> str:
        """Normalise currency code to uppercase regardless of source formatting."""
        return v.upper()

    @field_validator("status")
    @classmethod
    def status_lowercase(cls, v: str) -> str:
        """Normalise status to lowercase for consistent comparison."""
        return v.lower().strip()

    @field_validator("reference_id")
    @classmethod
    def strip_reference(cls, v: str) -> str:
        """Strip surrounding whitespace from reference IDs — a frequent raw-export issue."""
        return v.strip()

    model_config = {
        "json_encoders": {Decimal: str},
        "populate_by_name": True,
    }


class LedgerRecord(BaseModel):
    """
    Represents a single order / payment record from the merchant's internal
    accounting or order-management system (OMS / ERP / custom ledger).

    This model captures what the merchant *recorded* at the time the customer
    placed or completed an order. It is the source-of-truth for the merchant's
    expected cash flow.

    Real-world quirks to be aware of
    ----------------------------------
    - ``reference_id`` is typically the merchant's own order ID or a payment
      reference stored at checkout (e.g. ``"ORD-2024-03-15-001234"``).
      It may or may not contain the same token as the gateway's ``reference_id``;
      truncation, prefix stripping, and case differences are common.
    - ``amount`` is the *gross order* value — what the customer paid before any
      gateway or platform fees were deducted. Consequently it is usually *higher*
      than the corresponding GatewayRecord amount by the fee percentage.
    - ``timestamp`` is the *order creation* or *payment confirmation* time in
      the merchant's local timezone (often IST, not UTC). Off-by-5h30m errors
      are a frequent source of false mismatches.
    - ``status`` uses business-domain terminology:
        "paid"      → customer completed payment (maps to GW "captured")
        "settled"   → funds confirmed received in merchant account
        "pending"   → awaiting payment confirmation
        "failed"    → payment not completed
        "returned"  → customer refund processed internally
    - Missing records arise when payment webhooks were not received or not
      processed correctly by the merchant's backend.

    Attributes
    ----------
    transaction_id : str
        Merchant-internal unique identifier for this ledger entry.
        Example: ``"LED-20240315-001234"``
    reference_id : str
        Payment reference stored at the time of checkout — intended to match
        the gateway's ``reference_id`` but often formatted differently.
        Example: ``"ORD-20240315-123456"`` vs gateway's ``"RAZORPAY20240315123456"``
    amount : Decimal
        Gross order value as recorded at checkout, before any fee deductions.
        Always positive; refund entries carry ``status="returned"``.
    currency : str
        ISO-4217 currency code. Defaults to ``"INR"``.
    timestamp : datetime
        Datetime of order placement or payment confirmation. May be stored in
        IST or as a naive datetime — normalised to UTC on ingestion.
    status : str
        Merchant-internal status string. Normalised to ``TransactionStatus``
        by the reconciliation engine before comparison.
    counterparty : str
        Order or customer reference as used by the merchant's system — often
        an order number or customer ID, not the VPA / phone used by the gateway.
        Example: ``"CUST-98765"``, ``"ORDER-2024-001234"``
    """

    transaction_id : str          = Field(..., description="Merchant-internal ledger entry ID")
    reference_id   : str          = Field(..., description="Payment reference stored at checkout")
    amount         : Decimal      = Field(..., gt=0, decimal_places=2, description="Gross order amount (before fees)")
    currency       : str          = Field(default="INR", min_length=3, max_length=3, description="ISO-4217 currency code")
    timestamp      : datetime     = Field(..., description="Order placement or payment confirmation datetime")
    status         : str          = Field(..., description="Merchant-internal status string")
    counterparty   : str          = Field(..., description="Order / customer reference in the merchant's system")

    @field_validator("currency")
    @classmethod
    def currency_uppercase(cls, v: str) -> str:
        """Normalise currency code to uppercase regardless of source formatting."""
        return v.upper()

    @field_validator("status")
    @classmethod
    def status_lowercase(cls, v: str) -> str:
        """Normalise status to lowercase for consistent comparison."""
        return v.lower().strip()

    @field_validator("reference_id")
    @classmethod
    def strip_reference(cls, v: str) -> str:
        """Strip surrounding whitespace from reference IDs — a frequent raw-export issue."""
        return v.strip()

    model_config = {
        "json_encoders": {Decimal: str},
        "populate_by_name": True,
    }


# ---------------------------------------------------------------------------
# Reconciliation Output Model
# ---------------------------------------------------------------------------

class ReconciliationRecord(BaseModel):
    """
    Captures the outcome of pairing one GatewayRecord against one LedgerRecord.

    Produced by the matching engine (``matcher.py``) after the full matching
    cascade (exact → fuzzy → unmatched residuals) has been run.  One
    ReconciliationRecord is created for every GatewayRecord and every
    LedgerRecord — even those that could not be paired (in which case one of
    the two source fields will be ``None``).

    Downstream consumers
    --------------------
    - ``exceptions.py`` reads ``match_status`` and ``discrepancies`` to
      classify each non-MATCHED record into an actionable exception category.
    - ``report.py`` aggregates these records to compute match rates and
      build the final CSV / JSON / Rich-table report.

    Attributes
    ----------
    gateway_record : Optional[GatewayRecord]
        The gateway-side transaction. ``None`` if this record represents a
        LedgerRecord with no corresponding gateway entry
        (MatchStatus.UNMATCHED_LEDGER).
    ledger_record : Optional[LedgerRecord]
        The ledger-side transaction. ``None`` if this record represents a
        GatewayRecord with no corresponding ledger entry
        (MatchStatus.UNMATCHED_GATEWAY).
    match_status : MatchStatus
        Result of the reconciliation attempt for this pair.
    discrepancies : list[str]
        Names of fields that differ between the two records beyond configured
        tolerances (e.g. ``["amount", "timestamp"]``).  Empty for MATCHED records.
    amount_delta : Optional[Decimal]
        Signed difference (gateway.amount − ledger.amount). Negative means the
        gateway settled less than the ledger expected (typical due to fees).
        ``None`` when one side is missing.
    timestamp_delta_seconds : Optional[float]
        Signed difference in seconds (gateway.timestamp − ledger.timestamp).
        Positive means the gateway timestamp is later (expected for settlement lag).
        ``None`` when one side is missing.
    matched_at : datetime
        UTC datetime when the reconciliation engine produced this record.
    """

    gateway_record           : Optional[GatewayRecord] = Field(default=None)
    ledger_record            : Optional[LedgerRecord]  = Field(default=None)
    match_status             : MatchStatus             = Field(..., description="Outcome of the reconciliation attempt")
    discrepancies            : list[str]               = Field(default_factory=list, description="Fields that differ beyond tolerance")
    amount_delta             : Optional[Decimal]       = Field(default=None, description="gateway.amount − ledger.amount")
    timestamp_delta_seconds  : Optional[float]         = Field(default=None, description="gateway.timestamp − ledger.timestamp in seconds")
    matched_at               : datetime                = Field(..., description="UTC datetime this record was produced")

    @model_validator(mode="after")
    def at_least_one_source(self) -> "ReconciliationRecord":
        """
        Enforce that every ReconciliationRecord has at least one source record.
        A pair with both sides None is logically invalid and indicates a bug
        in the matching engine.
        """
        if self.gateway_record is None and self.ledger_record is None:
            raise ValueError(
                "ReconciliationRecord must have at least one of "
                "gateway_record or ledger_record set."
            )
        return self

    model_config = {
        "json_encoders": {Decimal: str},
        "populate_by_name": True,
    }
