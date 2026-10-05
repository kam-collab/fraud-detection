"""Request / response models for the scoring API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """One payment in the assignment schema. `request_status` is not required: the fraud
    decision is made before the payment outcome is known."""

    request_id: Optional[str] = Field(None, description="Generated when omitted")
    request_time: Optional[datetime] = Field(None, description="Defaults to the service clock")
    service_type: str = Field(..., examples=["upi"])
    device_id: str = Field(..., examples=["DEV0001234"])
    merchant_id: str = Field(..., examples=["M100101"])
    merchant_state: Optional[str] = None
    merchant_city: Optional[str] = None
    merchant_type: str = Field(..., examples=["low"])
    mcc_code: int = Field(..., examples=[5411])
    mcc_title: Optional[str] = None
    issuer_bank: Optional[str] = None
    currency_code: str = "INR"
    amount: float = Field(..., examples=[850.0])
    request_status: Optional[str] = None


class ReviewDecision(BaseModel):
    decision: Literal["approve", "decline"]
