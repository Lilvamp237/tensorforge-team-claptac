"""Shared contract between the ML pipeline (Person A) and every adapter (Person B). Change only by agreement."""

from typing import Literal

from pydantic import BaseModel, Field

from .config import MODEL_VERSION


class Risk(BaseModel):
    label: Literal["low", "medium", "high"]
    probability: float


class Content(BaseModel):
    type: Literal["local_job", "foreign_job", "student_visa", "visa_pr", "unknown"] = "unknown"
    language: str = "unknown"
    claimed_country: str | None = None
    claimed_agency: str | None = None


class Claim(BaseModel):
    claim: str
    status: Literal["consistent", "contradicted", "unverifiable"]
    source: str | None = None


class Verification(BaseModel):
    agency_status: Literal["matched", "not_matched", "unknown"] = "unknown"
    claims: list[Claim] = Field(default_factory=list)


class Novelty(BaseModel):
    score: float = 0.0
    status: Literal["familiar", "unfamiliar_pattern"] = "familiar"


class Campaign(BaseModel):
    campaign_id: str
    similar_items: int


class ScanResult(BaseModel):
    scan_id: str
    source: Literal["text", "url", "file", "collector"]
    url: str | None = None
    risk: Risk
    model_probability: float | None = None  # text-model score before the official-record adjustment
    needs_review: bool
    content: Content
    indicators: list[str]
    verification: Verification
    novelty: Novelty
    campaign: Campaign | None = None
    model_version: str = MODEL_VERSION
    disclaimer: str = "This is a risk assessment, not a confirmed fraud determination."
