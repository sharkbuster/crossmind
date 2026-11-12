"""Shared domain models (API contracts + internal value objects)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .enums import (
    ApprovalDecision,
    Channel,
    Intent,
    RiskLevel,
    TaskStatus,
    TaskType,
)


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


# --------------------------------------------------------------------- retrieval
class SourceChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str = ""
    uri: str = ""
    text: str
    score: float = 0.0
    embed_score: float = 0.0
    lexical_score: float = 0.0
    rank: int = 0


class Citation(BaseModel):
    index: int
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str = ""
    score: float = 0.0
    text: str = ""


class RetrievedContext(BaseModel):
    chunks: list[SourceChunk] = Field(default_factory=list)
    best_score: float = 0.0
    no_answer: bool = False
    strategy: str = "hybrid"
    degraded: bool = False
    degradation_reason: str = ""
    latency_ms: int = 0


class RagAnswer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    sources: list[SourceChunk] = Field(default_factory=list)
    grounded: bool = False
    no_answer: bool = False
    warnings: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    latency_ms: int = 0


# --------------------------------------------------------------------- business
class Product(BaseModel):
    sku: str
    name: str
    category: str
    price_usd: float
    cost_usd: float
    weight_g: int
    rating: float = 0.0
    monthly_sales: int = 0
    tags: list[str] = Field(default_factory=list)
    description: str = ""

    @property
    def margin(self) -> float:
        if self.price_usd <= 0:
            return 0.0
        return round((self.price_usd - self.cost_usd) / self.price_usd, 4)


# --------------------------------------------------------------------- chat API
class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=64)
    locale: str = Field(default="zh-CN", max_length=16)
    channel: Channel = Channel.AMAZON
    product_sku: str | None = Field(default=None, max_length=64)
    require_citations: bool = True
    options: dict[str, Any] = Field(default_factory=dict)


class AgentStep(BaseModel):
    node: str
    status: str
    detail: str = ""
    latency_ms: int = 0
    tokens: int = 0


class ChatResponse(BaseModel):
    task_id: str
    trace_id: str
    session_id: str
    intent: Intent
    status: TaskStatus
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    sources: list[Citation] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    latency_ms: int = 0
    requires_approval: bool = False
    approval_reason: str = ""
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------- tasks
class TaskRecord(BaseModel):
    id: str
    tenant_id: str
    type: TaskType
    status: TaskStatus
    risk: RiskLevel = RiskLevel.LOW
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    trace_id: str = ""
    error: str = ""
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: ApprovalDecision
    reviewer: str = Field(default="human", max_length=64)
    comment: str = Field(default="", max_length=1000)


# --------------------------------------------------------------------- tracing
class Span(BaseModel):
    span_id: str
    parent_id: str | None = None
    name: str
    start_ms: float
    end_ms: float | None = None
    status: str = "ok"
    attributes: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end_ms is None:
            return 0.0
        return round(self.end_ms - self.start_ms, 3)


class TraceRecord(BaseModel):
    trace_id: str
    tenant_id: str
    route: str
    started_at: str
    duration_ms: float = 0.0
    status: str = "ok"
    spans: list[Span] = Field(default_factory=list)


# --------------------------------------------------------------------- evaluation
class EvalCase(BaseModel):
    id: str
    question: str
    answerable: bool = True
    expected_doc_ids: list[str] = Field(default_factory=list)
    expected_key_points: list[str] = Field(default_factory=list)
    notes: str = ""


class EvalCaseResult(BaseModel):
    id: str
    answerable: bool
    answered: bool
    no_answer: bool
    grounded: bool
    hit: bool
    reciprocal_rank: float
    citation_precision: float
    faithfulness: float
    key_point_coverage: float
    latency_ms: int


class EvalMetrics(BaseModel):
    cases: int
    hit_at_k: float
    mrr: float
    refusal_accuracy: float
    citation_precision: float
    faithfulness: float
    key_point_coverage: float
    avg_latency_ms: float


class EvalReport(BaseModel):
    mode: str
    generated_at: str = Field(default_factory=now_iso)
    top_k: int
    threshold: float
    metrics: EvalMetrics
    results: list[EvalCaseResult] = Field(default_factory=list)
