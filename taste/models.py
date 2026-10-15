"""taste-6 wire shapes — mirrors contract v1 (taste-bench/contract/models.py).

Self-contained by design: variants are independent implementations, the
conformance suite is the arbiter of contract-equivalence. If contract bumps,
this file updates deliberately.

taste-6 additions over the control shapes:
  + PacketRequest.turn_ctx        — I4 per-turn injection (optional)
  + CurationProposal/CurateRequest — H6 working-set mutations (why REQUIRED)
  + ResolveRequest                — owner resolve gains accept/rationale
  + WorkingSet* models            — the bounded curated context
  + make_steps/get_steps          — A5 procedural patterns in variant_meta
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def _now() -> datetime:
    return datetime.now(UTC)


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _require_rationale(v: str) -> str:
    """NEVER rule 1: a rationale must carry content, not whitespace."""
    if not v or not v.strip():
        raise ValueError("rationale is required (NEVER rule 1)")
    return v


class Polarity(str, Enum):
    DO = "do"
    NEVER = "never"
    WHY = "why"


class Category(str, Enum):
    STRUCTURE = "structure"
    AESTHETIC = "aesthetic"
    PROCEDURAL = "procedural"
    DOCTRINE = "doctrine"
    ANTI = "anti"


class Provenance(str, Enum):
    PERSONA = "persona"
    DECISION = "decision"
    CORRECTION = "correction"
    INFERENCE = "inference"
    AMENDMENT = "amendment"


class LayerState(str, Enum):
    PROVISIONAL = "provisional"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETIRED = "retired"


class Standard(BaseModel):
    id: str = Field(default_factory=lambda: _id("std"))
    profile_id: str = "owner"
    category: Category
    polarity: Polarity
    directive: str
    rationale: str = ""
    confidence: float = 0.5
    provenance: Provenance = Provenance.PERSONA
    state: LayerState = LayerState.PROVISIONAL
    supersedes: list[str] = Field(default_factory=list)
    created: datetime = Field(default_factory=_now)
    updated: datetime = Field(default_factory=_now)
    variant_meta: dict[str, Any] = Field(default_factory=dict)


class Directive(BaseModel):
    polarity: Polarity
    text: str
    standard_id: str | None = None


class Exemplar(BaseModel):
    text: str
    why: str = ""


class Constraint(BaseModel):
    hard: bool = True
    text: str


class Packet(BaseModel):
    profile_id: str = "owner"
    task_ctx: dict[str, Any] = Field(default_factory=dict)
    directives: list[Directive] = Field(default_factory=list)
    exemplars: list[Exemplar] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    free_degrees: list[str] = Field(default_factory=list)
    variant_meta: dict[str, Any] = Field(default_factory=dict)


class Candidate(BaseModel):
    label: str
    content_ref: str = ""
    content: str | None = None
    score: float = 0.0


class DecisionRecord(BaseModel):
    id: str = Field(default_factory=lambda: _id("dec"))
    profile_id: str = "owner"
    task_ctx: dict[str, Any] = Field(default_factory=dict)
    candidates: list[Candidate] = Field(default_factory=list)
    pick: str
    rationale: str
    agent_role: str = ""
    provenance: str = "owner_pick"
    ts: datetime = Field(default_factory=_now)
    variant_meta: dict[str, Any] = Field(default_factory=dict)

    _rationale_required = field_validator("rationale")(_require_rationale)


class InterviewRequest(BaseModel):
    profile_id: str = "owner"
    task_ctx: dict[str, Any] = Field(default_factory=dict)


class InterviewQuestion(BaseModel):
    id: str
    text: str
    options: list[str] = Field(default_factory=list)
    resolves: str = ""


class InterviewResponse(BaseModel):
    questions: list[InterviewQuestion] = Field(default_factory=list)
    coverage: dict[str, list[str]] = Field(
        default_factory=lambda: {"resolved": [], "gaps": []}
    )


class MirrorRequest(BaseModel):
    profile_id: str = "owner"
    task_ctx: dict[str, Any] = Field(default_factory=dict)
    interview_answers: dict[str, str] = Field(default_factory=dict)


class MirrorResponse(BaseModel):
    mirror: str
    assumptions: list[str] = Field(default_factory=list)
    vetoes_hit: list[str] = Field(default_factory=list)
    inferred: bool = False
    gate: str = "required"


class PacketRequest(BaseModel):
    profile_id: str = "owner"
    task_ctx: dict[str, Any] = Field(default_factory=dict)
    answers: dict[str, str] = Field(default_factory=dict)
    guidelines: list[str] = Field(default_factory=list)
    # I4: present → slim per-turn bundle; absent → full task packet
    turn_ctx: dict[str, Any] | None = None


class ScoreRequest(BaseModel):
    profile_id: str = "owner"
    task_ctx: dict[str, Any] = Field(default_factory=dict)
    candidates: list[Candidate]


class CandidateScore(BaseModel):
    ref: str
    score: float
    breakdown: dict[str, float] = Field(default_factory=dict)
    violations: list[str] = Field(default_factory=list)


class ScoreResult(BaseModel):
    scores: list[CandidateScore] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    profile_id: str = "owner"
    task_ctx: dict[str, Any] = Field(default_factory=dict)
    pick: str
    rationale: str = Field(min_length=1)  # empty → 422, NEVER rule 1
    rejected: list[str] = Field(default_factory=list)
    # D2 coupling surface: optional candidate CONTENTS — present → the
    # layers pulled toward the winner are reinforced; absent → no
    # confidence moves (the wire never fabricates evidence)
    candidates: list[Candidate] = Field(default_factory=list)

    _rationale_required = field_validator("rationale")(_require_rationale)


class FeedbackResponse(BaseModel):
    accepted: bool
    record_id: str | None = None
    extracted: dict[str, int] = Field(default_factory=dict)


class HoneRequest(BaseModel):
    profile_id: str = "owner"
    segment_ctx: dict[str, Any] = Field(default_factory=dict)
    draft: str | None = None


class HoneResponse(BaseModel):
    candidates: list[Candidate] = Field(default_factory=list)
    pick: str = ""
    score: float = 0.0
    iterate: bool = False


class IngestRequest(BaseModel):
    profile_id: str = "owner"
    docs: list[dict[str, str]] = Field(default_factory=list)


class Restatement(BaseModel):
    layer: Standard
    implications: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    candidates: list[Standard] = Field(default_factory=list)
    restatements: list[Restatement] = Field(default_factory=list)


class ImportRequest(BaseModel):
    profile_id: str = "owner"
    standards: list[Standard] = Field(default_factory=list)


class MetaResponse(BaseModel):
    name: str
    version: str
    contract: str = "v1"
    permutation: dict[str, str] = Field(default_factory=dict)


# ------------------------------------------------------------------ H6 — curation

CURATION_ACTIONS = ("pin", "demote", "suggest_retire")


class CurationProposal(BaseModel):
    """A working-set mutation the AGENT proposes — the owner resolves it.

    `why` is mandatory: the rationale law generalizes to curation.
    """

    action: str = Field(pattern="^(pin|demote|suggest_retire)$")
    standard_id: str
    why: str
    weight: float | None = Field(default=None, gt=0, le=1)

    _why_required = field_validator("why")(_require_rationale)


class CurateRequest(BaseModel):
    profile_id: str = "owner"
    proposal: CurationProposal


class ResolveRequest(BaseModel):
    accept: bool = False
    rationale: str | None = None


# ------------------------------------------------------------------ H6 — working set

class WorkingSetEntry(BaseModel):
    profile_id: str = "owner"
    standard_id: str
    slot: int
    weight: float = Field(default=1.0, gt=0, le=1)
    pinned_at: datetime = Field(default_factory=_now)


class WorkingSetLayerInfo(BaseModel):
    """A WS row joined to its layer — the only view packets and owners get."""

    slot: int
    standard_id: str
    weight: float
    pinned_at: datetime
    directive: str
    category: Category
    polarity: Polarity
    state: LayerState
    confidence: float


class WorkingSetResponse(BaseModel):
    entries: list[WorkingSetLayerInfo] = Field(default_factory=list)
    count: int = 0


# ------------------------------------------------------------------ A5 — steps

def make_steps(directives: list[str]) -> list[dict[str, Any]]:
    """A5: a standard can be a multi-step pattern, not just a rule."""
    return [{"step": i + 1, "directive": d} for i, d in enumerate(directives)]


def get_steps(s: Standard) -> list[dict[str, Any]]:
    """Steps ride in variant_meta (contract portability) — list order is law."""
    return list(s.variant_meta.get("steps") or [])
