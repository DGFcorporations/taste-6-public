"""hone — F1 permutation: best-of-3, taste-scored (held with control).

Generate 3 candidates → score → pick → iterate flag if under threshold.
The driver (candidate generator) is pluggable: real wiring calls the flash
model per segment; the offline default perturbs the draft deterministically
so the loop is fully testable without model spend.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable

from .core import StandardsEngine
from .inject import compile_packet
from .models import (
    Candidate,
    HoneRequest,
    HoneResponse,
    PacketRequest,
)
from .score import score_candidates

SCORE_THRESHOLD = 0.55
N_CANDIDATES = 3
HONE_LIMIT = 15


def task_key(ctx: dict) -> str:
    """Segments of one task share a hone budget. Attendance does not reset it."""
    for field in ("task_ref", "ref"):
        value = ctx.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    relevant = {key: value for key, value in ctx.items() if key not in {"segment", "attended"}}
    canonical = json.dumps(relevant, sort_keys=True, separators=(",", ":"), default=str)
    return "task_" + hashlib.sha256(canonical.encode()).hexdigest()

Driver = Callable[[str | None, dict, list[str]], list[str]]

HEDGE_WORDS = ["just", "really", "very", "basically", "actually",
               "in today's world", "it's important to note"]


def _offline_driver(draft: str | None, packet_directives: list[str],
                    segment: str) -> list[str]:
    """Deterministic 3-candidate generator — conformance-safe, zero spend.

    A: tightened draft (hedges stripped).  B: restructured (sentences
    reordered, longest-first → verdict-first bias).  C: draft + first
    do-directive applied as a prefix frame.
    """
    base = (draft or "").strip()
    tight = base
    for w in HEDGE_WORDS:
        tight = re.sub(rf"\b{re.escape(w)}\b[:,]?\s*", "", tight,
                       flags=re.I)
    tight = re.sub(r"\s+", " ", tight).strip()

    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", base) if s.strip()]
    reordered = " ".join(sorted(sents, key=len, reverse=True)) or base

    framed = base
    if packet_directives:
        framed = f"{packet_directives[0].rstrip('.')} — {base}"

    return [tight or base, reordered or base, framed or base]


def hone_segment(engine: StandardsEngine, req: HoneRequest,
                 driver: Driver | None = None) -> HoneResponse:
    reservation = engine.store.reserve_hone(
        req.profile_id, task_key(req.segment_ctx), HONE_LIMIT)
    ctx = {**req.segment_ctx, "attended": False}
    packet = compile_packet(engine, PacketRequest(
        profile_id=req.profile_id, task_ctx=ctx))
    do_rules = [d.text for d in packet.directives
                if d.polarity.value == "do"]

    gen = driver or (lambda draft, _ctx, rules:
                     _offline_driver(draft, rules,
                                     req.segment_ctx.get("segment", "")))
    raw = gen(req.draft, req.segment_ctx, do_rules)

    candidates = [Candidate(label=chr(65 + i), content=t)
                  for i, t in enumerate(raw[:N_CANDIDATES])]
    scored = score_candidates(engine, req.profile_id, req.segment_ctx,
                              candidates)
    best = max(scored, key=lambda s: s.score)

    return HoneResponse(
        candidates=candidates, pick=best.ref, score=best.score,
        iterate=best.score < SCORE_THRESHOLD and reservation < HONE_LIMIT,
    )
