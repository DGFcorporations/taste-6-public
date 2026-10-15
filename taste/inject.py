"""inject — I4 permutation: dual-mode packet compiler.

task mode: control-shaped full bundle (contract §2 shape) — the wire packet.
turn mode: slim bundle — working set by weight FIRST (what I've committed
to leads), then fresh retrieval for THIS turn's ctx, under strict-subset
caps (5/3/1 vs 8/4/2), so the turn packet is structurally ≤ the task
packet for any corpus state.

The working set is a NEW injection path — B7 is enforced again at render
time: pinned layers below the confidence floor never reach a packet,
even though apply_amendment already refused to pin sub-floor targets.
Belt and suspenders; the invariant holds either way.

Deterministic for identical corpus state in both modes (contract §4).
"""

from __future__ import annotations

from .core import CONFIDENCE_FLOOR, StandardsEngine
from .models import (
    Constraint,
    Directive,
    Exemplar,
    Packet,
    PacketRequest,
    Polarity,
)

import re

SIM_THRESHOLD = 0.40          # a do/why layer must be relevant to apply
MAX_DIRECTIVES = 8           # task packet size is bounded — every token is cost
MAX_NEVERS = 4               # guardrails always apply, bounded
MAX_EXEMPLARS = 2

TURN_MAX_DIRECTIVES = 5      # strict subsets of the task caps
TURN_MAX_NEVERS = 3
TURN_MAX_EXEMPLARS = 1
TURN_TOP_N = 6               # slimmer retrieval window for turns

# Domain synonym keywords for boosting relevance
DOMAIN_KEYWORDS = {
    "media": ["video", "audio", "mp4", "render", "youtube", "tiktok", "reels", "voice", "sync", "lip", "thumbnail", "kallaway", "hormozi", "elevenlabs", "qwen", "deepgram", "camera", "frame", "cut"],
    "content": ["copy", "text", "headline", "hook", "story", "framing", "script", "editorial", "email", "angle", "jargon", "fluff"],
    "engineering": ["code", "bug", "fix", "feature", "git", "commit", "test", "tdd", "branch", "refactor", "security", "fail-closed", "clean", "architecture"],
    "systems": ["container", "docker", "service", "systemd", "server", "port", "daemon", "proxy", "cloudflare", "postgres", "redis", "bridge", "rclone"],
    "governance": ["gate", "audit", "compliance", "phi", "legal", "secret", "env", "credential", "permission", "sterile-field"],
}


def score_standard_relevance(s, query: str, tokens: set[str], hit_score: float = 0.0) -> float:
    text = (s.directive + " " + (s.rationale or "")).lower()
    score = hit_score * 10.0
    for t in tokens:
        if len(t) > 3 and t in text:
            score += 3.0
        elif t in text:
            score += 1.0
    q_low = query.lower()
    for domain, kws in DOMAIN_KEYWORDS.items():
        if any(k in q_low for k in kws):
            if any(k in text for k in kws):
                score += 5.0
    conf = float(s.confidence if s.confidence is not None else 0.5)
    return score * (0.5 + conf)



def compile_packet(engine: StandardsEngine,
                   req: PacketRequest) -> Packet:
    if req.turn_ctx is not None:
        return _compile_turn(engine, req)
    return _compile_task(engine, req)


def _query(req: PacketRequest) -> str:
    query = " ".join(
        str(v) for v in req.task_ctx.values() if isinstance(v, str))
    for ans in req.answers.values():
        query += " " + ans
    return query.strip() or "general task"


def _compile_task(engine: StandardsEngine, req: PacketRequest) -> Packet:
    """Control-shaped bundle — identical mechanics to taste-1."""
    query = _query(req)
    hits = engine.retrieve(query, req.profile_id)
    hit_dict = {s.id: sim for s, sim in hits}
    tokens = set(re.findall(r"\w+", query.lower()))

    directives: list[Directive] = []
    exemplars: list[Exemplar] = []

    # Rank NEVER rules by task relevance + confidence, break ties by s.id for determinism
    active_nevers = [
        s for s in engine.active(req.profile_id)
        if s.polarity == Polarity.NEVER
    ]
    ranked_nevers = sorted(
        active_nevers,
        key=lambda s: (score_standard_relevance(s, query, tokens, hit_dict.get(s.id, 0.0)), s.id),
        reverse=True,
    )
    selected_nevers = sorted(ranked_nevers[:MAX_NEVERS], key=lambda s: s.id) if ranked_nevers else []
    directives.extend(
        Directive(polarity=s.polarity, text=s.directive, standard_id=s.id)
        for s in selected_nevers)

    ranked_hits = sorted(
        hits,
        key=lambda item: (score_standard_relevance(item[0], query, tokens, item[1]), item[1]),
        reverse=True,
    )
    for s, sim in ranked_hits:
        if s.polarity == Polarity.NEVER:
            continue  # already emitted from the active set above
        if sim < SIM_THRESHOLD:
            continue
        if s.polarity == Polarity.WHY:
            if len(exemplars) < MAX_EXEMPLARS:
                exemplars.append(Exemplar(text=s.directive, why=s.rationale))
            continue
        if len(directives) >= MAX_DIRECTIVES:
            break
        directives.append(Directive(
            polarity=s.polarity, text=s.directive, standard_id=s.id))

    # owner-supplied task guidelines land as hard constraints — they outrank
    # accumulated taste for THIS task (fresh signal law)
    constraints = [Constraint(hard=True, text=g) for g in req.guidelines]

    # NEVER rule 6 — one bounded free degree minimum, always
    free_degrees = [
        "voice and rhythm within the directives — vary phrasing, "
        "keep the pattern alive"]

    return Packet(
        profile_id=req.profile_id, task_ctx=req.task_ctx,
        directives=directives, exemplars=exemplars,
        constraints=constraints, free_degrees=free_degrees,
        variant_meta={"engine": "taste-6", "mode": "task", "hits": len(hits)})


def _compile_turn(engine: StandardsEngine, req: PacketRequest) -> Packet:
    """Slim, hot bundle — called per model call, not per task."""
    turn_ctx = req.turn_ctx or {}
    query = (_query(req) + " " +
             " ".join(str(v) for v in turn_ctx.values()
                      if isinstance(v, str))).strip() or "current turn"

    # fresh retrieval for THIS turn
    hits = engine.retrieve(query, req.profile_id, top_n=TURN_TOP_N)
    hit_dict = {s.id: sim for s, sim in hits}
    tokens = set(re.findall(r"\w+", query.lower()))

    # working set first, by weight — B7 re-checked at render time
    ws_all = engine.store.get_working_set(req.profile_id)
    ws = [w for w in ws_all if w.confidence >= CONFIDENCE_FLOOR]
    ws_ids = {w.standard_id for w in ws}

    directives: list[Directive] = []
    exemplars: list[Exemplar] = []

    # Rank NEVER rules by task relevance + confidence
    active_nevers = [
        s for s in engine.active(req.profile_id)
        if s.polarity == Polarity.NEVER
    ]
    ranked_nevers = sorted(
        active_nevers,
        key=lambda s: (score_standard_relevance(s, query, tokens, hit_dict.get(s.id, 0.0)), s.id),
        reverse=True,
    )
    selected_nevers = sorted(ranked_nevers[:TURN_MAX_NEVERS], key=lambda s: s.id) if ranked_nevers else []
    nevers = [
        Directive(polarity=s.polarity, text=s.directive, standard_id=s.id)
        for s in selected_nevers
    ]

    for w in ws:
        if w.polarity == Polarity.NEVER:
            continue  # guardrails come from the active set, below
        if w.polarity == Polarity.WHY:
            if len(exemplars) < TURN_MAX_EXEMPLARS:
                exemplars.append(Exemplar(text=w.directive, why=""))
            continue
        if len(directives) >= TURN_MAX_DIRECTIVES:
            break
        directives.append(Directive(
            polarity=w.polarity, text=w.directive, standard_id=w.standard_id))

    # guardrails join after the working set — they always apply but they
    # don't outrank what the owner already committed to
    directives.extend(nevers)

    # fresh retrieval for THIS turn fills the remaining slim slots
    ranked_hits = sorted(
        hits,
        key=lambda item: (score_standard_relevance(item[0], query, tokens, item[1]), item[1]),
        reverse=True,
    )
    for s, sim in ranked_hits:
        if s.id in ws_ids:
            continue
        if s.polarity == Polarity.NEVER:
            continue  # already emitted from the active set above
        if sim < SIM_THRESHOLD:
            continue
        if s.polarity == Polarity.WHY:
            if len(exemplars) < TURN_MAX_EXEMPLARS:
                exemplars.append(Exemplar(text=s.directive, why=s.rationale))
            continue
        if len(directives) >= TURN_MAX_DIRECTIVES:
            break
        directives.append(Directive(
            polarity=s.polarity, text=s.directive, standard_id=s.id))


    constraints = [Constraint(hard=True, text=g) for g in req.guidelines]
    free_degrees = [
        "voice and rhythm within the working set — vary phrasing, "
        "keep the pattern alive"]

    return Packet(
        profile_id=req.profile_id, task_ctx=req.task_ctx,
        directives=directives, exemplars=exemplars,
        constraints=constraints, free_degrees=free_degrees,
        variant_meta={"engine": "taste-6", "mode": "turn",
                      "ws_size": len(ws), "hits": len(hits)})
