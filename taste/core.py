"""core — the standards engine (taste-6: B7 confidence-gated retrieval,
evidence-driven confidence, working-set purge hooks, ONE feedback handler).

The state machine is the law:

    provisional → active → superseded|retired
       (owner confirm)      (correction wins)   (never deleted — NEVER 5)

B7 is the variant's spine: below-floor actives are excluded from retrieval
ENTIRELY — do, why, never alike. Owner confirmation seeds a layer AT the
floor, so confirmed taste is injectable from the first moment; only
decayed or imported-low layers drop out.

`record_feedback` is the single decision path — api, mcp_server and cli
all call it, so reinforce/decay can never diverge by surface.
"""

from __future__ import annotations

import re

from .embed import cosine, embed
from .models import (
    Candidate,
    Category,
    DecisionRecord,
    FeedbackResponse,
    LayerState,
    Polarity,
    Provenance,
    Standard,
)
from .store import Store

CONFIDENCE_FLOOR = 0.55   # B7: state_confidence floor — quality gate AS retrieval
TOP_N = 12                # retrieval window (held with control)
REINFORCE_DELTA = 0.05    # D2: evidence moves confidence in small steps


def locally_approved(standard: Standard) -> bool:
    """A portable ``state: active`` is not local owner approval.

    Import stamps ``variant_meta.local_approved = False``. The engine
    contract says import never activates anything. Rows this engine
    confirmed itself omit the flag, so older confirmed corpora keep
    applying.
    """
    return standard.variant_meta.get("local_approved") is not False


class StandardsEngine:
    def __init__(self, store: Store):
        self.store = store

    # ---------------------------------------------------------- write path
    def propose(self, standard: Standard) -> Standard:
        """New layers ALWAYS arrive provisional — propose-never-apply."""
        standard.state = LayerState.PROVISIONAL
        vec = embed([standard.directive + " " + standard.rationale])[0]
        self.store.put_standard(standard, vec)
        return standard

    def confirm(self, sid: str, rationale: str | None = None) -> bool:
        """Owner confirms a comprehension restatement → layer activates.

        D2: confirmation is direct evidence — confidence seeds at the
        B7 floor so confirmed taste reaches packets immediately.
        """
        s = self.store.get_standard(sid)
        if not s:
            return False
        if rationale is not None:
            s.rationale = rationale
            self.store.set_rationale(sid, rationale)
        if not s.rationale.strip():
            raise ValueError(f"{sid}: cannot activate without rationale")
        meta = dict(s.variant_meta)
        meta["local_approved"] = True
        with self.store.tx():
            self.store.set_variant_meta(sid, meta)
            self.store.set_confidence(sid, max(s.confidence, CONFIDENCE_FLOOR))
            self.store.set_state(sid, LayerState.ACTIVE)
        return True

    def supersede(self, old_id: str, new: Standard) -> Standard:
        """Correction wins: old layer marked superseded (kept forever),
        new layer links back via supersedes. Key rule: a supersede purges
        the working-set slot automatically."""
        new.supersedes = [old_id]
        new.state = LayerState.PROVISIONAL
        self.propose(new)
        old = self.store.get_standard(old_id)
        with self.store.tx():
            self.store.set_state(old_id, LayerState.SUPERSEDED)
            if old:
                self.store.purge_ws_standard(old.profile_id, old_id)
        return new

    def import_standards(self, profile_id: str, standards: list[Standard]) -> dict:
        """Portable import. Declared state is kept. Nothing starts applying.

        The engine contract forbids api.py from writing store rows. This
        is the only import writer. ``local_approved`` stays false until
        ``confirm``.
        """
        existing = {
            (s.profile_id, re.sub(r"\s+", " ", s.directive.lower()))
            for s in self.store.list_standards(profile_id)
        }
        imported = 0
        for s in standards:
            key = (s.profile_id, re.sub(r"\s+", " ", s.directive.lower()))
            if key in existing:
                continue
            existing.add(key)
            if (s.provenance in (Provenance.AMENDMENT, Provenance.INFERENCE)
                    or not s.rationale.strip()):
                s.state = LayerState.PROVISIONAL
            meta = dict(s.variant_meta)
            meta["local_approved"] = False
            s.variant_meta = meta
            self.store.put_standard(
                s, embed([s.directive + " " + s.rationale])[0])
            if s.state != LayerState.ACTIVE:
                self.store.purge_ws_standard(s.profile_id, s.id)
            imported += 1
        return {"imported": imported, "skipped": len(standards) - imported}

    def retire(self, sid: str) -> bool:
        s = self.store.get_standard(sid)
        if not s:
            return False
        with self.store.tx():
            self.store.set_state(sid, LayerState.RETIRED)
            self.store.purge_ws_standard(s.profile_id, sid)
        return True

    def reinforce(self, sid: str, delta: float):
        """Confirmed picks boost; contradictions decay. Bounded [0, 1]."""
        s = self.store.get_standard(sid)
        if s:
            self.store.set_confidence(
                sid, min(1.0, max(0.0, s.confidence + delta)))

    # ---------------------------------------------------------- read path
    def retrieve(self, query: str, profile_id: str = "owner",
                 category: str | None = None, top_n: int = TOP_N
                 ) -> list[tuple[Standard, float]]:
        """B7: cosine top-N over active, at-or-above-floor standards.

        Below-floor actives never appear here — in any polarity. The
        audit surfaces (mirror veto, score never-check) read
        `active()` directly and stay ungated by design.
        """
        cat = Category(category) if category else None
        rows = self.store.list_standards(
            profile_id, state=LayerState.ACTIVE, category=cat,
            with_vectors=True)
        candidates = [(s, v) for s, v in rows
                      if v is not None and s.confidence >= CONFIDENCE_FLOOR
                      and locally_approved(s)]
        if not candidates:
            return []
        qv = embed([query])[0]
        scored = [(s, cosine(qv, v)) for s, v in candidates]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_n]

    def active(self, profile_id: str = "owner") -> list[Standard]:
        return [s for s in self.store.list_standards(
            profile_id, state=LayerState.ACTIVE) if locally_approved(s)]

    def coverage(self, profile_id: str, task_query: str) -> dict:
        """What taste already resolves for this task + where the gaps are.
        Feeds the adaptive interview (E1)."""
        hits = self.retrieve(task_query, profile_id)
        by_cat: dict[str, int] = {}
        for s, sim in hits:
            if sim > 0.45:
                by_cat[s.category.value] = by_cat.get(s.category.value, 0) + 1
        all_cats = {"structure", "aesthetic", "procedural", "doctrine", "anti"}
        return {
            "resolved": sorted(by_cat.keys()),
            "gaps": sorted(all_cats - set(by_cat)),
        }

    # ---------------------------------------------------------- feedback (the one handler)
    def record_feedback(self, profile_id: str, task_ctx: dict, pick: str,
                        rationale: str, rejected: list[str] | None = None,
                        candidates: list[Candidate] | None = None,
                        ) -> FeedbackResponse:
        """Owner decision → corpus atom + provisional why-layer.

        D2 coupling: when candidate CONTENT is available, the do/why
        layers pulled toward the winner are reinforced and the ones
        pulled toward a rejected candidate decay. No content → no
        confidence movement (the wire never fabricates evidence).
        """
        rejected = rejected or []
        candidates = candidates or []

        rec = DecisionRecord(
            profile_id=profile_id, task_ctx=task_ctx,
            candidates=candidates, pick=pick, rationale=rationale,
            agent_role=str(task_ctx.get("role", "")),
            provenance="owner_pick")
        self.store.put_decision(rec)

        self._feedback_confidence(profile_id, pick, rejected, candidates)

        # every owner pick becomes a provisional why-layer — the rationale
        # is the reusable pattern
        layer = Standard(
            profile_id=profile_id, category=Category.DOCTRINE,
            polarity=Polarity.WHY,
            directive=(f"When choosing for "
                       f"{task_ctx.get('type', 'this kind of task')},"
                       f" prefer what won as '{pick}'"),
            rationale=rationale, provenance=Provenance.DECISION,
            variant_meta={"decision_id": rec.id})
        self.propose(layer)

        return FeedbackResponse(accepted=True, record_id=rec.id,
                                extracted={"candidates": 1})

    def _feedback_confidence(self, profile_id: str, pick: str,
                             rejected: list[str], candidates: list[Candidate]):
        by_label = {c.label: c for c in candidates}
        winner = by_label.get(pick)
        loser_texts = [c.content or c.content_ref
                       for r in rejected
                       if (c := by_label.get(r)) is not None
                       and (c.content or c.content_ref)]
        win_text = (winner.content or winner.content_ref
                    if winner is not None else "")
        if not winner or not win_text or not loser_texts:
            return
        active_do_why = [s for s in self.active(profile_id)
                         if s.polarity in (Polarity.DO, Polarity.WHY)]
        if not active_do_why:
            return
        layer_vecs = embed(
            [s.directive + " " + s.rationale for s in active_do_why])
        win_vec = embed([win_text])[0]
        loser_vecs = embed(loser_texts)
        for s, lv in zip(active_do_why, layer_vecs, strict=True):
            sim_win = cosine(lv, win_vec)
            sim_lose = max(cosine(lv, x) for x in loser_vecs)
            if sim_win > sim_lose + 1e-9:
                self.reinforce(s.id, REINFORCE_DELTA)
            elif sim_lose > sim_win + 1e-9:
                self.reinforce(s.id, -REINFORCE_DELTA)
