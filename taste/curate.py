"""curate — H6 permutation: the self-editing brain (propose-only).

The agent drafts and proposes working-set mutations; ONLY the owner's
resolve applies them. Propose-never-apply is absolute:

    suggest_for(task_ctx)   → drafts evidence-backed proposals (pure)
    propose(proposal)       → inferred_queue(kind=amendment)  [NEVER applies]
    apply_amendment(...)    → owner resolve path: validate + apply atomically

Validation happens at APPLY time because the corpus can change between
proposal and resolve — a stale target 422s and the resolve still
completes, so nothing half-applies and nothing clogs the queue.

`apply_amendment` is the ONLY writer of working_set rows in the engine,
and it is reachable only from the owner's resolve endpoint.
"""

from __future__ import annotations

from uuid import uuid4

from .core import CONFIDENCE_FLOOR, StandardsEngine
from .models import CurationProposal, LayerState
from .store import Store

DEFAULT_PIN_WEIGHT = 0.8


class Curator:
    def __init__(self, engine: StandardsEngine):
        self.engine = engine
        self.store: Store = engine.store

    # ---------------------------------------------------------- propose (never applies)
    def propose(self, profile_id: str, proposal: CurationProposal) -> str:
        """Queue the mutation for the owner. The `why` rides the payload."""
        if self.store.get_standard(proposal.standard_id) is None:
            raise ValueError(
                f"standard {proposal.standard_id} not found — nothing to "
                f"{proposal.action}")
        iid = f"amd_{uuid4().hex[:8]}"
        self.store.enqueue_inferred(iid, profile_id, "amendment", {
            "action": proposal.action,
            "standard_id": proposal.standard_id,
            "why": proposal.why,
            "weight": proposal.weight,
        })
        return iid

    # ---------------------------------------------------------- apply (owner resolve only)
    def apply_amendment(self, profile_id: str, iid: str, accept: bool,
                        rationale: str | None = None) -> dict:
        """Resolve a queued amendment. Accept applies the mutation —
        validate + mutate + resolve commit atomically; a refusal (stale
        target, full set) still marks the row resolved and changes nothing."""
        row = self.store.get_inferred(iid)
        if row is None:
            raise KeyError(f"queue item {iid} not found")
        if row["resolved"]:
            raise ValueError(f"queue item {iid} is already resolved")
        if row["kind"] != "amendment":
            raise ValueError(
                f"{iid} is a {row['kind']}, not an amendment — resolve it "
                f"on its own surface")
        if not accept:
            self.store.resolve_inferred(iid)
            return {"resolved": iid, "applied": False}

        applied = False
        try:
            with self.store.tx():
                self._mutate(profile_id, row["payload"])
                self.store.resolve_inferred(iid)
                applied = True
        except ValueError:
            # refusal: nothing applied, but the resolve completes
            self.store.resolve_inferred(iid)
            raise
        return {"resolved": iid, "applied": applied,
                "rationale": rationale or ""}

    def _mutate(self, profile_id: str, payload: dict):
        action = payload["action"]
        sid = payload["standard_id"]
        s = self.store.get_standard(sid)
        if action == "pin":
            if s is None:
                raise ValueError(f"standard {sid} not found")
            if s.state != LayerState.ACTIVE:
                raise ValueError(
                    f"{sid} is {s.state.value} — only active layers pin")
            if s.confidence < CONFIDENCE_FLOOR:
                raise ValueError(
                    f"{sid} confidence {s.confidence:.2f} is below the B7 "
                    f"floor ({CONFIDENCE_FLOOR}) — not pin-able")
            self.store.put_ws_entry(
                profile_id, sid,
                weight=payload.get("weight") or DEFAULT_PIN_WEIGHT)
        elif action == "demote":
            members = {w.standard_id
                       for w in self.store.get_working_set(profile_id)}
            if sid not in members:
                raise ValueError(f"{sid} is not in the working set")
            self.store.purge_ws_standard(profile_id, sid)
        elif action == "suggest_retire":
            if s is None:
                raise ValueError(f"standard {sid} not found")
            if s.state not in (LayerState.PROVISIONAL, LayerState.ACTIVE):
                raise ValueError(
                    f"{sid} is already {s.state.value} — nothing to retire")
            self.engine.retire(sid)  # also purges its WS slot
        else:
            raise ValueError(f"unknown curation action: {action}")

    # ---------------------------------------------------------- the self-editing brain
    def suggest_for(self, profile_id: str, task_ctx: dict,
                    top_n: int = 3) -> list[CurationProposal]:
        """Draft curation proposals from retrieval evidence — pure function.

        + pin: retrieval hits for this task family that are not yet
          committed to the working set (weight = their similarity)
        + demote: working-set members that decayed below the B7 floor —
          dead weight the agent is carrying

        Drafting enqueues nothing and applies nothing; the agent submits
        drafts through `propose()`, the owner resolves.
        """
        query = " ".join(
            str(v) for v in task_ctx.values() if isinstance(v, str)
        ) or "general task"
        members = self.store.get_working_set(profile_id)
        member_ids = {w.standard_id for w in members}

        proposals: list[CurationProposal] = []
        for s, sim in self.engine.retrieve(query, profile_id, top_n=top_n):
            if s.id in member_ids:
                continue
            proposals.append(CurationProposal(
                action="pin", standard_id=s.id,
                why=(f"top retrieval hit (sim {sim:.2f}) for this task "
                     f"family, not yet committed to the working set"),
                weight=max(0.1, round(sim, 2))))
            if len(proposals) >= top_n:
                break
        for w in members:
            if w.confidence < CONFIDENCE_FLOOR:
                proposals.append(CurationProposal(
                    action="demote", standard_id=w.standard_id,
                    why=(f"decayed to {w.confidence:.2f} — below the B7 "
                         f"floor, dead weight in the working set")))
        return proposals
