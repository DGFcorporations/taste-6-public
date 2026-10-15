"""taste-6 MCP server — the fleet interface (contract §5).

Exposes the four contract tools over stdio so any MCP-speaking seat can
reach the engine without an HTTP client. Tools map 1:1 to their HTTP
routes with identical request/response semantics — including I4:
taste_packet accepts turn_ctx and returns the slim per-turn bundle.

    taste_packet    -> POST /packet
    taste_score     -> POST /score
    taste_feedback  -> POST /feedback
    taste_recall    -> GET /standards (filtered)

Run: python -m taste.mcp_server   (env TASTE6_DB selects the store)
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

from mcp.server.mcpserver import MCPServer

from .core import StandardsEngine
from .inject import compile_packet
from .models import (
    Candidate,
    FeedbackRequest,
    PacketRequest,
)
from .score import score_candidates
from .store import Store

TOOL_DESCRIPTIONS = {
    "taste_packet":
        "Compile the taste packet for a task — the bundle that travels "
        "with the work order. Pass turn_ctx for the slim per-turn bundle "
        "(I4): {segment, draft_so_far, turn_index}.",
    "taste_score":
        "Score candidate output(s) against the owner's taste.",
    "taste_feedback":
        "Log an owner decision — rationale REQUIRED. This is how the "
        "corpus grows. Optionally pass candidate contents so taste can "
        "move confidence toward what won.",
    "taste_recall":
        "Recall taste standards for a profile, filtered by state or "
        "category. Identical semantics to GET /standards.",
}


def create_tools(store: Store, engine: StandardsEngine) -> dict[str, Callable]:
    """The four contract tools as plain callables — identical request and
    response semantics to their HTTP routes. create_server registers them;
    tests call them directly."""

    def taste_check(profile_id: str = "owner") -> str:
        active = engine.active(profile_id)
        all_stds = store.list_standards(profile_id)
        cats: dict[str, int] = {}
        pols: dict[str, int] = {}
        for s in active:
            c = s.category.value if hasattr(s.category, "value") else str(s.category)
            p = s.polarity.value if hasattr(s.polarity, "value") else str(s.polarity)
            cats[c] = cats.get(c, 0) + 1
            pols[p] = pols.get(p, 0) + 1
        return json.dumps({
            "status": "healthy",
            "name": "taste-6",
            "active_standards": len(active),
            "total_standards": len(all_stds),
            "categories": cats,
            "polarities": pols,
        })

    def taste_packet(task_ctx: dict, profile_id: str = "owner",
                     answers: dict | None = None,
                     guidelines: list[str] | None = None,
                     turn_ctx: dict | None = None) -> str:
        pkt = compile_packet(engine, PacketRequest(
            profile_id=profile_id, task_ctx=task_ctx,
            answers=answers or {}, guidelines=guidelines or [],
            turn_ctx=turn_ctx))
        return pkt.model_dump_json()

    def taste_score(candidates: list[dict], task_ctx: dict | None = None,
                    profile_id: str = "owner") -> str:
        cands = [Candidate(**c) for c in candidates]
        scored = score_candidates(engine, profile_id, task_ctx or {}, cands)
        return json.dumps({"scores": [s.model_dump(mode="json")
                                      for s in scored]})

    def taste_feedback(pick: str, rationale: str,
                       task_ctx: dict | None = None,
                       profile_id: str = "owner",
                       rejected: list[str] | None = None,
                       candidates: list[dict] | None = None) -> str:
        if not rationale or not rationale.strip():
            return _error_payload(422, "rationale required (NEVER rule 1)")
        req = FeedbackRequest(
            profile_id=profile_id, task_ctx=task_ctx or {}, pick=pick,
            rationale=rationale, rejected=rejected or [],
            candidates=[Candidate(**c) for c in candidates or []])
        resp = engine.record_feedback(
            req.profile_id, req.task_ctx, req.pick, req.rationale,
            rejected=req.rejected, candidates=req.candidates)
        return resp.model_dump_json()

    def taste_recall(profile_id: str = "owner", state: str | None = None,
                     category: str | None = None) -> str:
        if category == "check":
            return taste_check(profile_id)
        from .models import Category, LayerState

        rows = store.list_standards(
            profile_id,
            state=LayerState(state) if state else None,
            category=Category(category) if category else None)
        return json.dumps({"standards": [s.model_dump(mode="json")
                                          for s in rows],
                           "count": len(rows)})

    return {
        "taste_packet": taste_packet,
        "taste_score": taste_score,
        "taste_feedback": taste_feedback,
        "taste_recall": taste_recall,
    }




def _error_payload(status: int, message: str) -> str:
    return json.dumps({"error": {"code": f"e{status}", "message": message}})


def create_server(store: Store, engine: StandardsEngine) -> MCPServer:
    server = MCPServer("taste-6")
    for name, fn in create_tools(store, engine).items():
        server.tool(description=TOOL_DESCRIPTIONS[name])(fn)
    return server


store = Store()
engine = StandardsEngine(store)
server = create_server(store, engine)


def main():
    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
