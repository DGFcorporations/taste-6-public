"""segments — J2 permutation: dynamic, agent-shaped decomposition.

The agent's own process taste proposes the segments per task_ctx: a
matching A5 procedural pattern (retrieval-gated, so B7 applies) turns
its steps into the segment plan — the decomposition the agent actually
believes in, not a fixed map. The control's type map remains as fallback.
Each segment gets its own packet + hone budget (~10–15 iterations total
across a task, bound by the driver).
"""

from __future__ import annotations

from .core import StandardsEngine
from .models import get_steps

SEGMENT_MAP: dict[str, list[str]] = {
    "write_thread": ["hook", "body", "close"],
    "write_doc": ["lead", "structure", "detail", "close"],
    "security_review": ["triage", "findings", "verdict"],
    "daily_digest": ["numbers", "flags", "recommendation"],
    "decision_brief": ["verdict", "tradeoffs", "recommendation"],
    "default": ["draft", "refine", "finalize"],
}

MATCH_SIM = 0.30          # looser than the packet gate — segments are structural


def propose_segments(task_ctx: dict, engine: StandardsEngine,
                     profile_id: str = "owner") -> list[str]:
    """J2: procedural taste first, type map as fallback."""
    query = (" ".join(str(v) for v in task_ctx.values()
                      if isinstance(v, str))).strip() or "general task"
    for s, sim in engine.retrieve(query, profile_id, top_n=6):
        steps = get_steps(s)
        if steps and s.category.value == "procedural" and sim > MATCH_SIM:
            return [st["directive"] for st in steps]
    return SEGMENT_MAP.get(str(task_ctx.get("type", "")),
                           SEGMENT_MAP["default"])
