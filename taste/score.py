"""score — D1 permutation: rules scorer (held with control).

Two components, both free and deterministic:

  negative — lexical violation check: quoted strings inside `never`
             directives matched exactly; residual directive phrases checked
             by normalized containment. Each hit is a violation.
  positive — cosine of the candidate against the centroid of relevant
             `do`/`why` layers + exemplars: does it sit inside the
             taste-shaped region?

score = positive − 0.5 × violations, clamped to [−1, 1].

B7 note: the positive half inherits the confidence gate through
engine.retrieve(); the never-check reads actives directly (ungated) —
violation DETECTION never silently weakens when a guardrail decays.
"""

from __future__ import annotations

import re

from .core import StandardsEngine
from .embed import cosine, embed
from .models import Candidate, CandidateScore, Polarity

QUOTED = re.compile(
    r'"([^"]{4,})"|“([^”]{4,})”'      # double / smart quotes
    r"|(?<![\w'])'([^']{4,})'(?!\w)"  # single quotes, apostrophe-safe
)
LEADIN = re.compile(r"^(never|avoid|do not|don't|no)\s+", re.I)


def _quoted_strings(text: str) -> list[str]:
    return [next(g for g in m if g) for m in QUOTED.findall(text)]


def _forbidden_strings(directive: str) -> list[str]:
    """Extract the concrete forbidden material from a never-directive."""
    quoted = _quoted_strings(directive)
    stripped = LEADIN.sub("", directive).strip().rstrip(".")
    phrases = quoted + ([stripped] if len(stripped.split()) >= 4 else [])
    return [p.lower() for p in phrases]


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower())


def score_candidates(engine: StandardsEngine, profile_id: str,
                     task_ctx: dict, candidates: list[Candidate]
                     ) -> list[CandidateScore]:
    query = " ".join(str(v) for v in task_ctx.values() if isinstance(v, str))
    hits = engine.retrieve(query or "general", profile_id)

    # guardrails apply unconditionally: every active never-rule is checked
    nevers = [s for s in engine.active(profile_id)
              if s.polarity == Polarity.NEVER]
    positives = [s for s, sim in hits
                 if s.polarity in (Polarity.DO, Polarity.WHY) and sim > 0.3]

    centroid = None
    if positives:
        vecs = embed([s.directive + " " + s.rationale for s in positives])
        dim = len(vecs[0])
        centroid = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]

    out: list[CandidateScore] = []
    for cand in candidates:
        text = cand.content or cand.content_ref
        norm = _norm(text)

        violations: list[str] = []
        for s in nevers:
            for phrase in _forbidden_strings(s.directive):
                if phrase and phrase in norm:
                    violations.append(f"violates {s.id}: '{phrase}'")

        positive = 0.0
        if centroid is not None and text.strip():
            positive = max(0.0, cosine(embed([text])[0], centroid))

        score = max(-1.0, min(1.0, positive - 0.5 * len(violations)))
        out.append(CandidateScore(
            ref=cand.label, score=round(score, 4),
            breakdown={
                "positive": round(positive, 4),
                "negative": float(len(violations)),
                "bloat_veto": float(len(violations)),
                "leverage_fit": round(positive, 4),
                "cost_tokens": float(len(text) // 4),
            },
            violations=violations))
    return out
