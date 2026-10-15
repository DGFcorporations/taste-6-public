"""ingest — cold-start pipeline: docs → candidate layers + restatements.

Deterministic extraction (the control's bias — LLM-assisted extraction
is a later axis to test). Rule-bearing lines become provisional standards;
nothing activates here, ever (contract NEVER rules).

A5 divergence: numbered runs (≥2 consecutive numbered items, unbroken by
prose) fuse into ONE procedural standard carrying its steps in
variant_meta — a standard can be a multi-step pattern, not just a rule.
Bulleted lines keep control semantics: bullets list independent rules
(often polarity-mixed), numbers signal an ordered process.
"""

from __future__ import annotations

import json
import re

from .comprehend import restate
from .models import (
    Category,
    IngestResponse,
    Polarity,
    Provenance,
    Restatement,
    Standard,
    make_steps,
)

NEVER_RE = re.compile(
    r"\b(never|avoid|do not|don't|no\b.*\bever|ban|cut|ditch|scrap)\b", re.I)
WHY_RE = re.compile(r"\b(because|why|reason|since)\b", re.I)
RULE_RE = re.compile(
    r"\b(never|always|avoid|prefer|lead|keep|use|must|should|"
    r"cut|ban|require|start|end|write|organize|verify)\b", re.I)
BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
NUM_ITEM_RE = re.compile(r"^\s*\d+[.)]\s+(.*)$")

CAT_HINTS = [
    (Category.ANTI, re.compile(r"slop|generic|bloat|never|forbidden", re.I)),
    (Category.STRUCTURE, re.compile(
        r"structure|organiz|lead|order|first|section|hook|close", re.I)),
    (Category.AESTHETIC, re.compile(
        r"tone|voice|style|rhythm|sentence|word|punchy|warm", re.I)),
    (Category.PROCEDURAL, re.compile(
        r"step|process|protocol|verify|check|test|commit|handoff", re.I)),
    (Category.DOCTRINE, re.compile(
        r"principle|doctrine|belief|value|rule\b", re.I)),
]


def _classify(line: str, section: str) -> tuple[Category, Polarity]:
    text = line.lower()
    for cat, rx in CAT_HINTS:
        if rx.search(section) or rx.search(text):
            category = cat
            break
    else:
        category = Category.DOCTRINE
    if NEVER_RE.search(text):
        return (Category.ANTI if category == Category.DOCTRINE else category,
                Polarity.NEVER)
    if WHY_RE.search(text):
        return category, Polarity.WHY
    return category, Polarity.DO


def _mk(profile_id: str, category: Category, polarity: Polarity,
        directive: str, rationale: str, source: str, **meta) -> Standard:
    return Standard(
        profile_id=profile_id, category=category, polarity=polarity,
        directive=directive.rstrip("."), rationale=rationale,
        provenance=Provenance.PERSONA,
        variant_meta={"source_doc": source, **meta})


# ------------------------------------------------------------------ A5 run fusing

def _split_runs(text: str) -> tuple[list[tuple[list[str], str]], list[tuple[str, str]]]:
    """(numbered runs, atomic lines) — prose breaks a run; a run of one
    falls back to the atomic pipeline."""
    runs: list[tuple[list[str], str]] = []
    atomics: list[tuple[str, str]] = []
    section = ""
    current: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            section = line.lstrip("#").strip()
            continue
        m = NUM_ITEM_RE.match(line)
        if m:
            current.append(m.group(1).strip())
            continue
        # non-numbered line breaks the run
        if current:
            if len(current) >= 2:
                runs.append((current, section))
            else:
                atomics.append((current[0], section))
            current = []
        if BULLET_RE.match(line):
            line = BULLET_RE.sub("", line)
        if RULE_RE.search(line) and 4 <= len(line.split()) <= 80:
            atomics.append((line.rstrip("."), section))
    if current:
        if len(current) >= 2:
            runs.append((current, section))
        else:
            atomics.append((current[0], section))
    return runs, atomics


def _fuse_run(profile_id: str, doc_id: str, steps: list[str],
              section: str) -> Standard | None:
    """Fuse a numbered run into one procedural standard.

    Returns None when step-length filtering thins the run below two
    steps — the caller falls back to atomic extraction so a single
    surviving item never masquerades as a 'pattern'.
    """
    steps = [s.rstrip(".") for s in steps if 4 <= len(s.split()) <= 80]
    if len(steps) < 2:
        return None
    first = steps[0]
    directive = (first if len(first.split()) >= 4
                 else f"Follow the {len(steps)}-step pattern")
    return Standard(
        profile_id=profile_id, category=Category.PROCEDURAL,
        polarity=Polarity.DO, directive=directive,
        rationale="persona-declared process",
        provenance=Provenance.PERSONA,
        variant_meta={"source_doc": doc_id, "source_section": section,
                      "pattern": "numbered-run",
                      "steps": make_steps(steps)})


# ------------------------------------------------------------------ profile

def _ingest_profile(profile_id: str, doc_id: str, data: dict
                    ) -> list[Standard]:
    """taste_profile.json → candidate standards.

    Mapping (from the Cognitive Scaffolding blueprint):
      veto_triggers[]            → NEVER/anti guardrail layers
      axioms{k:v}                → DO/doctrine (string v = directive text)
      author_profile             → DO/doctrine (decision speed, risk)
      aesthetic_and_tone{k:v}    → DO/aesthetic ("voice: punchy, direct")
      agent_lanes{lane:[caps]}   → procedural, lane-tagged for packet scoping
    """
    out: list[Standard] = []
    why = "persona-declared preference"

    for trig in data.get("veto_triggers", []):
        out.append(_mk(profile_id, Category.ANTI, Polarity.NEVER,
                       trig, why, doc_id, trigger="veto"))

    for k, v in (data.get("axioms") or {}).items():
        if v is True:
            d = k.replace("_", " ")
        elif v is False or v is None:
            continue
        elif isinstance(v, (int, float)):
            d = f"{k.replace('_', ' ')}: {v}"
        else:
            d = str(v)
        out.append(_mk(profile_id, Category.DOCTRINE, Polarity.DO,
                       d, why, doc_id, axiom=k))

    for k, v in (data.get("author_profile") or {}).items():
        out.append(_mk(profile_id, Category.DOCTRINE, Polarity.DO,
                       f"{k.replace('_', ' ')}: {v}", why, doc_id,
                       profile_key=k))

    for k, v in (data.get("aesthetic_and_tone") or {}).items():
        out.append(_mk(profile_id, Category.AESTHETIC, Polarity.DO,
                       f"{k.replace('_', ' ')}: {v}", why, doc_id,
                       aesthetic_key=k))

    for lane, caps in (data.get("agent_lanes") or {}).items():
        out.append(_mk(profile_id, Category.PROCEDURAL, Polarity.DO,
                       f"{lane} lane operates: {', '.join(caps)}",
                       why, doc_id, lane=lane, capabilities=caps))

    return out


def ingest_docs(profile_id: str, docs: list[dict[str, str]]
                ) -> IngestResponse:
    candidates: list[Standard] = []
    restatements: list[Restatement] = []

    for doc in docs:
        text = doc.get("text", "")
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict) and (
                "axioms" in parsed or "veto_triggers" in parsed):
            new = _ingest_profile(profile_id, doc.get("id", ""), parsed)
            candidates.extend(new)
            restatements.extend(restate(s) for s in new)
            continue

        runs, atomics = _split_runs(text)
        for steps, section in runs:
            fused = _fuse_run(profile_id, doc.get("id", ""), steps, section)
            if fused is None:
                # step-length filter thinned the run below a pattern —
                # the original items fall back to atomic extraction
                atomics.extend((s, section) for s in steps)
                continue
            candidates.append(fused)
            restatements.append(restate(fused))

        for line, section in atomics:
            category, polarity = _classify(line, section)
            rationale = ""
            m = re.search(r"because\s+(.+)$", line, re.I)
            if m:
                line, rationale = line[: m.start()].strip(" .—-"), m.group(1)
            if not rationale:
                # self-described taste: "you declared it" is the honest why;
                # provisional anyway until owner confirmation refines it
                rationale = "persona-declared preference"
            std = Standard(
                profile_id=profile_id, category=category, polarity=polarity,
                directive=line, rationale=rationale,
                provenance=Provenance.PERSONA,
                variant_meta={"source_doc": doc.get("id", ""),
                              "source_section": section})
            candidates.append(std)
            restatements.append(restate(std))

    return IngestResponse(candidates=candidates, restatements=restatements)
