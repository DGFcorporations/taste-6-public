"""taste-6 engine behavior — scorer/mirror/hone/interview parity with the
control (held axes D1/E1/F1 + mirror), plus the full-loop test once the
API surface lands. Offline: hashing embedder via conftest.
"""

from taste.models import (
    Candidate,
    Category,
    LayerState,
    Polarity,
    Provenance,
    Standard,
)


def _std(directive, polarity=Polarity.DO, rationale="because taste",
         category=Category.DOCTRINE, confidence=0.6):
    return Standard(profile_id="owner", category=category, polarity=polarity,
                    directive=directive, rationale=rationale,
                    state=LayerState.PROVISIONAL,
                    provenance=Provenance.PERSONA, confidence=confidence)


def _confirm_std(engine, directive, **kw):
    s = engine.propose(_std(directive, **kw))
    engine.confirm(s.id)
    return s


# ---------------------------------------------------------------- full loop

PERSONA = """# Persona: Owner
- Never use generic openers like "In today's world".
- Lead with the angle, then the structure, then the build.
- Prefer short declarative sentences because they carry authority.
"""


def test_full_loop(client):
    # 1. ingest persona → provisional candidates
    r = client.post("/ingest", json={
        "profile_id": "owner",
        "docs": [{"id": "p", "text": PERSONA, "format": "md"}]})
    assert r.status_code == 200
    assert len(r.json()["candidates"]) >= 3

    # 2. owner confirms via comprehension gate → active, seeded at floor
    for s in client.get("/standards").json()["standards"]:
        if s["state"] == "provisional":
            client.post(f"/standards/{s['id']}/confirm")
    active = [s for s in client.get("/standards").json()["standards"]
              if s["state"] == "active"]
    assert len(active) >= 3
    assert all(s["confidence"] >= 0.55 for s in active), \
        "confirmed layers are injectable from the first moment (D2)"

    # 3. packet compiles directives from confirmed taste
    pkt = client.post("/packet", json={
        "profile_id": "owner",
        "task_ctx": {"type": "write_doc", "brief": "ship notes"}}).json()
    never_texts = [d["text"] for d in pkt["directives"]
                   if d["polarity"] == "never"]
    assert any("generic openers" in t for t in never_texts)

    # 4. scorer catches the violation taste forbids
    r = client.post("/score", json={
        "profile_id": "owner", "task_ctx": {"type": "write_doc"},
        "candidates": [
            {"label": "A",
             "content": "In today's world, shipping matters. "
                        "We shipped evidence capture; here's what changed."},
            {"label": "B",
             "content": "Shipped: evidence capture. What changed, why it matters."}]})
    scores = {s["ref"]: s for s in r.json()["scores"]}
    assert scores["A"]["violations"], "generic opener must be caught"
    assert scores["A"]["score"] < scores["B"]["score"]

    # 5. hone picks the better candidate offline
    h = client.post("/hone", json={
        "profile_id": "owner",
        "segment_ctx": {"type": "write_doc", "segment": "lead"},
        "draft": "In today's world, things are hard. We shipped a fix."}).json()
    assert h["pick"] and h["candidates"]

    # 6. feedback lands + extracts a provisional why-layer
    before = client.get("/standards").json()["count"]
    r = client.post("/feedback", json={
        "profile_id": "owner", "task_ctx": {"type": "write_doc"},
        "pick": "B", "rationale": "verdict-first beats throat-clearing"})
    assert r.status_code == 200
    assert client.get("/standards").json()["count"] == before + 1


# ---------------------------------------------------------------- D1 scorer

def test_scorer_flags_quoted_never_violation(engine):
    from taste.score import score_candidates

    _confirm_std(engine,
                 'never use generic openers like "in today\'s world"',
                 polarity=Polarity.NEVER, category=Category.ANTI)
    out = score_candidates(engine, "owner", {"type": "write_doc"}, [
        Candidate(label="A",
                 content="In today's world, things are hard. We shipped."),
        Candidate(label="B", content="Shipped: the fix. What changed."),
    ])
    by_ref = {c.ref: c for c in out}
    assert by_ref["A"].violations, "quoted never-rule violation caught"
    assert by_ref["A"].score < by_ref["B"].score, "violation costs score"
    for c in out:
        assert "positive" in c.breakdown and "negative" in c.breakdown, \
            "contract §3: breakdown needs positive + negative"


def test_scorer_breakdown_contract_minimum_on_empty_corpus(engine):
    from taste.score import score_candidates

    out = score_candidates(engine, "owner", {"type": "x"},
                           [Candidate(label="A", content="plain text")])
    assert "positive" in out[0].breakdown
    assert "negative" in out[0].breakdown


# ---------------------------------------------------------------- mirror

def test_mirror_gate_required_when_attended(engine, store):
    from taste.mirror import build_mirror
    from taste.models import MirrorRequest

    resp = build_mirror(engine, store, MirrorRequest(
        profile_id="owner", task_ctx={"type": "write_doc", "attended": True}))
    assert resp.mirror
    assert resp.gate == "required" and resp.inferred is False


def test_mirror_inferred_enqueues_review(engine, store):
    from taste.mirror import build_mirror
    from taste.models import MirrorRequest

    resp = build_mirror(engine, store, MirrorRequest(
        profile_id="owner", task_ctx={"type": "write_doc", "attended": False}))
    assert resp.inferred is True and resp.gate == "queued"
    assert any(p["kind"] == "mirror"
               for p in store.pending_inferred("owner")), \
        "inferred mirrors queue for owner review (NEVER rule 3)"


def test_mirror_elimination_filter_flags_veto_amid_extra_words(engine, store):
    from taste.mirror import build_mirror
    from taste.models import MirrorRequest

    _confirm_std(engine, "never use generic openers in newsletters",
                 polarity=Polarity.NEVER, category=Category.ANTI)
    resp = build_mirror(engine, store, MirrorRequest(
        profile_id="owner",
        task_ctx={"type": "write_doc", "attended": True,
                  "brief": "Write an article using generic openers for our "
                           "newsletter audience about quarterly notes"}))
    assert resp.vetoes_hit, "coverage measured against the RULE, not task length"


def test_mirror_elimination_quiet_on_unrelated_task(engine, store):
    from taste.mirror import build_mirror
    from taste.models import MirrorRequest

    _confirm_std(engine, "never use generic openers in newsletters",
                 polarity=Polarity.NEVER, category=Category.ANTI)
    resp = build_mirror(engine, store, MirrorRequest(
        profile_id="owner",
        task_ctx={"type": "write_doc", "attended": True,
                  "brief": "quarterly invoice summary"}))
    assert resp.vetoes_hit == []


# ---------------------------------------------------------------- F1 hone

def test_hone_bon3_offline_driver_strips_hedges(engine):
    from taste.hone import hone_segment
    from taste.models import HoneRequest

    resp = hone_segment(engine, HoneRequest(
        profile_id="owner",
        segment_ctx={"type": "write_doc", "segment": "lead"},
        draft="In today's world, just basically things are hard. "
              "We shipped a fix."))
    assert len(resp.candidates) == 3, "BoN-3"
    assert resp.pick in {"A", "B", "C"}
    assert any("basically" not in (c.content or "") for c in resp.candidates), \
        "offline driver A: hedges stripped"


def test_hone_iterates_when_score_under_threshold(engine):
    from taste.hone import hone_segment
    from taste.models import HoneRequest

    resp = hone_segment(engine, HoneRequest(
        profile_id="owner", segment_ctx={"type": "write_doc"},
        draft="zzz"))
    assert resp.candidates and resp.pick
    assert resp.iterate is True, "empty corpus: score 0.0 under 0.55 → iterate"


# ---------------------------------------------------------------- E1 interview

def test_interview_ceiling_three_on_empty_corpus(engine):
    from taste.interview import run_interview

    resp = run_interview(engine, "owner", {"type": "write_doc"})
    assert set(resp.coverage) == {"resolved", "gaps"}
    assert len(resp.questions) == 3, "empty corpus: all 5 categories gap, cap 3"
    for q in resp.questions:
        assert q.id and q.text


def test_interview_omits_resolved_gaps(engine):
    from taste.interview import run_interview

    _confirm_std(engine, "tone voice style words carry",
                 category=Category.AESTHETIC)
    resp = run_interview(engine, "owner",
                         {"type": "write_doc", "brief": "voice tone"})
    resolves = {q.resolves for q in resp.questions}
    assert "aesthetic" not in resolves, "never re-ask what taste resolves"
    assert len(resp.questions) == 3, "4 gaps remain, still capped at 3"
