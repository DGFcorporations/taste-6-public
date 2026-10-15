"""taste-6 variant-specific checks — each test pins one divergence from
the control (BLUEPRINT "Variant-specific checks" + the mechanics each
depends on). Offline: hashing embedder via conftest.
"""

import pytest

from taste.models import Category, LayerState, Polarity, Provenance, Standard


def _std(directive="Lead with the angle", polarity=Polarity.DO,
         state=LayerState.ACTIVE, rationale="because taste",
         provenance=Provenance.PERSONA, confidence=0.6,
         category=Category.DOCTRINE):
    return Standard(profile_id="owner", category=category,
                    polarity=polarity, directive=directive,
                    rationale=rationale, state=state,
                    provenance=provenance, confidence=confidence)


def _put(store, s, vec=None):
    store.put_standard(s, vec or [0.1, 0.2, 0.3, 0.4])


# ---------------------------------------------------------------- C3 — embedder

def test_c3_pins_bge_small_en_v1_5():
    from taste import embed

    assert embed.BGE_MODEL == "BAAI/bge-small-en-v1.5", "C3 is BGE-small"
    assert embed.DIM == 384


def test_hash_embedder_deterministic_normalized():
    from taste.embed import DIM, HashingEmbedder

    h = HashingEmbedder()
    a1 = h.embed(["ship notes verdict first"])[0]
    a2 = h.embed(["ship notes verdict first"])[0]
    assert a1 == a2, "same text -> same vector"
    assert len(a1) == DIM
    assert abs(sum(v * v for v in a1) - 1.0) < 1e-6, "unit norm"


def test_get_embedder_hash_forced_and_offline_fallback(monkeypatch):
    from taste import embed

    monkeypatch.setenv("TASTE_EMBED", "hash")
    embed.get_embedder.cache_clear()
    assert isinstance(embed.get_embedder(), embed.HashingEmbedder)

    class Boom:
        def __init__(self):
            raise RuntimeError("no model on this box")

    monkeypatch.setattr(embed, "BGEEmbedder", Boom)
    monkeypatch.delenv("TASTE_EMBED")
    embed.get_embedder.cache_clear()
    assert isinstance(embed.get_embedder(), embed.HashingEmbedder)
    embed.get_embedder.cache_clear()


def test_cosine_identity_and_orthogonality():
    from taste.embed import cosine

    v = [1.0, 0.0, 0.0]
    assert abs(cosine(v, v) - 1.0) < 1e-9
    assert abs(cosine(v, [0.0, 1.0, 0.0])) < 1e-9


def test_bge_integration_skipped_when_unavailable():
    """C3 real-model sanity — skipped when the model can't load (offline)."""
    import os

    if os.environ.get("TASTE_EMBED") == "hash":
        pytest.skip("hash mode forced — offline run")
    from taste.embed import BGEEmbedder

    try:
        emb = BGEEmbedder()
    except Exception:
        pytest.skip("BGE-small unavailable on this box")
    vecs = emb.embed(["ship the verdict first", "ship the verdict first"])
    assert len(vecs[0]) == 384
    assert vecs[0] == vecs[1], "same text embeds identically"


# ---------------------------------------------------------------- B7 + lifecycle (D1+D2)

def test_propose_lands_provisional_with_vector(engine, store):
    s = engine.propose(_std("ship the verdict"))
    assert s.state == LayerState.PROVISIONAL
    (std, vec) = store.list_standards("owner", with_vectors=True)[0]
    assert std.state == LayerState.PROVISIONAL
    assert vec is not None, "layers carry an embedding on write"


def test_confirm_seeds_confidence_at_floor(engine, store):
    s = engine.propose(_std("Lead with the verdict",
                            state=LayerState.PROVISIONAL, confidence=0.5))
    engine.confirm(s.id)
    row = store.get_standard(s.id)
    assert row.state == LayerState.ACTIVE
    assert row.confidence == 0.55, "owner confirmation is direct evidence"


def test_confirm_never_lowers_confidence(engine, store):
    s = engine.propose(_std("Proven layer",
                            state=LayerState.PROVISIONAL, confidence=0.9))
    engine.confirm(s.id)
    assert store.get_standard(s.id).confidence == 0.9


def test_b7_excludes_below_floor_entirely(engine, store):
    """Blueprint check: low-confidence active layers excluded from
    retrieval entirely — do, why, AND never, even at best similarity."""
    below_do = _std("ship the verdict first now", confidence=0.40)
    below_never = _std("never ship the verdict sloppy",
                       polarity=Polarity.NEVER, confidence=0.40)
    at_floor = _std("lead angle notes", confidence=0.55)
    from taste.embed import embed
    for s in (below_do, below_never, at_floor):
        s.state = LayerState.ACTIVE
        store.put_standard(s, embed(
            [s.directive + " " + s.rationale])[0])  # the import path
    hits = engine.retrieve("ship verdict", "owner")
    ids = [s.id for s, _ in hits]
    assert below_do.id not in ids, "below-floor do excluded despite best match"
    assert below_never.id not in ids, "below-floor never excluded"
    assert at_floor.id in ids, "at-floor active still retrieved"


def test_confidence_lifecycle_reinforce_decay_bounded(engine, store):
    s = engine.propose(_std("ship verdict first", confidence=0.5))
    engine.confirm(s.id)
    assert store.get_standard(s.id).confidence == 0.55
    engine.reinforce(s.id, 0.05)
    assert store.get_standard(s.id).confidence == pytest.approx(0.60)
    engine.reinforce(s.id, 10.0)
    assert store.get_standard(s.id).confidence == 1.0, "capped at 1.0"
    engine.reinforce(s.id, -10.0)
    assert store.get_standard(s.id).confidence == 0.0, "floored at 0.0"


def test_feedback_reinforces_winner_layers_decays_losers(engine, store):
    from taste.models import Candidate

    winner_layer = engine.propose(_std("ship notes verdict first"))
    engine.confirm(winner_layer.id)
    loser_layer = engine.propose(_std("completely unrelated filler words"))
    engine.confirm(loser_layer.id)
    cands = [
        Candidate(label="B", content="ship notes verdict first today"),
        Candidate(label="A", content="completely unrelated filler words here"),
    ]
    resp = engine.record_feedback(
        "owner", {"type": "write_doc"}, pick="B",
        rationale="B ships the verdict first", rejected=["A"],
        candidates=cands)
    assert resp.accepted and resp.record_id
    assert store.get_standard(winner_layer.id).confidence > 0.55, \
        "layers pulled toward the winner gain confidence"
    assert store.get_standard(loser_layer.id).confidence < 0.55, \
        "layers pulled toward a rejected candidate decay"


def test_feedback_without_content_moves_no_confidence(engine, store):
    pinned = engine.propose(_std("ship verdict first"))
    engine.confirm(pinned.id)
    before = store.get_standard(pinned.id).confidence
    resp = engine.record_feedback(
        "owner", {"type": "x"}, pick="A", rationale="won on vibes",
        rejected=["B"], candidates=[])
    assert resp.accepted
    assert store.get_standard(pinned.id).confidence == before, \
        "no scorable signal — no confidence movement"
    extracted = [s for s in store.list_standards("owner")
                 if s.provenance == Provenance.DECISION]
    assert extracted, "the why-layer extraction still runs"


def test_supersede_purges_working_set_slot(engine, store):
    s = engine.propose(_std("lead with the angle"))
    engine.confirm(s.id)
    store.put_ws_entry("owner", s.id, weight=0.8)
    newer = _std("lead with the verdict", provenance=Provenance.CORRECTION,
                 state=LayerState.PROVISIONAL)
    engine.supersede(s.id, newer)
    assert store.get_standard(s.id).state == LayerState.SUPERSEDED
    assert store.get_standard(s.id) is not None, "NEVER deleted"
    assert store.ws_slot_count("owner") == 0, "supersede evicts the slot"
    assert store.get_working_set("owner") == []


def test_retire_purges_working_set_slot(engine, store):
    s = engine.propose(_std("lead with the angle"))
    engine.confirm(s.id)
    store.put_ws_entry("owner", s.id, weight=0.8)
    engine.retire(s.id)
    assert store.get_standard(s.id).state == LayerState.RETIRED
    assert store.ws_slot_count("owner") == 0, "retire evicts the slot"


def test_confirm_without_rationale_is_refused(engine, store):
    s = engine.propose(_std("never pad", rationale="",
                            state=LayerState.PROVISIONAL))
    with pytest.raises(ValueError):
        engine.confirm(s.id)
    assert store.get_standard(s.id).state == LayerState.PROVISIONAL


# ---------------------------------------------------------------- A5 — procedural patterns

STEPS_DOC = """# Release flow
1. Verify the build passes.
2. Commit with evidence links.
3. Hand off to the publisher.
"""


def test_a5_numbered_sequence_fuses_into_one_procedural_standard():
    from taste.ingest import ingest_docs
    from taste.models import get_steps

    resp = ingest_docs("owner", [{"id": "doc", "text": STEPS_DOC,
                                   "format": "md"}])
    fused = [c for c in resp.candidates if get_steps(c)]
    assert len(fused) == 1, "a numbered run is ONE pattern, not N rules"
    proc = fused[0]
    assert proc.category == Category.PROCEDURAL
    assert proc.state == LayerState.PROVISIONAL, "never auto-activates"
    assert [st["directive"] for st in get_steps(proc)] == [
        "Verify the build passes",
        "Commit with evidence links",
        "Hand off to the publisher",
    ]
    assert get_steps(proc)[0]["step"] == 1


def test_a5_single_numbered_item_stays_atomic():
    from taste.ingest import ingest_docs
    from taste.models import get_steps

    doc = "1. Verify the build passes.\n"
    resp = ingest_docs("owner", [{"id": "d", "text": doc}])
    assert resp.candidates, "single numbered line still extracts"
    for c in resp.candidates:
        assert get_steps(c) == [], "one step is not a pattern"


def test_a5_prose_breaks_the_run():
    from taste.ingest import ingest_docs
    from taste.models import get_steps

    doc = ("1. Verify the build passes.\n"
           "Some prose between the items.\n"
           "2. Commit the work.\n")
    resp = ingest_docs("owner", [{"id": "d", "text": doc}])
    for c in resp.candidates:
        assert get_steps(c) == [], "interrupted runs never fuse"
    assert len(resp.candidates) == 2, "each item extracts atomically"


def test_ingest_atomic_rule_has_no_steps():
    from taste.ingest import ingest_docs
    from taste.models import get_steps

    resp = ingest_docs("owner", [{
        "id": "d", "text": 'Never use generic openers because they are slop.\n'}])
    (c,) = resp.candidates
    assert c.polarity == Polarity.NEVER
    assert get_steps(c) == [], "plain rules carry no steps"


def test_a5_steps_enumerated_in_restatement():
    from taste.comprehend import restate
    from taste.models import make_steps

    s = _std("Follow the release pattern", category=Category.PROCEDURAL)
    s.variant_meta["steps"] = make_steps(["verify", "commit"])
    joined = " ".join(restate(s).implications)
    assert "verify" in joined and "commit" in joined, \
        "restatement tells the owner what each step means"


# ---------------------------------------------------------------- I4 — dual-mode packets

def _corpus(engine):
    """Confirmed actives above the floor: do/never/why layers."""
    made = {}
    for key, d, pol in [
        ("angle", "Lead with the angle, then the structure", Polarity.DO),
        ("verdict", "Open with the verdict, never the windup", Polarity.DO),
        ("generic", "never use generic openers like 'in today's world'",
         Polarity.NEVER),
        ("why_short", "short sentences carry authority because they read as "
         "conviction", Polarity.WHY),
    ]:
        s = engine.propose(_std(d, polarity=pol))
        engine.confirm(s.id)
        made[key] = s
    return made


def _pkt(engine, task_ctx=None, turn_ctx=None, guidelines=None):
    from taste.models import PacketRequest

    return _compile(engine, PacketRequest(
        profile_id="owner", task_ctx=task_ctx or {}, guidelines=guidelines or [],
        turn_ctx=turn_ctx))


def _compile(engine, req):
    from taste.inject import compile_packet

    return compile_packet(engine, req)


def test_task_packet_control_parity(engine):
    _corpus(engine)
    pkt = _pkt(engine, task_ctx={"type": "write_doc", "brief": "ship notes"})
    texts = [d.text for d in pkt.directives]
    assert any("generic openers" in t for t in texts), \
        "never-rules apply unconditionally"
    assert pkt.free_degrees, "NEVER rule 6"
    assert pkt.variant_meta["mode"] == "task"
    assert pkt.variant_meta["engine"] == "taste-6"


def test_guidelines_land_as_hard_constraints(engine):
    _corpus(engine)
    pkt = _pkt(engine, task_ctx={"type": "x"},
               guidelines=["ship today, no exceptions"])
    assert any(c.text == "ship today, no exceptions" and c.hard
               for c in pkt.constraints)


def test_turn_packet_is_ws_first_and_marked(engine, store):
    from taste.models import PacketRequest

    made = _corpus(engine)
    store.put_ws_entry("owner", made["verdict"].id, weight=0.9)
    store.put_ws_entry("owner", made["why_short"].id, weight=0.5)
    pkt = _compile(engine, PacketRequest(
        profile_id="owner", task_ctx={"type": "write_doc"},
        turn_ctx={"segment": "lead", "draft_so_far": "Verdict: shipped.",
                  "turn_index": 1}))
    assert pkt.variant_meta["mode"] == "turn"
    assert pkt.variant_meta["ws_size"] == 2
    assert pkt.directives[0].text == "Open with the verdict, never the windup", \
        "working set by weight leads the turn packet"
    assert pkt.exemplars[0].text.startswith("short sentences"), \
        "WS why-layers render as the exemplar"


def test_turn_packet_empty_ws_falls_back_to_retrieval(engine):
    _corpus(engine)
    pkt = _pkt(engine, task_ctx={"type": "write_doc",
                                 "brief": "lead angle structure"},
               turn_ctx={"segment": "lead"})
    assert pkt.variant_meta["ws_size"] == 0
    assert pkt.directives, "retrieval-only turn packet still carries taste"
    assert pkt.free_degrees


def test_ws_member_below_floor_cannot_bypass_b7(engine, store):
    """The WS is a new injection path — B7 must hold at render time."""
    made = _corpus(engine)
    store.put_ws_entry("owner", made["angle"].id, weight=0.9)
    engine.reinforce(made["angle"].id, -0.1)  # 0.55 -> below floor
    pkt = _pkt(engine, task_ctx={"type": "write_doc",
                                 "brief": "lead angle structure"},
               turn_ctx={"segment": "lead"})
    texts = [d.text for d in pkt.directives]
    assert all("Lead with the angle" not in t for t in texts), \
        "decayed WS member must not slip past the confidence gate"
    assert pkt.variant_meta["ws_size"] == 0, \
        "sub-floor members don't count as rendered working set"


def test_per_turn_packet_smaller_than_task_packet(engine, store):
    """Blueprint check: byte-compare on a corpus that fills the caps."""
    do_layers = []
    for i in range(8):
        s = engine.propose(_std(f"ship notes quality keep the point {i}"))
        engine.confirm(s.id)
        do_layers.append(s)
    for i in range(2):
        s = engine.propose(_std(f"ship notes quality why it matters {i}",
                                polarity=Polarity.WHY))
        engine.confirm(s.id)
    for i in range(4):
        s = engine.propose(_std(f"never ship notes quality slop {i}",
                                polarity=Polarity.NEVER))
        engine.confirm(s.id)
    for s in do_layers[:2]:
        store.put_ws_entry("owner", s.id, weight=0.9)

    base = {"profile_id": "owner",
            "task_ctx": {"type": "write_doc", "brief": "ship notes quality"}}
    from taste.models import PacketRequest

    task = _compile(engine, PacketRequest(**base))
    turn = _compile(engine, PacketRequest(**base, turn_ctx={
        "segment": "lead", "draft_so_far": "ship notes", "turn_index": 1}))
    task_json = task.model_dump_json()
    turn_json = turn.model_dump_json()
    assert task.directives and turn.directives
    assert len(turn.directives) <= len(task.directives)
    assert len(turn_json) < len(task_json), \
        f"turn {len(turn_json)}B must be slimmer than task {len(task_json)}B"


def test_packets_deterministic_both_modes(engine, store):
    made = _corpus(engine)
    store.put_ws_entry("owner", made["angle"].id, weight=0.9)
    a_task = _pkt(engine, task_ctx={"type": "write_doc", "brief": "ship notes"})
    b_task = _pkt(engine, task_ctx={"type": "write_doc", "brief": "ship notes"})
    assert a_task.model_dump_json() == b_task.model_dump_json()
    a_turn = _pkt(engine, task_ctx={"type": "write_doc", "brief": "ship notes"},
                  turn_ctx={"segment": "lead", "turn_index": 3})
    b_turn = _pkt(engine, task_ctx={"type": "write_doc", "brief": "ship notes"},
                  turn_ctx={"segment": "lead", "turn_index": 3})
    assert a_turn.model_dump_json() == b_turn.model_dump_json()


# ---------------------------------------------------------------- H6 — curation loop

def _curator(engine):
    from taste.curate import Curator

    return Curator(engine)


def _pin(sid, weight=0.9, why="wins every time in this task family"):
    from taste.models import CurationProposal

    return CurationProposal(action="pin", standard_id=sid, why=why,
                            weight=weight)


def _demote(sid, why="not relevant to this task family"):
    from taste.models import CurationProposal

    return CurationProposal(action="demote", standard_id=sid, why=why)


def test_amendment_lands_queued_never_applied(engine, store):
    """Blueprint check: agent pin/demote lands in inferred_queue, never
    applies directly."""
    made = _corpus(engine)
    cur = _curator(engine)
    iid = cur.propose("owner", _pin(made["angle"].id))
    row = store.get_inferred(iid)
    assert row["kind"] == "amendment"
    assert row["payload"]["why"].strip(), "the why travels with the queue row"
    assert store.ws_slot_count("owner") == 0, "propose NEVER applies"
    assert store.get_standard(made["angle"].id).state == LayerState.ACTIVE


def test_propose_rejects_unknown_standard(engine):
    cur = _curator(engine)
    with pytest.raises(ValueError, match="not found"):
        cur.propose("owner", _pin("std_missing"))


def test_resolve_accept_applies_and_reject_does_not(engine, store):
    """Blueprint check: accept mutates, reject never does."""
    made = _corpus(engine)
    cur = _curator(engine)
    iid = cur.propose("owner", _pin(made["angle"].id))
    cur.apply_amendment("owner", iid, accept=False)
    assert store.ws_slot_count("owner") == 0, "reject leaves the WS untouched"
    assert store.get_inferred(iid)["resolved"] is True

    iid2 = cur.propose("owner", _pin(made["verdict"].id, weight=0.9))
    cur.apply_amendment("owner", iid2, accept=True)
    assert store.ws_slot_count("owner") == 1
    (e,) = store.get_working_set("owner")
    assert e.standard_id == made["verdict"].id
    assert e.weight == 0.9


def test_apply_refuses_stale_target_but_resolves(engine, store):
    """Stale proposal: apply 422s (here ValueError), the resolve still
    completes, nothing half-applies."""
    prov = engine.propose(_std("unconfirmed taste"))
    cur = _curator(engine)
    iid = cur.propose("owner", _pin(prov.id))
    with pytest.raises(ValueError, match="provisional"):
        cur.apply_amendment("owner", iid, accept=True)
    assert store.ws_slot_count("owner") == 0
    assert store.get_inferred(iid)["resolved"] is True, \
        "the resolve completes even on refusal"


def test_apply_full_ws_refuses_at_apply_not_propose(engine, store):
    """Blueprint check: the 20-slot bound is enforced at resolve; the
    proposal itself always lands in the queue."""
    cur = _curator(engine)
    extra = None
    for i in range(21):
        s = engine.propose(_std(f"layer number {i}"))
        engine.confirm(s.id)
        if i < 20:
            store.put_ws_entry("owner", s.id)
        else:
            extra = s
    iid = cur.propose("owner", _pin(extra.id))
    with pytest.raises(ValueError, match="full"):
        cur.apply_amendment("owner", iid, accept=True)
    assert store.ws_slot_count("owner") == 20


def test_apply_demote_requires_membership_then_removes(engine, store):
    made = _corpus(engine)
    cur = _curator(engine)
    iid = cur.propose("owner", _demote(made["angle"].id))
    with pytest.raises(ValueError, match="not in the working set"):
        cur.apply_amendment("owner", iid, accept=True)

    store.put_ws_entry("owner", made["angle"].id, weight=0.8)
    iid2 = cur.propose("owner", _demote(made["angle"].id))
    cur.apply_amendment("owner", iid2, accept=True)
    assert store.ws_slot_count("owner") == 0


def test_apply_suggest_retire_retires_candidate(engine, store):
    decayed = _corpus(engine)["angle"]
    engine.reinforce(decayed.id, -0.1)  # 0.55 -> below floor
    cur = _curator(engine)
    from taste.models import CurationProposal

    iid = cur.propose("owner", CurationProposal(
        action="suggest_retire", standard_id=decayed.id,
        why="contradicted twice, keeps decaying"))
    cur.apply_amendment("owner", iid, accept=True)
    assert store.get_standard(decayed.id).state == LayerState.RETIRED
    assert store.get_standard(decayed.id) is not None, "NEVER deleted"
    assert store.ws_slot_count("owner") == 0


def test_suggest_for_drafts_evidence_backed_proposals(engine, store):
    made = _corpus(engine)
    store.put_ws_entry("owner", made["angle"].id, weight=0.9)
    engine.reinforce(made["angle"].id, -0.1)  # WS member decays below floor
    cur = _curator(engine)
    proposals = cur.suggest_for(
        "owner", {"type": "write_doc", "brief": "ship notes"})
    by_action: dict[str, list] = {}
    for p in proposals:
        by_action.setdefault(p.action, []).append(p)
        assert p.why.strip(), "every draft carries why"
    demoted_ids = {p.standard_id for p in by_action.get("demote", [])}
    pinned_ids = {p.standard_id for p in by_action.get("pin", [])}
    assert made["angle"].id in demoted_ids, \
        "decayed WS member drafted for demotion"
    assert made["angle"].id not in pinned_ids, "already committed — not re-pinned"
    assert pinned_ids, "fresh retrieval hits drafted for pinning"
    # drafting is pure: it enqueues nothing, applies nothing
    assert store.pending_inferred("owner") == []
    assert store.ws_slot_count("owner") == 1


# ---------------------------------------------------------------- J2 — dynamic segments

def _proc(engine, directive, steps):
    from taste.models import make_steps

    s = _std(directive, category=Category.PROCEDURAL)
    s.variant_meta["steps"] = make_steps(steps)
    s = engine.propose(s)
    engine.confirm(s.id)
    return s


def test_j2_segments_from_procedural_steps(engine):
    """Blueprint check: the agent's own A5 patterns shape decomposition."""
    from taste.segments import propose_segments

    _proc(engine, "ship notes quality release pattern",
          ["verify the build", "commit the work", "hand off"])
    segs = propose_segments(
        {"type": "release", "brief": "ship notes quality"}, engine)
    assert segs == ["verify the build", "commit the work", "hand off"], \
        "steps ARE the segment plan when the pattern matches"


def test_j2_falls_back_to_type_map_when_no_pattern(engine):
    from taste.segments import propose_segments

    assert propose_segments({"type": "security_review"}, engine) == \
        ["triage", "findings", "verdict"]
    assert propose_segments({"type": "never_seen_before"}, engine) == \
        ["draft", "refine", "finalize"], "default lane for unknown types"


def test_j2_ignores_below_floor_patterns(engine):
    from taste.segments import propose_segments

    s = _proc(engine, "ship notes quality release pattern",
              ["ship notes quality verify", "ship notes quality commit"])
    engine.reinforce(s.id, -0.1)  # below the B7 floor
    assert propose_segments(
        {"type": "release", "brief": "ship notes quality"}, engine) == \
        ["draft", "refine", "finalize"], "decayed patterns don't shape work"


# ---------------------------------------------------------------- HTTP surface

def test_meta_declares_the_permutation(client):
    body = client.get("/meta").json()
    assert body["contract"] == "v1"
    assert body["name"] == "taste-6"
    assert body["permutation"] == {
        "schema": "A5-procedural", "retrieval": "B7-confidence",
        "embed": "C3-BGE-small", "scorer": "D1-rules",
        "interview": "E1-adaptive", "hone": "F1-BoN3",
        "storage": "G1-sqlite-wal", "memory": "H6-self-edit",
        "injection": "I4-per-turn", "pipeline": "J2-dynamic",
    }


def _confirm_all(client):
    for s in client.get("/standards").json()["standards"]:
        if s["state"] == "provisional":
            client.post(f"/standards/{s['id']}/confirm")


def test_curate_propose_queues_and_422s_without_why(client):
    client.post("/ingest", json={"profile_id": "owner", "docs": [
        {"id": "p", "text": "Never use filler words because bloat.\n"}]})
    _confirm_all(client)
    sid = client.get("/standards").json()["standards"][0]["id"]

    r = client.post("/curate/propose", json={
        "profile_id": "owner",
        "proposal": {"action": "pin", "standard_id": sid, "why": "   "}})
    assert r.status_code == 422, "the why law generalizes to curation"
    assert client.get("/review-queue").json()["pending"] == []

    r = client.post("/curate/propose", json={
        "profile_id": "owner",
        "proposal": {"action": "pin", "standard_id": sid,
                     "why": "wins in this family", "weight": 0.9}})
    assert r.status_code == 200
    iid = r.json()["queued"]
    pending = client.get("/review-queue").json()["pending"]
    assert any(p["id"] == iid and p["kind"] == "amendment"
               for p in pending)
    assert client.get("/working-set").json()["count"] == 0, \
        "the proposal applied nothing"


def test_resolve_endpoint_applies_amendment_on_accept(client):
    client.post("/ingest", json={"profile_id": "owner", "docs": [
        {"id": "p", "text": "Never use filler words because bloat.\n"}]})
    _confirm_all(client)
    sid = client.get("/standards").json()["standards"][0]["id"]
    client.post("/curate/propose", json={
        "profile_id": "owner",
        "proposal": {"action": "pin", "standard_id": sid,
                     "why": "wins in this family", "weight": 0.85}})
    iid = client.get("/review-queue").json()["pending"][-1]["id"]

    r = client.post(f"/review-queue/{iid}/resolve",
                    json={"accept": True, "rationale": "agreed"})
    assert r.status_code == 200 and r.json()["applied"] is True
    ws = client.get("/working-set").json()
    assert ws["count"] == 1
    (e,) = ws["entries"]
    assert e["standard_id"] == sid
    assert e["weight"] == 0.85
    assert e["state"] == "active"
    assert e["directive"], "joined layer context rides the WS view"


def test_resolve_endpoint_reject_leaves_state_untouched(client):
    client.post("/ingest", json={"profile_id": "owner", "docs": [
        {"id": "p", "text": "Never use filler words because bloat.\n"}]})
    _confirm_all(client)
    sid = client.get("/standards").json()["standards"][0]["id"]
    client.post("/curate/propose", json={
        "profile_id": "owner",
        "proposal": {"action": "pin", "standard_id": sid,
                     "why": "wins in this family"}})
    iid = client.get("/review-queue").json()["pending"][-1]["id"]
    r = client.post(f"/review-queue/{iid}/resolve",
                    json={"accept": False})
    assert r.status_code == 200 and r.json()["applied"] is False
    assert client.get("/working-set").json()["count"] == 0
    assert client.get("/review-queue").json()["pending"] == []


def test_turn_packet_over_http(client):
    client.post("/ingest", json={"profile_id": "owner", "docs": [
        {"id": "p", "text": "Never use filler words because bloat.\n"}]})
    _confirm_all(client)
    r = client.post("/packet", json={
        "profile_id": "owner",
        "task_ctx": {"type": "write_doc", "brief": "ship notes"},
        "turn_ctx": {"segment": "lead", "draft_so_far": "Verdict: shipped.",
                     "turn_index": 1}})
    assert r.status_code == 200
    pkt = r.json()
    assert pkt["variant_meta"]["mode"] == "turn"
    assert pkt["free_degrees"]
    task = client.post("/packet", json={
        "profile_id": "owner",
        "task_ctx": {"type": "write_doc", "brief": "ship notes"}}).json()
    assert task["variant_meta"]["mode"] == "task"


def test_working_set_endpoint_shape(client):
    client.post("/ingest", json={"profile_id": "owner", "docs": [
        {"id": "p", "text": "Never use filler words because bloat.\n"}]})
    _confirm_all(client)
    sid = client.get("/standards").json()["standards"][0]["id"]
    client.post("/curate/propose", json={
        "profile_id": "owner",
        "proposal": {"action": "pin", "standard_id": sid,
                     "why": "wins in this family"}})
    iid = client.get("/review-queue").json()["pending"][-1]["id"]
    client.post(f"/review-queue/{iid}/resolve", json={"accept": True})
    ws = client.get("/working-set").json()
    assert set(ws) == {"entries", "count"}
    assert ws["entries"][0]["pinned_at"], "ISO8601 from the owner's resolve"


def test_a5_short_steps_do_not_forge_a_pattern():
    """Edge: if step-length filtering drops a run below two steps, the
    items fall back to atomic rules — never a 1-step 'pattern'."""
    from taste.ingest import ingest_docs
    from taste.models import get_steps

    doc = ("1. Verify the build passes first.\n"
           "2. Commit it now.\n")
    resp = ingest_docs("owner", [{"id": "d", "text": doc}])
    for c in resp.candidates:
        assert get_steps(c) == [], "filtered-thin runs never fuse"


def test_a5_steps_round_trip_through_export_import(client):
    """Blueprint check: A5 steps survive the portability round-trip
    inside variant_meta (contract §2 — variant extras ride variant_meta)."""
    doc = ("# Release flow\n"
           "1. Verify the build passes first.\n"
           "2. Commit the finished work.\n")
    client.post("/ingest", json={"profile_id": "owner",
        "docs": [{"id": "d", "text": doc, "format": "md"}]})
    rows = [s for s in client.get("/standards").json()["standards"]
            if s["variant_meta"].get("steps")]
    assert rows, "ingested procedural pattern carries steps"
    exported = rows[0]

    moved = {**exported, "profile_id": "p2"}
    r = client.post("/standards/import",
                    json={"profile_id": "p2", "standards": [moved]})
    assert r.json()["imported"] == 1
    landed = [s for s in client.get(
        "/standards", params={"profile_id": "p2"}).json()["standards"]
        if s["id"] == exported["id"]][0]
    assert landed["variant_meta"]["steps"] == exported["variant_meta"]["steps"], \
        "steps survive export -> import"


# ---------------------------------------------------------------- working set mechanics

def test_ws_slot_allocation_takes_lowest_free_slot(store):
    e1 = store.put_ws_entry("owner", "std_a", weight=0.9)
    e2 = store.put_ws_entry("owner", "std_b", weight=0.8)
    assert (e1.slot, e2.slot) == (1, 2)
    store.del_ws_entry("owner", 1)
    e3 = store.put_ws_entry("owner", "std_c", weight=0.7)
    assert e3.slot == 1, "freed slots are reused lowest-first"


def test_ws_order_is_weight_then_slot_deterministic(store):
    xs = [_std(f"layer {i}") for i in "xyz"]
    for s in xs:
        _put(store, s)
    store.put_ws_entry("owner", xs[0].id, weight=0.5)   # slot 1
    store.put_ws_entry("owner", xs[1].id, weight=0.9)   # slot 2
    store.put_ws_entry("owner", xs[2].id, weight=0.5)   # slot 3
    ordered = [e.standard_id for e in store.get_working_set("owner")]
    assert ordered == [xs[1].id, xs[0].id, xs[2].id], \
        "weight desc, slot tiebreak"


def test_ws_read_joins_layer_state_hides_non_active(store):
    a = _std("Active layer")
    b = _std("Dead layer", state=LayerState.SUPERSEDED)
    _put(store, a)
    _put(store, b)
    store.put_ws_entry("owner", a.id)
    store.put_ws_entry("owner", b.id)
    ids = {e.standard_id for e in store.get_working_set("owner")}
    assert ids == {a.id}, "self-heal: non-active members never visible"


def test_ws_purge_removes_entries_for_standard(store):
    a = _std("A")
    _put(store, a)
    store.put_ws_entry("owner", a.id)
    store.purge_ws_standard("owner", a.id)
    assert store.get_working_set("owner") == []
    assert store.ws_slot_count("owner") == 0


def test_ws_capacity_20_enforced_at_store(store):
    for i in range(20):
        store.put_ws_entry("owner", f"std_{i}", weight=1.0 - i / 100)
    with pytest.raises(ValueError, match="full"):
        store.put_ws_entry("owner", "std_extra", weight=0.5)


def test_ws_repin_updates_weight_in_place(store):
    a = _std("A")
    _put(store, a)
    e1 = store.put_ws_entry("owner", a.id, weight=0.5)
    e2 = store.put_ws_entry("owner", a.id, weight=0.9)
    assert e2.slot == e1.slot, "re-pin moves no slot"
    assert store.ws_slot_count("owner") == 1
    assert store.get_working_set("owner")[0].weight == 0.9


def test_ws_entries_carry_layer_context(store):
    a = _std("Visible layer", confidence=0.8)
    _put(store, a)
    store.put_ws_entry("owner", a.id, weight=0.9)
    (e,) = store.get_working_set("owner")
    assert e.directive == "Visible layer"
    assert e.state == LayerState.ACTIVE
    assert e.confidence == 0.8
