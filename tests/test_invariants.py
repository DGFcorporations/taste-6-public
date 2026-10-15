"""Contract + control invariant pins for taste-6.

Rationale law, minimum wire shapes, and the new variant models.
Offline: hashing embedder via conftest.
"""

import pytest
from pydantic import ValidationError

from taste.models import (
    Category,
    DecisionRecord,
    FeedbackRequest,
    LayerState,
    Polarity,
    Provenance,
    Standard,
)

STANDARD_MINIMUM_FIELDS = (
    "id", "profile_id", "category", "polarity", "directive", "rationale",
    "confidence", "provenance", "state", "supersedes", "created", "updated",
)


# Rule 3 — rationale required for decisions -------------------------------
@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_decision_record_rejects_blank_rationale(bad):
    with pytest.raises(ValidationError):
        DecisionRecord(pick="A", rationale=bad)


def test_decision_record_accepts_real_rationale():
    assert DecisionRecord(pick="A", rationale="verdict-first won").pick == "A"


# §2 minimum shape survives the wire --------------------------------------
def test_standard_minimum_shape_round_trips():
    s = Standard(category=Category.DOCTRINE, polarity=Polarity.DO,
                 directive="Lead with the angle", rationale="attention")
    d = s.model_dump(mode="json")
    for k in STANDARD_MINIMUM_FIELDS:
        assert k in d, f"contract §2 minimum field {k} missing on export"
    assert Standard.model_validate(d).id == s.id


# Curation proposals carry the rationale law (the `why` law generalizes) ---
def test_curation_proposal_requires_why():
    from taste.models import CurationProposal

    with pytest.raises(ValidationError):
        CurationProposal(action="pin", standard_id="std_x", why="   ")


def test_curation_proposal_validates_action():
    from taste.models import CurationProposal

    with pytest.raises(ValidationError):
        CurationProposal(action="promote", standard_id="std_x", why="real reason")
    ok = CurationProposal(action="pin", standard_id="std_x", why="real reason")
    assert ok.action == "pin"


# I4 — PacketRequest gains an optional turn_ctx ----------------------------
def test_packet_request_accepts_turn_ctx():
    from taste.models import PacketRequest

    r = PacketRequest(task_ctx={"type": "write_doc"},
                      turn_ctx={"segment": "lead", "turn_index": 2})
    assert r.turn_ctx["segment"] == "lead"
    assert PacketRequest(task_ctx={"type": "x"}).turn_ctx is None


# A5 — steps live in variant_meta ------------------------------------------
def test_a5_steps_helpers_round_trip():
    from taste.models import get_steps, make_steps

    s = Standard(category=Category.PROCEDURAL, polarity=Polarity.DO,
                 directive="Follow the release pattern",
                 rationale="repeatable process")
    s.variant_meta["steps"] = make_steps(["verify", "commit", "hand off"])
    assert [st["directive"] for st in get_steps(s)] == \
        ["verify", "commit", "hand off"]
    assert get_steps(s)[0]["step"] == 1

    plain = Standard(category=Category.DOCTRINE, polarity=Polarity.DO,
                     directive="no steps here")
    assert get_steps(plain) == [], "non-procedural layers carry no steps"


# import never silently activates (pin at model level) ---------------------
def test_amendment_provenance_defaults_to_amendment_enum():
    assert Provenance.AMENDMENT.value == "amendment"


def test_layer_state_enum_is_the_contract_vocabulary():
    assert {s.value for s in LayerState} == \
        {"provisional", "active", "superseded", "retired"}


# ---------------------------------------------------------------- CLI

def _proposed(app, directive="Never pad prose"):
    from taste.models import Category, Polarity, Provenance, Standard

    engine = app.state.engine
    s = Standard(profile_id="owner", category=Category.AESTHETIC,
                 polarity=Polarity.NEVER, directive=directive,
                 rationale="density carries authority",
                 provenance=Provenance.PERSONA)
    engine.propose(s)
    return s


def test_cli_standards_state_filter(app, capsys, monkeypatch):
    from taste import cli

    _proposed(app)
    monkeypatch.setattr(cli, "create_app", lambda: app)
    cli.main(["standards", "--state=active"])
    assert "Never pad prose" not in capsys.readouterr().out
    cli.main(["standards", "--state=provisional"])
    assert "Never pad prose" in capsys.readouterr().out


def test_cli_curate_queue_resolve_working_set(app, capsys, monkeypatch):
    from taste import cli

    s = _proposed(app)
    app.state.engine.confirm(s.id)
    monkeypatch.setattr(cli, "create_app", lambda: app)
    cli.main(["curate", "--action=pin", "--std", s.id,
              "--why", "dense prose wins this family"])
    assert "queued" in capsys.readouterr().out
    iid = next(p["id"] for p in app.state.store.pending_inferred("owner")
               if p["kind"] == "amendment")
    cli.main(["working-set"])
    assert "empty" in capsys.readouterr().out.lower(), "propose ≠ apply"
    cli.main(["queue"])
    assert "amendment" in capsys.readouterr().out
    cli.main(["resolve", "--id", iid, "--accept"])
    cli.main(["working-set"])
    assert "Never pad prose" in capsys.readouterr().out


def test_cli_packet_turn_mode(app, capsys, monkeypatch):
    from taste import cli

    s = _proposed(app)
    app.state.engine.confirm(s.id)
    monkeypatch.setattr(cli, "create_app", lambda: app)
    cli.main(["packet", "--type=write_doc", "--turn-segment=lead",
              "--turn-index=2"])
    assert '"mode": "turn"' in capsys.readouterr().out
    cli.main(["packet", "--type=write_doc"])
    assert '"mode": "task"' in capsys.readouterr().out


def test_cli_feedback_and_ingest(app, capsys, monkeypatch, tmp_path):
    from taste import cli

    monkeypatch.setattr(cli, "create_app", lambda: app)
    cli.main(["feedback", "--pick=B", "--why=verdict first wins"])
    assert "provisional why-layer" in capsys.readouterr().out
    doc = tmp_path / "p.md"
    doc.write_text("# Persona\n- Never use filler words because bloat.\n")
    cli.main(["ingest", str(doc)])
    assert "candidate layers" in capsys.readouterr().out


# ---------------------------------------------------------------- MCP tools (§5)

def test_mcp_server_registers_exactly_the_four_contract_tools(app):
    import asyncio

    from taste.mcp_server import create_server

    server = create_server(app.state.store, app.state.engine)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"taste_packet", "taste_score", "taste_feedback",
                     "taste_recall"}, "§5 is the full MCP list — it never grows"


def test_mcp_tool_semantics_match_http(app):
    import json

    from taste.mcp_server import create_tools

    s = _proposed(app, 'never use "filler words" in prose')
    app.state.engine.confirm(s.id)
    tools = create_tools(app.state.store, app.state.engine)

    pkt = json.loads(tools["taste_packet"](
        task_ctx={"type": "write_doc"}, profile_id="owner"))
    assert pkt["variant_meta"]["mode"] == "task"
    turn = json.loads(tools["taste_packet"](
        task_ctx={"type": "write_doc"}, profile_id="owner",
        turn_ctx={"segment": "lead", "turn_index": 1}))
    assert turn["variant_meta"]["mode"] == "turn", "I4 flows through MCP too"

    scored = json.loads(tools["taste_score"](
        candidates=[{"label": "A", "content": "filler words everywhere"}],
        task_ctx={"type": "x"}, profile_id="owner"))
    assert scored["scores"][0]["ref"] == "A", "§5: maps to POST /score shape"

    fb = json.loads(tools["taste_feedback"](
        pick="A", rationale="it ships", task_ctx={"type": "x"},
        profile_id="owner"))
    assert fb["accepted"] is True and fb["record_id"]

    bad = json.loads(tools["taste_feedback"](
        pick="A", rationale="   ", profile_id="owner"))
    assert "error" in bad, "NEVER rule 1 enforced at the tool boundary"

    recalled = json.loads(tools["taste_recall"](
        profile_id="owner", state="active"))
    assert recalled["standards"], "§5: maps to GET /standards (filtered)"
    assert all(r["state"] == "active" for r in recalled["standards"])

# ---------------------------------------------------------------- HTTP pins

def _std_payload(directive, rationale="because taste", state="active",
                 confidence=0.7, provenance="persona"):
    return {
        "profile_id": "owner", "category": "aesthetic",
        "polarity": "never", "directive": directive,
        "rationale": rationale, "confidence": confidence,
        "provenance": provenance, "state": state, "supersedes": [],
    }


def _import(client, standards):
    """Import, then confirm rows submitted as already active.

    Declared active state is not local approval. Packet and veto tests
    confirm the row. Quarantine is pinned in test_import_approval.py.
    """
    resp = client.post("/standards/import", json={
        "profile_id": "owner", "standards": standards})
    rows = {r["directive"]: r
            for r in client.get("/standards").json()["standards"]}
    for raw in standards:
        if (raw.get("state", "provisional") == "active"
                and str(raw.get("rationale") or "").strip()
                and raw.get("provenance", "persona") not in
                ("amendment", "inference")):
            row = rows[raw["directive"]]
            if (row.get("variant_meta") or {}).get("local_approved") is False:
                client.post(
                    f"/standards/{row['id']}/confirm",
                    json={"rationale": raw["rationale"]})
    return resp


def test_import_dedups_within_a_batch(client):
    s = _std_payload("Never open with a generic hook")
    body = _import(client, [s, s]).json()
    assert body == {"imported": 1, "skipped": 1}
    assert client.get("/standards").json()["count"] == 1


def test_import_dedups_against_existing_corpus(client):
    s = _std_payload("Never open with a generic hook")
    _import(client, [s])
    assert _import(client, [s]).json() == {"imported": 0, "skipped": 1}


def test_import_without_rationale_stays_provisional(client):
    s = _std_payload("Never pad with filler", rationale="")
    _import(client, [s])
    row = client.get("/standards").json()["standards"][0]
    assert row["state"] == "provisional", \
        "no rationale → nothing activates (NEVER rule 1)"


def test_import_amendment_provenance_never_activates(client):
    s = _std_payload("Agent-inferred taste", provenance="amendment")
    _import(client, [s])
    row = client.get("/standards").json()["standards"][0]
    assert row["state"] == "provisional", \
        "agent-proposed standards arrive provisional (conformance §4)"


def test_import_purges_ws_when_landing_non_active(client):
    """A re-import that lands non-active must evict its working-set slot."""
    s = _std_payload("Pinned then demoted by re-import")
    _import(client, [s])
    sid = client.get("/standards").json()["standards"][0]["id"]
    client.post(f"/standards/{sid}/confirm")
    client.post("/curate/propose", json={
        "profile_id": "owner",
        "proposal": {"action": "pin", "standard_id": sid,
                     "why": "wins in this family", "weight": 0.9}})
    iid = client.get("/review-queue").json()["pending"][-1]["id"]
    client.post(f"/review-queue/{iid}/resolve",
                json={"accept": True, "rationale": "owner agrees"})
    assert client.get("/working-set").json()["count"] == 1

    # re-import the same layer (same id, edited directive) landing as
    # provisional — the physical slot must be purged, not just hidden
    s2 = _std_payload("Pinned then re-imported as provisional (edited)",
                      state="provisional")
    s2["id"] = sid
    _import(client, [s2])
    assert client.get("/working-set").json()["count"] == 0, \
        "import landing non-active evicts the slot (no orphaned pin)"


def test_provisional_layers_never_reach_packets(client):
    client.post("/ingest", json={"profile_id": "owner", "docs": [
        {"id": "p", "text": "- Never use filler words.\n", "format": "md"}]})
    pkt = client.post("/packet", json={
        "profile_id": "owner", "task_ctx": {"type": "write_doc"}}).json()
    assert pkt["directives"] == [], "provisional must not reach packets"


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_feedback_without_rationale_422_contract_error_shape(client, bad):
    r = client.post("/feedback", json={
        "profile_id": "owner", "task_ctx": {}, "pick": "A",
        "rationale": bad})
    assert r.status_code == 422
    body = r.json()
    assert "error" in body and "code" in body["error"] and \
        "message" in body["error"], "contract §1 error shape"


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_feedback_request_rejects_blank_rationale(bad):
    with pytest.raises(ValidationError):
        FeedbackRequest(pick="A", rationale=bad)


def test_supersede_keeps_history_over_http(client):
    s = _std_payload("Never open with a generic hook")
    _import(client, [s])
    sid = client.get("/standards").json()["standards"][0]["id"]
    client.post(f"/standards/{sid}/confirm")
    newer = {
        "profile_id": "owner", "category": "aesthetic", "polarity": "never",
        "directive": "Never open with a cliché hook at all",
        "rationale": "clichés read as slop", "provenance": "correction",
        "state": "provisional", "supersedes": [],
    }
    r = client.post(f"/standards/{sid}/supersede", json=newer)
    assert r.status_code == 200
    old = [x for x in client.get("/standards").json()["standards"]
           if x["id"] == sid][0]
    assert old["state"] == "superseded", "kept forever, never deleted (NEVER 5)"
    rows = client.get("/standards").json()["standards"]
    assert any(x["supersedes"] == [sid] for x in rows), \
        "the correction links back"


# ------------------------------------------------------- audit regressions

def test_never_rule_in_packet_without_retrieval_hit(client):
    """Guardrails are not relevance picks — an active never-rule reaches
    the packet even when retrieval never surfaces it (task mode)."""
    _import(client, [_std_payload("Never use emoji in output",
                                  rationale="owner hates emoji")])
    pkt = client.post("/packet", json={
        "profile_id": "owner",
        "task_ctx": {"type": "database_migration_planning"}}).json()
    texts = [d["text"] for d in pkt["directives"] if d["polarity"] == "never"]
    assert "Never use emoji in output" in texts


def test_never_rule_in_turn_packet_without_retrieval_or_ws(client):
    """Turn mode: a never-rule absent from BOTH the working set and the
    turn's retrieval window still reaches the slim packet."""
    _import(client, [_std_payload("Never use emoji in output",
                                  rationale="owner hates emoji")])
    pkt = client.post("/packet", json={
        "profile_id": "owner",
        "task_ctx": {"type": "database_migration_planning"},
        "turn_ctx": {"turn": "idx-scan rationale"}}).json()
    texts = [d["text"] for d in pkt["directives"] if d["polarity"] == "never"]
    assert "Never use emoji in output" in texts


def test_mirror_veto_long_rule_short_task(client):
    """Long rule + short task: ≥3 hits covering ≥half the task fires."""
    _import(client, [_std_payload(
        "never alpha beta gamma delta epsilon zeta theta",
        rationale="long forbidden list")])
    r = client.post("/mirror", json={
        "profile_id": "owner",
        "task_ctx": {"brief": "alpha beta gamma", "attended": True}}).json()
    assert r["vetoes_hit"], "3 hits covering the whole task must veto"


def test_mirror_veto_short_rule_fires(client):
    _import(client, [_std_payload("No emoji",
                                  rationale="owner hates emoji")])
    r = client.post("/mirror", json={
        "profile_id": "owner",
        "task_ctx": {"brief": "emoji everywhere", "attended": True}}).json()
    assert r["vetoes_hit"], "rule-side coverage must fire a short rule"


def test_mirror_veto_unrelated_task_clean(client):
    _import(client, [_std_payload("No emoji",
                                  rationale="owner hates emoji")])
    r = client.post("/mirror", json={
        "profile_id": "owner",
        "task_ctx": {"brief": "database migration planning",
                     "attended": True}}).json()
    assert r["vetoes_hit"] == []
