"""taste-6 API — the contract surface over UDS/Granian/FastAPI.

Every §3 endpoint maps 1:1 to CONTRACT.md. create_app() is the conformance
entry point (TASTE_APP=taste.api:create_app + TASTE_APP_FACTORY=1).

Variant additions over the control's owner surface:
  + POST /packet accepts an optional turn_ctx (I4 — per-turn bundles)
  + POST /curate/propose   — H6: agent proposes a WS mutation (why required)
  + GET  /working-set      — the bounded curated context, joined to layers
  + POST /review-queue/{iid}/resolve gains {accept, rationale} — an
    accepted amendment is applied HERE, the only write path to the WS.

Errors follow contract §1: {"error": {"code", "message"}} with 4xx/5xx.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .core import StandardsEngine
from .curate import Curator
from .hone import hone_segment
from .ingest import ingest_docs
from .inject import compile_packet
from .interview import run_interview
from .mirror import build_mirror
from .models import (
    CurateRequest,
    FeedbackRequest,
    FeedbackResponse,
    HoneRequest,
    HoneResponse,
    ImportRequest,
    IngestRequest,
    IngestResponse,
    InterviewRequest,
    InterviewResponse,
    LayerState,
    MetaResponse,
    MirrorRequest,
    MirrorResponse,
    Packet,
    PacketRequest,
    ResolveRequest,
    ScoreRequest,
    ScoreResult,
    Standard,
    WorkingSetResponse,
)
from .score import score_candidates
from .store import HoneBudgetExhausted, Store

PERMUTATION = {
    "schema": "A5-procedural", "retrieval": "B7-confidence",
    "embed": "C3-BGE-small", "scorer": "D1-rules",
    "interview": "E1-adaptive", "hone": "F1-BoN3",
    "storage": "G1-sqlite-wal", "memory": "H6-self-edit",
    "injection": "I4-per-turn", "pipeline": "J2-dynamic",
}


def _error_body(status: int, message: str) -> dict:
    return {"error": {"code": f"e{status}", "message": message}}


def create_app(db_path: str | Path | None = None) -> FastAPI:
    store = Store(db_path) if db_path else Store()
    engine = StandardsEngine(store)
    curator = Curator(engine)

    app = FastAPI(title="taste-6", version="0.1.0")
    app.state.store = store
    app.state.engine = engine
    app.state.curator = curator

    @app.exception_handler(HTTPException)
    async def _http_error(_request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code,
                            content=_error_body(exc.status_code,
                                                str(exc.detail)))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request,
                                exc: RequestValidationError):
        first = exc.errors()[:1]
        message = first[0].get("msg", "invalid request") if first else "invalid request"
        prefix = "Value error, "
        if message.startswith(prefix):
            message = message[len(prefix):]
        return JSONResponse(status_code=422,
                            content=_error_body(422, message))

    # ------------------------------------------------------------ meta & check
    @app.get("/meta", response_model=MetaResponse)
    def meta():
        return MetaResponse(name="taste-6", version="0.1.0",
                            permutation=PERMUTATION)

    @app.get("/check")
    def check(profile_id: str = "owner"):
        active = engine.active(profile_id)
        all_stds = store.list_standards(profile_id)
        cats: dict[str, int] = {}
        pols: dict[str, int] = {}
        for s in active:
            c = s.category.value if hasattr(s.category, "value") else str(s.category)
            p = s.polarity.value if hasattr(s.polarity, "value") else str(s.polarity)
            cats[c] = cats.get(c, 0) + 1
            pols[p] = pols.get(p, 0) + 1
        return {
            "status": "healthy",
            "name": "taste-6",
            "version": "0.1.0",
            "active_standards": len(active),
            "total_standards": len(all_stds),
            "categories": cats,
            "polarities": pols,
        }


    # ------------------------------------------------------------ task flow
    @app.post("/interview", response_model=InterviewResponse)
    def interview(req: InterviewRequest):
        return run_interview(engine, req.profile_id, req.task_ctx)

    @app.post("/mirror", response_model=MirrorResponse)
    def mirror(req: MirrorRequest):
        return build_mirror(engine, store, req)

    @app.post("/packet", response_model=Packet)
    def packet(req: PacketRequest):
        return compile_packet(engine, req)

    @app.post("/score", response_model=ScoreResult)
    def score(req: ScoreRequest):
        return ScoreResult(scores=score_candidates(
            engine, req.profile_id, req.task_ctx, req.candidates))

    @app.post("/hone", response_model=HoneResponse)
    def hone(req: HoneRequest):
        try:
            return hone_segment(engine, req)
        except HoneBudgetExhausted as exc:
            raise HTTPException(429, str(exc)) from exc

    # ------------------------------------------------------------ learning
    @app.post("/feedback", response_model=FeedbackResponse)
    def feedback(req: FeedbackRequest):
        return engine.record_feedback(
            req.profile_id, req.task_ctx, req.pick, req.rationale,
            rejected=req.rejected, candidates=req.candidates)

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(req: IngestRequest):
        resp = ingest_docs(req.profile_id, req.docs)
        for cand in resp.candidates:
            engine.propose(cand)   # lands provisional — never auto-active
        return resp

    # ------------------------------------------------------------ corpus
    @app.get("/standards")
    def standards(profile_id: str = "owner",
                  state: str | None = None):
        st = None
        if state:
            try:
                st = LayerState(state)
            except ValueError:
                raise HTTPException(
                    422, f"invalid state: {state!r} — expected one of "
                    f"{[s.value for s in LayerState]}") from None
        rows = store.list_standards(profile_id, state=st)
        return {"standards": [s.model_dump(mode="json") for s in rows],
                "count": len(rows)}

    @app.post("/standards/import")
    def standards_import(req: ImportRequest):
        return engine.import_standards(req.profile_id, list(req.standards))

    # ------------------------------------------------------------ owner authority
    @app.post("/standards/{sid}/confirm")
    def confirm(sid: str, body: dict | None = None):
        rationale = (body or {}).get("rationale")
        try:
            ok = engine.confirm(sid, rationale=rationale)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        if not ok:
            raise HTTPException(404, f"standard {sid} not found")
        return {"activated": sid}

    @app.post("/standards/{sid}/supersede")
    def supersede(sid: str, new: Standard):
        try:
            return engine.supersede(sid, new)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    # ------------------------------------------------------------ review queue
    @app.get("/review-queue")
    def review_queue(profile_id: str = "owner"):
        return {"pending": store.pending_inferred(profile_id)}

    @app.post("/review-queue/{iid}/resolve")
    def resolve(iid: str, body: ResolveRequest | None = None):
        body = body or ResolveRequest()
        row = store.get_inferred(iid)
        if row is None:
            raise HTTPException(404, f"queue item {iid} not found")
        if row["kind"] == "amendment":
            try:
                result = curator.apply_amendment(
                    row["profile_id"], iid, accept=body.accept,
                    rationale=body.rationale)
            except ValueError as e:
                raise HTTPException(422, str(e)) from e
            return result
        store.resolve_inferred(iid)
        return {"resolved": iid, "applied": False}

    # ------------------------------------------------------------ H6 curation
    @app.post("/curate/propose")
    def curate_propose(req: CurateRequest):
        try:
            iid = curator.propose(req.profile_id, req.proposal)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return {"queued": iid, "kind": "amendment"}

    @app.get("/working-set", response_model=WorkingSetResponse)
    def working_set(profile_id: str = "owner"):
        entries = store.get_working_set(profile_id)
        return WorkingSetResponse(
            entries=entries, count=len(entries))

    return app


app = create_app()
