"""taste CLI — operator surface for taste-6.

    taste ingest <file.md>                — extract candidate layers
    taste standards [--state=..]          — list corpus
    taste confirm <std_id>                — activate a provisional layer
    taste packet --type=write_doc          — compile the task packet
           [--turn-segment=lead --turn-index=2]  — per-turn bundle (I4)
    taste feedback --pick=A --why="..."   — log an owner decision
    taste queue                           — pending inferred items
    taste working-set                     — the bounded curated context
    taste curate --action=pin --std=ID --why="..." [--weight=0.8]
                                          — propose a WS mutation (H6)
    taste resolve --id=<iid> --accept [--rationale="..."]
                                          — owner resolves an amendment
    taste serve [--sock PATH]             — print the serve command
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .api import create_app
from .ingest import ingest_docs
from .inject import compile_packet
from .models import LayerState, PacketRequest


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(prog="taste")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest")
    pi.add_argument("file")
    ps = sub.add_parser("standards")
    ps.add_argument("--state", default=None)
    pc = sub.add_parser("confirm")
    pc.add_argument("std_id")
    pp = sub.add_parser("packet")
    pp.add_argument("--type", default="default")
    pp.add_argument("--turn-segment", default=None)
    pp.add_argument("--turn-index", type=int, default=None)
    pf = sub.add_parser("feedback")
    pf.add_argument("--pick", required=True)
    pf.add_argument("--why", required=True)
    sub.add_parser("queue")
    sub.add_parser("working-set")
    pu = sub.add_parser("curate")
    pu.add_argument("--action", required=True,
                    choices=["pin", "demote", "suggest_retire"])
    pu.add_argument("--std", required=True)
    pu.add_argument("--why", required=True)
    pu.add_argument("--weight", type=float, default=None)
    pv = sub.add_parser("resolve")
    pv.add_argument("--id", required=True)
    pv.add_argument("--accept", action="store_true")
    pv.add_argument("--rationale", default=None)
    pserve = sub.add_parser("serve")
    pserve.add_argument("--sock", default="./taste6.sock")

    args = p.parse_args(argv)

    if args.cmd == "serve":
        print(f"  granian --interface asgi --uds {args.sock} taste.api:app")
        print(f"  # or: uvicorn --uds {args.sock} taste.api:app")
        return 0

    app = create_app()
    engine, store = app.state.engine, app.state.store

    if args.cmd == "ingest":
        text = Path(args.file).read_text()
        resp = ingest_docs("owner", [{"id": args.file, "text": text}])
        for c in resp.candidates:
            engine.propose(c)
        print(f"{len(resp.candidates)} candidate layers (provisional):")
        for r in resp.restatements:
            print(f"  [{r.layer.polarity.value}] {r.layer.directive}")
            for imp in r.implications[:2]:
                print(f"      -> {imp}")

    elif args.cmd == "standards":
        for s in store.list_standards(
                "owner", state=LayerState(args.state) if args.state else None):
            print(f"[{s.state.value:11}] [{s.polarity.value:5}] "
                  f"{s.id}  {s.directive[:70]}")

    elif args.cmd == "confirm":
        print("activated" if engine.confirm(args.std_id) else "not found")

    elif args.cmd == "packet":
        turn_ctx = None
        if args.turn_segment or args.turn_index is not None:
            turn_ctx = {"segment": args.turn_segment or "",
                        "turn_index": args.turn_index or 0}
        pkt = compile_packet(engine, PacketRequest(
            task_ctx={"type": args.type}, turn_ctx=turn_ctx))
        print(pkt.model_dump_json(indent=2))

    elif args.cmd == "feedback":
        resp = engine.record_feedback("owner", {}, args.pick, args.why)
        print(f"decision {resp.record_id} logged; "
              f"1 provisional why-layer extracted")

    elif args.cmd == "queue":
        for item in store.pending_inferred("owner"):
            print(f"[{item['kind']}] {item['id']}  {item['ts']}")

    elif args.cmd == "working-set":
        entries = store.get_working_set("owner")
        if not entries:
            print("working set: empty — curate with `taste curate`, "
                  "owner resolves with `taste resolve`")
            return 0
        print(f"working set: {len(entries)}/20 slots")
        for e in entries:
            print(f"  [{e.slot:2}] w={e.weight:.2f} [{e.state.value}] "
                  f"{e.standard_id}  {e.directive[:60]}")

    elif args.cmd == "curate":
        from .models import CurationProposal

        proposal = CurationProposal(
            action=args.action, standard_id=args.std, why=args.why,
            weight=args.weight)
        try:
            iid = app.state.curator.propose("owner", proposal)
        except ValueError as e:
            print(f"refused: {e}")
            return 1
        print(f"queued amendment {iid} — propose-never-apply; "
              f"resolve with `taste resolve --id {iid} --accept`")

    elif args.cmd == "resolve":
        try:
            result = app.state.curator.apply_amendment(
                "owner", args.id, accept=args.accept,
                rationale=args.rationale)
        except (KeyError, ValueError) as e:
            print(f"refused: {e}")
            return 1
        state = "applied" if result["applied"] else "rejected"
        print(f"amendment {args.id} resolved: {state}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
