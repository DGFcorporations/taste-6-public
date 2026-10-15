"""taste-6 — self-editing agent variant of the TASTE cohort.

Permutation: A5 procedural patterns · B7 confidence-gated retrieval ·
C3 BGE-small · D1 rules scorer · E1 adaptive interview · F1 BoN-3 hone ·
G1 SQLite WAL store · H6 self-edit working set (propose-only) ·
I4 per-turn injection · J2 dynamic segments. See VARIANT.md.

The agent curates a bounded 20-slot working set of layers it proposes
into — the owner confirms every mutation. Retrieval is a confidence
floor: only battle-tested layers reach a packet.
"""

__version__ = "0.1.0"
