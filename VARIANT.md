# VARIANT: taste-6 — self-editing agent

**Thesis under test:** can an agent curate its own working taste set?
Letta-inspired — the agent maintains a bounded working set of layers it
proposes/promotes/demotes (propose-only — owner still approves activation).
Per-turn injection instead of per-task.

## Permutation

| Axis | Choice | Divergence from control |
|---|---|---|
| A schema | A5 procedural patterns (multi-step units, not just rules) | process taste |
| B retrieval | B7 confidence-gated (only layers ≥ threshold apply) | quality floor as retrieval |
| C embedding | C3 BGE-small-en-v1.5 | retrieval-tuned alternative |
| D scorer | D1 rules | held w/ control |
| E interview | E1 adaptive | held w/ control |
| F hone | F1 BoN-3 | held w/ control |
| G storage | G1 SQLite WAL | held w/ control |
| H memory | **H6 self-edit working set (propose-only)** | the axis under test |
| I injection | **I4 per-turn injection** (fresh taste per model call) | the axis under test |
| J pipeline | J2 dynamic decomposition | agent-shaped segments |

## The question it answers

If agents keep their own taste working set well-curated (with owner
gating), the engine scales to more profiles and more domains without
retrieval tuning per task. If self-edit drifts, it dies week 3 — that's
what the cohort is for.
