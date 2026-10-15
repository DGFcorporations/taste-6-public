"""driver — the model that fills connective tissue (P4 wiring).

The model is just the driver — taste supplies the output. Config below is
the owner's spec from the TASTE heuristic research: high-conviction
inference params tuned against generic creative drift.

EXCLUDED BY OWNER DIRECTIVE: local generation models (Ollama, vLLM,
llama.cpp, any on-box LLM). Drivers are flash-tier API models only.
(Local *embedding* — BGE-small — is the C3 axis and unrelated to
generation.)
"""

from __future__ import annotations

# Inference parameters — owner's spec (Project Taste §4)
TEMPERATURE = 0.35        # against generic creative drift
TOP_P = 0.85              # high-conviction vocabulary only
PRESENCE_PENALTY = 0.10   # varied angles across candidates
REPETITION_PENALTY = 1.05 # no circular justification
MAX_TOKENS = 1024         # punchy outputs, no conversational bloat

# Flash-tier driver candidates — cloud APIs only, cheapest sufficient
DRIVER_CANDIDATES = [
    "gemini-2.5-flash",
    "grok-4-fast",
    "qwen-flash",          # DashScope API — cloud, not local
    "deepseek-chat",
]

# The driver protocol — P4 wires real providers behind this
DriverFn = "Callable[[Optional[str], dict, list[str]], list[str]]"


def call_driver(prompt: str, n: int = 3,
                provider: str | None = None) -> list[str]:
    """Generate n candidates via the configured flash driver.

    P4: implement provider dispatch (LiteLLM or direct SDK). Until then the
    offline deterministic generator in hone.py keeps the loop honest.
    """
    raise NotImplementedError(
        "flash driver wiring lands in P4 — pass the offline driver for now")
