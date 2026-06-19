"""Input ring — the first guardrail, on the raw user query before it enters the graph.

Two concerns, both at the *input* boundary (cf. the output ring, which checks groundedness):
  * **prompt injection** — attempts to override the system/instructions ("ignore previous…").
  * **PII** — query-side personal data we shouldn't retrieve on or log verbatim.

This is a fast HEURISTIC first pass (regex/keyword). It is deliberately a placeholder for a
trained classifier — the industry tool here is **Meta Prompt-Guard-86M** (a fine-tuned
DeBERTa for jailbreak/injection detection), which we'll swap in behind this same
``scan()`` signature. Heuristics catch the obvious and cost ~nothing; the model catches the
subtle. Per our ground rule we adopt the tool, but ship the cheap gate first so the ring
exists end-to-end on day one.

``scan()`` returns the list of flags (empty == clean). Policy (fail-closed on injection) lives
in the graph node, not here — this only *detects*.
"""
from __future__ import annotations

import re

# Override / jailbreak phrasings. Lowercased substring match — crude but high-precision.
_INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (all |the )?(previous|prior|above)",
    r"forget (everything|all|your) (instructions|context)",
    r"you are now",
    r"new instructions?:",
    r"system prompt",
    r"reveal (your |the )?(system )?prompt",
    r"act as (if|though|a)",
    r"developer mode",
    r"do anything now|\bDAN\b",
]

# PII signatures. Coarse on purpose — recall over precision at a guardrail.
_PII_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "email": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
    "phone": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
}

_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]
_PII_RE = {k: re.compile(p) for k, p in _PII_PATTERNS.items()}


def scan(text: str) -> list[str]:
    """Return guardrail flags for a query. ``[]`` means clean.

    Flags: ``"injection"`` and/or ``"pii:<kind>"`` (e.g. ``"pii:email"``)."""
    flags: list[str] = []
    if any(rx.search(text) for rx in _INJECTION_RE):
        flags.append("injection")
    for kind, rx in _PII_RE.items():
        if rx.search(text):
            flags.append(f"pii:{kind}")
    return flags
