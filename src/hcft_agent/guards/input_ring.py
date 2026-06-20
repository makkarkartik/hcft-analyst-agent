"""Input ring — the first guardrail, on the raw user query before it enters the graph.

Two concerns at the input boundary:
  * **prompt injection** — a trained classifier (:mod:`injection`), keyword regex as fallback.
  * **PII** — Microsoft Presidio detects AND **redacts** (:mod:`pii`); we enforce, not just flag.

This module is a thin orchestrator over the two industry implementations, with cheap regex
fallbacks so the ring still works (degraded) if a model fails to load — fail-open on the *tool*,
but the detection itself is fail-closed (a positive blocks). Policy (what to do on a hit) lives
in the graph's ``input_guard`` node; this layer only detects + redacts.
"""
from __future__ import annotations

import re
from functools import lru_cache

from .injection import InjectionGuard
from .pii import PIIGuard

# --- regex fallbacks (used only if the models fail to load) ---
_INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (all |the )?(previous|prior|above)",
    r"forget (everything|all|your) (instructions|context)",
    r"you are now", r"new instructions?:", r"system prompt",
    r"reveal (your |the )?(system )?prompt", r"developer mode", r"do anything now|\bDAN\b",
]
_PII_PATTERNS = {
    "US_SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "EMAIL_ADDRESS": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
    "PHONE_NUMBER": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
}
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]
_PII_RE = {k: re.compile(p) for k, p in _PII_PATTERNS.items()}


@lru_cache(maxsize=1)
def _injection_guard() -> InjectionGuard:
    return InjectionGuard()


@lru_cache(maxsize=1)
def _pii_guard() -> PIIGuard:
    return PIIGuard()


def is_injection(text: str) -> tuple[bool, float]:
    """(is_injection, score). Model first; regex fallback if the model can't load."""
    try:
        return _injection_guard().is_injection(text)
    except Exception:
        return any(rx.search(text) for rx in _INJECTION_RE), 0.0


def redact(text: str) -> tuple[str, list[str]]:
    """(redacted_text, pii_entity_types). Presidio; regex fallback. Enforcement, not just flags."""
    try:
        return _pii_guard().redact(text)
    except Exception:
        ents = [k for k, rx in _PII_RE.items() if rx.search(text)]
        out = text
        for k in ents:
            out = _PII_RE[k].sub(f"<{k}>", out)
        return out, sorted(set(ents))


def scan(text: str) -> list[str]:
    """Compatibility helper (used by the UI): flags only, no redaction.
    Returns e.g. ['injection', 'pii:EMAIL_ADDRESS']."""
    flags: list[str] = []
    if is_injection(text)[0]:
        flags.append("injection")
    try:
        ents = _pii_guard().scan(text)
    except Exception:
        ents = [k for k, rx in _PII_RE.items() if rx.search(text)]
    flags += [f"pii:{e}" for e in ents]
    return flags
