"""Prompt-injection / jailbreak classifier — a purpose-trained model, not keyword heuristics.

Default: ``protectai/deberta-v3-base-prompt-injection-v2`` (open, non-gated; the model LLM Guard
ships). Swap to Meta's ``Llama-Prompt-Guard-2-86M`` by setting ``INJECTION_MODEL`` + ``HF_TOKEN``
and accepting its license (it's gated). The label mapping is normalized so either model works:
we take the probability mass on the *malicious* class regardless of how it's named.

Lazy-loaded; runs ~tens of ms on GPU per query (small DeBERTa), so it sits cheaply in the hot path.
"""
from __future__ import annotations

from hcft_agent.config import settings

# class names that mean "this is an attack", across model conventions
_MALICIOUS = {"INJECTION", "JAILBREAK", "MALICIOUS", "UNSAFE", "LABEL_1"}


class InjectionGuard:
    def __init__(self) -> None:
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            import torch
            from transformers import pipeline

            print(f"[guard] loading injection classifier {settings.injection_model} ...")
            self._pipe = pipeline(
                "text-classification", model=settings.injection_model,
                device=0 if torch.cuda.is_available() else -1,
                truncation=True, max_length=512, top_k=None,  # return all class scores
            )
        return self._pipe

    @staticmethod
    def _malicious_mass(class_scores: list[dict]) -> float:
        return max((s["score"] for s in class_scores if s["label"].upper() in _MALICIOUS), default=0.0)

    def score(self, text: str) -> float:
        """P(malicious) in 0..1 — the mass on the injection/jailbreak class."""
        if not text.strip():
            return 0.0
        return self._malicious_mass(self._load()(text)[0])  # [0] -> this input's class list

    def scores(self, texts: list[str]) -> list[float]:
        """Batch P(malicious) — one forward pass for many chunks (the context-ring scan path).
        Empty/blank texts score 0.0 without hitting the model."""
        idx = [i for i, t in enumerate(texts) if t and t.strip()]
        out = [0.0] * len(texts)
        if not idx:
            return out
        results = self._load()([texts[i] for i in idx])  # list[list[{label,score}]]
        for i, res in zip(idx, results):
            out[i] = self._malicious_mass(res)
        return out

    def is_injection(self, text: str) -> tuple[bool, float]:
        s = self.score(text)
        return s >= settings.injection_threshold, s
