"""Output ring — the inline groundedness guard (fast, deterministic, non-circular).

Wraps **Vectara HHEM-2.1-open**, a purpose-trained cross-encoder that scores
``P(hypothesis is factually consistent with premise)`` in one forward pass. We use it as the
*live* counterpart to the offline RAGAS-faithfulness metric: same concept (is the answer
grounded in the retrieved context?), but ~tens of ms instead of the 2-5 s a RAGAS LLM
pipeline costs — because every answer must pass through it.

Why a trained model and not an LLM judge here:
  * deterministic — same (context, answer) -> same score, so the refuse/answer verdict is stable;
  * fast — one cross-encoder pass, no network;
  * non-circular — it's not an LLM grading another LLM's output.

The policy (score < threshold -> refuse rather than fabricate) lives in the graph's output-guard
node; this class only *scores*. Lazy-loaded so importing it is free.
"""
from __future__ import annotations

from hcft_agent.config import settings


class GroundednessGuard:
    def __init__(self) -> None:
        self._model = None

    def _get_model(self):
        if self._model is None:
            from transformers import AutoModelForSequenceClassification

            print(f"[guard] loading groundedness model {settings.hhem_model} ...")
            # HHEM ships a custom model class -> trust_remote_code; exposes .predict(pairs).
            self._model = AutoModelForSequenceClassification.from_pretrained(
                settings.hhem_model, trust_remote_code=True
            )
            # HHEM-2.1-open is T5-based (relative position embeddings -> NO hard length limit;
            # verified empirically it scores evidence placed past token 512). Its bundled T5
            # tokenizer ships model_max_length=512, which only emits a *spurious* "sequence too
            # long" warning — no truncation actually happens. Raise it to the model's real ~8k
            # support so the warning stops misleading. (Vectara misspells the attr as 'tokenzier'.)
            tok = getattr(self._model, "tokenzier", None) or getattr(self._model, "tokenizer", None)
            if tok is not None:
                tok.model_max_length = 8192
        return self._model

    def score(self, context: str, answer: str) -> float:
        """P(answer is grounded in context), 0..1. Higher == better grounded.

        Empty context or empty answer -> 0.0 (nothing to be grounded in / nothing said)."""
        if not context.strip() or not answer.strip():
            return 0.0
        from hcft_agent.obs.telemetry import trace_block

        with trace_block("guard.hhem", run_type="tool"):
            # HHEM expects (premise, hypothesis) == (evidence, claim) == (context, answer).
            scores = self._get_model().predict([(context, answer)])
            return float(scores[0])

    def is_grounded(self, context: str, answer: str) -> tuple[bool, float]:
        """Convenience: (passes_threshold, score) using ``settings.grounded_min_score``."""
        s = self.score(context, answer)
        return s >= settings.grounded_min_score, s
