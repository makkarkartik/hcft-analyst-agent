"""LLM-judge layer — the slow, offline counterpart to the deterministic anchors in
``agent_eval``. Two judges from DELIBERATELY different model families:

  * **RAGAS** faithfulness + answer-relevancy — run on a **cross-family** model (Fireworks
    Llama-3.3-70B by default). Faithfulness decomposes the answer into atomic claims and runs an
    NLI check of each against the retrieved context (2+ sequential LLM calls — why this is OFFLINE
    only). Answer-relevancy back-generates questions from the answer and measures their embedding
    similarity to the real one — "is the answer actually responsive".
  * **DeepEval G-Eval** refusal-quality — a chain-of-thought, form-filling judge on the SAME
    family as the reader (OpenAI gpt-4o-mini). It scores, from the context alone, whether the
    answer-vs-refuse DECISION was correct. Same-family makes it the strict gate judge AND the one
    we validate with κ — a same-family judge is the circularity we most need to check.

This module exposes BUILDERS (`build_ragas`, `build_geval`) + per-row scorers so the LangSmith
``evaluate()`` path and the offline batch path share ONE implementation of each metric. Batch
wrappers (`ragas_scores`, `geval_refusal`) sit on top for the no-LangSmith path. Every entry point
degrades to ``{"error": ...}`` rather than throwing, so one missing dep never sinks a run.
"""
from __future__ import annotations

import asyncio
import re

from hcft_agent.config import settings

_SOURCE_SPLIT = re.compile(r"\n\n(?=\[Source\s+\d+\])")


def split_context(context: str) -> list[str]:
    """Split the assembled ``[Source N] ...`` block back into a list — RAGAS/DeepEval want the
    retrieved contexts as separate items, not one concatenated blob."""
    if not context:
        return []
    return [p.strip() for p in _SOURCE_SPLIT.split(context.strip()) if p.strip()]


# ============================================================ RAGAS (cross-family)
def _shim_ragas_vertexai() -> None:
    """ragas 0.4.3 still does `from langchain_community.chat_models.vertexai import ChatVertexAI`
    (+ `langchain_community.llms.VertexAI`), but langchain-community 0.4.x (the sunset release this
    env is on, alongside langchain-core 1.x) deleted those paths. We never use VertexAI, so register
    inert stubs before ragas imports — a 2-line upstream skew, bridged, not reimplemented."""
    import sys
    import types

    import langchain_community.llms as _llms
    if not hasattr(_llms, "VertexAI"):
        _llms.VertexAI = type("VertexAI", (), {})
    key = "langchain_community.chat_models.vertexai"
    if key not in sys.modules:
        m = types.ModuleType(key)
        m.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[key] = m


def build_ragas():
    """Build the cross-family (faithfulness, answer_relevancy) metric pair, judge LLM + embeddings
    wired. Raises if ragas/langchain is missing so the caller records a clean 'unavailable'."""
    _shim_ragas_vertexai()
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy

    judge = ChatOpenAI(                       # Fireworks via the OpenAI-compatible API = cross-family
        model=settings.ragas_judge_model, base_url=settings.ragas_judge_base_url,
        api_key=settings.ragas_judge_api_key, temperature=0.0,
    )
    emb = OpenAIEmbeddings(model=settings.ragas_embed_model, api_key=settings.orchestrator_api_key)
    llm = LangchainLLMWrapper(judge)
    ew = LangchainEmbeddingsWrapper(emb)
    return Faithfulness(llm=llm), ResponseRelevancy(llm=llm, embeddings=ew)


def ragas_score_row(faith, relev, question: str, answer: str, contexts: list[str]) -> tuple[float, float]:
    """Score one (question, answer, contexts) -> (faithfulness, answer_relevancy)."""
    from ragas.dataset_schema import SingleTurnSample

    sample = SingleTurnSample(user_input=question, response=answer,
                              retrieved_contexts=contexts or [""])

    async def go():
        return float(await faith.single_turn_ascore(sample)), float(await relev.single_turn_ascore(sample))

    return asyncio.run(go())


def is_cross_family() -> bool:
    return "openai" not in settings.ragas_judge_base_url


def ragas_scores(recs: list[dict]) -> dict:
    """Batch RAGAS over grounded, non-refused rows (offline path). Per-row + means."""
    rows = [r for r in recs
            if r["eval_kind"] != "unanswerable_rag" and not r["is_refusal"]
            and r.get("agent_answer") and r.get("context")]
    out: dict = {"n_scored": len(rows), "judge": settings.ragas_judge_model, "cross_family": is_cross_family()}
    if not rows:
        out["note"] = "no grounded non-refused rows to score"
        return out
    try:
        faith, relev = build_ragas()
    except Exception as e:
        out["error"] = f"ragas unavailable ({type(e).__name__}: {e})"
        return out

    per_row, faiths, arels = [], [], []
    for r in rows:
        try:
            f, a = ragas_score_row(faith, relev, r["question"], r["agent_answer"],
                                   split_context(r["context"]))
            per_row.append({"qa_id": r["qa_id"], "faithfulness": round(f, 3), "answer_relevancy": round(a, 3)})
            if f == f:  # not NaN
                faiths.append(f)
            if a == a:
                arels.append(a)
        except Exception as e:
            per_row.append({"qa_id": r["qa_id"], "error": f"{type(e).__name__}: {e}"})
    out["per_row"] = per_row
    out["faithfulness_mean"] = round(sum(faiths) / len(faiths), 3) if faiths else None
    out["answer_relevancy_mean"] = round(sum(arels) / len(arels), 3) if arels else None
    return out


# ============================================================ DeepEval G-Eval (same-family)
GEVAL_CRITERIA = (
    "You are auditing a healthcare-facility RAG answer for DECISION correctness given ONLY the "
    "retrieved context (no outside knowledge).\n"
    "An answer is APPROPRIATE when:\n"
    "  - it gives the SPECIFIC information the question asks for AND that information is directly "
    "stated in the context (a grounded, specific answer); OR\n"
    "  - it cleanly refuses ('not enough information ...') AND the context does NOT contain the "
    "specific information requested (an honest refusal).\n"
    "Penalize heavily: (a) answering with a figure/plan/fact the context does not actually state "
    "(fabrication or over-reach), and (b) refusing when the context plainly contains the exact "
    "answer (over-refusal). Being on-topic is NOT enough to justify an answer."
)


def build_geval():
    """Build the G-Eval 'Refusal Correctness' metric (raises if deepeval is missing)."""
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams

    return GEval(
        name="Refusal Correctness", criteria=GEVAL_CRITERIA,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                           LLMTestCaseParams.RETRIEVAL_CONTEXT],
        model=settings.geval_judge_model, threshold=settings.geval_threshold,
    )


def geval_score_row(metric, question: str, answer: str, contexts: list[str]) -> tuple[float, str]:
    """Score one row -> (score 0..1, reason)."""
    from deepeval.test_case import LLMTestCase

    tc = LLMTestCase(input=question, actual_output=answer or "",
                     retrieval_context=contexts or ["(empty)"])
    metric.measure(tc)
    return float(metric.score), (metric.reason or "")


def geval_refusal(recs: list[dict]) -> dict:
    """Batch G-Eval over all rows (offline path). Per-row score/verdict + mean + pass-rate."""
    out: dict = {"n_scored": 0, "judge": settings.geval_judge_model, "threshold": settings.geval_threshold}
    rows = [r for r in recs if r.get("question")]
    if not rows:
        return out
    try:
        metric = build_geval()
    except Exception as e:
        out["error"] = f"deepeval unavailable ({type(e).__name__}: {e})"
        return out

    per_row, scores = [], []
    for r in rows:
        try:
            s, reason = geval_score_row(metric, r["question"], r.get("agent_answer") or "",
                                        split_context(r.get("context", "")))
            per_row.append({"qa_id": r["qa_id"], "score": round(s, 3),
                            "appropriate": s >= settings.geval_threshold, "reason": reason[:280]})
            scores.append(s)
        except Exception as e:
            per_row.append({"qa_id": r["qa_id"], "error": f"{type(e).__name__}: {e}"})
    out["n_scored"] = len(scores)
    out["per_row"] = per_row
    out["mean_score"] = round(sum(scores) / len(scores), 3) if scores else None
    out["pass_rate"] = round(sum(1 for p in per_row if p.get("appropriate")) / len(scores), 3) if scores else None
    return out
