"""RAG chat agent — the P1 vertical slice, as a LangGraph state machine.

    input_guard --(injection?)--> refuse
         |
      retrieve --> grade --(relevant?)--> generate --> output_guard --(grounded?)--> END
         ^                  |                                          |
         |             (weak & retries<N)                        (ungrounded)
         +---- rewrite <----+                                         v
                            (retries exhausted) --------------------> refuse

Design choices we locked:
  * **span per node** (not one span for the whole run) so each step's eval/guard binds to its
    own span. Every node opens ``rag.<node>`` and writes its verdict to ``hcft.*`` attributes;
    LLM calls (generate/rewrite) auto-nest as OpenInference LLM spans inside.
  * **deterministic gates**, no gold at inference: grade = rerank-score threshold; retry cap = N.
  * **inline groundedness guard** (HHEM) on the output ring only — refuse > fabricate.
  * LLM-judge eval (RAGAS/DeepEval/G-Eval) stays OFFLINE; nothing judges in the hot path.

Run:  ``python -m hcft_agent.agents.rag_chat "your question"``
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, StateGraph

from hcft_agent import generate as gen
from hcft_agent.config import settings
from hcft_agent.guards import input_ring
from hcft_agent.guards.groundedness import GroundednessGuard
from hcft_agent.obs import attributes as A
from hcft_agent.obs.telemetry import get_tracer, init_telemetry
from hcft_agent.retriever import Retriever
from hcft_agent.agents.state import RagState

_tracer = get_tracer("hcft-agent")


# --- shared resources (load models once) ---
@lru_cache(maxsize=1)
def _retriever() -> Retriever:
    return Retriever()


@lru_cache(maxsize=1)
def _guard() -> GroundednessGuard:
    return GroundednessGuard()


@lru_cache(maxsize=1)
def _rewriter():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.orchestrator_model,
        base_url=settings.orchestrator_base_url,
        api_key=settings.orchestrator_api_key,
        temperature=0.0,
    )


# =========================== nodes ===========================

def input_guard(state: RagState) -> dict:
    """Scan the raw query; fail CLOSED on injection (refuse before retrieving)."""
    with _tracer.start_as_current_span("rag.input_guard") as span:
        q = state["question"]
        flags = input_ring.scan(q)
        injected = "injection" in flags
        A.set_attrs(span, {A.INPUT_FLAGS: flags or None})
        out: dict = {"input_flags": flags, "query": q, "retries": 0}
        if injected:
            out.update(degraded=True, degraded_reason="input_injection")
        return out


def retrieve(state: RagState) -> dict:
    """Dense + rerank over the current (possibly rewritten) query."""
    with _tracer.start_as_current_span("rag.retrieve") as span:
        cands = _retriever().retrieve(state["query"])
        retrieved_ids = [c["chunk_id"] for c in cands[: settings.context_top_k]]
        A.set_attrs(span, {A.RETRIEVED_IDS: retrieved_ids})
        return {"candidates": cands, "retrieved_ids": retrieved_ids}


def grade(state: RagState) -> dict:
    """Deterministic relevance gate: top reranked candidate must clear the threshold."""
    with _tracer.start_as_current_span("rag.grade") as span:
        cands = state.get("candidates") or []
        top = cands[0].get("rerank_score") if cands else None
        relevant = top is not None and top >= settings.grade_min_rerank_score
        A.set_attrs(span, {
            "hcft.grade.top_score": top,
            "hcft.grade.relevant": relevant,
            A.RETRIES: state.get("retries", 0),
        })
        return {"relevant": relevant}


def rewrite(state: RagState) -> dict:
    """Reformulate the query for a better retrieval pass (bounded by max_rewrites)."""
    with _tracer.start_as_current_span("rag.rewrite") as span:
        msg = (
            "Rewrite this healthcare-facility question to retrieve better evidence. "
            "Keep the intent; make entities/terms explicit. Return ONLY the rewritten query.\n\n"
            f"Question: {state['question']}"
        )
        new_q = (_rewriter().invoke(msg).content or "").strip() or state["query"]
        retries = state.get("retries", 0) + 1
        A.set_attrs(span, {A.RETRIES: retries, "hcft.rewrite.query": new_q})
        return {"query": new_q, "retries": retries}


def generate(state: RagState) -> dict:
    """Grounded answer from the top-k context (reader LLM auto-traced as a nested span)."""
    with _tracer.start_as_current_span("rag.generate") as span:
        result = gen.generate(state["question"], state.get("candidates") or [])
        A.set_attrs(span, {
            A.CITED_IDS: result["cited_ids"] or None,
            A.IS_REFUSAL: result["is_refusal"],
        })
        return {
            "answer": result["answer"],
            "cited_ids": result["cited_ids"],
            "context": result["context"],
            "is_refusal": result["is_refusal"],
        }


def output_guard(state: RagState) -> dict:
    """Inline groundedness check (HHEM). Below threshold -> refuse instead of fabricating."""
    with _tracer.start_as_current_span("rag.output_guard") as span:
        grounded, score = _guard().is_grounded(state.get("context", ""), state.get("answer", ""))
        out: dict
        if grounded or state.get("is_refusal"):
            out = {"grounded": grounded, "grounded_score": score, "terminal": "answer"}
        else:
            # answer not supported by context -> drop it, refuse honestly
            out = {
                "grounded": False, "grounded_score": score,
                "answer": gen.REFUSAL_TEXT, "is_refusal": True, "terminal": "refuse",
                "degraded": True, "degraded_reason": "ungrounded_output",
            }
        A.set_attrs(span, {
            A.GROUNDED: out["grounded"], A.GROUNDED_SCORE: score,
            A.OUTPUT_GROUNDED: out["grounded"], A.TERMINAL: out["terminal"],
            A.IS_REFUSAL: out.get("is_refusal", state.get("is_refusal", False)),
            A.DEGRADED: out.get("degraded"), A.DEGRADED_REASON: out.get("degraded_reason"),
        })
        return out


def refuse(state: RagState) -> dict:
    """Single terminal refusal node (injection or retrieval-exhausted)."""
    with _tracer.start_as_current_span("rag.refuse") as span:
        reason = state.get("degraded_reason", "no_relevant_context")
        A.set_attrs(span, {
            A.TERMINAL: "refuse", A.IS_REFUSAL: True,
            A.DEGRADED: True, A.DEGRADED_REASON: reason,
        })
        return {
            "answer": gen.REFUSAL_TEXT, "is_refusal": True, "terminal": "refuse",
            "degraded": True, "degraded_reason": reason,
        }


# =========================== routing ===========================

def _after_input(state: RagState) -> str:
    return "refuse" if "injection" in state.get("input_flags", []) else "retrieve"


def _after_grade(state: RagState) -> str:
    if state.get("relevant"):
        return "generate"
    return "rewrite" if state.get("retries", 0) < settings.max_rewrites else "refuse"


# =========================== build ===========================

@lru_cache(maxsize=1)
def build_app():
    """Compile the graph once. Telemetry is initialised here so spans export on first use."""
    init_telemetry("hcft-agent")
    g = StateGraph(RagState)
    for name, fn in [
        ("input_guard", input_guard), ("retrieve", retrieve), ("grade", grade),
        ("rewrite", rewrite), ("generate", generate), ("output_guard", output_guard),
        ("refuse", refuse),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("input_guard")
    g.add_conditional_edges("input_guard", _after_input,
                            {"retrieve": "retrieve", "refuse": "refuse"})
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _after_grade,
                            {"generate": "generate", "rewrite": "rewrite", "refuse": "refuse"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "output_guard")
    g.add_edge("output_guard", END)
    g.add_edge("refuse", END)
    return g.compile()


def answer(question: str) -> RagState:
    """Convenience entry: run the graph end-to-end for one question."""
    return build_app().invoke({"question": question})


if __name__ == "__main__":
    import sys

    from hcft_agent.obs.telemetry import flush

    q = " ".join(sys.argv[1:]) or "What infection control deficiencies were cited?"
    final = answer(q)
    print(f"\nQ: {q}")
    print(f"terminal={final.get('terminal')}  grounded={final.get('grounded')} "
          f"({final.get('grounded_score'):.3f})  refusal={final.get('is_refusal')}  "
          f"retries={final.get('retries')}  flags={final.get('input_flags')}")
    print(f"cited={final.get('cited_ids')}")
    print(f"\nA: {final.get('answer')}")
    flush()
