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
from hcft_agent.guards import context_ring, input_ring
from hcft_agent.guards.groundedness import GroundednessGuard
from hcft_agent.obs.telemetry import init_telemetry, tag
from hcft_agent.retriever import Retriever
from hcft_agent.agents.state import RagState

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
# Each node is auto-traced by LangGraph's native tracer (one nested run per node, with its
# input/output state). We don't open manual spans; we just attach the domain verdicts via
# tag() so the hcft.* values are queryable on the node's run.

def input_guard(state: RagState) -> dict:
    """Scan the raw query; fail CLOSED on injection, and REDACT PII before it's embedded/logged.

    PII redaction is enforcement, not just detection: the redacted query is what flows downstream
    (into Pinecone and onto the trace) — the raw PII never leaves this node. We redact only
    unambiguous identifiers, so retrieval-relevant entities (place/org names) are preserved."""
    q = state["question"]
    injected, inj_score = input_ring.is_injection(q)
    redacted, pii_entities = input_ring.redact(q)          # enforce: query is now PII-free
    flags = (["injection"] if injected else []) + [f"pii:{e}" for e in pii_entities]
    tag(**{"hcft.input_flags": flags or None, "hcft.injection_score": round(inj_score, 3),
           "hcft.pii_redacted": pii_entities or None})
    out: dict = {"input_flags": flags, "query": redacted, "retries": 0}
    if injected:
        out.update(degraded=True, degraded_reason="input_injection")
    return out


def retrieve(state: RagState) -> dict:
    """Dense + rerank over the current (possibly rewritten) query. The context ring (next node)
    owns ``retrieved_ids`` because it may quarantine chunks before they reach the window."""
    cands = _retriever().retrieve(state["query"])
    return {"candidates": cands}


def context_guard(state: RagState) -> dict:
    """Indirect-injection scan on the retrieved chunks (untrusted input) BEFORE they reach the
    reader. Quarantine poisoned chunks, answer from the clean remainder; if none survive, fail
    closed (route to refuse). ``retrieved_ids`` is the post-quarantine window the generator sees."""
    cands = state.get("candidates") or []
    clean, quarantined, max_score = context_ring.scan_context(cands)
    retrieved_ids = [c["chunk_id"] for c in clean[: settings.context_top_k]]
    q_ids = [q["chunk_id"] for q in quarantined]
    tag(**{"hcft.context_threat": max_score, "hcft.quarantined_ids": q_ids or None,
           "hcft.retrieved_ids": retrieved_ids})
    out: dict = {"candidates": clean, "retrieved_ids": retrieved_ids,
                 "quarantined_ids": q_ids, "context_threat": max_score}
    if not clean and quarantined:
        out.update(degraded=True, degraded_reason="poisoned_context")
    return out


def grade(state: RagState) -> dict:
    """Deterministic relevance gate: top reranked candidate must clear the threshold."""
    cands = state.get("candidates") or []
    top = cands[0].get("rerank_score") if cands else None
    relevant = top is not None and top >= settings.grade_min_rerank_score
    tag(**{"hcft.grade.top_score": top, "hcft.grade.relevant": relevant,
           "hcft.retries": state.get("retries", 0)})
    return {"relevant": relevant}


def rewrite(state: RagState) -> dict:
    """Reformulate the query for a better retrieval pass (bounded by max_rewrites)."""
    msg = (
        "Rewrite this healthcare-facility question to retrieve better evidence. "
        "Keep the intent; make entities/terms explicit. Return ONLY the rewritten query.\n\n"
        f"Question: {state['question']}"
    )
    new_q = (_rewriter().invoke(msg).content or "").strip() or state["query"]
    retries = state.get("retries", 0) + 1
    tag(**{"hcft.retries": retries, "hcft.rewrite.query": new_q})
    return {"query": new_q, "retries": retries}


def generate(state: RagState) -> dict:
    """Grounded answer from the top-k context (reader LLM auto-traced as a nested run)."""
    result = gen.generate(state["question"], state.get("candidates") or [])
    tag(**{"hcft.cited_ids": result["cited_ids"] or None, "hcft.is_refusal": result["is_refusal"]})
    return {
        "answer": result["answer"],
        "cited_ids": result["cited_ids"],
        "context": result["context"],
        "is_refusal": result["is_refusal"],
    }


def output_guard(state: RagState) -> dict:
    """Inline groundedness check (HHEM). If the answer isn't supported by the context, hand off to
    the single ``refuse`` sink (don't build the refusal here) — one place owns refusal construction."""
    grounded, score = _guard().is_grounded(state.get("context", ""), state.get("answer", ""))
    passed = grounded or state.get("is_refusal", False)
    out: dict = {"grounded": grounded, "grounded_score": score,
                 "terminal": "answer" if passed else "refuse"}
    if not passed:                          # ungrounded answer -> drop it, route to refuse
        out.update(degraded=True, degraded_reason="ungrounded_output")
    tag(**{"hcft.grounded": grounded, "hcft.grounded_score": score, "hcft.terminal": out["terminal"],
           "hcft.is_refusal": state.get("is_refusal", False) or not passed,
           "hcft.degraded_reason": out.get("degraded_reason")})
    return out


def refuse(state: RagState) -> dict:
    """Single terminal refusal node (injection or retrieval-exhausted)."""
    reason = state.get("degraded_reason", "no_relevant_context")
    tag(**{"hcft.terminal": "refuse", "hcft.is_refusal": True, "hcft.degraded_reason": reason})
    return {
        "answer": gen.REFUSAL_TEXT, "is_refusal": True, "terminal": "refuse",
        "degraded": True, "degraded_reason": reason,
    }


# =========================== routing ===========================

def _after_input(state: RagState) -> str:
    return "refuse" if "injection" in state.get("input_flags", []) else "retrieve"


def _after_context(state: RagState) -> str:
    # every retrieved chunk was quarantined as poisoned -> fail closed; else proceed to grade
    if not state.get("candidates") and state.get("quarantined_ids"):
        return "refuse"
    return "grade"


def _after_grade(state: RagState) -> str:
    if state.get("relevant"):
        return "generate"
    return "rewrite" if state.get("retries", 0) < settings.max_rewrites else "refuse"


def _after_output(state: RagState) -> str:
    # grounded answer (or an honest refusal from the reader) -> END; ungrounded -> the refuse sink
    return "end" if (state.get("grounded") or state.get("is_refusal")) else "refuse"


# =========================== build ===========================

@lru_cache(maxsize=1)
def build_app():
    """Compile the graph once. Telemetry is initialised here so spans export on first use."""
    init_telemetry("hcft-agent")
    g = StateGraph(RagState)
    for name, fn in [
        ("input_guard", input_guard), ("retrieve", retrieve), ("context_guard", context_guard),
        ("grade", grade), ("rewrite", rewrite), ("generate", generate),
        ("output_guard", output_guard), ("refuse", refuse),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("input_guard")
    g.add_conditional_edges("input_guard", _after_input,
                            {"retrieve": "retrieve", "refuse": "refuse"})
    g.add_edge("retrieve", "context_guard")
    g.add_conditional_edges("context_guard", _after_context,
                            {"grade": "grade", "refuse": "refuse"})
    g.add_conditional_edges("grade", _after_grade,
                            {"generate": "generate", "rewrite": "rewrite", "refuse": "refuse"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "output_guard")
    g.add_conditional_edges("output_guard", _after_output, {"end": END, "refuse": "refuse"})
    g.add_edge("refuse", END)
    return g.compile()


def warmup() -> None:
    """Fire one throwaway pass so the embedder/reranker CUDA kernels + Pinecone TLS are hot.
    Cold first-call JIT is ~25s; warm is ~0.5s. Call at server/eval startup so the cold cost
    lands here, not on a user's first real query. The UI does the equivalent in its loaders."""
    try:
        _retriever().retrieve("warmup")
        _guard().is_grounded("warmup context", "warmup answer")
        input_ring.is_injection("warmup")          # warms the injection model (input + context rings)
    except Exception:
        pass


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
