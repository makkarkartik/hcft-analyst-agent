"""M2 — self-corrective RAG (CRAG-style) research agent. ASYNC.

Graph (■ node, ◇ conditional edge, ↺ correction loop):

                          START
                            │
                            ▼
                      ┌───────────┐
            ┌────────▶│ retrieve  │◀────────────┐
            │         └─────┬─────┘              │
            │               ▼                    │
            │      ┌─────────────────┐           │
            │      │ grade_documents │           │
            │      └────────┬────────┘           │
            │               ▼                    │
            │        ◇ relevant? ──── no ──┐     │  ↺ retrieval-miss loop
            │               │ yes          │     │     (until retries == MAX)
            │               ▼              ▼     │
            │         ┌──────────┐   ┌──────────────┐
            │         │ generate │   │ rewrite_query│─┘
            │         └────┬─────┘   └──────────────┘
            │              ▼                ▲
            │   ┌────────────────────┐      │
            │   │ grade_groundedness │      │
            │   └─────────┬──────────┘      │
            │             ▼                 │
            │      ◇ grounded? ─── no ──────┘   ↺ groundedness loop
            │             │ yes                    (re-retrieve; see weakness note)
            │             ▼
            │            END
            └── (rewrite_query always re-enters retrieve)

Both ◇ edges also short-circuit to their "yes" branch once retries == MAX_RETRIES,
so neither loop can spin forever (recursion_limit=15 is the hard backstop).

Architecture notes (interrogate these):
- ASYNC nodes (`async def` + `ainvoke`). The blocking retriever call (GPU embed + network) is
  pushed off the event loop with `asyncio.to_thread`, so concurrent requests aren't serialized
  by it. (JD: "asynchronous programming ... multi-step agents".)
- Graders return STRUCTURED yes/no so a conditional edge can branch deterministically.
- Grader NODE does the LLM call and writes its verdict to state; the conditional EDGE is a cheap,
  pure function that only reads state (edges run hot — keep them I/O-free).
- Per-node models + prompts: orchestrator grades/rewrites; the swappable READER generates.
- Two independent stops prevent an infinite rewrite loop: the retry budget (domain logic) and
  LangGraph's `recursion_limit` (hard safety net).
"""

import asyncio
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from hcft_agent.config import settings
from hcft_agent.retriever import Retriever

MAX_RETRIES = 2

retriever = Retriever()

# Orchestrator model: grading + rewriting (judgment work).
orchestrator = ChatOpenAI(
    base_url=settings.orchestrator_base_url,
    api_key=settings.orchestrator_api_key,
    model=settings.orchestrator_model,
    temperature=0.0,
)
# Reader model: the swappable generation slot (frontier now, raft-3b in M9).
reader = ChatOpenAI(
    base_url=settings.reader_base_url,
    api_key=settings.reader_api_key,
    model=settings.reader_model,
    temperature=0.0,
)


class GradeDocuments(BaseModel):
    """Binary relevance of the retrieved context to the question."""
    binary_score: str = Field(description="'yes' if the context is relevant to the question, else 'no'")


class GradeGroundedness(BaseModel):
    """Whether the drafted answer is supported by the retrieved context."""
    binary_score: str = Field(description="'yes' if the answer is grounded in the context, else 'no'")


doc_grader = orchestrator.with_structured_output(GradeDocuments)
grounded_grader = orchestrator.with_structured_output(GradeGroundedness)


def _is_yes(score: str) -> bool:
    return score.strip().lower().startswith("y")


class RagState(TypedDict):
    question: str        # the user's original question (fixed)
    query: str           # working retrieval query (rewritten on a miss)
    documents: list[dict]
    context: str
    answer: str
    relevant: bool       # grade_documents verdict
    grounded: bool       # grade_groundedness verdict
    retries: int


# ---- nodes (async; do the work, write verdicts to state) ----
async def retrieve(state: RagState) -> dict:
    hits = await asyncio.to_thread(retriever.retrieve, state["query"])
    print(f"[retrieve] query={state['query']!r} -> {len(hits)} hits")
    return {"documents": hits, "context": retriever.build_context(hits)}


async def grade_documents(state: RagState) -> dict:
    verdict = await doc_grader.ainvoke([
        SystemMessage(
            "You grade whether retrieved context is relevant to a question. "
            "Answer 'yes' only if it contains information that could answer the question."
        ),
        HumanMessage(f"Question: {state['question']}\n\nContext:\n{state['context']}"),
    ])
    print(f"[grade_documents] relevant={verdict.binary_score}")
    return {"relevant": _is_yes(verdict.binary_score)}


async def rewrite_query(state: RagState) -> dict:
    better = (await orchestrator.ainvoke([
        SystemMessage(
            "Rewrite the user's question into a more effective retrieval query for a dense "
            "vector search over healthcare inspection reports. Return only the rewritten query."
        ),
        HumanMessage(state["question"]),
    ])).content
    print(f"[rewrite_query] -> {better!r}  (retry {state['retries'] + 1})")
    return {"query": better, "retries": state["retries"] + 1}


async def generate(state: RagState) -> dict:
    answer = (await reader.ainvoke([
        SystemMessage(
            "You are a healthcare-reports analyst. Answer ONLY from the provided context. "
            "If the context does not contain the answer, say you cannot find it. Cite [Source N]."
        ),
        HumanMessage(f"Context:\n{state['context']}\n\nQuestion: {state['question']}"),
    ])).content
    print("[generate] drafted answer")
    return {"answer": answer}


async def grade_groundedness(state: RagState) -> dict:
    verdict = await grounded_grader.ainvoke([
        SystemMessage("Answer 'yes' if the answer is grounded in the context, else 'no'."),
        HumanMessage(f"Context:\n{state['context']}\n\nAnswer:\n{state['answer']}"),
    ])
    print(f"[grade_groundedness] grounded={verdict.binary_score}")
    return {"grounded": _is_yes(verdict.binary_score)}


# ---- conditional edges (cheap, pure: only read state) ----
def route_after_grading(state: RagState) -> str:
    if state["relevant"] or state["retries"] >= MAX_RETRIES:
        return "generate"
    return "rewrite_query"


def route_after_groundedness(state: RagState) -> str:
    if state["grounded"] or state["retries"] >= MAX_RETRIES:
        return END
    return "rewrite_query"


# ---- wire the graph ----
builder = StateGraph(RagState)
builder.add_node("retrieve", retrieve)
builder.add_node("grade_documents", grade_documents)
builder.add_node("rewrite_query", rewrite_query)
builder.add_node("generate", generate)
builder.add_node("grade_groundedness", grade_groundedness)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade_documents")
builder.add_conditional_edges("grade_documents", route_after_grading,
                              {"generate": "generate", "rewrite_query": "rewrite_query"})
builder.add_edge("rewrite_query", "retrieve")
builder.add_edge("generate", "grade_groundedness")
builder.add_conditional_edges("grade_groundedness", route_after_groundedness,
                              {END: END, "rewrite_query": "rewrite_query"})

graph = builder.compile()


async def run(question: str) -> dict:
    return await graph.ainvoke(
        {"question": question, "query": question, "retries": 0},
        config={"recursion_limit": 15},
    )


if __name__ == "__main__":
    result = asyncio.run(run("What infection control deficiencies were cited?"))
    print("\n=== ANSWER ===")
    print(result["answer"])
