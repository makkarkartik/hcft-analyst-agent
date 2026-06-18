"""M3 — persistence (MongoDB checkpointer) + HITL interrupt() + time travel.

Three new capabilities on top of M2's self-corrective RAG:

1. CHECKPOINTING
   graph.compile(checkpointer=MongoDBSaver(...)) — after every super-step,
   the full RagState is serialized and written to MongoDB. Each conversation
   is identified by a thread_id in config["configurable"]. A crashed or
   restarted process can resume any thread from its last checkpoint.

2. HITL interrupt()
   The generate node calls interrupt(payload) AFTER drafting an answer.
   This raises GraphInterrupt internally — LangGraph saves the state to the
   checkpointer and returns the payload to the caller. The graph is now
   suspended. A human reviews the draft and calls:
       graph.ainvoke(Command(resume=value), config=same_thread_config)
   The graph wakes up exactly where it paused; the value passed to resume
   becomes the return value of interrupt(), and execution continues.

   WHY after generate (not before)?
   The analyst cares about the synthesized answer, not raw retrieved chunks.
   Pausing after generation lets a human approve or edit the final artifact
   before it reaches grade_groundedness and is released. Pre-generate
   interrupts make sense for cost-control gates in high-volume systems;
   post-generate interrupts are the quality/safety gate.

3. TIME TRAVEL
   graph.aget_state_history(config) streams every checkpoint for a thread,
   newest first. graph.aupdate_state(config, values, as_node=...) writes
   a new checkpoint forked from any prior one. Re-invoking from that config
   re-runs the graph from that point with the patched state — no replay of
   earlier nodes needed.

Graph topology is identical to M2 (same nodes, same edges). Only changes:
  - generate node calls interrupt() after drafting
  - compile() receives the checkpointer
  - build_rag_graph() is a factory so tests can inject any checkpointer
"""

import asyncio
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from hcft_agent.config import settings
from hcft_agent.retriever import Retriever

MAX_RETRIES = 2

retriever = Retriever()

orchestrator = ChatOpenAI(
    base_url=settings.orchestrator_base_url,
    api_key=settings.orchestrator_api_key,
    model=settings.orchestrator_model,
    temperature=0.0,
)
reader = ChatOpenAI(
    base_url=settings.reader_base_url,
    api_key=settings.reader_api_key,
    model=settings.reader_model,
    temperature=0.0,
)


class GradeDocuments(BaseModel):
    """Binary relevance of retrieved context to the question."""
    binary_score: str = Field(description="'yes' if relevant, else 'no'")


class GradeGroundedness(BaseModel):
    """Whether the answer is supported by the retrieved context."""
    binary_score: str = Field(description="'yes' if grounded, else 'no'")


doc_grader = orchestrator.with_structured_output(GradeDocuments)
grounded_grader = orchestrator.with_structured_output(GradeGroundedness)


def _is_yes(score: str) -> bool:
    return score.strip().lower().startswith("y")


class RagState(TypedDict):
    question: str        # original user question — immutable
    query: str           # working retrieval query — rewritten on misses
    documents: list[dict]
    context: str
    answer: str
    relevant: bool
    grounded: bool
    retries: int


# ---- nodes ----

async def retrieve(state: RagState) -> dict:
    hits = await asyncio.to_thread(retriever.retrieve, state["query"])
    print(f"  [retrieve] query={state['query']!r} -> {len(hits)} hits")
    return {"documents": hits, "context": retriever.build_context(hits)}


async def grade_documents(state: RagState) -> dict:
    verdict = await doc_grader.ainvoke([
        SystemMessage(
            "You grade whether retrieved context is relevant to a question. "
            "Answer 'yes' only if it contains information that could answer the question."
        ),
        HumanMessage(f"Question: {state['question']}\n\nContext:\n{state['context']}"),
    ])
    print(f"  [grade_documents] relevant={verdict.binary_score}")
    return {"relevant": _is_yes(verdict.binary_score)}


async def rewrite_query(state: RagState) -> dict:
    better = (await orchestrator.ainvoke([
        SystemMessage(
            "Rewrite into a more effective dense-retrieval query for healthcare inspection reports. "
            "Return only the rewritten query."
        ),
        HumanMessage(
            f"Original question: {state['question']}\n"
            f"Previous query that failed: {state['query']}\n"
            "Rewrite it differently."
        ),
    ])).content
    print(f"  [rewrite_query] -> {better!r}  (retry {state['retries'] + 1})")
    return {"query": better, "retries": state["retries"] + 1}


async def generate(state: RagState) -> dict:
    """Draft an answer, then PAUSE for human review via interrupt().

    interrupt(payload) suspends the graph here. The payload is returned to the
    caller of ainvoke/astream. The graph is frozen in the checkpointer.
    Execution resumes when Command(resume=value) is passed; `value` becomes
    the return value of interrupt().
    """
    draft = (await reader.ainvoke([
        SystemMessage(
            "You are a healthcare-reports analyst. Answer ONLY from the provided context. "
            "If the context does not contain the answer, say you cannot find it. Cite [Source N]."
        ),
        HumanMessage(f"Context:\n{state['context']}\n\nQuestion: {state['question']}"),
    ])).content

    print(f"  [generate] draft ready — interrupting for human review")

    # ← graph suspends here; payload surfaced to the human
    human_feedback = interrupt({
        "draft_answer": draft,
        "question": state["question"],
        "context_preview": state["context"][:400],
    })

    # human_feedback = whatever the human sends via Command(resume=...)
    approved = human_feedback.get("approved_answer", draft)
    edited = approved != draft
    print(f"  [generate] human {'edited' if edited else 'approved'} answer")
    return {"answer": approved}


async def grade_groundedness(state: RagState) -> dict:
    verdict = await grounded_grader.ainvoke([
        SystemMessage("Answer 'yes' if the answer is grounded in the context, else 'no'."),
        HumanMessage(f"Context:\n{state['context']}\n\nAnswer:\n{state['answer']}"),
    ])
    print(f"  [grade_groundedness] grounded={verdict.binary_score}")
    return {"grounded": _is_yes(verdict.binary_score)}


# ---- conditional edges (pure — no I/O) ----

def route_after_grading(state: RagState) -> str:
    if state["relevant"] or state["retries"] >= MAX_RETRIES:
        return "generate"
    return "rewrite_query"


def route_after_groundedness(state: RagState) -> str:
    if state["grounded"] or state["retries"] >= MAX_RETRIES:
        return END
    return "rewrite_query"


# ---- graph factory (checkpointer injected, not module-level) ----

def build_rag_graph(checkpointer):
    """Return a compiled graph with the given checkpointer.

    Keeping the graph as a factory (not a module-level singleton) means:
    - tests can inject a SqliteSaver or in-memory checkpointer
    - the production MongoDB saver is only instantiated in __main__ / app code
    - no circular import from checkpointer setup touching module state
    """
    builder = StateGraph(RagState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("generate", generate)
    builder.add_node("grade_groundedness", grade_groundedness)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade_documents")
    builder.add_conditional_edges(
        "grade_documents", route_after_grading,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("generate", "grade_groundedness")
    builder.add_conditional_edges(
        "grade_groundedness", route_after_groundedness,
        {END: END, "rewrite_query": "rewrite_query"},
    )
    return builder.compile(checkpointer=checkpointer)


# ----------------------------------------------------------------------------
# DEMO — split into independent phases you run one at a time:
#   python -m hcft_agent.graphs.m3_hitl_rag phase1   (run until it pauses)
#   python -m hcft_agent.graphs.m3_hitl_rag phase2   (resume — approve the draft)
#   python -m hcft_agent.graphs.m3_hitl_rag phase3   (time travel)
#
# State lives in MongoDB under a FIXED thread_id, so phase2 (a brand-new process)
# loads exactly where phase1 left off. That cross-process resume IS persistence.
# ----------------------------------------------------------------------------

THREAD_ID = "m3-demo"
QUESTION = "What infection control deficiencies were cited?"


def _config():
    return {"configurable": {"thread_id": THREAD_ID}, "recursion_limit": 15}


async def phase1():
    """Run the graph until generate() hits interrupt(), then exit (state saved to Mongo)."""
    with MongoDBSaver.from_conn_string(settings.mongo_uri, db_name="hcft_checkpoints") as cp:
        graph = build_rag_graph(cp)
        print("PHASE 1 — running until the graph pauses\n")
        async for chunk in graph.astream(
            {"question": QUESTION, "query": QUESTION, "retries": 0},
            config=_config(),
            stream_mode="updates",
        ):
            if "__interrupt__" in chunk:
                payload = chunk["__interrupt__"][0].value
                print("\n  *** GRAPH PAUSED — saved to Mongo, process can now exit ***")
                print(f"  Draft answer the human must review:\n  {payload['draft_answer'][:220]}...")
                return
        print("  (graph finished without pausing)")


async def phase2():
    """New process. Load the paused state from Mongo and resume by approving the draft."""
    with MongoDBSaver.from_conn_string(settings.mongo_uri, db_name="hcft_checkpoints") as cp:
        graph = build_rag_graph(cp)

        # Prove the graph is paused: read the saved state, show WHERE it stopped.
        snapshot = await graph.aget_state(_config())
        print("PHASE 2 — resuming a graph that was paused in a PREVIOUS process\n")
        print(f"  Graph is paused before node(s): {snapshot.next}")
        draft = snapshot.values.get("answer") or snapshot.tasks[0].interrupts[0].value["draft_answer"]
        print(f"  Saved draft:\n  {draft[:220]}...\n")

        print("  Human approves. Sending Command(resume=...)\n")
        final = await graph.ainvoke(
            Command(resume={"approved_answer": draft}),
            config=_config(),
        )
        print(f"  FINAL ANSWER:\n  {final.get('answer', '(none)')}")


async def phase3():
    """Inspect every saved checkpoint for the thread (the raw material of time travel)."""
    with MongoDBSaver.from_conn_string(settings.mongo_uri, db_name="hcft_checkpoints") as cp:
        graph = build_rag_graph(cp)
        print("PHASE 3 — list every checkpoint saved for this thread\n")
        checkpoints = [c async for c in graph.aget_state_history(_config())]
        print(f"  {len(checkpoints)} checkpoints (newest first):")
        for i, c in enumerate(checkpoints):
            step = c.metadata.get("step", "?")
            writes = list((c.metadata.get("writes") or {}).keys())
            print(f"    [{i:>2}] step={step:>3}  next={c.next}  last_write={writes}")


if __name__ == "__main__":
    import sys

    phases = {"phase1": phase1, "phase2": phase2, "phase3": phase3}
    choice = sys.argv[1] if len(sys.argv) > 1 else "phase1"
    if choice not in phases:
        print(f"usage: ... m3_hitl_rag.py [phase1|phase2|phase3]")
        sys.exit(1)
    asyncio.run(phases[choice]())
