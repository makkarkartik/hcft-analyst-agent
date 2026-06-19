"""The RAG chat agent's graph state.

A single ``TypedDict`` threaded through every node. ``total=False`` so nodes set only the
keys they own; LangGraph merges each node's returned partial dict into the running state.
The fields fall into three groups that mirror the pipeline map: control flow, the domain
verdicts we emit to ``hcft.*`` spans (eval reads these), and the reliability flags.
"""
from __future__ import annotations

from typing import TypedDict


class RagState(TypedDict, total=False):
    # --- control flow ---
    question: str            # original user question (immutable)
    query: str               # query actually used for retrieval (rewrites mutate this)
    candidates: list[dict]   # reranked candidates from Retriever.retrieve()
    context: str             # assembled context string fed to the reader
    answer: str
    retries: int             # rewrite attempts so far
    relevant: bool           # grade verdict (must be a declared channel or LangGraph drops it)

    # --- domain verdicts (-> hcft.* attributes; eval/guards read these) ---
    input_flags: list[str]   # input-ring findings, e.g. ["injection","pii"]
    retrieved_ids: list[str] # chunk ids in the context window
    cited_ids: list[str]     # chunk ids the answer actually cited
    grounded_score: float    # HHEM P(grounded); set by the output guard
    grounded: bool           # grounded_score >= threshold
    is_refusal: bool
    terminal: str            # "answer" | "refuse" | "error"

    # --- reliability ---
    degraded: bool           # produced via a fallback path?
    degraded_reason: str
