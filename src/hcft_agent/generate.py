"""Grounded generation — the reader node's core (LLM call, kept out of the graph file).

Uses ``langchain_openai.ChatOpenAI`` (not the raw OpenAI client) so OpenInference
auto-instruments the call as an LLM span with token/cost — the generate step shows up in the
trace for free, nested under our manual ``rag.generate`` span.

Contract the prompt enforces:
  * answer ONLY from the numbered sources; if they don't contain the answer, say so plainly
    (a clean refusal — refuse > fabricate);
  * cite the sources used as ``[Source N]`` so we can map citations back to chunk ids and
    later check whether what was cited actually supports the claim.
"""
from __future__ import annotations

import re

from hcft_agent.config import settings

_SYSTEM = (
    "You are a healthcare-facility analyst. Answer ONLY if the numbered sources contain the "
    "SPECIFIC information the question asks for. Being on the same topic is NOT enough: if the "
    "sources discuss the area but lack the exact figure, plan, year, population, or detail "
    "requested, you MUST reply EXACTLY: "
    "\"I don't have enough information in the provided sources to answer that.\" "
    "Do not infer, generalize, estimate, or fill gaps from outside knowledge — a partial or "
    "approximate answer assembled from related text is a failure; refusing is correct. When you "
    "do answer, every claim must be directly stated in a source, cited inline as [Source N]."
)

_CITE_RE = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)

REFUSAL_TEXT = "I don't have enough information in the provided sources to answer that."


def _client():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.reader_model,
        base_url=settings.reader_base_url,
        api_key=settings.reader_api_key,
        temperature=settings.gen_temperature,
    )


def build_context(question: str, hits: list[dict]) -> tuple[str, list, dict]:
    """Assemble the numbered-source context + the chat messages + the ``{N: chunk_id}`` map.
    Shared by :func:`generate` and :func:`stream` so the prompt has ONE source of truth."""
    window = hits[: settings.context_top_k]
    blocks, id_by_num = [], {}
    for i, h in enumerate(window, 1):
        id_by_num[i] = h["chunk_id"]
        text = (h.get("text") or "")[: settings.context_char_cap]
        blocks.append(f"[Source {i}] {text}")
    context = "\n\n".join(blocks)
    messages = [("system", _SYSTEM), ("human", f"Question: {question}\n\nSources:\n{context}\n\nAnswer:")]
    return context, messages, id_by_num


def finalize(answer: str, id_by_num: dict, context: str) -> dict:
    """Map ``[Source N]`` markers -> cited chunk ids and detect refusal. Shared post-processing."""
    answer = (answer or "").strip()
    cited_nums = {int(n) for n in _CITE_RE.findall(answer)}
    cited_ids = [id_by_num[n] for n in sorted(cited_nums) if n in id_by_num]
    return {
        "answer": answer, "cited_ids": cited_ids, "context": context,
        "is_refusal": answer.startswith(REFUSAL_TEXT[:40]),
    }


def generate(question: str, hits: list[dict]) -> dict:
    """Answer ``question`` from the top ``context_top_k`` ``hits``. Returns
    ``{answer, cited_ids, context, is_refusal}`` (cited ids mapped from ``[Source N]``)."""
    context, messages, id_by_num = build_context(question, hits)
    resp = _client().invoke(messages)
    return finalize(resp.content, id_by_num, context)


def stream(question: str, hits: list[dict]):
    """Token generator for live UIs. Yields text deltas as they arrive; the caller accumulates
    the full answer and can then call :func:`finalize` on it. Same prompt as :func:`generate`."""
    _, messages, _ = build_context(question, hits)
    for chunk in _client().stream(messages):
        if chunk.content:
            yield chunk.content
