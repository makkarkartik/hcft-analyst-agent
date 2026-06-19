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
    "You are a healthcare-facility analyst. Answer the question using ONLY the numbered "
    "sources provided. If the sources do not contain the answer, reply exactly: "
    "\"I don't have enough information in the provided sources to answer that.\" "
    "Do not use outside knowledge. Cite every source you rely on inline as [Source N]."
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


def generate(question: str, hits: list[dict]) -> dict:
    """Answer ``question`` from the top ``context_top_k`` ``hits`` (each a candidate dict with
    ``chunk_id`` + ``text``). Returns ``{answer, cited_ids, context, is_refusal}``.

    ``cited_ids`` are mapped from the model's ``[Source N]`` markers back to chunk ids, so a
    downstream check can verify the citation actually supports the statement."""
    window = hits[: settings.context_top_k]
    blocks, id_by_num = [], {}
    for i, h in enumerate(window, 1):
        id_by_num[i] = h["chunk_id"]
        text = (h.get("text") or "")[: settings.context_char_cap]
        blocks.append(f"[Source {i}] {text}")
    context = "\n\n".join(blocks)

    prompt = f"Question: {question}\n\nSources:\n{context}\n\nAnswer:"
    resp = _client().invoke([("system", _SYSTEM), ("human", prompt)])
    answer = (resp.content or "").strip()

    cited_nums = {int(n) for n in _CITE_RE.findall(answer)}
    cited_ids = [id_by_num[n] for n in sorted(cited_nums) if n in id_by_num]
    is_refusal = answer.startswith(REFUSAL_TEXT[:40])

    return {"answer": answer, "cited_ids": cited_ids, "context": context, "is_refusal": is_refusal}
