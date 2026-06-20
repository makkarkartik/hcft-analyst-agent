"""Dense Pinecone retrieval + BGE cross-encoder rerank, hydrated from MongoDB.

Port of the HCFT stage-06 retriever (`SLM_Fine_Tuning/src/06_rag.py`) with ONE change:
chunk text is hydrated from the local `hcft.chunks` MongoDB collection instead of the
sqlite text store. Everything on the query side is identical, because it MUST be — the
Pinecone `hcft` index was built with Qwen3-Embedding-4B (768-dim Matryoshka, normalized),
so queries must be embedded the same way or the cosine space won't line up.

Models load lazily and can be freed with `close()` (release VRAM before loading a generator).
"""

from __future__ import annotations

import gc
import os

from pymongo import MongoClient

from hcft_agent.config import settings


class Retriever:
    def __init__(self) -> None:
        self._embedder = None
        self._reranker = None
        self._index = None
        self._coll = None

    # ---- lazy resources ----
    def _get_index(self):
        if self._index is None:
            from pinecone import Pinecone

            key = settings.pinecone_api_key or os.environ.get("PINECONE_API_KEY")
            if not key:
                raise RuntimeError("PINECONE_API_KEY not set (put it in .env).")
            self._index = Pinecone(api_key=key).Index(settings.pinecone_index)
        return self._index

    def _get_collection(self):
        if self._coll is None:
            self._coll = MongoClient(settings.mongo_uri)[settings.mongo_db][settings.chunks_collection]
        return self._coll

    def _get_embedder(self):
        if self._embedder is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[retriever] loading query embedder {settings.embed_model} ({device}) ...")
            self._embedder = SentenceTransformer(
                settings.embed_model,
                device=device,
                truncate_dim=settings.embed_dim,
                model_kwargs={"torch_dtype": getattr(torch, settings.embed_dtype)},
                tokenizer_kwargs={"padding_side": "left"},
            )
            self._embedder.max_seq_length = settings.embed_max_seq_length
        return self._embedder

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            print(f"[retriever] loading reranker {settings.reranker_model} ...")
            self._reranker = CrossEncoder(settings.reranker_model, max_length=settings.rerank_max_length)
        return self._reranker

    # ---- query embedding (Qwen3 instruction on the QUERY side only) ----
    def embed_query(self, question: str):
        text = f"Instruct: {settings.query_instruction}\nQuery: {question}"
        return self._get_embedder().encode(
            [text], normalize_embeddings=settings.embed_normalize, convert_to_numpy=True
        )[0]

    # ---- hydrate chunk text + metadata from Mongo ----
    def _hydrate(self, chunk_ids: list[str]) -> dict[str, dict]:
        if not chunk_ids:
            return {}
        coll = self._get_collection()
        out: dict[str, dict] = {}
        for doc in coll.find({"_id": {"$in": chunk_ids}}):
            out[doc["_id"]] = doc
        return out

    # ---- core retrieval ----
    def _rank(self, question: str, rerank: bool = True) -> list[dict]:
        """Full candidate list in DENSE order (Pinecone score desc), hydrated, with a
        ``rerank_score`` added per candidate when ``rerank`` is set. NOT truncated — the
        caller decides. Shared by :meth:`retrieve` (pipeline) and :meth:`candidates` (eval).

        Each sub-stage is its own nested LangSmith run (embed / pinecone_query / hydrate /
        rerank) so the latency breakdown shows up under the caller's run — no ad-hoc timing."""
        from hcft_agent.obs.telemetry import trace_block

        with trace_block("retriever.embed", run_type="embedding"):
            vec = self.embed_query(question)

        with trace_block("retriever.pinecone_query", run_type="retriever"):
            res = self._get_index().query(
                vector=vec.tolist(), top_k=settings.dense_top_k, include_metadata=True
            )
        ids, dense = [], {}
        for m in res.get("matches", []):
            ids.append(m["id"])
            dense[m["id"]] = float(m["score"])

        with trace_block("retriever.hydrate", run_type="tool"):
            hydrated = self._hydrate(ids)
        cands = []
        for cid in ids:
            doc = hydrated.get(cid, {})
            cands.append({
                "chunk_id": cid,
                "dense_score": dense[cid],
                "text": doc.get("text", ""),
                "doc_id": doc.get("doc_id"),
                "hospital": doc.get("hospital"),
                "state": doc.get("state"),
                "year": doc.get("year"),
                "page_num": doc.get("page_num"),
            })

        if rerank and cands:
            with trace_block("retriever.rerank", run_type="chain"):
                scores = self._get_reranker().predict(
                    [(question, c["text"]) for c in cands], show_progress_bar=False
                )
                for c, s in zip(cands, scores):
                    c["rerank_score"] = float(s)
        return cands

    def retrieve(self, question: str, rerank: bool = True) -> list[dict]:
        """Pipeline retrieval: reranked (when enabled) and truncated to ``final_top_k``."""
        cands = self._rank(question, rerank=rerank)
        if rerank and cands and "rerank_score" in cands[0]:
            cands = sorted(cands, key=lambda c: c["rerank_score"], reverse=True)
        return cands[: settings.final_top_k]

    def candidates(self, question: str) -> list[dict]:
        """Eval hook: ALL ``dense_top_k`` candidates in dense order, each carrying
        ``dense_score`` and ``rerank_score`` — so retrieval metrics can be measured at both
        the dense (pre-rerank) and reranked stages from a single call."""
        return self._rank(question, rerank=True)

    # ---- assemble the context string fed to the LLM ----
    def build_context(self, hits: list[dict]) -> str:
        parts = []
        for i, h in enumerate(hits[: settings.context_top_k], 1):
            txt = (h.get("text") or "")[: settings.context_char_cap]
            parts.append(f"[Source {i}] {txt}")
        return "\n\n".join(parts)

    def close(self) -> None:
        self._embedder = None
        self._reranker = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    r = Retriever()
    hits = r.retrieve("What infection control deficiencies were cited?")
    for i, h in enumerate(hits, 1):
        rr = h.get("rerank_score")
        rr_s = f"rerank={rr:.3f} " if rr is not None else ""
        print(f"#{i:<2} {rr_s}dense={h['dense_score']:.3f} {h['hospital']} ({h['state']}) p{h['page_num']}")
        print(f"     {(h.get('text') or '')[:200].strip()}...\n")
    r.close()
