"""Dense vs Hybrid retrieval A/B on the v3 grounded eval set.

Same questions, two retrieval modes:
  * dense  — Pinecone (Qwen3) only
  * hybrid — dense + MongoDB $text (BM25-style) fused by Reciprocal Rank Fusion

Reports recall@pool (gold reached the candidate pool) and hit@context_k POST-rerank, both
EXACT (gold chunk) and DOC-LEVEL (gold's document) — the honest bracket. Multi-hop rows count a
hit if ANY gold chunk lands. Deterministic; no LLM.

    python scripts/retrieval_ab.py --limit 120
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from hcft_agent.config import settings
from hcft_agent.retriever import Retriever

V3 = Path(r"C:\Users\kartik\OMSCS\Personal Projects\SLM_Fine_Tuning\data\qa_eval_v3.jsonl")


def _gold(r: dict) -> list[str]:
    return [c for c in (r.get("source_chunk_ids") or [r.get("source_chunk_id")]) if c]


def _doc(cid: str) -> str:
    return cid.split("_p")[0]


def eval_mode(retr: Retriever, rows: list[dict], mode: str) -> dict:
    k, pool = settings.context_top_k, settings.dense_top_k
    rec = hit_exact = hit_doc = 0
    for r in rows:
        gold = set(_gold(r))
        gold_docs = {_doc(g) for g in gold}
        cands = retr.candidates(r["question"], mode=mode)
        if gold & {c["chunk_id"] for c in cands}:
            rec += 1
        top = sorted(cands, key=lambda c: c.get("rerank_score", 0.0), reverse=True)[:k]
        top_ids = {c["chunk_id"] for c in top}
        top_docs = {_doc(c["chunk_id"]) for c in top}
        hit_exact += 1 if gold & top_ids else 0
        hit_doc += 1 if gold_docs & top_docs else 0
    n = len(rows) or 1
    return {f"recall@{pool}": rec / n, f"hit@{k}_exact": hit_exact / n, f"hit@{k}_doc": hit_doc / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", type=Path, default=V3)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.testset, encoding="utf-8")]
    rows = [r for r in rows if r["eval_kind"] != "unanswerable_rag" and _gold(r)]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit]
    print(f"[A/B] {len(rows)} grounded v3 questions · k={settings.context_top_k} · pool={settings.dense_top_k}")

    r = Retriever()
    results = {}
    for mode in ("dense", "hybrid"):
        results[mode] = eval_mode(r, rows, mode)
        print(f"  {mode:7s} " + "  ".join(f"{k}={v:.3f}" for k, v in results[mode].items()))
    r.close()

    print("\n  lift (hybrid − dense):")
    for k in results["dense"]:
        d = results["hybrid"][k] - results["dense"][k]
        print(f"    {k:16s} {d:+.3f}")


if __name__ == "__main__":
    main()
