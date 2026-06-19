"""Standalone retrieval-quality harness — no agent, no LLM, fully deterministic.

For each gold-labeled test question we call ``Retriever.candidates()`` once (all
``dense_top_k`` candidates, with both dense and rerank scores) and locate the gold chunk's
rank at two stages:

  * pre-rerank  (dense order)  -> recall@pool, hit@1/k   (the first-stage retriever's job)
  * post-rerank (BGE order)    -> hit@1/k, MRR           (the reranker's job)

The metric that predicts RAG answer quality is **hit@context_top_k POST-rerank**: did the
gold chunk land in the top-k window the generator actually reads? Reranker lift =
hit@k(post) - hit@k(pre). Everything is judged against the gold ``source_chunk_id`` — never
the reranker — so there is no circularity (cf. nDCG-oracle caveat).

    python -m hcft_agent.eval.retrieval --limit 50     # quick pass
    python -m hcft_agent.eval.retrieval                 # full grounded test set

Requires Mongo up (text hydration) + Pinecone reachable; loads the embed+rerank models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hcft_agent.config import settings

# Gold-labeled curated test set (lives in the SLM project; reused as the eval substrate).
DEFAULT_TESTSET = Path(
    r"C:\Users\kartik\OMSCS\Personal Projects\SLM_Fine_Tuning\data\qa_v2\test.jsonl"
)


def load_grounded(path: Path, limit: int | None) -> list[dict]:
    """Grounded rows only — skip genuine-unanswerable / distractor rows (no retrievable gold)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("eval_kind") == "unanswerable_rag" or r.get("is_distractor"):
                continue
            if r.get("source_chunk_id"):
                rows.append(r)
    return rows[:limit] if limit else rows


def load_unanswerable(path: Path, limit: int | None) -> list[dict]:
    """Genuinely-unanswerable rows — the NEGATIVE class for grade-gate calibration: the
    corpus has no answer, so the top rerank score is the best a non-answerable query reaches."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("eval_kind") == "unanswerable_rag" or r.get("is_distractor"):
                rows.append(r)
    return rows[:limit] if limit else rows


def rank_of(chunk_id: str, ordered_ids: list[str]) -> int | None:
    """1-based rank of the gold chunk in an ordered id list, or None if absent."""
    try:
        return ordered_ids.index(chunk_id) + 1
    except ValueError:
        return None


def _pct(xs: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in 0..1); no numpy dependency."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = q * (len(s) - 1)
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def calibrate(testset: Path, limit: int | None) -> None:
    """Dump rerank-score distributions for the grade gate.

    POSITIVE = rerank_score of the gold chunk on answerable questions (we *want* to answer).
    NEGATIVE = top rerank_score on genuinely-unanswerable questions (we *want* to refuse).
    The gate threshold should sit between the two distributions. Prints percentiles + a
    suggested starting threshold (midpoint of the medians)."""
    from hcft_agent.retriever import Retriever

    pos_rows = load_grounded(testset, limit)
    neg_rows = load_unanswerable(testset, limit)
    print(f"[calibrate] positives(answerable)={len(pos_rows)}  negatives(unanswerable)={len(neg_rows)}")

    r = Retriever()
    pos, neg = [], []
    for row in pos_rows:
        cands = r.candidates(row["question"])
        by_id = {c["chunk_id"]: c.get("rerank_score", 0.0) for c in cands}
        if row["source_chunk_id"] in by_id:           # only when gold is retrievable
            pos.append(by_id[row["source_chunk_id"]])
    for row in neg_rows:
        cands = r.candidates(row["question"])
        if cands:
            neg.append(max(c.get("rerank_score", 0.0) for c in cands))
    r.close()

    # NB BGE-reranker-v2-m3 via CrossEncoder applies a sigmoid -> scores are 0..1 probabilities
    # (NOT raw logits) and saturate near 1.0 for the top candidate.
    print("\n=== grade-gate calibration (BGE-reranker-v2-m3, sigmoid 0..1) ===")
    for name, xs in [("POS gold-chunk score ", pos), ("NEG top-cand score  ", neg)]:
        if xs:
            print(f"  {name}  n={len(xs):<4} "
                  f"p10={_pct(xs,.1):.2f}  p25={_pct(xs,.25):.2f}  "
                  f"p50={_pct(xs,.5):.2f}  p75={_pct(xs,.75):.2f}  p90={_pct(xs,.9):.2f}")
    if pos and neg:
        suggest = (_pct(pos, .5) + _pct(neg, .5)) / 2
        overlap = _pct(neg, .9) >= _pct(pos, .1)   # distributions overlap -> weak discriminator
        print(f"\n  >> midpoint-of-medians = {suggest:.2f}")
        if overlap:
            print("  >> WARNING: POS/NEG overlap -- rerank_score saturates, weak answer/refuse "
                  "gate. Use a LOW floor (catch broken retrieval only); delegate refuse to the "
                  "output groundedness guard + generator refusal.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", type=Path, default=DEFAULT_TESTSET)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--calibrate", action="store_true",
                    help="dump rerank-score distributions to pick the grade-gate threshold")
    args = ap.parse_args()

    if args.calibrate:
        calibrate(args.testset, args.limit)
        return

    K = settings.context_top_k      # the window the generator reads (5)
    POOL = settings.dense_top_k     # candidate pool size (50)
    rows = load_grounded(args.testset, args.limit)
    print(f"[retrieval-eval] {len(rows)} grounded questions | pool={POOL} | context_k={K}")

    from hcft_agent.retriever import Retriever

    r = Retriever()
    a = dict(recall=0, h1_pre=0, h1_post=0, hk_pre=0, hk_post=0, mrr_pre=0.0, mrr_post=0.0)
    n = 0
    for i, row in enumerate(rows, 1):
        gold = row["source_chunk_id"]
        cands = r.candidates(row["question"])
        dense_order = [c["chunk_id"] for c in cands]
        rerank_order = [c["chunk_id"] for c in sorted(
            cands, key=lambda c: c.get("rerank_score", 0.0), reverse=True)]
        rp, rq = rank_of(gold, dense_order), rank_of(gold, rerank_order)
        a["recall"] += 1 if rp else 0
        a["h1_pre"] += 1 if rp and rp <= 1 else 0
        a["h1_post"] += 1 if rq and rq <= 1 else 0
        a["hk_pre"] += 1 if rp and rp <= K else 0
        a["hk_post"] += 1 if rq and rq <= K else 0
        a["mrr_pre"] += (1.0 / rp) if rp else 0.0
        a["mrr_post"] += (1.0 / rq) if rq else 0.0
        n += 1
        if i % 25 == 0:
            print(f"  ...{i}/{len(rows)}")
    r.close()

    if not n:
        print("no grounded rows — nothing to score")
        return
    p = lambda k: a[k] / n
    print(f"\n=== Retrieval over {n} grounded questions ===")
    print(f"  recall@{POOL:<3}          {p('recall'):.3f}   (gold reached the candidate pool)")
    print(f"  hit@1   pre / post    {p('h1_pre'):.3f} / {p('h1_post'):.3f}")
    print(f"  hit@{K}   pre / post    {p('hk_pre'):.3f} / {p('hk_post'):.3f}   reranker lift {p('hk_post') - p('hk_pre'):+.3f}")
    print(f"  MRR     pre / post    {a['mrr_pre'] / n:.3f} / {a['mrr_post'] / n:.3f}")
    print(f"\n  >> RAG-critical: hit@{K} POST-rerank = {p('hk_post'):.3f}  (gold in the generator's context window)")


if __name__ == "__main__":
    main()
