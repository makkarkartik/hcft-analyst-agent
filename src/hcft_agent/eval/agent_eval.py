"""End-to-end agent evaluation: run the RAG agent over a frozen slice, then score it.

Layered by the four-object taxonomy, with the deterministic anchors first (they break judge
circularity) and the LLM judges on top:

  outcome     refusal_accuracy          deterministic (is_refusal vs gold answerable/unanswerable)
  outcome     ROUGE-L / BERTScore       deterministic anchor vs the gold answer (grounded rows)
  component   retrieval hit@k           deterministic vs gold source_chunk_id(s)
  op-safety   HHEM groundedness         the live guard, scored offline too
  component   RAGAS faithfulness/a-rel  LLM judge  (added in run_eval; optional dep)
  outcome     DeepEval G-Eval refusal   LLM judge  (added in run_eval; optional dep)

Frozen slice: stratified + seeded so the SAME questions are scored every run (a real gate).

    python -m hcft_agent.eval.agent_eval --grounded 30 --unanswerable 15
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from hcft_agent.config import settings

DEFAULT_EVALSET = Path(
    r"C:\Users\kartik\OMSCS\Personal Projects\SLM_Fine_Tuning\data\qa_eval_v3.jsonl"
)


# --------------------------------------------------------------------- slice
def load_slice(path: Path, n_grounded: int, n_unanswerable: int, seed: int = 7) -> list[dict]:
    """Frozen, stratified slice: deterministic given (path, counts, seed)."""
    grounded, unans = [], []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        (unans if r["eval_kind"] == "unanswerable_rag" else grounded).append(r)
    grounded.sort(key=lambda r: r["qa_id"])
    unans.sort(key=lambda r: r["qa_id"])
    rng = random.Random(seed)
    rng.shuffle(grounded)
    rng.shuffle(unans)
    return grounded[:n_grounded] + unans[:n_unanswerable]


# --------------------------------------------------------------------- run the agent
def collect(rows: list[dict]) -> list[dict]:
    """Run the agent on each question; return records joining gold + agent output."""
    from hcft_agent.agents.rag_chat import answer, warmup

    warmup()
    recs = []
    for i, r in enumerate(rows, 1):
        s = answer(r["question"])
        gold_ids = r.get("source_chunk_ids") or [r.get("source_chunk_id")]
        recs.append({
            "qa_id": r["qa_id"], "question": r["question"],
            "eval_kind": r["eval_kind"], "hop_type": r.get("hop_type"),
            "gold_answer": r.get("answer") or "",
            "gold_chunk_ids": [c for c in gold_ids if c],
            "agent_answer": s.get("answer") or "",
            "is_refusal": bool(s.get("is_refusal")),
            "retrieved_ids": s.get("retrieved_ids") or [],
            "context": s.get("context") or "",
            "grounded_score": s.get("grounded_score"),
            "terminal": s.get("terminal"),
        })
        if i % 10 == 0:
            print(f"  ...ran {i}/{len(rows)}")
    return recs


# --------------------------------------------------------------------- deterministic metrics
def refusal_accuracy(recs: list[dict]) -> dict:
    """Refusal quality DECOMPOSED by retrieval — you can't grade the generator's refusal without
    knowing whether retrieval surfaced the answer. Three regimes:

      * answerable-in-context (grounded, gold chunk retrieved): should ANSWER → refusal is the
        real over-refusal (the generator's failure).
      * retrieval-miss (grounded, gold NOT retrieved): refusing is CORRECT (can't answer);
        answering = fabrication. This isolates retrieval failures from the generator.
      * unanswerable: should REFUSE; answering = fabrication.
    """
    def got_gold(r):
        return bool(set(r["gold_chunk_ids"]) & set(r["retrieved_ids"]))

    in_ctx = [r for r in recs if r["eval_kind"] != "unanswerable_rag" and got_gold(r)]
    miss = [r for r in recs if r["eval_kind"] != "unanswerable_rag" and not got_gold(r)]
    unans = [r for r in recs if r["eval_kind"] == "unanswerable_rag"]

    def rate(rows, cond):
        return round(sum(1 for r in rows if cond(r)) / len(rows), 3) if rows else None

    return {
        "answerable_in_context": {
            "n": len(in_ctx),
            "answered": sum(1 for r in in_ctx if not r["is_refusal"]),
            "over_refused": sum(1 for r in in_ctx if r["is_refusal"]),
            "answer_rate": rate(in_ctx, lambda r: not r["is_refusal"]),  # want HIGH
        },
        "unanswerable": {
            "n": len(unans),
            "correct_refusals": sum(1 for r in unans if r["is_refusal"]),
            "fabricated": sum(1 for r in unans if not r["is_refusal"]),
            "refuse_rate": rate(unans, lambda r: r["is_refusal"]),        # want HIGH
        },
        "grounded_retrieval_miss": {
            "n": len(miss),
            "correctly_refused": sum(1 for r in miss if r["is_refusal"]),
            "fabricated": sum(1 for r in miss if not r["is_refusal"]),    # answered w/o the evidence
        },
    }


def _doc_of(chunk_id: str) -> str:
    """chunk_id is '<doc_id>_p<page>_c<idx>' -> doc_id."""
    return chunk_id.split("_p")[0]


def retrieval_hit_at_k(recs: list[dict]) -> dict:
    """Bracket retrieval quality: EXACT (gold chunk in the window) vs DOC-LEVEL (gold's document
    in the window). The honest range — exact under-credits near-misses to adjacent chunks; the
    truth sits between the two. Grounded rows only."""
    grounded = [r for r in recs if r["eval_kind"] != "unanswerable_rag"]
    n = len(grounded) or 1
    exact = sum(1 for r in grounded if set(r["gold_chunk_ids"]) & set(r["retrieved_ids"]))
    doc = sum(1 for r in grounded
              if {_doc_of(c) for c in r["gold_chunk_ids"]} & {_doc_of(c) for c in r["retrieved_ids"]})
    k = settings.context_top_k
    return {f"hit@{k}_exact": exact / n, f"hit@{k}_doclevel": doc / n, "n_grounded": len(grounded)}


def overlap_metrics(recs: list[dict]) -> dict | None:
    """ROUGE-L + BERTScore of agent answer vs gold answer — deterministic anchor.
    Grounded, non-refused rows only (a refusal has no answer to compare)."""
    pairs = [(r["agent_answer"], r["gold_answer"]) for r in recs
             if r["eval_kind"] != "unanswerable_rag" and not r["is_refusal"] and r["gold_answer"]]
    if not pairs:
        return None
    preds, refs = [p for p, _ in pairs], [g for _, g in pairs]
    out: dict = {"n_scored": len(pairs)}
    try:
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        out["rougeL"] = sum(sc.score(g, p)["rougeL"].fmeasure for p, g in pairs) / len(pairs)
    except Exception as e:
        out["rougeL"] = f"unavailable ({type(e).__name__})"
    try:
        from bert_score import score as bert_score
        _, _, f1 = bert_score(preds, refs, lang="en", verbose=False)
        out["bertscore_f1"] = f1.mean().item()
    except Exception as e:
        out["bertscore_f1"] = f"unavailable ({type(e).__name__})"
    return out


def report(recs: list[dict]) -> dict:
    res = {
        "n": len(recs),
        "refusal": refusal_accuracy(recs),
        "retrieval": retrieval_hit_at_k(recs),
        "overlap_anchor": overlap_metrics(recs),
    }
    print("\n=== Agent eval (deterministic layer) ===")
    print(json.dumps(res, indent=2))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", type=Path, default=DEFAULT_EVALSET)
    ap.add_argument("--grounded", type=int, default=30)
    ap.add_argument("--unanswerable", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dump", type=Path, default=None, help="write per-record JSONL (for LLM judges)")
    args = ap.parse_args()

    rows = load_slice(args.evalset, args.grounded, args.unanswerable, args.seed)
    print(f"[eval] frozen slice: {len(rows)} ({args.grounded} grounded + {args.unanswerable} unanswerable)")
    recs = collect(rows)
    report(recs)
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[eval] dumped {len(recs)} records -> {args.dump} (feed to run_eval for LLM judges)")


if __name__ == "__main__":
    main()
