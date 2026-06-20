"""Shared report assembly + gate baseline — the ONE place that turns raw metric dicts into the
stage×substage ``eval_report.json`` the dashboard and the pytest gate both read.

Two producers feed this with the SAME shapes:
  * ``experiment.py``  — the LangSmith ``evaluate()`` path (scores linked to runs);
  * ``run_eval.py``    — the offline batch path (no network, fast gate iteration).

Keeping the builder here means both paths emit byte-identical schema, so the dashboard/gate never
care which one ran.
"""
from __future__ import annotations

import json

from hcft_agent.config import PROJECT_ROOT, settings

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORTS_DIR / "eval_report.json"
BASELINE_PATH = REPORTS_DIR / "eval_baseline.json"

# Headline metrics the regression gate guards (path in report["flat"], direction). The tolerance
# (run_eval/test) lets a small slice wobble without a false alarm; the gate fails on a real drop.
GATES = [
    ("retrieval.recall_at_pool",            "higher"),
    ("retrieval.hit_at_k_exact",            "higher"),
    ("refusal.unanswerable.refuse_rate",    "higher"),
    ("refusal.answerable.answer_rate",      "higher"),
    ("refusal.unanswerable.fabricated",     "lower"),
    ("generation.ragas_faithfulness",       "higher"),
    ("generation.hhem_mean",                "higher"),
    ("validation.cohen_kappa",              "higher"),
]
GATE_TOL = 0.05


def over_refusal_cases(recs: list[dict]) -> list[dict]:
    """The answerable-in-context rows the agent REFUSED — the 'over-refusal' cost of the strict
    contract. We surface each (not just the count) so we can tell a TRUE over-refusal on a
    single-hop row whose gold was retrieved from a multi-hop row where only SOME gold chunks made
    the window (refusing there is defensible, not a bug)."""
    cases = []
    for r in recs:
        if r["eval_kind"] == "unanswerable_rag" or not r["is_refusal"]:
            continue
        got = set(r.get("gold_chunk_ids", [])) & set(r.get("retrieved_ids", []))
        if not got:
            continue  # gold not retrieved -> correctly refused, not an over-refusal
        n_gold = len(r.get("gold_chunk_ids", []))
        cases.append({
            "qa_id": r["qa_id"], "hop_type": r.get("hop_type"), "question": r["question"],
            "gold_retrieved": f"{len(got)}/{n_gold}",
            "partial_multihop": r.get("hop_type") == "multi" and len(got) < n_gold,
        })
    return cases


def _metric(key, value, kind, method, n=None, higher_better=True, gate=False):
    return {"key": key, "value": value, "kind": kind, "method": method,
            "n": n, "higher_better": higher_better, "gate": gate}


def build_report(recs, ragas, geval, kappa, ab, *, experiment=None) -> dict:
    """Assemble the consolidated report. ``ragas``/``geval``/``kappa`` are the judge dicts (shape
    from judges.py / validate.py); ``ab`` is the optional 120q retrieval A/B json; ``experiment``
    is the LangSmith experiment name/url when produced via evaluate()."""
    from hcft_agent.eval import agent_eval

    det = agent_eval.report(recs)              # prints + returns the deterministic block
    ref = det["refusal"]; ret = det["retrieval"]; ov = det["overlap_anchor"] or {}
    k = settings.context_top_k

    hscores = [r["grounded_score"] for r in recs if isinstance(r.get("grounded_score"), (int, float))]
    hhem_mean = round(sum(hscores) / len(hscores), 3) if hscores else None
    hhem_min = round(min(hscores), 3) if hscores else None

    if ab:
        hy = ab.get("hybrid", {})
        recall = hy.get(f"recall@{settings.dense_top_k}")
        hit_exact = hy.get(f"hit@{k}_exact")
        hit_doc = hy.get(f"hit@{k}_doc")
        ret_n = ab.get("n")
        ret_method = f"hybrid (dense+BM25 RRF) A/B vs gold chunk_id · {ret_n}q · deterministic"
    else:
        recall = None
        hit_exact = ret.get(f"hit@{k}_exact")
        hit_doc = ret.get(f"hit@{k}_doclevel")
        ret_n = ret.get("n_grounded")
        ret_method = f"slice records (top-{k} window only) vs gold chunk_id · deterministic"

    flat = {
        "retrieval.recall_at_pool": recall,
        "retrieval.hit_at_k_exact": hit_exact,
        "retrieval.hit_at_k_doc": hit_doc,
        "generation.ragas_faithfulness": ragas.get("faithfulness_mean"),
        "generation.ragas_answer_relevancy": ragas.get("answer_relevancy_mean"),
        "generation.rougeL": ov.get("rougeL") if isinstance(ov.get("rougeL"), (int, float)) else None,
        "generation.bertscore_f1": ov.get("bertscore_f1") if isinstance(ov.get("bertscore_f1"), (int, float)) else None,
        "generation.hhem_mean": hhem_mean,
        "refusal.answerable.answer_rate": ref["answerable_in_context"]["answer_rate"],
        "refusal.answerable.over_refused": ref["answerable_in_context"]["over_refused"],
        "refusal.unanswerable.refuse_rate": ref["unanswerable"]["refuse_rate"],
        "refusal.unanswerable.fabricated": ref["unanswerable"]["fabricated"],
        "refusal.retrieval_miss.fabricated": ref["grounded_retrieval_miss"]["fabricated"],
        "refusal.geval_quality": geval.get("mean_score"),
        "validation.cohen_kappa": kappa.get("cohen_kappa"),
        "validation.judge_accuracy": kappa.get("judge_accuracy"),
    }

    n_g = sum(1 for r in recs if r["eval_kind"] != "unanswerable_rag")
    n_u = len(recs) - n_g
    stages = [
        {"stage": "Retrieval", "subtitle": "dense (Pinecone) + BM25 (Mongo) → RRF → rerank → window",
         "substages": [
            {"name": "candidate pool", "metrics": [
                _metric("recall@%d" % settings.dense_top_k, recall, "component",
                        "gold reached the pool — the ceiling on RAG quality · " + ret_method, ret_n, gate=True)]},
            {"name": "context window (post-rerank)", "metrics": [
                _metric("hit@%d exact" % k, hit_exact, "component", "gold chunk in the generator's window · " + ret_method, ret_n, gate=True),
                _metric("hit@%d doc-level" % k, hit_doc, "component", "gold's DOCUMENT in the window (honest upper bracket)", ret_n)]},
         ]},
        {"stage": "Generation", "subtitle": "answer ONLY from sources · grounded · cited",
         "substages": [
            {"name": "faithfulness", "metrics": [
                _metric("RAGAS faithfulness", ragas.get("faithfulness_mean"), "component (LLM judge, cross-family)",
                        f"claim-decomposition + NLI vs context · {ragas.get('judge')}", ragas.get("n_scored"), gate=True),
                _metric("ROUGE-L", flat["generation.rougeL"], "anchor (non-LLM)", "lexical overlap vs gold answer", ov.get("n_scored")),
                _metric("BERTScore-F1", flat["generation.bertscore_f1"], "anchor (non-LLM)", "semantic similarity vs gold answer", ov.get("n_scored"))]},
            {"name": "answer relevancy", "metrics": [
                _metric("RAGAS answer-relevancy", ragas.get("answer_relevancy_mean"), "component (LLM judge, cross-family)",
                        "back-generated-question ↔ real-question similarity (responsiveness)", ragas.get("n_scored"))]},
            {"name": "groundedness guard (HHEM, live)", "metrics": [
                _metric("HHEM mean", hhem_mean, "op-safety (deterministic)", "Vectara cross-encoder P(answer grounded) — the inline guard, scored offline", len(hscores), gate=True),
                _metric("HHEM min", hhem_min, "op-safety (deterministic)", "worst-grounded answer in the slice", len(hscores))]},
         ]},
        {"stage": "Refusal / routing (outcome)", "subtitle": "decomposed by retrieval — you can't grade refusal without knowing if the answer was retrievable",
         "substages": [
            {"name": "answerable-in-context (should ANSWER)", "metrics": [
                _metric("answer rate", ref["answerable_in_context"]["answer_rate"], "outcome (deterministic)",
                        "gold retrieved → answering is correct; refusing = TRUE over-refusal", ref["answerable_in_context"]["n"], gate=True),
                _metric("over-refused", ref["answerable_in_context"]["over_refused"], "outcome (deterministic)",
                        "count of true over-refusals (the cost of the strict contract)", ref["answerable_in_context"]["n"], higher_better=False)]},
            {"name": "unanswerable (should REFUSE)", "metrics": [
                _metric("refuse rate", ref["unanswerable"]["refuse_rate"], "outcome (deterministic)",
                        "no answer in corpus → refusing is correct", ref["unanswerable"]["n"], gate=True),
                _metric("fabricated", ref["unanswerable"]["fabricated"], "outcome (deterministic)",
                        "answered when it should have refused (hallucination)", ref["unanswerable"]["n"], higher_better=False, gate=True)]},
            {"name": "grounded retrieval-miss (gold NOT retrieved → REFUSE)", "metrics": [
                _metric("fabricated", ref["grounded_retrieval_miss"]["fabricated"], "outcome (deterministic)",
                        "answered without the evidence in context", ref["grounded_retrieval_miss"]["n"], higher_better=False)]},
            {"name": "refusal-decision quality (G-Eval)", "metrics": [
                _metric("G-Eval score", geval.get("mean_score"), "outcome (LLM judge, same-family)",
                        f"CoT judge of answer-vs-refuse correctness from context alone · {geval.get('judge')}", geval.get("n_scored"))]},
         ]},
        {"stage": "Judge validation", "subtitle": "is the LLM judge trustworthy, or rubber-stamping?",
         "substages": [
            {"name": f"G-Eval vs {kappa.get('anchor', 'deterministic')} anchor", "metrics": [
                _metric("Cohen's κ", kappa.get("cohen_kappa"), "validation (non-LLM anchor)",
                        f"judge↔anchor agreement on 'was the decision appropriate' · {kappa.get('band', 'n/a')} · n={kappa.get('n_paired')}", kappa.get("n_paired"), gate=True),
                _metric("judge accuracy", kappa.get("judge_accuracy"), "validation (non-LLM anchor)",
                        "raw fraction where judge matched the anchor", kappa.get("n_paired"))]},
         ]},
    ]

    return {
        "meta": {
            "slice": "frozen, stratified, seeded", "n": len(recs),
            "n_grounded": n_g, "n_unanswerable": n_u,
            "reader_model": settings.reader_model,
            "experiment": experiment,
            "judges": {
                "geval": f"openai:{settings.geval_judge_model} (same-family gate judge)",
                "ragas": f"{settings.ragas_judge_model} ({'cross-family' if ragas.get('cross_family') else 'same-family — set RAGAS_JUDGE to a different family'})",
                "kappa_anchor": kappa.get("anchor", "deterministic"),
            },
            "retrieval_ab": bool(ab),
        },
        "stages": stages,
        "flat": flat,
        "over_refusals": over_refusal_cases(recs),
        "judge_errors": {"ragas": ragas.get("error"), "geval": geval.get("error"), "kappa": kappa.get("error")},
    }


def write_report(report: dict) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[report] -> {REPORT_PATH}")


def set_baseline(report: dict) -> dict:
    base = {}
    for path, direction in GATES:
        v = report["flat"].get(path)
        if not isinstance(v, (int, float)):
            continue
        bound = round(v - GATE_TOL, 4) if direction == "higher" else round(v + GATE_TOL, 4)
        base[path] = {"value": v, "direction": direction, "bound": bound, "tol": GATE_TOL}
    REPORTS_DIR.mkdir(exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(base, indent=2), encoding="utf-8")
    print(f"[report] baseline frozen ({len(base)} gated metrics) -> {BASELINE_PATH}")
    return base
