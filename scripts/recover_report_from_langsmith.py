"""Salvage the eval report from a LangSmith experiment whose process wedged AFTER posting all the
per-row scores but BEFORE reaching build_report() (the per-row asyncio.run() teardown hang).

Everything we need is already on the server: each experiment run carries the agent's output
(run.outputs) + a link to its dataset example (gold), and each evaluator score is attached as
feedback. So we reconstruct the same records + judge dicts experiment._records_from_results would
have built, then run the SHARED build_report / judge_kappa / set_baseline — byte-identical output to
a clean run, minus the 1-2 rows the judges never finished.

    python scripts/recover_report_from_langsmith.py --experiment hcft-agent-1f2854b7 \
        --retrieval-ab reports/retrieval_ab.json --set-baseline
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["LANGSMITH_TRACING"] = "false"
from dotenv import load_dotenv

load_dotenv()

from hcft_agent.config import settings
from hcft_agent.eval import judges
from hcft_agent.eval.report import REPORTS_DIR, build_report, set_baseline, write_report
from hcft_agent.eval.validate import judge_kappa

DATASET = "hcft-agent-eval-v3"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--retrieval-ab", dest="ab", type=Path, default=None)
    ap.add_argument("--set-baseline", action="store_true")
    args = ap.parse_args()

    from langsmith import Client
    c = Client()

    ds = next(c.list_datasets(dataset_name=args.dataset))
    examples = {str(e.id): e for e in c.list_examples(dataset_id=ds.id)}
    runs = list(c.list_runs(project_name=args.experiment, is_root=True))
    print(f"[recover] experiment {args.experiment}: {len(runs)} runs, {len(examples)} dataset examples")

    recs, ragas_rows, geval_rows = [], [], []
    f_all, a_all, g_all = [], [], []
    for r in runs:
        ex = examples.get(str(r.reference_example_id))
        if ex is None:
            continue
        out, ref = (r.outputs or {}), (ex.outputs or {})
        scores = {f.key: f.score for f in c.list_feedback(run_ids=[r.id])}
        qid = ref.get("qa_id")
        recs.append({
            "qa_id": qid, "question": (ex.inputs or {}).get("question"),
            "eval_kind": ref.get("eval_kind"), "hop_type": ref.get("hop_type"),
            "gold_answer": ref.get("gold_answer") or "", "gold_chunk_ids": ref.get("gold_chunk_ids") or [],
            "agent_answer": out.get("answer") or "", "is_refusal": bool(out.get("is_refusal")),
            "retrieved_ids": out.get("retrieved_ids") or [], "context": out.get("context") or "",
            "grounded_score": out.get("grounded_score"), "terminal": out.get("terminal"),
        })
        f, a = scores.get("ragas_faithfulness"), scores.get("ragas_answer_relevancy")
        if f is not None or a is not None:
            ragas_rows.append({"qa_id": qid, "faithfulness": f, "answer_relevancy": a})
            if isinstance(f, (int, float)): f_all.append(f)
            if isinstance(a, (int, float)): a_all.append(a)
        g = scores.get("geval_refusal")
        if isinstance(g, (int, float)):
            geval_rows.append({"qa_id": qid, "score": g, "appropriate": g >= settings.geval_threshold})
            g_all.append(g)

    ragas = {
        "n_scored": len(f_all), "judge": settings.ragas_judge_model,
        "cross_family": judges.is_cross_family(), "per_row": ragas_rows,
        "faithfulness_mean": round(sum(f_all) / len(f_all), 3) if f_all else None,
        "answer_relevancy_mean": round(sum(a_all) / len(a_all), 3) if a_all else None,
        "error": None,
    }
    geval = {
        "n_scored": len(g_all), "judge": settings.geval_judge_model,
        "threshold": settings.geval_threshold, "per_row": geval_rows,
        "mean_score": round(sum(g_all) / len(g_all), 3) if g_all else None,
        "pass_rate": round(sum(1 for r in geval_rows if r["appropriate"]) / len(g_all), 3) if g_all else None,
        "error": None,
    }
    print(f"[recover] scored rows: faithfulness={len(f_all)} relevancy={len(a_all)} geval={len(g_all)} | records={len(recs)}")

    ab = json.loads(args.ab.read_text(encoding="utf-8")) if args.ab else None
    kappa = judge_kappa(recs, geval)
    report = build_report(recs, ragas, geval, kappa, ab, experiment=args.experiment)
    REPORTS_DIR.mkdir(exist_ok=True)
    write_report(report)

    if report["over_refusals"]:
        print(f"\n[over-refusal] {len(report['over_refusals'])} answerable-in-context rows refused:")
        for cse in report["over_refusals"]:
            tag = " (multi-hop partial — defensible)" if cse["partial_multihop"] else " (TRUE over-refusal)"
            print(f"   - {cse['qa_id']} gold {cse['gold_retrieved']}{tag}")
    if args.set_baseline:
        set_baseline(report)


if __name__ == "__main__":
    main()
