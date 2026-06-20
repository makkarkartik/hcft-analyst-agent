"""OFFLINE eval path — deterministic anchors + LLM judges + κ, no network beyond the judge APIs.

Use this when you want the gate's ``eval_report.json`` WITHOUT going through LangSmith (fast
iteration, CI without a tracer, or re-judging a dumped records file). For the linked-to-runs path
that logs an Experiment over a Dataset, use ``hcft_agent.eval.experiment`` instead — both emit the
identical report schema (built in ``report.py``).

    python -m hcft_agent.eval.run_eval --records ...\\data\\eval_records_v3.jsonl \\
        --retrieval-ab reports\\retrieval_ab.json --set-baseline
    python -m hcft_agent.eval.run_eval --collect 20 10 --no-judges        # det. layer only, no API spend
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hcft_agent.eval import agent_eval
from hcft_agent.eval.judges import geval_refusal, ragas_scores
from hcft_agent.eval.report import build_report, set_baseline, write_report
from hcft_agent.eval.validate import judge_kappa

DEFAULT_RECORDS = Path(
    r"C:\Users\kartik\OMSCS\Personal Projects\SLM_Fine_Tuning\data\eval_records_v3.jsonl"
)


def get_records(args) -> list[dict]:
    if args.collect:
        ng, nu = args.collect
        rows = agent_eval.load_slice(args.evalset, ng, nu, args.seed)
        print(f"[run_eval] collecting fresh: {len(rows)} rows through the live agent ...")
        recs = agent_eval.collect(rows)
        if args.dump:
            with open(args.dump, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[run_eval] dumped fresh records -> {args.dump}")
        return recs
    recs = [json.loads(l) for l in open(args.records, encoding="utf-8")]
    print(f"[run_eval] loaded {len(recs)} records from {args.records}")
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    ap.add_argument("--evalset", type=Path, default=agent_eval.DEFAULT_EVALSET)
    ap.add_argument("--collect", nargs=2, type=int, metavar=("NG", "NU"),
                    help="re-run the agent fresh on NG grounded + NU unanswerable")
    ap.add_argument("--dump", type=Path, default=None, help="when --collect, also write the records here")
    ap.add_argument("--retrieval-ab", dest="ab", type=Path, default=None,
                    help="reports/retrieval_ab.json from scripts/retrieval_ab.py --out")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-judges", action="store_true", help="deterministic layer only (no API spend)")
    ap.add_argument("--set-baseline", action="store_true", help="freeze the gate thresholds from this run")
    args = ap.parse_args()

    recs = get_records(args)
    ab = json.loads(args.ab.read_text(encoding="utf-8")) if args.ab else None

    if args.no_judges:
        ragas = {"note": "skipped (--no-judges)"}; geval = {"note": "skipped"}; kappa = {"note": "skipped"}
    else:
        print("[run_eval] RAGAS (cross-family) ...");          ragas = ragas_scores(recs)
        print("[run_eval] G-Eval (refusal correctness) ...");  geval = geval_refusal(recs)
        print("[run_eval] Cohen's κ (judge vs anchor) ...");   kappa = judge_kappa(recs, geval)

    report = build_report(recs, ragas, geval, kappa, ab)
    write_report(report)

    orc = report["over_refusals"]
    if orc:
        print(f"\n[over-refusal] {len(orc)} answerable-in-context rows were refused:")
        for c in orc:
            flag = " (multi-hop partial — defensible)" if c["partial_multihop"] else " (TRUE over-refusal)"
            print(f"   - {c['qa_id']} gold {c['gold_retrieved']}{flag}\n       {c['question'][:90]}")

    if args.set_baseline:
        set_baseline(report)


if __name__ == "__main__":
    main()
