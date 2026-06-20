"""LangSmith-native eval: an Experiment over a Dataset, with every score attached to the run that
produced it — so a faithfulness dip links straight to the trace, and successive agent improvements
are comparable example-by-example on the SAME dataset (not stranded in a separate metrics UI).

Shape:
  * **Dataset** — the frozen slice, one example per qa_id (inputs={"question"}, reference outputs
    carry gold). Stable substrate; create once, reuse forever.
  * **target** — the live agent (``rag_chat.answer``). Each example -> one traced run.
  * **evaluators** (attached per run): deterministic refusal-correctness + retrieval hit (vs gold,
    non-circular), HHEM groundedness (the live guard), and the LLM judges RAGAS faithfulness/
    answer-relevancy (cross-family) + G-Eval refusal-quality (same-family).
  * **summary evaluators** (experiment-level): the refusal-by-regime rates.
  * after evaluate(): reconstruct records from the run outputs, compute Cohen's κ (judge vs
    deterministic anchor), and emit the SAME ``eval_report.json`` the dashboard + gate read.

    python -m hcft_agent.eval.experiment --grounded 20 --unanswerable 10 \\
        --retrieval-ab reports\\retrieval_ab.json --set-baseline
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Windows: the default Proactor event loop raises "Event loop is closed" and can HANG when
# asyncio.run() (RAGAS scores each row in its own short-lived loop) tears down httpx connections —
# that's what wedged the last run at row 29/30. The Selector loop closes cleanly. Set before any loop.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from hcft_agent.config import settings
from hcft_agent.eval import agent_eval, judges
from hcft_agent.eval.report import REPORTS_DIR, build_report, set_baseline, write_report
from hcft_agent.eval.validate import judge_kappa

DATASET = "hcft-agent-eval-v3"
RECORDS_FRESH = REPORTS_DIR / "eval_records_fresh.jsonl"


# ----------------------------------------------------------------- dataset
def sync_dataset(client, name: str, rows: list[dict], recreate: bool = False) -> str:
    """Create the dataset from the frozen slice if absent (idempotent). One example per qa_id;
    gold travels in the reference outputs so deterministic evaluators stay non-circular."""
    existing = next((d for d in client.list_datasets(dataset_name=name)), None)
    if existing and recreate:
        client.delete_dataset(dataset_id=existing.id)
        existing = None
    if existing:
        n = sum(1 for _ in client.list_examples(dataset_id=existing.id))   # 0.8.x has no count_examples
        if n >= len(rows):
            print(f"[experiment] reusing dataset '{name}' ({n} examples)")
            return existing.id
        print(f"[experiment] dataset '{name}' has {n} < {len(rows)} — recreating")
        client.delete_dataset(dataset_id=existing.id)

    ds = client.create_dataset(dataset_name=name,
                               description="HCFT agent frozen eval slice (grounded + unanswerable)")
    client.create_examples(
        dataset_id=ds.id,
        inputs=[{"question": r["question"]} for r in rows],
        outputs=[{
            "qa_id": r["qa_id"], "eval_kind": r["eval_kind"], "hop_type": r.get("hop_type"),
            "gold_answer": r.get("answer") or "",
            "gold_chunk_ids": [c for c in (r.get("source_chunk_ids") or [r.get("source_chunk_id")]) if c],
        } for r in rows],
    )
    print(f"[experiment] created dataset '{name}' with {len(rows)} examples")
    return ds.id


# ----------------------------------------------------------------- target (the agent)
# IMPORTANT: evaluate() runs the target in a worker thread, and loading/using torch+CUDA models in a
# non-main thread SEGFAULTS on Windows. So we run the agent in the MAIN thread up front and hand
# evaluate() a pure dict-lookup target — only the CPU/API judges (deterministic + RAGAS + G-Eval)
# then run in evaluate()'s threads, never the GPU agent.
def _extract(state: dict) -> dict:
    """JSON-serializable, evaluator-relevant fields (drop the heavy `candidates`)."""
    return {
        "answer": state.get("answer") or "",
        "is_refusal": bool(state.get("is_refusal")),
        "retrieved_ids": state.get("retrieved_ids") or [],
        "context": state.get("context") or "",
        "grounded_score": state.get("grounded_score"),
        "terminal": state.get("terminal"),
    }


def precompute_outputs(rows: list[dict]) -> dict:
    """Run the live agent on every question in the MAIN thread; return {question: output_dict}.

    CONFIRMED root cause of the Windows segfault: a LangSmith background thread (Client or tracer)
    active while torch initializes CUDA crashes the process. So this runs with tracing forced OFF
    and is called BEFORE any LangSmith Client exists — models load + the agent runs with no LangSmith
    thread anywhere near torch."""
    import os

    from hcft_agent.agents.rag_chat import answer, build_app, warmup

    os.environ["LANGSMITH_TRACING"] = "false"
    print("[experiment] warming models in the main thread, tracing off (segfault guard) ...")
    warmup()
    build_app()                                   # compile the graph (this calls init_telemetry)...
    os.environ["LANGSMITH_TRACING"] = "false"     # ...then keep the WHOLE precompute untraced
    out = {}
    for i, r in enumerate(rows, 1):
        out[r["question"]] = _extract(answer(r["question"]))
        if i % 5 == 0:
            print(f"   ...agent ran {i}/{len(rows)}")
    return out


def make_target(precomputed: dict):
    def target(inputs: dict) -> dict:
        return precomputed[inputs["question"]]          # instant dict lookup, thread-safe, no GPU
    return target


# ----------------------------------------------------------------- helpers
def _got_gold(out: dict, ref: dict) -> bool:
    return bool(set(ref.get("gold_chunk_ids", [])) & set(out.get("retrieved_ids", [])))


def _doc(cid: str) -> str:
    return cid.split("_p")[0]


# ----------------------------------------------------------------- evaluators (per run)
def ev_refusal_correct(run, example) -> dict:
    """Deterministic: the agent SHOULD answer iff grounded AND gold retrieved, else refuse."""
    out, ref = run.outputs or {}, example.outputs or {}
    should_answer = ref.get("eval_kind") != "unanswerable_rag" and _got_gold(out, ref)
    appropriate = (not out.get("is_refusal")) == should_answer
    return {"key": "refusal_correct", "score": int(appropriate),
            "comment": "should_answer" if should_answer else "should_refuse"}


def ev_retrieval_hit_exact(run, example) -> dict:
    out, ref = run.outputs or {}, example.outputs or {}
    if ref.get("eval_kind") == "unanswerable_rag":
        return {"key": "retrieval_hit_exact", "score": None, "comment": "n/a (unanswerable)"}
    return {"key": "retrieval_hit_exact", "score": int(_got_gold(out, ref))}


def ev_retrieval_hit_doc(run, example) -> dict:
    out, ref = run.outputs or {}, example.outputs or {}
    if ref.get("eval_kind") == "unanswerable_rag":
        return {"key": "retrieval_hit_doc", "score": None, "comment": "n/a (unanswerable)"}
    gold_docs = {_doc(c) for c in ref.get("gold_chunk_ids", [])}
    got_docs = {_doc(c) for c in out.get("retrieved_ids", [])}
    return {"key": "retrieval_hit_doc", "score": int(bool(gold_docs & got_docs))}


def ev_hhem(run, example) -> dict:
    """The live groundedness guard's own score, surfaced as a feedback (non-refusal rows only)."""
    out = run.outputs or {}
    if out.get("is_refusal") or not isinstance(out.get("grounded_score"), (int, float)):
        return {"key": "hhem_grounded", "score": None, "comment": "n/a (refusal/no score)"}
    return {"key": "hhem_grounded", "score": float(out["grounded_score"])}


# --- LLM judges: build the heavy objects once, reuse across rows ---
_RAGAS = {"built": False, "faith": None, "relev": None, "err": None, "row_err": None}
_GEVAL = {"built": False, "metric": None, "err": None}


def _ragas():
    if not _RAGAS["built"]:
        _RAGAS["built"] = True
        try:
            _RAGAS["faith"], _RAGAS["relev"] = judges.build_ragas()
        except Exception as e:
            _RAGAS["err"] = f"{type(e).__name__}: {e}"
    return _RAGAS


def _geval():
    if not _GEVAL["built"]:
        _GEVAL["built"] = True
        try:
            _GEVAL["metric"] = judges.build_geval()
        except Exception as e:
            _GEVAL["err"] = f"{type(e).__name__}: {e}"
    return _GEVAL


def ev_ragas_faithfulness(run, example) -> dict:
    out, ref = run.outputs or {}, example.outputs or {}
    if ref.get("eval_kind") == "unanswerable_rag" or out.get("is_refusal") or not out.get("context"):
        return {"key": "ragas_faithfulness", "score": None, "comment": "n/a (refusal/unanswerable)"}
    rg = _ragas()
    if rg["err"]:
        return {"key": "ragas_faithfulness", "score": None, "comment": rg["err"]}
    try:
        f, _ = judges.ragas_score_row(rg["faith"], rg["relev"], example.inputs["question"],
                                      out["answer"], judges.split_context(out["context"]))
        return {"key": "ragas_faithfulness", "score": round(f, 3)}
    except Exception as e:
        _RAGAS["row_err"] = f"{type(e).__name__}: {e}"
        return {"key": "ragas_faithfulness", "score": None, "comment": _RAGAS["row_err"]}


def ev_ragas_relevancy(run, example) -> dict:
    out, ref = run.outputs or {}, example.outputs or {}
    if ref.get("eval_kind") == "unanswerable_rag" or out.get("is_refusal") or not out.get("context"):
        return {"key": "ragas_answer_relevancy", "score": None, "comment": "n/a (refusal/unanswerable)"}
    rg = _ragas()
    if rg["err"]:
        return {"key": "ragas_answer_relevancy", "score": None, "comment": rg["err"]}
    try:
        _, a = judges.ragas_score_row(rg["faith"], rg["relev"], example.inputs["question"],
                                      out["answer"], judges.split_context(out["context"]))
        return {"key": "ragas_answer_relevancy", "score": round(a, 3)}
    except Exception as e:
        return {"key": "ragas_answer_relevancy", "score": None, "comment": f"{type(e).__name__}: {e}"}


def ev_geval(run, example) -> dict:
    out = run.outputs or {}
    gv = _geval()
    if gv["err"]:
        return {"key": "geval_refusal", "score": None, "comment": gv["err"]}
    try:
        s, reason = judges.geval_score_row(gv["metric"], example.inputs["question"],
                                           out.get("answer", ""), judges.split_context(out.get("context", "")))
        return {"key": "geval_refusal", "score": round(s, 3), "comment": reason[:280]}
    except Exception as e:
        return {"key": "geval_refusal", "score": None, "comment": f"{type(e).__name__}: {e}"}


# ----------------------------------------------------------------- summary evaluators (experiment-level)
def summ_answer_rate(runs, examples) -> dict:
    """Answer-rate on answerable-in-context rows (gold retrieved) — over-refusal's complement."""
    num = den = 0
    for run, ex in zip(runs, examples):
        out, ref = run.outputs or {}, ex.outputs or {}
        if ref.get("eval_kind") == "unanswerable_rag" or not _got_gold(out, ref):
            continue
        den += 1
        num += 0 if out.get("is_refusal") else 1
    return {"key": "answerable_answer_rate", "score": (num / den) if den else None}


def summ_unanswerable_refuse_rate(runs, examples) -> dict:
    num = den = 0
    for run, ex in zip(runs, examples):
        out, ref = run.outputs or {}, ex.outputs or {}
        if ref.get("eval_kind") != "unanswerable_rag":
            continue
        den += 1
        num += 1 if out.get("is_refusal") else 0
    return {"key": "unanswerable_refuse_rate", "score": (num / den) if den else None}


# ----------------------------------------------------------------- post-hoc: rebuild records + report
def _records_from_results(results) -> tuple[list[dict], dict, dict]:
    """Rebuild agent_eval-shaped records + the ragas/geval judge dicts from the ExperimentResults,
    so report.build_report (the single source of truth) runs unchanged."""
    recs, ragas_rows, geval_rows = [], [], []
    f_all, a_all, g_all = [], [], []
    for item in results:
        try:
            run, ex = item["run"], item["example"]
            out, ref = run.outputs or {}, ex.outputs or {}
            evs = {er.key: er.score for er in (item.get("evaluation_results") or {}).get("results", [])}
            qid = ref.get("qa_id")
            recs.append({
                "qa_id": qid, "question": ex.inputs.get("question"),
                "eval_kind": ref.get("eval_kind"), "hop_type": ref.get("hop_type"),
                "gold_answer": ref.get("gold_answer") or "", "gold_chunk_ids": ref.get("gold_chunk_ids") or [],
                "agent_answer": out.get("answer") or "", "is_refusal": bool(out.get("is_refusal")),
                "retrieved_ids": out.get("retrieved_ids") or [], "context": out.get("context") or "",
                "grounded_score": out.get("grounded_score"), "terminal": out.get("terminal"),
            })
            f, a = evs.get("ragas_faithfulness"), evs.get("ragas_answer_relevancy")
            if f is not None or a is not None:
                ragas_rows.append({"qa_id": qid, "faithfulness": f, "answer_relevancy": a})
                if isinstance(f, (int, float)): f_all.append(f)
                if isinstance(a, (int, float)): a_all.append(a)
            g = evs.get("geval_refusal")
            if isinstance(g, (int, float)):
                geval_rows.append({"qa_id": qid, "score": g, "appropriate": g >= settings.geval_threshold})
                g_all.append(g)
        except Exception as e:
            print(f"[experiment] skipped a result row: {type(e).__name__}: {e}")

    ragas = {
        "n_scored": len(ragas_rows), "judge": settings.ragas_judge_model,
        "cross_family": judges.is_cross_family(), "per_row": ragas_rows,
        "faithfulness_mean": round(sum(f_all) / len(f_all), 3) if f_all else None,
        "answer_relevancy_mean": round(sum(a_all) / len(a_all), 3) if a_all else None,
        "error": _RAGAS["err"] or (_RAGAS["row_err"] if not f_all and not a_all else None),
    }
    geval = {
        "n_scored": len(g_all), "judge": settings.geval_judge_model,
        "threshold": settings.geval_threshold, "per_row": geval_rows,
        "mean_score": round(sum(g_all) / len(g_all), 3) if g_all else None,
        "pass_rate": round(sum(1 for r in geval_rows if r["appropriate"]) / len(g_all), 3) if g_all else None,
        "error": _GEVAL["err"],
    }
    return recs, ragas, geval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", type=Path, default=agent_eval.DEFAULT_EVALSET)
    ap.add_argument("--grounded", type=int, default=20)
    ap.add_argument("--unanswerable", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--recreate", action="store_true", help="rebuild the dataset from scratch")
    ap.add_argument("--retrieval-ab", dest="ab", type=Path, default=None)
    ap.add_argument("--no-judges", action="store_true", help="deterministic evaluators only (no API spend)")
    ap.add_argument("--set-baseline", action="store_true")
    args = ap.parse_args()

    import os

    from langsmith import Client, evaluate
    from hcft_agent.obs.telemetry import flush, init_telemetry

    rows = agent_eval.load_slice(args.evalset, args.grounded, args.unanswerable, args.seed)
    print(f"[experiment] frozen slice: {len(rows)} ({args.grounded} grounded + {args.unanswerable} unanswerable)")

    # 1. Run the agent (loads GPU models) FIRST, before any LangSmith Client exists — a LangSmith
    #    background thread active during torch CUDA init segfaults on Windows (confirmed exit 139).
    precomputed = precompute_outputs(rows)

    # 2. Only now touch LangSmith (no torch loading past here). Keep the GLOBAL tracer OFF on purpose:
    #    evaluate() builds the experiment + attaches every score through the Client directly, so it
    #    needs NO env tracer. With tracing ON, the only thing that emits traces during evaluate() is the
    #    judges (RAGAS/G-Eval) — and they orphan into the tracing project as independent root traces
    #    (RAGAS's asyncio.run() drops the run-tree contextvars; DeepEval owns its client; and the agent
    #    itself ran untraced in precompute, so there's no parent run to nest under). Off = clean
    #    experiment, scores still per-run, no orphan noise. init_telemetry() only ensures the API key /
    #    project env are loaded for the Client.
    init_telemetry("hcft-agent")
    os.environ["LANGSMITH_TRACING"] = "false"
    client = Client()
    ds_id = sync_dataset(client, args.dataset, rows, recreate=args.recreate)
    evaluators = [ev_refusal_correct, ev_retrieval_hit_exact, ev_retrieval_hit_doc, ev_hhem]
    if not args.no_judges:
        _ragas(); _geval()                              # build judge objects before evaluate()'s threads
        evaluators += [ev_ragas_faithfulness, ev_ragas_relevancy, ev_geval]

    results = evaluate(
        make_target(precomputed),
        data=args.dataset,
        evaluators=evaluators,
        summary_evaluators=[summ_answer_rate, summ_unanswerable_refuse_rate],
        experiment_prefix="hcft-agent",
        metadata={"reader": settings.reader_model, "retrieval_mode": settings.retrieval_mode,
                  "ragas_judge": settings.ragas_judge_model, "geval_judge": settings.geval_judge_model},
        max_concurrency=4,                     # agent already ran in main; only API/CPU judges run here
    )

    exp_name = (getattr(results, "experiment_name", None)
                or getattr(getattr(results, "_manager", None), "experiment_name", None))
    print(f"\n[experiment] LangSmith experiment: {exp_name}")

    # Rebuild records + judge dicts from the run-attached scores, then emit the shared report.
    recs, ragas, geval = _records_from_results(list(results))
    REPORTS_DIR.mkdir(exist_ok=True)
    with open(RECORDS_FRESH, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[experiment] fresh records -> {RECORDS_FRESH} (re-judge cheaply via run_eval --records)")
    ab = json.loads(args.ab.read_text(encoding="utf-8")) if args.ab else None
    kappa = judge_kappa(recs, geval)
    report = build_report(recs, ragas, geval, kappa, ab, experiment=exp_name)
    write_report(report)

    if report["over_refusals"]:
        print(f"\n[over-refusal] {len(report['over_refusals'])} answerable-in-context rows refused:")
        for c in report["over_refusals"]:
            tag = " (multi-hop partial — defensible)" if c["partial_multihop"] else " (TRUE over-refusal)"
            print(f"   - {c['qa_id']} gold {c['gold_retrieved']}{tag}")

    if args.set_baseline:
        set_baseline(report)
    flush()


if __name__ == "__main__":
    main()
