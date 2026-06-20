"""Judge validation by Cohen's κ — the circularity-breaker for the LLM-judge layer.

A G-Eval score on its own is just one model's opinion of another model's output; if both are the
same family, a shared blind spot passes twice and we'd never know. κ fixes that by measuring how
well the G-Eval verdict agrees with a NON-LLM anchor on the SAME yes/no question — *"was the
agent's answer-vs-refuse decision appropriate?"* — over the same rows.

  * **anchor (rater A)** — DETERMINISTIC, from gold. The agent SHOULD answer iff the row is
    grounded AND its gold chunk was actually retrieved; otherwise it should refuse. So the anchor
    label is ``appropriate = (should_answer == answered)``. No LLM touches this — that's the point.
  * **judge (rater B)** — the G-Eval verdict (``score ≥ threshold`` -> appropriate).

κ near 1 means the LLM judge tracks ground truth and we can trust it where no gold exists (real
free-text answers); κ near 0 means it's noise and the deterministic layer must carry the gate.
Interpretation bands follow Landis & Koch. Drop a human-labeled file in
``settings.kappa_human_labels`` to swap the anchor for true human labels (gold-standard κ).
"""
from __future__ import annotations

import json
from pathlib import Path

from hcft_agent.config import settings


def _should_answer(r: dict) -> bool:
    """Deterministic 'the correct decision is to ANSWER': grounded row whose gold chunk was
    actually retrieved. Grounded-but-not-retrieved and unanswerable both -> should refuse."""
    if r["eval_kind"] == "unanswerable_rag":
        return False
    return bool(set(r.get("gold_chunk_ids", [])) & set(r.get("retrieved_ids", [])))


def _anchor_label(r: dict) -> int:
    """1 if the agent's decision matched the deterministic 'should answer/refuse', else 0."""
    answered = not r["is_refusal"]
    return int(answered == _should_answer(r))


def _human_anchor() -> dict[str, int]:
    """Optional human labels: ``{qa_id: appropriate(0/1)}`` from a JSONL of
    ``{"qa_id": ..., "appropriate": true/false}``. Empty if the file isn't configured/found."""
    path = settings.kappa_human_labels
    if not path or not Path(path).exists():
        return {}
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out[d["qa_id"]] = int(bool(d["appropriate"]))
    return out


def _kappa_band(k: float) -> str:
    if k < 0:      return "worse than chance"
    if k < 0.20:   return "slight"
    if k < 0.40:   return "fair"
    if k < 0.60:   return "moderate"
    if k < 0.80:   return "substantial"
    return "almost perfect"


def judge_kappa(recs: list[dict], geval: dict) -> dict:
    """Cohen's κ between the G-Eval verdict and the anchor (human if configured, else
    deterministic). Pairs rows by ``qa_id``; rows the judge errored on are skipped."""
    out: dict = {"anchor": "deterministic", "metric": "cohen_kappa"}
    geval_by_id = {p["qa_id"]: p for p in geval.get("per_row", []) if "appropriate" in p}
    if not geval_by_id:
        out["error"] = "no usable G-Eval verdicts (judge unavailable or all errored)"
        return out

    human = _human_anchor()
    if human:
        out["anchor"] = "human"

    judge_labels, anchor_labels, paired = [], [], []
    for r in recs:
        qid = r["qa_id"]
        if qid not in geval_by_id:
            continue
        jl = int(bool(geval_by_id[qid]["appropriate"]))
        al = human.get(qid) if human else _anchor_label(r)
        if al is None:        # human file present but this row unlabeled -> skip
            continue
        judge_labels.append(jl)
        anchor_labels.append(al)
        paired.append({"qa_id": qid, "judge": jl, "anchor": al, "agree": int(jl == al)})

    n = len(paired)
    out["n_paired"] = n
    if n == 0:
        out["error"] = "no rows had both an anchor and a judge verdict"
        return out
    out["judge_accuracy"] = round(sum(p["agree"] for p in paired) / n, 3)

    try:
        from sklearn.metrics import cohen_kappa_score
        # If either rater is constant, κ is undefined (0/0); report observed agreement instead.
        if len(set(judge_labels)) < 2 or len(set(anchor_labels)) < 2:
            out["cohen_kappa"] = None
            out["note"] = "κ undefined (a rater gave a single class); see judge_accuracy"
        else:
            k = float(cohen_kappa_score(anchor_labels, judge_labels))
            out["cohen_kappa"] = round(k, 3)
            out["band"] = _kappa_band(k)
    except Exception as e:
        out["error"] = f"sklearn unavailable ({type(e).__name__}: {e})"
    out["pairs"] = paired
    return out
