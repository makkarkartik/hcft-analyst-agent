"""Render reports/eval_report.json -> docs/eval_scores.html — the eval scoreboard at every stage
and substage, colour-coded by metric KIND (so a deterministic anchor never reads like an LLM
judge), with each gated metric's pass/fail vs the frozen baseline.

Data-driven on purpose: rerun the experiment, rerun this, the page refreshes. Matches the dark
theme of docs/pipeline_map.html.

    python scripts/build_eval_dashboard.py
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "eval_report.json"
BASELINE = ROOT / "reports" / "eval_baseline.json"
CTX_EVAL = ROOT / "reports" / "context_guard_eval.json"
CTX_BASE = ROOT / "reports" / "context_guard_baseline.json"
OUT = ROOT / "docs" / "eval_scores.html"

# metric KIND -> css class (first matching substring wins)
KIND_CLASS = [
    ("validation", "k-val"), ("anchor", "k-anchor"), ("op-safety", "k-safety"),
    ("component", "k-comp"), ("outcome", "k-outcome"),
]
# flat-path for each (stage metric key) so we can look up the baseline gate. Built from report.flat.


def kind_class(kind: str) -> str:
    for sub, cls in KIND_CLASS:
        if sub in kind:
            return cls
    return "k-comp"


def fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".") if v != int(v) else f"{int(v)}"
    return str(v)


def flat_key_for(metric: dict, stage: str) -> str | None:
    """Map a stage metric to its report.flat path, so we can read the baseline gate verdict."""
    k = metric["key"].lower()
    table = {
        ("Retrieval", "recall"): "retrieval.recall_at_pool",
        ("Retrieval", "exact"): "retrieval.hit_at_k_exact",
        ("Retrieval", "doc"): "retrieval.hit_at_k_doc",
        ("Generation", "faithfulness"): "generation.ragas_faithfulness",
        ("Generation", "relevancy"): "generation.ragas_answer_relevancy",
        ("Generation", "rouge"): "generation.rougeL",
        ("Generation", "bertscore"): "generation.bertscore_f1",
        ("Generation", "hhem mean"): "generation.hhem_mean",
    }
    for (st, sub), path in table.items():
        if stage.startswith(st) and sub in k:
            return path
    if "answer rate" in k:        return "refusal.answerable.answer_rate"
    if "refuse rate" in k:        return "refusal.unanswerable.refuse_rate"
    if k == "fabricated":         return "refusal.unanswerable.fabricated"   # approximate (unanswerable)
    if "κ" in metric["key"] or "kappa" in k: return "validation.cohen_kappa"
    if "judge accuracy" in k:     return "validation.judge_accuracy"
    return None


def gate_badge(metric: dict, stage: str, baseline: dict, flat: dict) -> str:
    if not metric.get("gate"):
        return '<span class="badge ungated">not gated</span>'
    path = flat_key_for(metric, stage)
    spec = baseline.get(path) if path else None
    cur = flat.get(path) if path else None
    if spec is None or not isinstance(cur, (int, float)):
        return '<span class="badge ungated">no baseline</span>'
    bound, direction = spec["bound"], spec["direction"]
    ok = cur >= bound if direction == "higher" else cur <= bound
    arrow = "≥" if direction == "higher" else "≤"
    cls = "pass" if ok else "fail"
    label = "PASS" if ok else "REGRESSED"
    return (f'<span class="badge {cls}">{label}</span>'
            f'<span class="bound">vs {fmt(spec["value"])} (gate {arrow} {fmt(bound)})</span>')


def metric_card(m: dict, stage: str, baseline: dict, flat: dict) -> str:
    cls = kind_class(m["kind"])
    n = f'<span class="n">n={m["n"]}</span>' if m.get("n") is not None else ""
    return f"""
      <div class="card {cls}">
        <div class="val">{fmt(m["value"])}</div>
        <div class="mkey">{escape(m["key"])} {n}</div>
        <div class="kind">{escape(m["kind"])}</div>
        <div class="method">{escape(m["method"])}</div>
        <div class="gate">{gate_badge(m, stage, baseline, flat)}</div>
      </div>"""


def render(report: dict, baseline: dict) -> str:
    meta, flat = report["meta"], report["flat"]
    j = meta.get("judges", {})
    exp = meta.get("experiment")
    exp_html = f'<b>experiment</b> {escape(str(exp))}' if exp else '<b>experiment</b> (offline run — no LangSmith)'

    stages_html = []
    for stage in report["stages"]:
        subs = []
        for sub in stage["substages"]:
            cards = "".join(metric_card(m, stage["stage"], baseline, flat) for m in sub["metrics"])
            subs.append(f'<div class="sub"><div class="sub-h">{escape(sub["name"])}</div>'
                        f'<div class="cards">{cards}</div></div>')
        stages_html.append(f"""
      <section class="stage">
        <div class="stage-h"><span class="stage-t">{escape(stage["stage"])}</span>
          <span class="stage-sub">{escape(stage["subtitle"])}</span></div>
        {''.join(subs)}
      </section>""")

    # over-refusal + judge-error panels
    orc = report.get("over_refusals") or []
    orc_rows = "".join(
        f'<li><code>{escape(c["qa_id"])}</code> gold {escape(c["gold_retrieved"])}'
        f'{" · multi-hop partial (defensible)" if c["partial_multihop"] else " · TRUE over-refusal"}'
        f' — {escape(c["question"][:90])}</li>' for c in orc)
    orc_html = (f'<section class="panel"><h3>Over-refusals ({len(orc)})</h3>'
                f'<ul>{orc_rows}</ul></section>') if orc else ""

    # context-ring (indirect-injection) guard panel — its own eval + gate
    ctx_html = ""
    if CTX_EVAL.exists():
        c = json.loads(CTX_EVAL.read_text(encoding="utf-8"))
        cb = json.loads(CTX_BASE.read_text(encoding="utf-8")) if CTX_BASE.exists() else {}
        def cbadge(key, higher):
            spec, cur = cb.get(key), c.get(key)
            if not spec or not isinstance(cur, (int, float)):
                return '<span class="badge ungated">no baseline</span>'
            ok = cur >= spec["bound"] if higher else cur <= spec["bound"]
            return f'<span class="badge {"pass" if ok else "fail"}">{"PASS" if ok else "REGRESSED"}</span>'
        ctx_html = f"""
      <section class="stage">
        <div class="stage-h" style="border-left-color:var(--safety);background:linear-gradient(90deg,var(--safety-bg),transparent)">
          <span class="stage-t">Context ring — indirect-injection scan</span>
          <span class="stage-sub">retrieved chunks are untrusted input · quarantine poisoned before the reader · n={c['n_benign']} benign / {c['n_poisoned']} poisoned · threshold {c['threshold']}</span></div>
        <div class="sub"><div class="cards">
          <div class="card k-safety"><div class="val">{fmt(c['detection_recall'])}</div>
            <div class="mkey">detection recall</div><div class="kind">op-safety (deterministic probe)</div>
            <div class="method">planted injections quarantined (want high)</div>
            <div class="gate">{cbadge('detection_recall', True)}</div></div>
          <div class="card k-safety"><div class="val">{fmt(c['false_positive_rate'])}</div>
            <div class="mkey">false-positive rate</div><div class="kind">op-safety (deterministic probe)</div>
            <div class="method">clean chunks wrongly quarantined (want low — costs retrieval hits)</div>
            <div class="gate">{cbadge('false_positive_rate', False)}</div></div>
          <div class="card k-safety"><div class="val">{fmt(c.get('precision'))}</div>
            <div class="mkey">precision</div><div class="kind">op-safety (deterministic probe)</div>
            <div class="method">benign p90 score {fmt(c['benign_score_dist']['p90'])} · poisoned p50 {fmt(c['poisoned_score_dist']['p50'])}</div>
            <div class="gate"><span class="badge ungated">diagnostic</span></div></div>
        </div></div>
      </section>"""

    errs = {k: v for k, v in (report.get("judge_errors") or {}).items() if v}
    err_html = (f'<section class="panel err"><h3>Judge availability</h3><ul>'
                + "".join(f'<li><b>{escape(k)}</b>: {escape(str(v))}</li>' for k, v in errs.items())
                + '</ul></section>') if errs else ""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>HCFT Agent — Eval Scoreboard</title>
<style>
  :root {{
    --bg:#0d1117; --panel:#161b22; --ink:#e6edf3; --muted:#8b949e; --row:#21262d;
    --comp:#1f6feb; --comp-bg:#0d2540; --outcome:#238636; --outcome-bg:#0f2a17;
    --anchor:#6e7681; --anchor-bg:#1c2128; --safety:#1f9e9e; --safety-bg:#0c2626;
    --val:#a371f7; --val-bg:#1d1430; --amber:#d8a23a;
    --pass:#3fb950; --fail:#f85149;
  }}
  *{{box-sizing:border-box}}
  body{{margin:0;padding:26px 30px 64px;background:var(--bg);color:var(--ink);
    font:14px/1.45 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
  h1{{font-size:21px;margin:0 0 4px}}
  .sub-line{{color:var(--muted);font-size:13px;margin:0 0 6px}}
  .meta{{background:var(--panel);border:1px solid var(--row);border-radius:10px;padding:12px 16px;
    margin:14px 0 22px;font-size:12.5px;color:var(--muted);display:flex;gap:26px;flex-wrap:wrap}}
  .meta b{{color:var(--ink)}}
  .legend{{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-bottom:22px}}
  .legend .sw{{display:inline-block;width:11px;height:11px;border-radius:3px;vertical-align:-1px;margin-right:5px}}
  .stage{{background:var(--panel);border:1px solid var(--row);border-radius:12px;margin-bottom:22px;overflow:hidden}}
  .stage-h{{padding:13px 18px;border-bottom:1px solid var(--row);border-left:4px solid var(--comp);
    background:linear-gradient(90deg,var(--comp-bg),transparent)}}
  .stage-t{{font-weight:700;font-size:15px}} .stage-sub{{color:var(--muted);font-size:12.5px;margin-left:10px}}
  .sub{{padding:12px 18px}} .sub+.sub{{border-top:1px solid var(--row)}}
  .sub-h{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:10px}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}}
  .card{{border-radius:10px;padding:12px 14px;border:1px solid var(--row);background:#10151c}}
  .card .val{{font-size:26px;font-weight:700;line-height:1.1}}
  .card .mkey{{font-weight:600;font-size:13px;margin-top:3px}}
  .card .n{{color:var(--muted);font-weight:400;font-size:11px;margin-left:4px}}
  .card .kind{{font-size:11px;color:var(--muted);margin-top:2px}}
  .card .method{{font-size:11.5px;color:#c9d1d9;opacity:.85;margin-top:7px}}
  .card .gate{{margin-top:9px;font-size:11px}}
  .k-comp{{border-left:4px solid var(--comp)}} .k-outcome{{border-left:4px solid var(--outcome)}}
  .k-anchor{{border-left:4px solid var(--anchor)}} .k-safety{{border-left:4px solid var(--safety)}}
  .k-val{{border-left:4px solid var(--val)}}
  .badge{{padding:1px 8px;border-radius:999px;border:1px solid currentColor;font-weight:600}}
  .badge.pass{{color:var(--pass)}} .badge.fail{{color:var(--fail)}} .badge.ungated{{color:var(--muted)}}
  .bound{{color:var(--muted);margin-left:8px}}
  .panel{{background:var(--panel);border:1px solid var(--row);border-radius:12px;padding:14px 18px;margin-bottom:18px}}
  .panel h3{{margin:0 0 8px;font-size:14px}} .panel.err{{border-color:#5a2a2a}}
  .panel ul{{margin:0;padding-left:18px}} .panel li{{font-size:12.5px;margin:3px 0;color:#c9d1d9}}
  code{{background:#0d1117;border:1px solid var(--row);border-radius:4px;padding:0 4px;font-size:11.5px}}
</style></head><body>

  <h1>HCFT Agent — Eval Scoreboard</h1>
  <p class="sub-line">Every pipeline stage × substage, scored. Deterministic anchors break judge
    circularity; LLM judges sit on top; κ validates the judges. Each score is also a LangSmith
    feedback on the run that produced it.</p>

  <div class="meta">
    <span><b>slice</b> {escape(meta.get("slice",""))} · n={meta.get("n")} ({meta.get("n_grounded")} grounded / {meta.get("n_unanswerable")} unanswerable)</span>
    <span><b>reader</b> {escape(str(meta.get("reader_model")))}</span>
    <span><b>G-Eval</b> {escape(str(j.get("geval")))}</span>
    <span><b>RAGAS</b> {escape(str(j.get("ragas")))}</span>
    <span><b>κ anchor</b> {escape(str(j.get("kappa_anchor")))}</span>
    <span>{exp_html}</span>
  </div>

  <div class="legend">
    <span><span class="sw" style="background:var(--comp)"></span>component</span>
    <span><span class="sw" style="background:var(--outcome)"></span>outcome</span>
    <span><span class="sw" style="background:var(--anchor)"></span>anchor (non-LLM)</span>
    <span><span class="sw" style="background:var(--safety)"></span>op-safety</span>
    <span><span class="sw" style="background:var(--val)"></span>validation (κ)</span>
    <span><span class="badge pass">PASS</span> / <span class="badge fail">REGRESSED</span> vs frozen baseline</span>
  </div>

  {''.join(stages_html)}
  {ctx_html}
  {orc_html}
  {err_html}

</body></html>"""


def main() -> None:
    if not REPORT.exists():
        raise SystemExit(f"missing {REPORT} — run `python -m hcft_agent.eval.experiment ...` first")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
    OUT.write_text(render(report, baseline), encoding="utf-8")
    print(f"[dashboard] -> {OUT}")


if __name__ == "__main__":
    main()
