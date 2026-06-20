"""Regression gate — turns the eval report into a CI guard.

The flow is deliberately two-step so the gate is cheap and deterministic in CI:
  1. someone runs ``python -m hcft_agent.eval.run_eval ... --set-baseline`` once to FREEZE the
     accepted scores into ``reports/eval_baseline.json`` (committed);
  2. every later run regenerates ``reports/eval_report.json`` (the expensive part — agent + judges),
     and THIS test asserts no gated metric fell past its tolerance band. No model calls happen
     here; we only compare two JSON files.

A 'higher' metric must stay ≥ ``baseline − tol``; a 'lower' metric (fabrications, over-refusals)
must stay ≤ ``baseline + tol``. The band absorbs 30-row sampling noise so the gate fails on a real
regression, not a coin-flip. Metrics that were unavailable at baseline time (e.g. κ undefined, a
judge dep missing) are skipped, not failed — the gate guards what it can actually measure.

    pytest tests/test_eval_gate.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "eval_report.json"
BASELINE = ROOT / "reports" / "eval_baseline.json"


def _load(p: Path) -> dict:
    if not p.exists():
        pytest.skip(f"{p.name} missing — run `python -m hcft_agent.eval.run_eval ... --set-baseline` first")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def flat() -> dict:
    return _load(REPORT)["flat"]


@pytest.fixture(scope="module")
def baseline() -> dict:
    return _load(BASELINE)


def test_baseline_has_gates(baseline):
    assert baseline, "baseline is empty — freeze it with --set-baseline"


@pytest.mark.parametrize("metric", [
    "retrieval.recall_at_pool", "retrieval.hit_at_k_exact",
    "refusal.unanswerable.refuse_rate", "refusal.answerable.answer_rate",
    "refusal.unanswerable.fabricated", "generation.ragas_faithfulness",
    "generation.hhem_mean", "validation.cohen_kappa",
])
def test_no_regression(metric, flat, baseline):
    spec = baseline.get(metric)
    if spec is None:
        pytest.skip(f"{metric} not in baseline (was unavailable when frozen)")
    cur = flat.get(metric)
    if not isinstance(cur, (int, float)):
        pytest.skip(f"{metric} unavailable in current report ({cur!r})")

    bound, direction = spec["bound"], spec["direction"]
    if direction == "higher":
        assert cur >= bound, (
            f"{metric} REGRESSED: {cur:.3f} < {bound:.3f} (baseline {spec['value']:.3f} − tol {spec['tol']})"
        )
    else:
        assert cur <= bound, (
            f"{metric} REGRESSED: {cur:.3f} > {bound:.3f} (baseline {spec['value']:.3f} + tol {spec['tol']})"
        )
