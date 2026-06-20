"""Quality gate for the context ring (indirect-injection scan).

Same two-step contract as the main eval gate: `eval_context_guard.py --set-baseline` freezes the
accepted detection/FP, then this test (no model calls) asserts a later run didn't regress past the
tolerance band. Detection must stay ≥ baseline−tol; false-positive rate ≤ baseline+tol.

    python scripts/eval_context_guard.py --n 60          # produce reports/context_guard_eval.json
    pytest tests/test_context_guard.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "reports" / "context_guard_eval.json"
BASELINE = ROOT / "reports" / "context_guard_baseline.json"


def _load(p: Path) -> dict:
    if not p.exists():
        pytest.skip(f"{p.name} missing — run `python scripts/eval_context_guard.py --set-baseline` first")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def result() -> dict:
    return _load(EVAL)


@pytest.fixture(scope="module")
def baseline() -> dict:
    return _load(BASELINE)


@pytest.mark.parametrize("metric", ["detection_recall", "false_positive_rate"])
def test_no_regression(metric, result, baseline):
    spec = baseline.get(metric)
    if spec is None:
        pytest.skip(f"{metric} not in baseline")
    cur = result.get(metric)
    if not isinstance(cur, (int, float)):
        pytest.skip(f"{metric} unavailable ({cur!r})")
    bound, direction = spec["bound"], spec["direction"]
    if direction == "higher":
        assert cur >= bound, f"{metric} REGRESSED: {cur} < {bound} (baseline {spec['value']})"
    else:
        assert cur <= bound, f"{metric} REGRESSED: {cur} > {bound} (baseline {spec['value']})"
