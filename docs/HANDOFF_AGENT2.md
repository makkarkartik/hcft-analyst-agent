# Handoff → Agent 2 (tool-using analyst)

**Read this first in a fresh session.** It is deliberately *principles-first*: it anchors on the
reusable judgment and substrate, not the play-by-play of how Agent 1 was built — so it primes good
Agent-2 *design*, not Agent-1 *mimicry*. The full build record is in [`SESSION_LOG.md`](SESSION_LOG.md);
the system is pictured in [`agent_anatomy.html`](agent_anatomy.html); the deep dive is
[`RAG_AGENT_TUTORIAL.md`](RAG_AGENT_TUTORIAL.md).

---

## 1. Where Agent 1 leaves you

A complete, measured, gated **RAG chat agent** (P1): LangGraph state machine, hybrid retrieval, three
guardrail rings, an offline LLM-judge eval stack, a baselined pytest gate, and LangSmith
Experiment-over-Dataset tracking. It covers **3 of the 4 eval objects** (component / outcome /
operational-safety) and **3 of the 4 guardrail rings** (input / context / output). The two gaps —
**trajectory** eval and the **action ring** — are exactly what Agent 2 exists to fill.

*(Headline numbers + the LangSmith experiment link are in §6, filled from the frozen-slice run.)*

## 2. Locked conventions — the judgment to carry forward (do NOT re-litigate)

These are decisions already made and defended. Apply them to Agent 2 by default.

1. **Prefer industry tools; own the *decision*, adopt the *implementation*.** No hand-rolled
   equivalents when a standard exists (LangGraph, Presidio, Prompt-Guard-2, RAGAS, DeepEval, HHEM,
   sklearn). What you own is the *design*: topology, which metric binds to which step, thresholds.
   **Graceful-degradation fallbacks are allowed** (regex if a model can't load) — resilience, not a
   second implementation; enforcement stays fail-closed.
2. **No guard without an offline eval twin and a baselined gate.** Every inline check earns a
   measurement (e.g. HHEM gate ⇄ RAGAS faithfulness; context ring ⇄ detection-recall/FP probe). If
   you add an action-ring guard, you add its eval + gate in the same change.
3. **Eval trust order:** deterministic anchors first (no circularity), LLM judges on top, **Cohen's
   κ validates the judge against a non-LLM anchor**. Attribute every outcome to the stage that owns
   it (the refusal-by-retrieval decomposition is the canonical example — a blended metric hides
   everything).
4. **The three-axis mental model** (use it to place every new check): **🔒 security guardrail**
   (vs attacker · fail-closed) · **↩ quality gate** (vs model error · degrade→refuse) · **📊 offline
   twin** (dataset · measure-only). Observability **fails open** (never blocks).
5. **Tracing:** native LangSmith, **span per node**, attach domain verdicts via `tag()` →
   `hcft.*`. (OTel was tried and reverted — thread-local context broke across LangGraph worker
   threads.)
6. **Eval is linked to runs**, not a separate metrics silo: a LangSmith `evaluate()` Experiment over
   a stable **Dataset**, so scores attach to traces and versions compare example-by-example.
7. **Measure before you fix.** (E.g. the HHEM "512 truncation" non-bug; the context-ring truncation
   that the probe *did* catch and we fixed with windowing.)
8. **The user architects; you implement.** Surface decisions, recommend, then build. Review files
   before running; don't relocate/delete without sign-off.

## 3. Reusable substrate (import, don't rebuild)

| What | Where | Reuse for Agent 2 |
|---|---|---|
| Data layer | Pinecone `hcft` (519,555 vectors) + Mongo `hcft.chunks` (519,555 docs, BM25 text index) | the corpus + the `metric_lookup` tool's backing store |
| Settings | `config.py` — single source, env-overridable | add Agent-2 knobs here, nowhere else |
| Report + gate | `eval/report.py` (stage×substage schema + baseline), `tests/test_*` (tolerance-band pytest gate) | **extend `report.py` with a trajectory stage**; add a trajectory gate |
| Judges | `eval/judges.py` (RAGAS + G-Eval builders + per-row scorers), `eval/validate.py` (κ) | reuse verbatim for outcome scoring; add a trajectory judge if needed |
| LangSmith path | `eval/experiment.py` (Dataset + evaluators + summary), `eval/run_eval.py` (offline twin) | clone the pattern; evaluators become trajectory-aware |
| Telemetry | `obs/telemetry.py` (`init_telemetry`, `trace_block`, `tag`) | span-per-node + tool spans |
| Guards | `guards/` — `input_ring`, `context_ring`, `groundedness` (the ring pattern) | **the action ring is new code** but mirrors this structure |
| Dashboard | `scripts/build_eval_dashboard.py` → `docs/eval_scores.html` | regenerates as-is once trajectory metrics are in the report |

## 4. Environment gotchas (Windows) — will save you an hour

- **Console encoding:** set `$env:PYTHONUTF8 = "1"` before any eval script — the agent/eval print
  unicode (κ, →, ·) and Windows cp1252 will crash on it.
- **MongoDB runs in Docker** (`docker compose up -d`, container `hcft-mongo` :27017) — needs **Docker
  Desktop running first**. The 519k chunks live in a persistent volume.
- **PNG render of the HTML diagrams:** Windows **Controlled Folder Access blocks `chrome.exe`** from
  writing into the Projects folder. Render to `$HOME/_hcft_png/…` with headless Chrome, then `cp` in.
- **Judge models:** `gpt-4o-mini` (OpenAI — reader + G-Eval), `gpt-oss-120b` via **Fireworks** (the
  only non-Chinese serverless LLM the key reaches — cross-model RAGAS judge), `Prompt-Guard-2-86M`
  (gated; license accepted, `HF_TOKEN` in `.env`). The Fireworks key has **no Llama/Gemma/Nemotron**
  serverless access — don't assume those slugs.
- **venv:** `.venv` is pinned (`torch 2.5.1+cu124`, `transformers 4.54.1`, `numpy<2`). RAGAS needed a
  2-line `sys.modules` shim for its dead `langchain_community.chat_models.vertexai` import on
  langchain 1.x (in `judges.build_ragas`) — leave it.
- **Secrets:** `.env`, `reports/`, `.deepeval/`, `docs/eval_scores.html` are gitignored. Keep it that
  way. Baselines (`reports/*baseline.json`) are local; re-freeze with `--set-baseline`.

## 5. Open Agent-1 items (parked, not blocking)

See [`V2_BACKLOG.md`](V2_BACKLOG.md). Highlights: over-refusal tuning, the over-generic QA filter
(`--write` + re-baseline), a true cross-*vendor* RAGAS judge, a human-anchor κ slice, and lifting
context-ring detection recall (~0.70 — dilution cases need segmentation / a stronger model; FP is
already low and the output guard backstops misses).

## 6. Agent-1 results (frozen slice — filled from the run)

> _TO FILL after the LangSmith experiment lands: retrieval recall@50 / hit@5 (dense vs hybrid),
> refusal-by-regime rates, RAGAS faithfulness/relevancy, G-Eval, κ, HHEM mean; the LangSmith
> experiment URL; the context-guard detection-recall/FP. These also live in the scoreboard
> (`build_eval_dashboard.py`) and the frozen baselines under `reports/`._

---

## 7. Agent 2 — the brief

**Goal:** a **tool-using analyst** that calls RAG as *one* tool among several, so it finally
exercises the two uncovered cells: **trajectory** evaluation and the **action ring**.

**Capabilities (candidate tools — confirm scope with the user before building):**
- `rag_search` — Agent 1, wrapped as a tool.
- `metric_lookup` — exact figure for a (hospital/tract, metric, year) from Mongo metadata.
- `compare` / `aggregate` — across hospitals / states / years (the thing pure RAG can't do).

**What's new (and must each ship with an eval + gate, per convention 2):**
- **Trajectory eval** (the new object): tool-selection accuracy, step efficiency (no needless
  loops), trajectory-match vs a gold tool sequence. Add a `Trajectory` stage to `report.py`.
- **Action ring** (the new guardrail): **tool allowlist**, **schema-validated arguments** (reject
  malformed/over-broad calls), **sandboxed execution**, optional **HITL** gate for sensitive tools.
  Mirror the `guards/` ring pattern; add its detection/precision eval + gate.

**Build order (same discipline as Agent 1):** agent → guardrails → eval harness → gate → diagram.
Then regenerate the anatomy picture — the `trajectory ✗ N/A` and `action ✗ N/A` cells in the
scorecard go live.

**Reuse checklist:** `report.py` (+ trajectory stage) · `judges.py`/`validate.py` (outcome scoring) ·
`experiment.py` (Dataset + evaluators) · `telemetry.py` (span per tool call) · the pytest gate
pattern · the dashboard generator.

## 8. First moves in the fresh session
1. Read this doc + glance at `agent_anatomy.html`.
2. Bring up the stack (Docker → `docker compose up -d`; `$env:PYTHONUTF8="1"`).
3. Ask the user to confirm Agent-2 tool scope (the list in §7) before writing code.
4. Build the tool-using graph; then its action-ring guards; then the trajectory eval + gate.
