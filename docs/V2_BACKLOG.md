# V2 Backlog — deferred Agent-1 (P1 RAG chat) polish

These are **project-specific tuning/cleanup** items, parked on purpose. Agent 1 is a complete,
measured, gated vertical slice; these would refine *its* numbers but don't add a new
agent-engineering concept, so they're low-leverage for the learning/interview goal. Pick them up
in a V2 hardening pass (or never).

| # | Item | Why deferred | How to do it |
|---|------|--------------|--------------|
| 1 | **Over-refusal tuning** | The 2–3/10 over-refusals on answerable-in-context are a tuning knob, not a concept. Don't soften the contract blindly — it would regress the unanswerable refuse-rate. | Inspect the fresh over-refusal cases (`run_eval` prints them); only relax the prompt if they're true single-hop misses, then re-run the gate to confirm no unanswerable regression. |
| 2 | **Over-generic QA filter** | The frozen slice already scores fine; cleaning 7/2039 rows is hygiene. | `python scripts/filter_generic_qa.py --write`, point the eval at `qa_eval_v3_filtered.jsonl`, re-`--set-baseline`. |
| 3 | **True cross-vendor RAGAS judge** | Current judge is `gpt-oss-120b` (cross-model but OpenAI vendor lineage). A non-OpenAI judge (Gemma/Nemotron/Claude) is a stronger circularity break — but the κ deterministic anchor already carries that load. | One env var: `RAGAS_JUDGE_MODEL` / `RAGAS_JUDGE_BASE_URL` / `RAGAS_JUDGE_API_KEY`. No code change. |
| 4 | **Human-anchor κ slice** | κ currently validates G-Eval against the *deterministic* anchor (legit, non-LLM). A ~15-row human-labeled slice upgrades it to true human-κ. | Label ~15 rows `{qa_id, appropriate}` → `KAPPA_HUMAN_LABELS` path. |
| 5 | ✅ **DONE — Indirect-injection scan on retrieved chunks** | Shipped into Agent 1, not deferred: the `context_guard` node ([context_ring.py](../src/hcft_agent/guards/context_ring.py)) quarantines poisoned chunks before the reader, fail-closed if all are poisoned, with its own eval + gate ([eval_context_guard.py](../scripts/eval_context_guard.py), [test_context_guard.py](../tests/test_context_guard.py)). Reuses the input-ring classifier, so it inherits any Prompt-Guard-2 swap. | — |
| 6 | **Bigger frozen slice** | 30 rows gives wide CIs; 100+ tightens the gate. Mechanical. | Bump `--grounded/--unanswerable`, re-baseline. |
| 7 | **SLM reader swap A/B** (`raft-3b`) | Cross-project milestone — closes the loop with SLM_Fine_Tuning. Tracked separately as a feature, not V2 tuning. | Point `READER_*` at the Ollama-served fine-tune; run the eval as an A/B vs gpt-4o-mini. |
| 8 | **Context-ring detection recall (~0.70)** | Windowing fixed the 512-truncation evasion (0.63→0.70 at FP 0.017), but the residual misses are *dilution* cases where Prompt-Guard-2 itself doesn't fire on a tiny payload buried in a long benign chunk. We chose low FP on purpose (the output groundedness guard backstops a missed injection). Lifting recall needs a model/segmentation change, not tuning. | Sentence-level segmentation (score each sentence) and/or a stronger indirect-injection model; keep FP low. Measure with `eval_context_guard.py`. |

Everything here is reversible/additive — none of it blocks building Agents 2+.
