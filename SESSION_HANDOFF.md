# SESSION HANDOFF — 2026-06-13

For the next chat session. Read `PLAN.md` (rev 2026-06-13) and `DECISIONS.md` first; this file is
the *delta* since then plus the live build state and what to do next.

## TL;DR

- Project: **HCFT Analyst Agent** — a LangGraph supervisor over the HCFT corpus (519,555 chunks),
  scoped to the **Cisco Senior Technical Lead (Agentic Frameworks)** JD.
- Method is now **"AI builds, you architect"** (M0–M1 were hand-typed as proof; from M2 the AI
  authors code and the learner interrogates design / writes the `DECISIONS.md` entries).
- This session: (1) folded the JD into `PLAN.md` as 5 changes, (2) authored **M2 async
  self-corrective RAG**, (3) had a long **evaluation-metrics debate** that is NOT yet written into
  `DECISIONS.md` — that is the top pending task.

## What changed this session

### 1. `PLAN.md` re-scoped to the JD (rev 2026-06-13) — the "all 5"
1. Added a **JD coverage map** (every min/preferred qual → milestone, with ✅/⬆️/⚠️ markers).
2. **Eval + observability elevated to primary** → M5 tracing, M6 trajectory + RAGAS + regression gate.
3. **Async graphs from M2 on** (`async def` nodes, `astream`, blocking work via `asyncio.to_thread`).
4. **MCP server (M8) + Security/access-control (M7) promoted** from Phase-2 to scoped Tier-1 builds.
5. **Delivery/leadership framed**: `PLAN.md` = roadmap, `DECISIONS.md` = risk/decision log;
   Snowflake bridged from Databricks as **Tier-3 designed & cost-modeled** (not operated).
- Honesty ladder: Tier-1 = built locally, Tier-3 = designed & cost-modeled. M2–M9 all Tier-1 is
  >24h if gold-plated, so each milestone goes to "working + measured" then stops.

### 2. `DECISIONS.md` rule 1 updated to the new method.

### 3. Built `src/hcft_agent/graphs/m2_research_rag.py` — async self-corrective RAG
- Flow: `retrieve → grade_documents → (relevant? generate : rewrite_query↺) ; generate →
  grade_groundedness → (grounded? END : rewrite_query↺)`. ASCII diagram is in the module docstring.
- Five defensible decisions: node-does-LLM-call / edge-is-pure; structured-output routing
  (`with_structured_output`); two loop-breakers (`MAX_RETRIES=2` + `recursion_limit=15`); real
  async (`asyncio.to_thread` for the blocking retriever, `ainvoke` for LLMs); `question` (immutable)
  vs `query` (rewritten) split.
- Reader uses the **`READER_*` slot** (the M9 swap seam), graders use the orchestrator slot.
- Fixed two issues from the learner's stub: import is `hcft_agent.retriever` (singular); reader
  pointed at orchestrator slot → moved to reader slot.
- **Known v1 weakness (logged, not yet fixed):** a groundedness failure routes back to `retrieve`,
  which re-fetches the *same* docs. Often groundedness fails because *generation* drifted, not
  retrieval. Stronger design: groundedness-fail → constrained re-`generate`/refuse; only
  relevance-fail → re-retrieve. **Decision pending** — split the paths or keep v1 + log trade-off.
- **NOT yet run** against the live corpus.

## The evaluation debate (this session's core — MUST be written into `DECISIONS.md`)

Learner's challenge: "we're relying on an LLM for relevance — just a signal; shouldn't we measure
ROUGE-L / BLEU / RAGAS instead?" Resolution reached:

- **Two layers, don't conflate.** In-graph graders = *runtime control signals*, reference-free
  (no gold answer exists at inference time), must be cheap → LLM binary score is correct here.
  Metrics (ROUGE-L/BERTScore/RAGAS) = *offline measurement* in M6 against a gold set.
- **BLEU**: machine-translation metric (modified n-gram **precision** + brevity penalty, geometric
  mean, no recall, corpus-scale). *Applicable* to offline answer-vs-gold, but a **weak, high-variance
  proxy** for short grounded-QA answers (punishes valid paraphrase, blind to grounding). Not
  forbidden — just not the metric to decide on. Prefer **ROUGE-L** (recall-flavored, LCS) as the
  lexical signal.
- **Metric ladder:** lexical-precision (BLEU) → lexical-recall (ROUGE-L) → semantic (BERTScore) →
  grounding (RAGAS faithfulness).
- **RAGAS faithfulness** = decompose answer into atomic claims (LLM) → NLI-check each claim is
  entailed by retrieved context (LLM) → score = supported/total. Reference-free.
  **Answer-relevancy** = back-generate N questions from the answer (LLM) → embed → mean cosine
  similarity to the original question. Context-precision (rank-aware) and context-recall (needs the
  gold answer) cover the retriever side.
- **Key insight (learner's):** RAGAS faithfulness/answer-relevancy **are still LLM-as-judge**.
  RAGAS doesn't escape judge dependency — it just *conditions the judge better* (atomic binary
  calls + arithmetic formula + judged against a concrete artifact → lower variance). Real failure
  modes remain: judge variance/non-determinism, family self-preference (cf. HCFT dropping GPT-4o as
  judge), correlated domain errors.
- **What actually breaks circularity:** (a) a **human-labeled anchor** slice → report judge
  agreement (κ / P-R), the "evaluate the evaluator" move; (b) a **non-LLM reference metric**
  (ROUGE-L/BERTScore) that doesn't share the judge's failure modes.
- **Proposed M6 metric set (to lock):** RAGAS faithfulness + context-precision + context-recall;
  **ROUGE-L or BERTScore** (one non-LLM lexical/semantic signal); **trajectory / tool-call accuracy**;
  **grader-agreement vs a human-labeled slice**. Three *independent* lenses so a judge blind-spot
  can't silently pass. Plus the in-graph grader is itself evaluated (report its precision/recall).

## Current build state

- ✅ Mongo in Docker (`hcft-mongo`, `127.0.0.1:27017`); 519,555 chunks migrated + indexed.
- ✅ `retriever.py` — Qwen3 embed → Pinecone `hcft` → BGE-v2-m3 rerank → Mongo hydrate (lazy-loaded).
- ✅ `config.py` — orchestrator slot + reader slot (both `gpt-4o-mini` now), embed/rerank params.
- ✅ M0 (`scratch/m0_hello.py`, `scratch/m0_fanout.py`), M1 (`graphs/m1_react.py`) — done + logged.
- 🟡 **M2** (`graphs/m2_research_rag.py`) — authored, lint-clean, **not run yet**.
- Models: OpenAI `gpt-4o-mini` everywhere (Fireworks 404'd earlier; `OPENAI_API_KEY` in `.env`).
- Interpreter: agent repo has its own `.venv`; run as `.\.venv\Scripts\python.exe -m hcft_agent...`.
  Shell is PowerShell (`;` not `&&`). Keep `numpy<2`.
- `scratch/` is gitignored (learning artifacts never pushed; repo = project artifacts only).

## Doc drift to fix next session (don't skip)

- `PLAN.md` renumbered milestones to **M2…M9** (M7=security, M8=MCP, M9=reader swap), but
  `DECISIONS.md` still has the **old** numbering (M5=streaming, M6=eval, M7=reader swap, no
  security/MCP sections). **Action:** re-number `DECISIONS.md` headers to match `PLAN.md` and add
  empty M7 (security/AST sandbox) and M8 (MCP) sections.

## Next steps (in order)

1. **Write the M2 evaluation decision into `DECISIONS.md`** (the debate above) and **lock the M6
   metric set**. ← highest priority; it's the JD's core and it's only in this handoff right now.
2. **Decide the M2 groundedness-loop fix** (split groundedness-fail → re-generate/refuse vs keep v1).
3. **Run M2** against the live corpus; do the predict-before-run log; fill the M2 `DECISIONS.md`
   experiment (LLM grader vs cheap heuristic: tokens/latency vs refusal accuracy).
4. **Fix the `DECISIONS.md` ↔ `PLAN.md` numbering drift.**
5. Proceed to **M3** (persistence Sqlite→Mongo + `interrupt()` HITL + time travel).
6. Commit M2 (one commit, message = the design decision, per working rule 4).

## Guardrails (carry forward)

- Method: AI authors from M2; learner writes every `DECISIONS.md` entry + does predict-before-run.
- The fine-tune (`raft-3b-r64-v2_2`) is the **reader only**, never the orchestrator.
- vLLM / Snowflake = designed & cost-modeled, not operated. No `HCFT_Medium` content in this repo.
