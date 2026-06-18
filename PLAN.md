# PLAN.md — locked scope (rev 2026-06-13, JD-aligned)

**Target role:** Senior Technical Lead, Autonomous Agentic Frameworks & System Integration
(Cisco CX). A **build-and-lead** role: design/build/deploy agents *and* own roadmap, risk, and
delivery. This plan is scoped to that JD.

Primary goal: **interview readiness for a senior agentic-systems lead** — defensible judgment
backed by a real, measured build. Secondary goal: a public portfolio piece connecting to the
HCFT fine-tuning project. Budget: **~24h hands-on**; anything beyond is **designed & cost-modeled**
(honesty ladder), not faked. Changes here require a decision logged in `DECISIONS.md`.

## Locked decisions

1. **What we're building:** the HCFT Analyst Agent — a LangGraph supervisor routing between
   (a) a self-corrective RAG research agent, (b) a MongoDB analytics agent, and (c) a
   map-reduce synthesis agent, over the existing HCFT corpus (519,555 chunks).
2. **Working method (rev 2026-06-13 — "AI builds, you architect"):** M0–M1 were hand-typed and
   stand as proof-of-hands. From M2 on, the AI authors implementation and we spend the reclaimed
   time on **architect/lead work**: trade-off interrogation, failure-mode analysis, alternative
   architectures, eval/observability/security rigor, and system-design reps. Retention rituals
   kept: predict-before-run, **learner writes every `DECISIONS.md` entry**, per-file Q&A. One
   commit per milestone. Rationale: senior interviews test judgment, not syntax Cursor writes.
3. **Stack:** Pinecone `hcft` (reused) + BGE-v2-m3 rerank + MongoDB in Docker (text hydration,
   analytics, LangGraph checkpointer). Graphs are **async** (`async def` nodes, `astream`). DONE:
   Mongo migration + indexes; real retriever (Qwen3 embed → Pinecone → rerank → Mongo hydrate).
4. **Models:** public tool-capable model (OpenAI `gpt-4o-mini`) as orchestrator + graders. The
   reader/generation slot is OpenAI-compatible and swappable (frontier now → `raft-3b` in M9).
5. **Reader swap:** `raft-3b-r64-v2_2` merged with PEFT `merge_and_unload()` (rsLoRA-aware) →
   GGUF → Ollama. The fine-tune is NEVER the orchestrator. vLLM stays designed-not-operated.
6. **Headline deliverable:** frontier reader vs `raft-3b` reader in the same slot — groundedness,
   refusal accuracy, latency, cost/query — plus the evals-driven-development story (the JD's core).
7. **Public repo rules:** code public; reproducible quickstart = FAISS + bundled subset; no
   HCFT_Medium content ever.

## JD coverage map (living checklist)

> Source of truth = `JD.md` (verbatim posting + full reconciliation table). Summary below.

Minimum quals:
- ✅ Design/develop/deploy agents & orchestration patterns → M2–M4.
- ⬆️ **Evals-driven development + observability for non-deterministic systems** → M5 + M6 (JD core; elevated to primary).
- ✅ Python; LangChain/LangGraph → throughout.
- ⬆️ **Asynchronous programming (multi-step agents)** → async graphs from M2.
- ✅ **MCP / tool-integration for enterprise** → M8 (built basic).
- ⚠️ **Cloud data platform (Snowflake)** → bridge from Databricks (SLM stage-02); Snowflake + Cortex **designed & cost-modeled**, not operated. *(Real gap — transferable, not faked.)*
- ⚠️ Jira/GitHub → GitHub used; **Jira not used** — concede + frame Agile/SDLC/backlog.

Preferred:
- ✅ Frontier **and** open-source LLM ecosystems → M9 frontier-vs-`raft-3b`.
- ✅ Understand LLMs / **train & focus on specific areas** → SLM RAFT fine-tune project (strong evidence).
- ✅ **LLM optimization: prompting, fine-tuning, context mgmt, reduce hallucinations** (JD "Your Impact") → SLM RAFT + M2 self-corrective/groundedness RAG.
- ✅ NLP & prompt engineering → per-node system prompts (M2+).
- ⬆️ **Enterprise security & agentic access control** → M7 (tool permission scoping, AST allowlist + sandboxed test-before-freeze, HITL approval).
- ✅ AI dev tools (Claude Code/Codex/**Cortex**) → Cursor used to build; Cortex in Tier-3 design.
- ✅ Leadership / deliverables / risk to stakeholders → `PLAN.md` = roadmap; `DECISIONS.md` = risk/decision log; stakeholder summary (to produce).

Verbal-only (don't build): JD names **AutoGen / CrewAI** beside LangGraph — be ready to contrast
when-to-pick-which rather than building all three.

## Schedule (Tier-1 = built locally; Tier-3 = designed & cost-modeled)

| # | Milestone | Tier | JD hook |
|---|---|---|---|
| M0 | Hello graph (hand-built) ✅ | 1 | state/reducers/super-steps |
| M1 | ReAct loop (hand-built) ✅ | 1 | tool loop, prebuilt vs hand-rolled |
| M2 | Self-corrective RAG, **async** | 1 | cycles, structured-output routing, hallucination control |
| M3 | Persistence (Sqlite→Mongo) + HITL `interrupt()` + time travel | 1 | durable execution, human-in-the-loop |
| M4 | Supervisor + specialists + `Send` map-reduce (+ swarm variant) | 1 | multi-agent orchestration |
| M5 | Streaming + **observability/tracing** | 1 | observability for non-determinism |
| M6 | **Evals-driven dev:** trajectory + outcome + RAGAS faithfulness + regression gate | 1 | evals (JD core) |
| M7 | **Security & access control:** tool permission scoping, AST allowlist + sandboxed test-before-freeze, HITL approval | 1 | enterprise security/access |
| M8 | **MCP server** exposing the tools | 1 (basic) | MCP / enterprise tool integration |
| M9 | Reader swap (`raft-3b` via Ollama) + comparison eval | 1 (fallback designed) | frontier + OSS ecosystems |
| — | Snowflake + Cortex serving path | 3 | cloud data platform (bridge from Databricks) |
| — | Leadership artifacts (roadmap, risk log, stakeholder summary) + factsheet + arch diagram | — | lead/delivery |

> Time risk: M2–M9 all Tier-1 is more than 24h if everything is gold-plated. Rule: get each to a
> **working, measured** state, then stop; push depth into `DECISIONS.md` rather than more code.

## Out of scope (do not gold-plate)

- New retriever/dataset work (BM25/RRF hybrid stays Phase-2 in the HCFT repo).
- Polished UI — console/SSE streaming or bare Gradio only.
- Airflow, long-term cross-thread memory store (stretch only).
- Any further fine-tuning. Model weaknesses get system-level fixes (grading/refusal gates).

## Definition of done

1. M2–M9 each reach a working, measured state; README checkboxes ticked; comparison table filled.
2. `DECISIONS.md` complete — every milestone has interview-answer + trade-off + failure-mode notes.
3. You can whiteboard the architecture and answer the JD's judgment topics (evals, observability,
   multi-agent cost, security, async, frontier-vs-OSS) out loud, pointing at code you shipped.
