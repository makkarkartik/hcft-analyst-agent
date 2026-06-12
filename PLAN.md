# PLAN.md — locked scope (2026-06-12)

Primary goal: **LangGraph interview readiness through hands-on implementation.**
Secondary goal: a public portfolio piece connecting to the HCFT fine-tuning project.
Budget: **24 hours**. Changes to this plan require a deliberate decision logged in `DECISIONS.md`.

## Locked decisions

1. **What we're building:** the HCFT Analyst Agent — a LangGraph supervisor routing between
   (a) a self-corrective RAG research agent, (b) a MongoDB analytics agent, and (c) a
   map-reduce synthesis agent, over the existing HCFT corpus (519,555 chunks).
2. **Working method:** all LangGraph code hand-written (AI = reviewer only); predict-before-run;
   factsheet entry per milestone written immediately; one commit per milestone stating the
   design decision. (Full rules at the top of `DECISIONS.md`.)
3. **Stack:** Pinecone `hcft` (reused, unchanged) + BGE-v2-m3 rerank + MongoDB in Docker
   (text hydration, analytics, LangGraph checkpointer). DONE: migration + indexes.
4. **Models:** tool-capable public model (Fireworks / gpt-4o-mini) as orchestrator + graders
   for all milestones. The reader/generation slot is OpenAI-compatible and swappable.
5. **Reader swap (M7):** `raft-3b-r64-v2_2` merged with PEFT `merge_and_unload()` (rsLoRA-aware
   — see README lineage note) → GGUF → Ollama. The fine-tune is NEVER the orchestrator.
   vLLM stays architected-not-operated. Timebox 3h; fallback = transformers behind FastAPI shim.
6. **Headline deliverable:** same 20-question eval, frontier reader vs raft-3b reader in the
   same slot — groundedness, refusal accuracy, latency, cost/query. Supporting experiments:
   agentic vs plain RAG (M2/M6), supervisor vs single agent (M4), checkpointer overhead (M3).
7. **Public repo rules:** code public; reproducible quickstart = FAISS + bundled subset
   (strangers can't reach Pinecone/local Ollama); no HCFT_Medium content ever.

## Schedule (24h total, ~3h buffer)

| # | Milestone | Budget | Core interview topics |
|---|---|---|---|
| M0 | Hello graph, built cold as self-test | 0.75h | state, reducers, super-steps |
| M1 | ReAct loop from scratch + retriever tool | 3h | tool loop, prebuilt vs hand-rolled |
| M2 | Self-corrective RAG subgraph | 3h | cycles, structured-output routing, recursion_limit |
| M3 | Checkpointing (Sqlite→Mongo), interrupt(), time travel | 3h | durable execution, HITL, forking |
| M4 | Supervisor + Send map-reduce + comparison experiment | 4h | Send vs Command, subgraph state mapping |
| M5 | Streaming + tracing | 2.5h | stream modes, astream_events |
| M6 | Eval + retries/hardening | 2.5h | RetryPolicy, headline numbers |
| M7 | Reader swap + comparison eval | 3h | serving boundary, rsLoRA merge, template parity |
| — | Factsheet rehearsal + README/diagram polish | 2h | everything, out loud |

## Out of scope (do not gold-plate)

- New retriever/dataset work (BM25/RRF hybrid stays Phase-2 in the HCFT repo).
- Polished UI — console streaming or bare Gradio only.
- vLLM, MCP server, Airflow, long-term memory store (stretch only if under budget after M7).
- Any further fine-tuning. Model weaknesses get system-level fixes (grading/refusal gates).

## Definition of done

1. All milestone checkboxes in README ticked, with the comparison table filled with real numbers.
2. `DECISIONS.md` complete — every milestone has its interview answer + trade-off entry.
3. You can answer the M0–M7 interview topics out loud, pointing at code you wrote.
