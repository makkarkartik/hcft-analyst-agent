# reference/ — study artifacts only

Earlier hand-built LangGraph milestones, kept for reference and *not* part of the current build:

- `m1_react.py` — ReAct loop from scratch
- `m2_research_rag.py` — self-corrective RAG (grade → rewrite → generate), async
- `m3_hitl_rag.py` — persistence (Mongo checkpointer) + HITL `interrupt()` + time travel

The fresh, **library-first** system lives under `src/hcft_agent/` (`agents/`, `obs/`, `eval/`,
`guards/`). Rationale and plan: see [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) and
[`docs/SESSION_LOG.md`](../docs/SESSION_LOG.md).
