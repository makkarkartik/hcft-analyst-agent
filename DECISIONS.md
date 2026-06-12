# DECISIONS.md — design decisions & trade-offs (filled per milestone)

Working rules for this project (retention > speed):

1. All LangGraph-specific code is hand-written; AI assistance is reviewer-only for graph code.
2. Predict-before-run: before the first execution of any graph, write down expected node
   order and state after each super-step; record where the prediction was wrong.
3. Each milestone closes with its entry here, written immediately — interview-answer first,
   trade-off second.
4. One commit per milestone; the message states the design decision, not the feature.

---

## M0 — Hello graph (state, reducers, conditional edge)

- **Interview answer to own:** what a reducer is; why two parallel nodes writing one key
  without a reducer is an error; the Pregel/super-step execution model.
- **Prediction log:**
- **Trade-off hit:**

## M1 — ReAct loop from scratch vs `create_react_agent`

- **Interview answer to own:** what the prebuilt compiles to; when to take the abstraction.
- **Experiment:** same task both ways — LoC, debuggability, where the prebuilt's defaults bite.
- **Decision:**

## M2 — Self-corrective RAG

- **Interview answer to own:** cycles + conditional edges + `recursion_limit`; structured
  output as a routing mechanism.
- **Experiment:** LLM grader vs cheap heuristic — tokens/latency per query vs refusal accuracy.
- **Decision:**

## M3 — Persistence, HITL, time travel

- **Interview answer to own:** checkpoint = state per super-step per thread; interrupt vs
  static breakpoint; how `Command(resume=...)` re-enters the graph; `update_state` forking.
- **Experiment:** checkpointer overhead (off vs SqliteSaver vs MongoDBSaver, per-turn latency).
- **Decision:**

## M4 — Supervisor multi-agent + Send map-reduce

- **Interview answer to own:** `Send` vs `Command`; subgraph state-key mapping; supervisor vs
  swarm trade-offs.
- **Experiment:** supervisor vs single tool-calling agent on the same 20 questions —
  quality / latency / token cost. (When is multi-agent worth it?)
- **Decision:**

## M5 — Streaming + observability

- **Interview answer to own:** what each `stream_mode` (`values`/`updates`/`messages`/`custom`)
  emits and when to use which; `astream_events`.
- **Decision:**

## M6 — Eval + hardening

- **Interview answer to own:** `RetryPolicy` semantics; failure routing on tool errors.
- **Headline result:** agentic RAG vs plain RAG (groundedness, refusal acc, latency, cost).
- **Decision:**

## M7 — Reader swap: raft-3b-r64-v2_2 via Ollama

- **Interview answer to own:** why the fine-tune is the *reader*, not the orchestrator
  (no tool-calling training); rsLoRA merge gotcha (see README lineage note — α/√r, 8× scale
  difference vs naive α/r merge); train/serve prompt-template parity.
- **Headline result:** frontier reader vs raft-3b reader, same slot, same eval set.
- **Decision:**
