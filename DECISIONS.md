# DECISIONS.md — design decisions & trade-offs (filled per milestone)

Working rules for this project (retention > speed):

1. **Method (rev 2026-06-13 — "AI builds, you architect"):** M0–M1 hand-typed (proof-of-hands).
   From M2 on the AI authors implementation; the learner spends the time on architect/lead work —
   trade-off interrogation, failure-mode analysis, alternative architectures, eval/observability/
   security rigor. Retention kept via predict-before-run, learner-written entries, per-file Q&A.
   Rationale: the target role (Senior Technical Lead) tests judgment, not syntax.
2. Predict-before-run: before the first execution of any graph, write down expected node
   order and state after each super-step; record where the prediction was wrong.
3. Each milestone closes with its entry here, written immediately — interview-answer first,
   trade-off second.
4. One commit per milestone; the message states the design decision, not the feature.
5. **Expanded scope (2026-06-13):** beyond the base patterns, explicitly cover — trajectory /
   tool-call eval (not just outcome eval); orchestrator+specialist and swarm/handoff variants;
   per-node system prompts with per-stage evaluation; and dynamically-generated tools gated by
   an AST allowlist + sandboxed test-before-freeze. Built where cheap/high-signal; studied at
   the factsheet level otherwise (honesty ladder).

---

## Scaffold — FastAPI deferred to a post-M5 thin wrapper (2026-06-12)

- **Decision:** no service layer before the graphs exist. Instead, `src/hcft_agent/` is a
  proper package from day one: graphs are `build_*_graph(...)` factories (checkpointer/models
  injected, no module-level state), config centralized in `config.py`. A FastAPI wrapper
  (`/threads/{id}/messages` → `graph.astream()` over SSE) is added after M5 when streaming +
  checkpointing exist — wrapping a finished graph is ~1h; designing endpoints before the
  graphs means refactoring after every milestone.
- **Trade-off:** delays the "it's a real service" demo in exchange for spending the 24h budget
  on LangGraph concepts (the interview goal); FastAPI skill already evidenced elsewhere.

## M0 — Hello graph (state, reducers, conditional edge)

- **Interview answer to own:** what a reducer is; why two parallel nodes writing one key
  without a reducer is an error; the Pregel/super-step execution model.

  In LangGraph, a single state key cannot be updated by two different nodes in the same step —
  the system throws `InvalidUpdateError` (`INVALID_CONCURRENT_GRAPH_UPDATE`). A **reducer** is a
  function `(existing_value, new_value) -> merged_value` that defines how concurrent (or
  sequential) writes to a key are combined. The **default** channel is `LastValue`: it holds one
  value and fails loudly if more than one write arrives in a step. To allow multiple writers,
  you opt in with an annotated key, e.g. `messages: Annotated[list, add_messages]` — `add_messages`
  is the reducer most chat agents use, appending new messages and updating existing ones by id.

  The two nodes wrote in parallel because both are reachable from `START`, so they land in the
  **same super-step**. LangGraph executes a super-step by running all eligible nodes, collecting
  their writes, and applying them together at the tick boundary — so two writes to one
  `LastValue` key collide.

- **Prediction log:** Predicted that once the reducer was added, both notes would be appended to
  the `notes` list. Confirmed: `{'notes': ['from researcher', 'from analyst']}`.

- **Trade-off hit:** `LastValue` (fail loud on concurrent writes) vs a reducer that combines them.
  Failing loud is the right default: silently picking a winner would hide nondeterministic bugs.
  Best of all, where possible, design the graph so a key has a single writer and the question
  doesn't arise.

## M1 — ReAct loop from scratch vs `create_react_agent`

- **Interview answer to own:** what the prebuilt compiles to; when to take the abstraction.

  `create_react_agent` compiles to an `agent` node + a `ToolNode` + a `tools_condition`
  conditional edge + a loop-back edge from `tools` to `agent` — i.e. exactly the graph I
  hand-built.

  `llm.bind_tools([...])` makes the model tool-aware by converting each tool's name, docstring,
  and parameters into a JSON tool-schema sent with the request, so the LLM can return a
  structured `tool_calls` request instead of plain text. Binding lets the LLM **ask**; the
  `ToolNode` is what actually **runs** the tool (the request/execute split).

  The `agent` node fires **twice**: `START → agent`, and the first call returns an `AIMessage`
  with `tool_calls`, so `tools_condition` (which inspects the **last message**) routes to
  `tools`; the `ToolNode` runs the query, appends a `ToolMessage`, and the edge loops back to
  `agent`. On the **second** call the tool result is already in `messages`, so the LLM returns an
  `AIMessage` with **no** `tool_calls` → `tools_condition` routes to `END`. The loop is
  model-controlled: it stops when the model stops requesting tools (could be 1 call for a
  no-tool question, or N for a multi-step one).

- **Experiment:** same task hand-rolled vs prebuilt — identical Human→AI(request)→Tool→AI(answer)
  loop. Hand-rolled ≈ 15 lines and every node is visible/editable; prebuilt = 1 line but the loop
  is hidden.

- **Decision:** hand-rolled gives fine-grained control (needed from M2 on, where we insert
  grading/rewrite nodes into the loop); the prebuilt gives simplicity but hides the cycle — take
  it only when the standard ReAct loop needs no customization.

## M2 — Self-corrective RAG

### Graph mechanics

- **Interview answer to own:** cycles + conditional edges + `recursion_limit`; structured
  output as a routing mechanism; async nodes; immutable-question / mutable-query split;
  node-does-LLM-call / edge-is-pure discipline.

  **Cycles and loop-breakers.** A conditional edge in LangGraph can return the name of any
  node — including one that was already visited. That is a cycle. LangGraph prevents infinite
  loops via `recursion_limit` (default 25 super-steps). We also add a `MAX_RETRIES = 2`
  counter in state so the graph self-terminates at the *business logic* level before the
  framework kills it. Two independent breakers because the framework limit is a hard stop
  with no graceful answer; the counter-based stop lets us return a "could not verify" answer
  instead of throwing.

  **Structured output as a routing signal.** Rather than parsing free-text like "yes" or
  "no" from the grader's response, we call `llm.with_structured_output(GradeDocuments)`,
  which forces the model to emit a JSON-validated Pydantic object with a `relevant: bool`
  field. The conditional edge reads `state["relevant"]` directly — zero string parsing,
  zero hallucination risk in the control signal. Same pattern for `GradeGroundedness`.

  **Async throughout.** All nodes are `async def`. The retriever is synchronous (blocking
  network calls to Pinecone + Mongo), so it runs in `asyncio.to_thread(retriever.retrieve,
  ...)` to avoid blocking the event loop. LLM calls use `.ainvoke()`. This keeps the graph
  composable into a larger async supervisor later (M4).

  **Immutable question / mutable query split.** `question` holds the original user question
  and is never touched after the first node. `query` is what actually goes to the retriever
  and is rewritten on each retry. This distinction matters for evaluation (always score
  against the original intent) and for debugging (you can see what the graph tried).

  **Node-does-LLM-call / edge-is-pure discipline.** Every side-effectful operation
  (embedding, LLM inference, Mongo lookup) lives inside a node. Conditional edges are pure
  functions over the state dict — no I/O, no randomness. This makes the routing logic
  trivially testable and keeps the graph's execution model predictable.

### Eval debate: two layers — in-graph graders vs offline metrics

  *(Resolution of the session debate — the JD's "evals-driven development" core.)*

  **Challenge raised:** the in-graph relevance and groundedness graders call an LLM — aren't
  they just "vibes"? Shouldn't we measure ROUGE-L / BLEU / RAGAS instead?

  **Resolution: don't conflate the two layers.**

  *Layer 1 — runtime control signals (in-graph graders):* at inference time there is no gold
  answer. The grader must decide in real time whether to accept or retry. This must be cheap,
  reference-free, and binary. An LLM binary score is the correct tool here — not a corpus
  metric, which requires a gold set.

  *Layer 2 — offline measurement (M6):* after a run, over a held-out set with gold answers,
  we compute proper metrics to evaluate the system's quality. This is where ROUGE-L, BERTScore,
  and RAGAS live.

  Conflating the two leads to either: (a) trying to run RAGAS at inference time (too slow,
  requires a gold answer that doesn't exist), or (b) never measuring offline quality at all
  because "the in-graph grader already checks."

### Metric ladder (what to use for offline measurement, in order of what they capture)

  **BLEU** — modified n-gram *precision* + brevity penalty, geometric mean across n-gram
  orders, computed at corpus scale (not per-sentence). Designed for machine translation.
  *Applicable* to answer-vs-gold offline, but a weak, high-variance proxy for short
  grounded-QA: punishes valid paraphrase, blind to grounding, no recall component.
  Do not use as the decision metric.

  **ROUGE-L** — longest-common-subsequence recall between generated answer and gold answer.
  Recall-flavored, tolerates word reordering, better proxy for "did the model say what the
  gold answer said." Preferred lexical signal for this task.

  **BERTScore** — embeds both strings and computes token-level cosine similarity. Captures
  semantic equivalence (paraphrase-tolerant) where ROUGE-L is purely lexical. Use alongside
  ROUGE-L to get two independent signals.

  **RAGAS faithfulness** — decompose the generated answer into atomic claims (LLM), then NLI-
  check whether each claim is entailed by the retrieved context (LLM), score = supported /
  total claims. Reference-free (no gold answer needed). Directly measures grounding —
  the property we care most about. Also available: *answer-relevancy* (back-generate N
  questions from the answer, embed them, cosine-sim to original question — measures whether
  the answer actually addresses the question); *context-precision* (rank-aware, does relevant
  context appear early?); *context-recall* (needs gold answer, measures retriever coverage).

### Key insight: RAGAS does not escape LLM-judge dependency

  RAGAS faithfulness and answer-relevancy are still LLM-as-judge. What RAGAS buys is *lower
  variance* — atomic binary calls, deterministic arithmetic formula, judge evaluated against
  a concrete artifact (the retrieved context) rather than asked for holistic impressions.
  But the failure modes remain: judge non-determinism, domain-correlated errors, and
  family self-preference (this is why the HCFT project dropped GPT-4o as judge — OpenAI-
  family self-preference of ~+3.3 pts on GPT-generated answers).

  **What actually breaks circularity:**
  (a) a *human-labeled anchor* slice — report judge agreement (κ / precision-recall) against
      it, i.e. "evaluate the evaluator." This bounds how much you can trust the judge's
      signal on the unlabeled set.
  (b) a *non-LLM reference metric* (ROUGE-L, BERTScore) that does not share the judge's
      failure modes. If ROUGE-L says 0.55 and RAGAS faithfulness says 0.9, something is wrong.

### Proposed M6 metric set (to lock when M6 is built)

  Three independent lenses so a single judge blind-spot cannot silently pass:
  1. **RAGAS faithfulness + context-precision + context-recall** — grounding and retriever
     quality, LLM-judged but conditioned on concrete artifacts.
  2. **ROUGE-L or BERTScore** — one non-LLM lexical/semantic signal to break circularity.
  3. **Trajectory / tool-call accuracy** — did the graph take the right path? Did
     `grade_documents` fire before `generate`? Were retries bounded? Evaluates the
     orchestration layer independently of answer quality.
  4. **Grader agreement vs human-labeled slice** — evaluate the in-graph grader itself:
     precision, recall, κ vs human labels on ~50 examples. Reports the grader's own error
     rate so downstream metrics can be interpreted with the right confidence interval.

### Known v1 weakness — groundedness-loop routing (decision pending)

  When `grade_groundedness` returns `grounded=False`, the current graph routes back to
  `retrieve`. This re-fetches the same documents (the query hasn't changed). The most common
  cause of groundedness failure is that *generation drifted*, not that retrieval was wrong —
  the model hallucinated despite having sufficient context.

  Stronger design: groundedness-fail → constrained re-`generate` (with an added instruction
  like "answer only from the provided context; do not add claims not in the sources") or
  explicit refusal. Only route to re-retrieve on relevance failure, where the retrieval was
  actually insufficient.

  **Pending decision:** split the paths (generate its own retry node) vs keep v1 and log the
  trade-off. Will decide after the first run reveals which failure mode actually dominates.

### Experiment (first two runs — 2026-06-13)

  Hypothesis: LLM grader catches hallucinations; adds ~0.3–0.6s latency per grader call.
  Prediction: most retries will be relevance-triggered, not groundedness-triggered. ✅ confirmed.

  | Question | Retries | Path | Grounded | Output |
  |---|---|---|---|---|
  | "What infection control deficiencies were cited?" | 0 | retrieve→grade→generate→grade→END | yes | Grounded answer citing LIJ Medical Center $4K fine [Source 1] |
  | "What is the average nurse staffing ratio across all hospitals?" | 2 (exhausted) | retrieve→grade→rewrite×2→retrieve×2→generate→grade→END | no (budget) | "I cannot find..." — correct refusal |

  **Observed failure modes:**

  1. **Rewriter collapse:** retry 1 and retry 2 produced the *identical* rewritten query —
     `"Average nurse staffing ratio in hospitals"`. Root cause: `rewrite_query` always rewrites
     from `state["question"]` (the original), never from `state["query"]` (the previous attempt).
     It has no memory of what it already tried and no signal about why the prior query failed.
     A bad query loops identically until `MAX_RETRIES` is hit.
     **Fix (logged, not yet applied):** pass the failed query explicitly:
     ```
     "Original question: {question}\nPrevious query that failed: {query}\nRewrite it differently."
     ```

  2. **Aggregate questions are unanswerable by RAG.** "Average across ALL hospitals" requires
     computing over 519,555 chunks — no single retrieved passage can answer it. RAG answers
     *lookup* questions; *aggregate* questions belong to the MongoDB analytics agent (M4).
     The supervisor in M4 will route these differently.

  3. **Groundedness-loop weakness masked by budget.** `grounded=no` fired on the second run,
     which in v1 would route back to `rewrite_query` — but `retries == MAX_RETRIES` saved it.
     The v1 weakness (re-fetching the same docs on groundedness failure) was never exercised
     because exhaustion happened first. Still a real design flaw for production.

- **Decision:** keep v1 routing; log all three failure modes above. The rewriter fix and
  groundedness split are improvements for v2 — both require only a prompt + edge change.
  Moving to M3 now; the graph does gracefully degrade (correct refusal on unanswerable
  questions) which is the minimum bar for a real system.

## M3 — Persistence, HITL, time travel

- **Interview answer to own:** checkpoint = state per super-step per thread; interrupt vs
  static breakpoint; how `Command(resume=...)` re-enters the graph; `update_state` forking.

  **Checkpointing:** `graph.compile(checkpointer=MongoDBSaver(...))` — after every super-step,
  the full state is serialized to MongoDB. Each conversation thread is identified by
  `config["configurable"]["thread_id"]`. A crashed or restarted process can resume any thread
  from its last checkpoint. 20 checkpoints were written for a single 2-phase HITL run — one
  per Pregel super-step, including `loop` boundary ticks.

  **Factory pattern:** `build_rag_graph(checkpointer)` injects the checkpointer at compile
  time. The graph nodes are unaware of persistence. Tests pass `InMemorySaver`; production
  passes `MongoDBSaver`. Checkpointer is never a module-level singleton.

  **`interrupt()` in LangGraph 1.x:** placing `interrupt(payload)` inside a node suspends
  the graph at that point. Critical version-specific behavior: `ainvoke()` does NOT raise
  `GraphInterrupt` to the caller — it saves state and returns the partial state silently.
  To detect an interrupt you must use `astream(stream_mode="updates")` and check for the
  `"__interrupt__"` key in each chunk. The interrupt payload is in
  `chunk["__interrupt__"][0].value`.

  **Resume:** `graph.ainvoke(Command(resume=value), config=same_thread_config)` — the graph
  wakes up exactly where `interrupt()` was called; `value` becomes its return value.

  **Time travel:** `graph.aupdate_state(old_config, new_values, as_node=X)` creates a forked
  checkpoint that looks like node X just ran with the patched values. Returns a new config
  pointing at the fork. The original thread is untouched. Re-invoking with the new config
  runs the graph forward from the fork point.

  **WHERE to interrupt (the design decision):** post-`generate` is the right gate for a
  healthcare analyst system — the analyst reviews the synthesized answer, not raw chunks.
  Pre-`generate` is a cost-control gate (avoid paying for generation on bad retrieval).

- **Experiment (2026-06-13):** 3-phase demo — interrupt detected, human approved, time travel
  forked from step=3 with patched query and re-ran graph from there. All three mechanisms
  confirmed working.

- **Decision:** `MongoDBSaver` via `from_conn_string()` context manager (not a singleton).
  Interrupt detection via `astream` not `ainvoke` (LangGraph 1.x behavior). Time travel
  via `aupdate_state(as_node=X)` fork pattern.

## M4 — Supervisor multi-agent + Send map-reduce

- **Interview answer to own:** `Send` vs `Command`; subgraph state-key mapping; supervisor vs
  swarm trade-offs.
- **Experiment:** supervisor vs single tool-calling agent on the same 20 questions —
  quality / latency / token cost. (When is multi-agent worth it?)
- **Decision:**

## M5 — Streaming + observability

- **Interview answer to own:** what each `stream_mode` (`values`/`updates`/`messages`/`custom`)
  emits and when to use which; `astream_events`; how to attach a LangSmith / OpenTelemetry
  tracer to a LangGraph run; what "observability for non-deterministic systems" means
  (trace every node's input/output, not just the final answer, because the path is the bug).
- **Decision:** *(fill after M5 build)*

## M6 — Evals-driven development: trajectory + outcome + RAGAS + regression gate

- **Interview answer to own:** the full metric set locked in M2 (two-layer model, metric
  ladder, grader self-evaluation, human anchor slice); trajectory evaluation vs outcome
  evaluation; what a regression gate is (automated check that a new graph version does not
  regress below the baseline on the held-out eval set, run in CI before merge).
- **Headline result:** agentic RAG vs plain RAG — groundedness (RAGAS faithfulness),
  refusal accuracy, ROUGE-L, latency, cost/query. *(Fill after run.)*
- **Decision:** *(fill after M6 build)*

## M7 — Security & access control: tool permission scoping, AST allowlist, HITL approval

- **Interview answer to own:** why agentic security is different from API security (the
  agent can compose tool calls the designer never anticipated — each tool is a capability
  grant, not just an endpoint); the AST allowlist pattern (parse the model's requested
  tool call as an AST, reject if it references forbidden modules or operations before
  executing); sandboxed test-before-freeze (run the generated code in a subprocess with
  no network/filesystem write access, capture output, let a human approve before the result
  is used); HITL `interrupt()` as the final approval gate for high-risk actions.
- **Decision:** *(fill after M7 build)*

## M8 — MCP server: exposing retrieval and analytics tools

- **Interview answer to own:** what MCP (Model Context Protocol) is (a standard wire protocol
  for tool discovery and invocation, so any MCP-compatible client can call the server's tools
  without knowing their implementation); why a senior TL cares (enterprise tool integration
  without bespoke adapters for every client); trade-off vs direct Python function binding
  (MCP adds a network hop and a schema contract, but decouples client from server — worth it
  for multi-team / multi-runtime use).
- **Decision:** *(fill after M8 build)*

## M9 — Reader swap: raft-3b-r64-v2_2 via Ollama

- **Interview answer to own:** why the fine-tune is the *reader*, not the orchestrator
  (no tool-calling training; a 3B model is not reliable for structured JSON tool-call output
  under arbitrary user queries); rsLoRA merge gotcha (α/√r scaling — a naive α/r merge
  produces an 8× amplitude error at r=64, α=16 → α/√r=2 vs α/r=0.25; must use PEFT
  `merge_and_unload()` which handles rsLoRA-aware scaling); train/serve prompt-template
  parity (the Llama-3.2 instruct template used at RAFT train time must be reproduced exactly
  at serve time via `tokenizer.apply_chat_template`).
- **Headline result:** frontier reader (gpt-4o-mini) vs raft-3b reader, same slot, same eval
  set — groundedness, refusal accuracy, ROUGE-L, latency, cost/query. *(Fill after M9 run.)*
- **Decision:** *(fill after M9 build)*
