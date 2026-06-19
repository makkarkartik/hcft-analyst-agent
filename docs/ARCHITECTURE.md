# System Architecture — abstraction model, graphs, per-stage guards/eval, plan

> The **operational** map: the abstraction model, the actual graph topology, the guardrail +
> eval hook at every stage, and the phased build plan with a live tracker. Companion to
> `CONCEPTS.md` (glossary) and `SESSION_LOG.md` (decisions).
> Model: **skills-first**; agents are bounded loops over skills; a stateful **chat agent** is
> the front door (cheap internal router); two doors (chat + MCP). Framework: **LangGraph**.

Legend in diagrams:  `[G]` = guardrail fires here · `[E]` = eval signal captured here.

---

## 0. Abstraction model

Layers, dumb → smart:
- **Tool** — one function (`retrieve()`, `mongo_aggregate()`, `render_chart()`). No judgment.
- **Skill** — a bounded capability + a **manifest** (IO schema · guards · eval metrics ·
  fallback · when-to-use). Fixed control flow; **directly callable**.
- **Agent** — a **bounded loop over skills/tools** with autonomy (decides, retries,
  self-corrects). Variable cost → must be capped.
- **Chat agent** — the stateful, top-level agent (the conversational front door).

**Skill vs agent test:** *does the path depend on intermediate results it can't know in
advance?* No → fixed pipeline → **skill**. Yes → observe/decide/loop → **agent**. Agents are
*made of* skills (a skill is an agent's tool).

| Capability | Skill / agent | Why |
|---|---|---|
| Visualize given data | **skill** | fixed transform (data → code → render) |
| NL → one aggregation | **skill** | steps known |
| Deep analysis | **agent** | open-ended, iterative investigation |
| RAG chat (self-correcting) | **agent** | retrieve→grade→rewrite loop |
| Code-gen w/ test-and-fix | **agent** | generate→check→sandbox→test→fix loop |
| Single-shot code-gen | skill | one pass |

**Framework = LangGraph** (low-level: explicit nodes/state/cycles), chosen over CrewAI/AutoGen
because this system's value is *measuring + guarding + controlling* the agent — which needs
explicit, inspectable control flow, per-stage guard injection, bounded loops, and HITL. An
opinionated role/task framework would hide exactly what we evaluate and fight the skills model.

---

## 1. Entry: two doors + the chat agent

**Two front doors, one set of skills:**
- **Conversational door** — the **chat agent** (stateful LangGraph graph); routing is its
  *first internal node*, not a stage before it.
- **Programmatic door** — **direct skill / MCP call** that skips the chat agent entirely (the
  caller already knows the skill).

```
USER MSG ─► CHAT AGENT  (LangGraph · owns conversation state)
              │
        [1] INPUT GUARD          [G] injection·jailbreak·PII·scope
              │
        [2] TRIAGE / ROUTER  (cheap: small model or embeddings)     [E] route correctness
              │
   ┌──────────┼─────────────┬───────────────┬─────────────┐
   ▼          ▼             ▼               ▼             ▼
 inline    RAG-chat     DEEP-ANALYSIS    CODE-GEN     clarify /
 (chit-    agent        agent            agent        refuse
 chat,     (grounded)   (analytic)       (code)       (ambiguous /
 follow-                                               out-of-scope)
 up, re-
 format)
   │          └─────────────┴───────────────┘
   │                  │ (answer + citations)
   └────────┬─────────┘
            ▼
       [3] OUTPUT GUARD          [G] groundedness·citations·PII-egress·schema
            │
       [4] EMIT TRACE → LangSmith + respond     [E] cost/latency; gate (offline)
```

**Three rules that keep this cheap and robust:**
1. **Routing is inside the chat agent**, not before it — the chat agent owns conversation state.
2. **Not every message dispatches.** Chit-chat / follow-ups / reformat ("now as a chart") are
   answered *inline* — no skill, no specialist cost. Only genuine new-task intents route out.
3. **Tiered router** (small model / embeddings); escalate to heavier reasoning only when
   ambiguous or a skill rejects — **course-correction**: the chat agent re-decides.

Dispatch: router node returns `Command(goto=<skill/agent>)`; each specialist is a compiled
subgraph. One OpenInference/OTel trace spans the whole run.

---

## 2. Specialist agents (bounded loops over skills)

Each is an **agent** (a capped loop); their tools are **skills** (retrieve, NL→query,
render-chart, …). "Visualize given data" is a *skill* reused by analysis/code-gen, not its
own agent.

### 2a. RAG CHAT
```
 retrieve ──►[G] indirect-injection scan on chunks ──► grade_docs ──┐
    ▲                                                               │ relevant?
    │  rewrite_query  (guard: retries ≤ MAX_RETRIES) ◄─── no ───────┤
    │                                                               ▼ yes
    └───────────────────────────────────  generate ──►[G] groundedness gate ──► answer │ refuse
```
- `[G]` retrieved-chunk injection · groundedness · citation provenance (`cited_ids ⊆ retrieved_ids`)
- `[E]` component (hit@k/MRR/nDCG) · trajectory (retrieve-before-generate, retries≤N, no
  rewriter-collapse) · outcome (ROUGE/BERT/RAGAS/judge) · refusal correctness

### 2b. DEEP ANALYSIS
```
 plan ──► NL→Mongo query ──►[G] query-safety (read-only · no $where/writes · cost cap) ──►
    execute aggregation ──► map-reduce synthesis ──►[G] groundedness/citation gate ──► report
```
- `[G]` NL→query safety · capability scoping (read-only Mongo handle)
- `[E]` trajectory (right tool chosen, query AST valid, routed correctly) · outcome · cost

### 2c. CODE-GEN  (the sandboxed coder)
```
 spec ──► generate code ──►[G] AST allowlist ──►[HITL] approval (interrupt) ──►
    [SANDBOX] subprocess (no net · FS-restricted · mem/CPU/time caps) ──► test-before-commit ──► artifact
```
- `[G]` AST allowlist · sandbox isolation · HITL approval · test gate (defense in depth)
- `[E]` trajectory (gated path followed in order) · AST-pass rate · sandbox-contained · tests-passed

---

## 3. Stage × guard × eval matrix

| Stage | Guardrail | Eval signal | Object |
|---|---|---|---|
| chat: input | injection/jailbreak/PII/scope | input_guard verdicts | operational |
| chat: router | — | route correctness | trajectory |
| rag: retrieve | indirect-injection on chunks | hit@k, MRR, nDCG | component |
| rag: grade/rewrite | retry cap | retries≤N, no-collapse | trajectory |
| rag: generate | groundedness + citations | ROUGE/BERT/RAGAS/judge, refusal acc | outcome |
| analysis: query | NL→query safety, read-only | query valid, right tool | trajectory/safety |
| codegen: generate | AST allowlist | AST-pass rate | safety |
| codegen: execute | sandbox + HITL + test gate | sandbox-contained, tests-passed | safety |
| chat: output | groundedness/citation/PII/schema | output_guard verdicts | operational |
| chat: emit | — | cost, latency, gate pass/fail | operational |

---

## 4. Implementation plan + tracker

Status: `☐ todo` · `◐ in progress` · `☑ done`.

| Phase | Deliverable | Components | Status |
|---|---|---|---|
| **P0 Skeleton** | telemetry + eval plumbing | OpenInference instrumentation + OTel→LangSmith, DeepEval+RAGAS gate + G-Eval, `hcft.*` attrs, config, repo restructure | ◐ |
| **P1 RAG chat** | first agent end-to-end | RAG agent + its `[G]`/`[E]` hooks, emits full trace → first real numbers | ☐ |
| **P2 Chat agent** | front door | input guard, cheap triage/router, inline handling, dispatch (Command), output guard, refuse | ☐ |
| **P3 Deep analysis** | analytic agent | NL→Mongo skill, query-safety guard, map-reduce synthesis | ☐ |
| **P4 Code-gen** | sandboxed coder | generate → AST allowlist → HITL → sandbox → test-before-commit (adopt sandbox tool) | ☐ |
| **P5 Eval harness** | make it measurable | anchor-slice labeling in LangSmith, benchmark extension (agent-shaped cases), regression gate in CI | ☐ |
| **P6 Second door** | direct invocation | expose skills via MCP / API (skips the chat agent) | ☐ |

Build principle: **vertical slice first.** P0+P1 get *one* agent fully traced, guarded, and
measured before widening to the chat agent and the other specialists — so the contract is
proven on real output before everything depends on it.

---

## 5. Per-stage eval × guard × fallback (per agent)

### RAG CHAT
| Stage | `[E]` eval | `[G]` guard | Fallback / degradation |
|---|---|---|---|
| embed query | embed latency | — | embed error → retry → local embed model |
| retrieve (dense) | hit@k, recall@k | — | Pinecone down → local mirror; 0 hits → widen top_k → BM25 → refuse |
| chunk injection scan | detector precision/recall | injection | flagged chunk dropped; all flagged → refuse |
| rerank (BGE) | nDCG@10, MRR, rerank lift | — | reranker error → keep dense order (degraded rank) |
| grade docs | grader κ vs human | — | grader down → proceed ungraded, mark low-confidence |
| rewrite (loop) | retries ≤ N, no-collapse | retry cap | max retries → graceful refuse (no infinite loop) |
| generate | ROUGE/BERT/RAGAS/judge | — | LLM timeout → retry → fallback model; else snippets + "couldn't synthesize" |
| groundedness gate | faithfulness, citation valid, refusal acc | groundedness, citations | ungrounded → 1 constrained re-gen → else refuse (never ship hallucination) |

### DEEP ANALYSIS
| Stage | `[E]` eval | `[G]` guard | Fallback / degradation |
|---|---|---|---|
| plan / decompose | plan correctness | — | plan fails → single-step fallback |
| NL→Mongo query | query valid, right-tool | query-safety (read-only, no `$where`) | unsafe/invalid → 1 regen → clarify or refuse |
| execute aggregation | query latency, result size | cost cap, capability scope | timeout/expensive → partial + warn; empty → "no matching data" |
| map-reduce synthesis | outcome quality, coverage | — | worker fails → synthesize from rest (flag partial); reduce fails → raw aggregates |
| groundedness/citation gate | faithfulness | groundedness | ungrounded → numbers-only (no narrative), or refuse |

### CODE-GEN
| Stage | `[E]` eval | `[G]` guard | Fallback / degradation |
|---|---|---|---|
| generate code | AST-pass rate, gen quality | — | gen fails → retry → template |
| AST allowlist | violation types, pass rate | AST allowlist | violation → 1 stricter regen → refuse ("can't produce safe code") |
| HITL approval | approval rate / time | HITL gate | reject → regen w/ feedback; timeout → hold (never auto-approve) |
| sandbox exec | contained?, resource use | sandbox (no net, caps) | runtime error → captured, no side effects; timeout/OOM → kill+report; escape → hard-fail |
| test-before-commit | tests-passed rate | test gate | fail → return code + failing tests, do NOT freeze; flaky → bounded retry |

---

## 6. Degradation & fallback policy (cross-cutting)

- **Degradation ladder** — every capability has a fallback chain ending in a *safe refusal*,
  never a crash or a hallucination. Terminal rule: **refuse > fabricate.**
- **Fail-closed guards, fail-open observability** — guardrails deny by default when they
  error (block/refuse on safety-critical paths); observability (LangSmith / trace export)
  must never block the user — if tracing fails, persist locally (Mongo) and proceed.
- **Circuit breakers** — repeated dependency failure (Pinecone / LLM / judge) trips a
  breaker → fast-fail to fallback instead of hanging every request.
- **Bounded everything** — retries, recursion, query cost, sandbox time/memory all capped;
  the cap's terminal branch is a graceful message.
- **Partial-result honesty** — degraded answers (dense-only ranking, partial map-reduce,
  ungraded retrieval) are *flagged degraded* in the trace, not silently passed as full quality.

Chat-agent level: router low-confidence → clarify or default to RAG chat (never silent
misroute); sub-agent exception → caught, graceful error + trace emitted (no 500); judge
offline → eval degrades to overlap + programmatic (judge is directional anyway).

---

## 7. Decisions log
- **Abstraction / orchestration** — ✅ 2026-06-19: skills-first; agents = bounded loops over
  skills; **chat agent** is the stateful front door with a *cheap internal router* (not a
  pre-stage); two doors (chat + direct/MCP); tiered routing + course-correction. See §0–§1.
- **Framework** — ✅ LangGraph (over CrewAI/AutoGen) — explicit control flow for eval/guards.
- **Run record** — ✅ 2026-06-19: OpenInference/OTel spans + DeepEval; no custom
  `AgentRunTrace` (SESSION_LOG §12).
- **Retrieval fallback chain** — FAISS local mirror vs widen-top_k → BM25 → refuse? Decide at P1.
- **Fail-open/closed policy** — guards fail *closed*, observability fails *open*. *(Rec: yes.)*
- **Router implementation** — LLM classifier vs embedding/rules vs hybrid. Decide at P2.
