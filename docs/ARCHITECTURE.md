# System Architecture — graphs, per-stage guards/eval, implementation plan

> The **operational** map: the actual graph topology, the guardrail + eval hook at every
> stage, how the chat graph triggers the specialist agents, and the phased build plan with
> a live tracker. Companion to `CONCEPTS.md` (the glossary) and `SESSION_LOG.md` (decisions).
> Orchestration pattern: **supervisor/router → specialist subgraphs** (pending final sign-off).

Legend in diagrams:  `[G]` = guardrail fires here · `[E]` = eval signal captured here.

---

## 1. Top-level: the CHAT graph (supervisor / router)

```
                         USER MESSAGE
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
   [1]  │  INPUT GUARD                                 │  [G] injection · jailbreak · PII · scope
        │  (on the user message)                       │  [E] input_guard verdicts → trace
        └─────────────────────────────────────────────┘
                              │  (blocked → refuse early)
                              ▼
        ┌─────────────────────────────────────────────┐
   [2]  │  ROUTER  — classify intent                   │  [E] route correctness (trajectory)
        └─────────────────────────────────────────────┘
              │            │            │            │
   grounded Q │  analytic/ │  code /    │  out-of-   │
              │  aggregate │  compute   │  scope     │
              ▼            ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐
   [3]  │ RAG CHAT │ │  DEEP    │ │ CODE-GEN │ │ REFUSE  │
        │ subgraph │ │ ANALYSIS │ │ subgraph │ │         │
        └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬────┘
             └────────────┴────────────┴────────────┘
                              │  (answer + citations + trace fragment)
                              ▼
        ┌─────────────────────────────────────────────┐
   [4]  │  OUTPUT GUARD                                │  [G] groundedness · citations · PII-egress · schema
        │  (on the specialist's answer)                │  [E] output_guard verdicts; outcome metrics
        └─────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
   [5]  │  EMIT TRACE → Langfuse + respond to user      │  [E] cost/latency; regression gate (offline)
        └─────────────────────────────────────────────┘
```

Dispatch mechanism: router returns a `Command(goto=<subgraph>)`; each specialist is a
compiled subgraph with its own state slice. One `AgentRunTrace` spans the whole run.

---

## 2. Specialist subgraphs

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
| **P0 Skeleton** | the contract + plumbing | `trace.py` (AgentRunTrace), gate runner, Langfuse docker + wiring, repo restructure | ☐ |
| **P1 RAG chat** | first agent end-to-end | RAG subgraph + its `[G]`/`[E]` hooks, emits full trace → first real numbers | ☐ |
| **P2 Chat graph** | orchestration | input guard, router, dispatch (Command), output guard, refuse path | ☐ |
| **P3 Deep analysis** | analytic agent | NL→Mongo, query-safety guard, map-reduce synthesis | ☐ |
| **P4 Code-gen** | sandboxed coder | generate → AST allowlist → HITL → sandbox → test-before-commit | ☐ |
| **P5 Eval harness** | make it measurable | anchor-slice labeling in Langfuse, benchmark extension (agent-shaped cases), regression gate in CI | ☐ |

Build principle: **vertical slice first.** P0+P1 get *one* agent fully traced, guarded, and
measured before widening to the router and the other two specialists — so the contract is
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
  error (block/refuse on safety-critical paths); observability (Langfuse / trace export)
  must never block the user — if tracing fails, persist locally (Mongo) and proceed.
- **Circuit breakers** — repeated dependency failure (Pinecone / LLM / judge) trips a
  breaker → fast-fail to fallback instead of hanging every request.
- **Bounded everything** — retries, recursion, query cost, sandbox time/memory all capped;
  the cap's terminal branch is a graceful message.
- **Partial-result honesty** — degraded answers (dense-only ranking, partial map-reduce,
  ungraded retrieval) are *flagged degraded* in the trace, not silently passed as full quality.

Chat-graph level: router low-confidence → clarify or default to RAG chat (never silent
misroute); sub-agent exception → caught, graceful error + trace emitted (no 500); judge
offline → eval degrades to overlap + programmatic (judge is directional anyway).

---

## 7. Open decisions
- **Orchestration pattern** — supervisor/router → subgraphs (proposed) vs single ReAct-with-tools
  vs hierarchical delegation. *Recommended: supervisor/router.* Awaiting sign-off.
- **Trace schema** — the 10-group `AgentRunTrace` (see SESSION_LOG / chat). Awaiting sign-off.
- **Retrieval fallback chain** — wire the FAISS local mirror as a Pinecone fallback, or just
  widen-top_k → BM25 → refuse? (FAISS needs the index ported.) To decide at P1.
- **Fail-open/closed policy** — confirm: guards fail *closed*, observability fails *open*. *(Rec: yes.)*
- **Router implementation** — LLM classifier vs embedding/rules vs hybrid. To decide at P2.
