# Concept Ledger — eval · validation · security/safety · observability

> The single index of every concept in play, so nothing slips. Companion to
> `SESSION_LOG.md` (which is the chronological *decision* log). Doubles as interview
> revision. Update as concepts are added or move from theory → built.

**Status legend:** `📖 theory` (covered/decided, not yet code) · `🔨 building` ·
`✅ done` (implemented or already measured) · `🔁 reused` (ported from the data layer).

---

## A. Evaluation — objects & methods

| Concept | One line | Where in our system | Status |
|---|---|---|---|
| Four-object taxonomy | component / outcome / trajectory / operational — the spine | `trace.py` feeds all four | 📖 |
| Component eval | each piece (retriever, reader) in isolation | `eval/` retrieval + reader@oracle | 📖 |
| Outcome eval | final answer vs intent | `eval/` outcome | 📖 |
| Trajectory eval | the *path* taken, independent of answer | `eval/` trajectory asserts | 📖 |
| Operational/safety eval | latency/cost + policy pass-fail | `eval/` + `guards/` | 📖 |
| Reference-based metric | needs a gold answer; deterministic | `eval/metrics` | 🔁 |
| Reference-free / LLM-judge | scales w/o gold; circular | `eval/judge` | 📖 |
| Programmatic assert | deterministic boolean over the trace | `eval/trajectory` | 📖 |
| Human eval | gold standard; used to anchor | the 100-ex slice | 📖 |
| Oracle vs RAG mode | isolate FT skill vs end-to-end retrieval | eval harness | ✅ |

## B. Metrics

| Concept | One line | Status |
|---|---|---|
| ROUGE-L | longest-common-subsequence lexical overlap | ✅ 0.497 |
| BERTScore | token-level semantic similarity | ✅ 0.544 |
| RAGAS faithfulness | answer → atomic claims → NLI vs context | 📖 |
| RAGAS relevancy / context-precision / context-recall | answer & retriever quality | 📖 |
| Retrieval: hit@k / MRR / nDCG@k / recall@k | ranking quality before the reader | ✅ hit@5 .84 |
| Perplexity | LM fit (SLM-level signal) | ✅ |

## C. LLM-as-judge

| Concept | One line | Status |
|---|---|---|
| LLM-as-judge | a model scoring another model's output | 📖 |
| Circularity / correlated error | judge shares failure modes with generator | 📖 |
| Self-preference bias | rates own/same-family higher | 📖 |
| Position bias | favors slot A or B in pairwise | 📖 |
| Verbosity bias | prefers longer answers | 📖 |
| Format/confidence bias | rewards confident/structured tone | 📖 |
| Sycophancy / anchoring | drifts toward hinted answer | 📖 |
| Judge non-determinism / variance | same input → different score | 📖 |
| Low discrimination | clusters everything at 4/5 | 📖 |
| **Breaker 1:** human-anchor slice | label ~100, measure judge vs humans | 📖 |
| Cohen's κ / precision-recall (of judge) | the judge's *own* error rate | 📖 |
| **Breaker 2:** non-LLM reference | ROUGE/BERT fail differently → convergent validity | 📖 |
| **Breaker 3:** cross-family panel | decorrelates self-preference | 📖 |
| RAGAS conditioning | lower variance, NOT an escape from circularity | 📖 |

## D. Benchmarking & validation

| Concept | One line | Status |
|---|---|---|
| Bespoke held-out set | public benchmarks ≈ useless for domain RAG | 🔁 (curated 448/12) |
| Split hygiene | source/entity-level splits, no leakage | ✅ |
| Stratification | test set mirrors corpus distribution | 📖 |
| Contamination / leakage | test not in train; judge never sees gold ref | 📖 |
| Refusal-axis coverage | include genuinely-unanswerable cases | ✅ (12 unanswerable) |
| Distractor design | wrong-chunk negatives teaching abstention | 🔁 |
| Confidence interval / bootstrap | CI on the mean (~±0.02–0.03 @ n≈460) | 📖 |
| Paired significance | paired bootstrap / Wilcoxon / McNemar | 📖 |
| Sample size / power | enough rows to detect the gap | 📖 |
| Regression gate | thresholds vs frozen baseline + noise margin, in CI | 📖 |
| Baseline freezing | the reference point gates compare against | 📖 |
| Anchor slice (100, binary) | the human-labeled calibration set | 📖 |

## E. Security & safety (guardrails)

| Concept | One line | Where | Status |
|---|---|---|---|
| Agentic ≠ API security | threat is the model misusing granted capabilities | framing | 📖 |
| Confused-deputy problem | legitimate authority steered to act maliciously | framing | 📖 |
| Prompt injection (direct) | user input overrides instructions | `guards/input` | 📖 |
| **Indirect prompt injection** | malicious instruction inside a retrieved chunk | `guards/input` (on retrieved) | 📖 |
| Jailbreak | bypass safety constraints | `guards/input` | 📖 |
| Input ring | injection / jailbreak / PII / scope, pre-model | `guards/input` | 📖 |
| Output ring | groundedness / citation / PII-egress / schema, pre-user | `guards/output` | 📖 |
| Action ring | the agent-specific hard ring | `guards/action` | 📖 |
| Capability scoping / least privilege | minimal tool set per agent | `guards/action` | 📖 |
| Tool allow-listing | only sanctioned tools per agent | `agents/` | 📖 |
| NL→query safety | read-only, no `$where`/writes, cost-bounded | `guards/action` | 📖 |
| AST allowlist | parse generated code, reject off-list nodes | `guards/action` | 📖 |
| Sandboxed execution | subprocess, no net, FS-restricted, resource caps | `guards/action` | 📖 |
| HITL approval / `interrupt()` | human gate before irreversible ops | `agents/` | 📖 |
| Test-before-commit | run generated code vs tests before use/freeze | `guards/action` | 📖 |
| Defense in depth | no single layer trusted; layer them | principle | 📖 |
| Eval/guardrail duality | same check = metric offline, gate online | `eval/` ↔ `guards/` | 📖 |
| PII detection / redaction | find & strip sensitive data | `guards/` | 📖 |
| Llama Guard / Prompt Guard | adopted input-ring classifiers | `guards/input` | 📖 (adopt) |
| NeMo Guardrails / Guardrails AI | guardrail libraries (input ring) | `guards/input` | 📖 (adopt) |

## F. Observability

| Concept | One line | Where | Status |
|---|---|---|---|
| Tracing / spans | per-step capture of inputs/outputs/latency/cost | `obs/` | 📖 |
| Structured trace schema | the Pydantic contract everything computes from | `trace.py` | 🔨 next |
| LangSmith (hosted) | reference LangGraph observability + datasets + annotation queues | `obs/` (env vars) | 📖 |
| Langfuse (self-host alt) | OSS equivalent; the choice under a data-residency constraint | (documented alternative) | 📖 |
| OpenTelemetry / OTel GenAI | vendor-neutral tracing standard | `obs/` exporter | 📖 |
| Annotation queue | labeling UI for the anchor slice | LangSmith | 📖 |
| Cost accounting / latency waterfall | token/$ + timing per run | `trace.py` + LangSmith | 📖 |

## G. Orchestration (LangGraph)

| Concept | One line | Where | Status |
|---|---|---|---|
| Supervisor / router | classify intent, dispatch to one specialist subgraph | chat graph | 📖 |
| Subgraph | a compiled graph used as a node in a parent graph | `agents/` | 📖 |
| `Command(goto=…)` | node-returned control transfer / dispatch | router | 📖 |
| `Send` | fan-out to parallel workers (map-reduce) | analysis agent | 📖 |
| State slice / reducer | per-subgraph state + merge semantics | all graphs | 🔁 |
| Vertical slice | one agent fully traced+guarded+measured before widening | build strategy | 📖 |
| HITL `interrupt()` | suspend graph for human approval, resume later | codegen / analysis | 📖 |

## H. Reliability — degradation & fallback

| Concept | One line | Where | Status |
|---|---|---|---|
| Degradation ladder | fallback chain ending in a safe refusal, never crash/fabricate | all agents | 📖 |
| Refuse > fabricate | terminal rule: refuse rather than ship a hallucination | groundedness gate | 📖 |
| Fail-closed (guards) | a guard that errors denies by default | `guards/` | 📖 |
| Fail-open (observability) | tracing failure never blocks the user | `obs/` | 📖 |
| Circuit breaker | trip to fast-fail after repeated dependency failure | retriever / LLM / judge | 📖 |
| Bounded everything | caps on retries/recursion/query-cost/sandbox time+mem | all | 📖 |
| Partial-result honesty | degraded answers flagged degraded in the trace | `trace.py` | 📖 |
| Retrieval fallback chain | Pinecone → local mirror → widen → BM25 → refuse | RAG retrieve | 📖 |

---

### How to use this
- A concept graduates `📖 → 🔨 → ✅` as it becomes code; flip the status when it lands.
- New concept introduced in conversation → add a row here in the same turn it's coined.
- For deep rationale on any row, see the matching section in `SESSION_LOG.md`.
