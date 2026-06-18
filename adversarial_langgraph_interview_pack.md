# Adversarial LangGraph Interview Pack — Cisco CX Senior Technical Lead

Purpose: run a rigorous 45-minute mock interview for a Senior Technical Lead, Autonomous Agentic Frameworks & System Integration role, using the HCFT Analyst Agent project as the evidence base.

Use mode: the interviewer should ask the questions, interrupt when the answer is vague, force trade-offs, and score based on design judgment rather than memorized syntax. Do not show the answer key during rehearsal.

---

## 1. Interview thesis

This is not a “tell me what LangGraph is” interview. It is a senior design-and-leadership interview. The candidate must defend an agentic architecture, explain concrete LangGraph mechanics, show eval/observability discipline for non-deterministic systems, handle enterprise security/tool integration, and honestly frame project gaps.

The role signal to test:

- Can the candidate design, build, and deploy autonomous/multi-agent systems?
- Can the candidate explain why LangGraph is the right orchestration layer versus a vanilla tool-calling agent?
- Can the candidate separate runtime control signals from offline eval metrics?
- Can the candidate describe observability and trajectory evaluation for non-deterministic systems?
- Can the candidate reason about enterprise integration: APIs, databases, MCP, Snowflake/Cortex, access control, SDLC, and stakeholder risk?
- Can the candidate lead: scope, prioritize, communicate gaps, and avoid overclaiming?

---

## 2. 45-minute structure

| Time | Segment | Goal | Interviewer posture |
|---:|---|---|---|
| 0:00–3:00 | Opening pressure | Force concise architecture framing | “You have 90 seconds. No buzzwords.” |
| 3:00–10:00 | Architecture whiteboard | Test system design and routing | Challenge every component: why not simpler? |
| 10:00–17:00 | LangGraph mechanics | Test actual graph understanding | Ask reducers, super-steps, cycles, edge purity, async. |
| 17:00–25:00 | Self-corrective RAG + evals | Test hallucination control and measurement maturity | Attack LLM graders as “vibes.” |
| 25:00–31:00 | Multi-agent/data/tool integration | Test supervisor, Mongo analytics, synthesis, MCP | Ask where RAG fails and what tool should handle it. |
| 31:00–37:00 | Persistence, HITL, observability, security | Test production readiness | Push on traceability, approvals, tool permissions. |
| 37:00–42:00 | Model strategy + OSS/frontier | Test fine-tune/reader/orchestrator judgment | Attack the 3B model and ask why it is not the orchestrator. |
| 42:00–45:00 | Leadership/gaps close | Test honesty and stakeholder communication | Push Snowflake, Jira, AutoGen/CrewAI, doc drift. |

---

## 3. Scorecard, 100 points

### A. Architecture and system ownership — 20

Strong answer:
- Frames HCFT Analyst Agent as a LangGraph supervisor over research RAG, Mongo analytics, and synthesis.
- Explains why different question classes need different agents.
- Identifies vector RAG as poor for global aggregation.
- Names cost/latency/quality trade-offs of multi-agent systems.
- Can draw the graph and state boundaries.

Red flags:
- “Multi-agent is better” with no trade-off.
- Routes everything to vector search.
- Cannot explain why a supervisor is needed.
- Cannot explain where human approval belongs.

### B. LangGraph mechanics — 20

Strong answer:
- Explains state, reducers, LastValue, concurrent update failure, and add_messages.
- Explains ReAct as agent node + ToolNode + conditional tool edge + loop.
- Explains cycles, recursion limits, and business-level retry limits.
- Explains structured output for routing.
- Keeps LLM calls inside nodes and edges pure.
- Understands async nodes and to_thread for blocking retriever calls.

Red flags:
- Thinks conditional edges call the LLM.
- Cannot explain why two parallel writers need a reducer.
- Cannot explain why the agent node fires twice in ReAct.
- Treats recursion_limit as the only safety mechanism.

### C. Evals and observability — 20

Strong answer:
- Separates runtime graders from offline metrics.
- Explains why RAGAS is still LLM-as-judge.
- Includes human-labeled anchor slice, non-LLM metric, trajectory eval, regression gate.
- Says observability must trace node inputs/outputs and graph path, not only final answer.
- Gives failure-mode examples: rewriter collapse, aggregate query misroute, groundedness loop.

Red flags:
- “The grader says it is grounded, so eval is done.”
- Claims RAGAS removes LLM judge bias.
- Uses BLEU as the main grounded-QA metric.
- Measures final answer only, not tool calls/path/retries.

### D. Enterprise integration, security, and access control — 15

Strong answer:
- Explains tools as capability grants.
- Uses scoped tools, schemas, authz, audit logs, HITL for high-risk actions.
- Explains AST allowlist and sandboxed test-before-freeze for generated code/tools.
- Explains MCP as enterprise tool discovery/invocation boundary and when it is worth the overhead.

Red flags:
- “We trust the model prompt.”
- Exposes broad DB write access to the agent.
- Cannot distinguish direct Python tool binding from MCP.
- No approval boundary for file writes or external actions.

### E. Model, RAG, and data strategy — 15

Strong answer:
- Explains reader vs orchestrator separation.
- Does not put a 3B RAFT fine-tune in charge of arbitrary structured tool calls.
- Explains frontier reader vs raft-3B comparison on groundedness, refusal accuracy, latency, cost/query.
- Mentions prompt-template parity and rsLoRA merge risk if asked.

Red flags:
- Claims the fine-tuned reader should orchestrate tools without tool-calling training.
- Ignores prompt-template mismatch.
- Treats fine-tuning as a substitute for retrieval-confidence gates.

### F. Leadership and delivery judgment — 10

Strong answer:
- Owns the roadmap and risk log.
- Concedes Snowflake/Jira gaps cleanly.
- Frames Databricks/Snowflake as transferable where true, not equivalent where not true.
- Can give stakeholder-friendly status/risk update.
- Prioritizes “working + measured” over gold-plating.

Red flags:
- Fakes Snowflake/Jira hands-on experience.
- Overclaims M8/M9 if not actually complete.
- Cannot explain scope cuts.
- Does not notice documentation drift.

Passing bar: 75+. Strong senior signal: 85+. Hire-with-reservations: 70–74 if gaps are honest and mechanics are solid. No-hire signal: below 70 or any serious overclaiming.

---

## 4. Opening prompt

“Pretend I am the Cisco hiring manager. You have 90 seconds. Explain the HCFT Analyst Agent architecture and why it proves you can lead autonomous agentic framework work. Do not list technologies. Explain the system, trade-offs, and what you measured.”

Expected strong answer:

“The system is a LangGraph supervisor over a healthcare report corpus. It routes lookup questions to a self-corrective RAG research graph, aggregate questions to MongoDB analytics, and multi-report brief generation to a map-reduce synthesis workflow with HITL approval. I used LangGraph because the hard part is not one tool call; it is controlled stateful execution: cycles, retries, checkpointing, routing, observability, and bounded failure. The project is designed around evals: in-graph graders are runtime control signals, while offline evaluation uses grounding, retrieval, trajectory, and human-anchor metrics. The leadership angle is that the roadmap is scoped by JD risk: core LangGraph and evals are Tier-1, Snowflake/Cortex is designed and cost-modeled, and gaps like Jira are acknowledged rather than faked.”

Adversarial follow-up:

“That sounds like a portfolio narrative. Give me the actual graph nodes and failure boundaries.”

Expected recovery:

Name the research RAG flow: retrieve → grade_documents → generate or rewrite_query loop; generate → grade_groundedness → END or retry/refuse. Mention MAX_RETRIES and recursion_limit. Name supervisor routing to research vs analytics vs synthesis. Mention refusal for unanswerable/unsupported questions.

---

## 5. Question bank with adversarial follow-ups and answer keys

### 5.1 Architecture: why LangGraph?

Question:
“Why didn’t you just build a normal LangChain tool-calling agent? Why LangGraph?”

Strong answer:
- A single ReAct agent is fine for a simple query/tool loop.
- This project needs explicit state, multiple branches, cycles, retry budgets, checkpointing, interrupts, streaming, and trajectory eval.
- LangGraph lets the team reason about execution paths and failure modes, not just final answers.
- Hand-rolled graph nodes make it possible to insert graders, rewrites, persistence, HITL, and specialist subgraphs.

Push:
“But isn’t that overengineering?”

Recovery:
“Yes, if the problem is only ‘call search then answer.’ It is not overengineering when requirements include durable multi-turn workflows, observability, HITL, policy gates, and multiple tool classes. I would still keep the graph small: use a single agent for simple lookup and add supervisor/specialists only where routing improves correctness or cost.”

---

### 5.2 State and reducers

Question:
“In LangGraph, two nodes write to the same state key in the same super-step. What happens?”

Strong answer:
- Default state channel is effectively single-writer LastValue.
- Concurrent writes to the same key fail loudly.
- Use a reducer when multiple writers are intended.
- For messages, add_messages appends/updates by message id.
- Prefer single-writer design where possible.

Push:
“Why not let the last writer win?”

Recovery:
“Because execution order can become nondeterministic. Silent last-writer-wins hides race conditions in an agent system; fail-loud is safer.”

---

### 5.3 ReAct mechanics

Question:
“You said you hand-built ReAct before using prebuilt abstractions. What does create_react_agent compile to?”

Strong answer:
- Agent node.
- ToolNode.
- Conditional tools edge inspecting the last AIMessage/tool_calls.
- Loop from tools back to agent.
- Stops when model returns no tool_calls.

Push:
“Why does the agent node run twice?”

Recovery:
“First call asks for a tool. ToolNode executes and appends ToolMessage. Second model call sees the tool result and produces final answer with no tool_calls, so conditional edge ends.”

---

### 5.4 Self-corrective RAG graph

Question:
“Walk me through your self-corrective RAG graph. I want node names, state transitions, and loop breakers.”

Strong answer:
- retrieve: query → Pinecone/Mongo/rerank docs.
- grade_documents: structured output relevant bool.
- if not relevant and retries left: rewrite_query → retrieve.
- if relevant: generate.
- grade_groundedness: structured output grounded bool.
- if grounded: END.
- if not grounded: current v1 routes to rewrite/retrieve, but stronger v2 should constrain re-generate or refuse when docs are sufficient.
- Loop breakers: MAX_RETRIES plus recursion_limit.

Push:
“Your groundedness failure goes back to retrieve. Isn’t that wrong?”

Recovery:
“Yes, often. Groundedness failure usually means generation drifted, not retrieval failed. I would split the path: relevance failure triggers re-retrieve/rewrite; groundedness failure triggers constrained regeneration or refusal. The current v1 logs this as a known weakness.”

---

### 5.5 Structured output as routing

Question:
“Why use with_structured_output for graders? Why not parse ‘yes’/‘no’?”

Strong answer:
- Routing needs typed control signals, not prose.
- Pydantic/structured object gives bool fields like relevant/grounded.
- Avoids brittle string parsing.
- Conditional edges stay pure and deterministic over state.

Push:
“But structured output can still be wrong.”

Recovery:
“Yes. Structured output improves reliability of the control interface; it does not guarantee semantic correctness. That is why the grader itself must be evaluated against human labels.”

---

### 5.6 Runtime graders vs offline evals

Question:
“You are using an LLM to grade relevance and groundedness. Isn’t that just vibes?”

Strong answer:
- There are two separate layers.
- Runtime graders are cheap, binary, reference-free control signals because no gold answer exists at inference time.
- Offline evals are measured on a held-out set with gold answers or human labels.
- Do not use the runtime grader as the final quality claim.

Push:
“Why not run ROUGE-L or RAGAS at inference time?”

Recovery:
“ROUGE-L needs a reference answer that does not exist at inference. RAGAS can be reference-free for faithfulness, but it is too heavy for every routing decision and still judge-based. Use it offline for measurement, not as the graph’s cheap control edge.”

---

### 5.7 RAGAS circularity

Question:
“You said RAGAS. But RAGAS uses an LLM judge too. How does that solve the problem?”

Strong answer:
- It does not fully solve judge dependency.
- It reduces variance by decomposing claims and checking against concrete retrieved context.
- The real circularity breakers are: human-labeled anchor slice and non-LLM metrics such as ROUGE-L/BERTScore.
- Report judge agreement: precision, recall, kappa.

Push:
“So if RAGAS says 0.95 faithfulness, are you done?”

Recovery:
“No. I would compare against human labels, trajectory correctness, refusal accuracy, and lexical/semantic reference metrics. A single high RAGAS number can hide judge blind spots.”

---

### 5.8 BLEU trap

Question:
“Would BLEU be your main metric for grounded QA?”

Strong answer:
- No.
- BLEU is precision-oriented and brittle for short QA.
- It punishes valid paraphrases and says nothing about grounding.
- Prefer RAGAS faithfulness/context metrics plus ROUGE-L or BERTScore and refusal accuracy.

Push:
“Is BLEU forbidden?”

Recovery:
“No, it can be an auxiliary lexical signal, but it should not drive the decision.”

---

### 5.9 Trajectory eval

Question:
“What does trajectory evaluation mean here?”

Strong answer:
- Evaluate the path, not just final text.
- Did the supervisor choose research vs analytics vs synthesis correctly?
- Did grade_documents fire before generate?
- Were retries bounded?
- Did the graph refuse when retrieval was insufficient?
- Did high-risk write actions hit HITL approval?

Push:
“Why does path matter if the final answer is correct?”

Recovery:
“Because non-deterministic systems can be accidentally correct. A correct answer via an unsafe tool path, unbounded retries, or missing approval gate is still a production failure.”

---

### 5.10 Aggregate query trap

Question:
“A user asks: ‘What is the average nurse staffing ratio across all hospitals?’ Your RAG retrieves five chunks. What should happen?”

Strong answer:
- Do not answer from top-k chunks.
- This is an aggregate over the corpus, not a lookup.
- Route to MongoDB analytics agent or refuse if required field is unavailable.
- The supervisor should classify aggregate intent.
- RAG can provide supporting report snippets, but should not pretend top-k equals global coverage.

Push:
“Couldn’t you just ask the LLM to estimate?”

Recovery:
“No. That would be fabricated precision. Either run an aggregation over structured data or say the corpus/index does not support that aggregate.”

---

### 5.11 Supervisor versus swarm

Question:
“When would you use supervisor-specialist routing versus a swarm/handoff architecture?”

Strong answer:
- Supervisor is better when routes are known, auditability matters, and enterprise policy wants centralized control.
- Swarm/handoff can be useful when tasks are open-ended and agent ownership naturally shifts.
- For healthcare/report analytics, supervisor is safer: predictable routing, clearer trace, easier access control.
- Can still use handoff pattern for specialist deep dives if traceability is maintained.

Push:
“Isn’t supervisor a bottleneck?”

Recovery:
“Yes, it can be. Mitigate with lightweight routing, async execution, and only invoke specialists when useful. Do not add agent hops if a direct tool call is cheaper and sufficient.”

---

### 5.12 Send map-reduce

Question:
“What problem does Send solve in LangGraph?”

Strong answer:
- Dynamic fan-out.
- Send creates parallel work items to the same node/subgraph with different state slices.
- Useful for map-reduce across reports/chunks.
- Reducers combine outputs.

Push:
“What can go wrong?”

Recovery:
“Reducer design, cost explosion, context window bloat, duplicate/contradictory summaries, and weak provenance. Need bounded fan-out, per-map citations, and reduce-stage faithfulness checks.”

---

### 5.13 Persistence, interrupt, time travel

Question:
“Explain checkpointing, interrupt, and time travel as if I am reviewing your production design.”

Strong answer:
- Checkpointer stores state per super-step per thread.
- Enables durable multi-turn execution and replay/debugging.
- interrupt pauses graph execution for human input or approval.
- Command(resume=...) resumes from the interrupted point.
- update_state/time travel can fork from a prior checkpoint for debugging or alternative decisions.

Push:
“Where would you place HITL?”

Recovery:
“Before irreversible or high-risk actions: file writes, external API mutations, generated-code freeze, sensitive data export, or stakeholder-facing synthesis.”

---

### 5.14 Observability

Question:
“What do you trace in an agentic system?”

Strong answer:
- Node inputs and outputs.
- Conditional edge decisions.
- Tool calls, args, return payload summaries, errors.
- Retrieved doc IDs/scores/rerank positions.
- Rewritten queries.
- Grader decisions and rationales if available.
- Token usage, latency, cost per node.
- Checkpoint/thread IDs.
- Final answer plus citations/refusal reason.

Push:
“Why not just log the final answer?”

Recovery:
“Because in agent systems the path is often the bug. Final-answer logging cannot reveal bad routing, unsafe tool calls, retry loops, retrieval misses, or grader false positives.”

---

### 5.15 Security and access control

Question:
“How is agentic security different from normal API security?”

Strong answer:
- A tool is a capability grant.
- The agent can compose tool calls in unexpected ways.
- Need least privilege, scoped tools, schema validation, allowlists, audit logs, rate limits, HITL gates, and environment isolation.
- Generated code/tools need AST allowlist and sandboxed testing before use.

Push:
“Isn’t a prompt saying ‘don’t do unsafe things’ enough?”

Recovery:
“No. Prompts are advisory. Permission boundaries must be enforced outside the model.”

---

### 5.16 MCP

Question:
“Why add MCP instead of binding Python functions directly?”

Strong answer:
- Direct binding is simpler in one Python runtime.
- MCP is useful when enterprise tools need a standard discovery/invocation protocol across clients and runtimes.
- It decouples client from tool implementation.
- Trade-off: network hop, schema/versioning, auth, deployment surface.
- Worth it for multi-team integration; not worth it for a local toy script.

Push:
“How would you secure MCP?”

Recovery:
“AuthN/AuthZ per tool, least privilege, audit logs, schema validation, network segmentation, no blanket DB credentials, and policy-enforced approval for high-risk tools.”

---

### 5.17 Reader versus orchestrator

Question:
“Why not use the fine-tuned raft-3B model as the orchestrator?”

Strong answer:
- It was trained/focused as a reader, not as a reliable arbitrary tool-calling orchestrator.
- Orchestrator needs structured output and robust tool routing under diverse user inputs.
- The reader slot is swappable behind OpenAI-compatible interface.
- Compare frontier vs raft-3B reader on same graph/eval set.

Push:
“Isn’t the fine-tune supposed to reduce hallucinations?”

Recovery:
“Yes, but model-level improvement does not replace system-level retrieval confidence, groundedness checks, refusal logic, and evals.”

---

### 5.18 rsLoRA merge gotcha

Question:
“You mention rsLoRA. What is the operational risk?”

Strong answer:
- The adapter was trained with rsLoRA at rank 64.
- Scaling is alpha/sqrt(r), not alpha/r.
- A manual merge using vanilla LoRA scaling under-scales by 8x for alpha=16, r=64.
- Use PEFT merge_and_unload and verify prompt-template parity/refusal behavior.

Push:
“So this is not just math trivia?”

Recovery:
“No. A wrong merge silently degrades the reader and can make evals look like architecture failure when it is a model-serving bug.”

---

### 5.19 Snowflake gap

Question:
“The JD asks for Snowflake. Where is it in your project?”

Strong answer:
- “I should be precise: I did not operate Snowflake in this build.”
- The project uses MongoDB locally and has transferable data-platform patterns.
- The Snowflake/Cortex path is designed/cost-modeled as Tier-3, not claimed as hands-on.
- I would frame Databricks/medallion ETL experience as transferable, not equivalent.

Push:
“Why should I believe you can handle Snowflake?”

Recovery:
“Because the core decisions transfer: schema design, data hydration, access control, query cost, lineage, and eval pipelines. But I would still ramp on Snowflake-specific operational details and not claim production experience I do not have.”

---

### 5.20 Jira/GitHub gap

Question:
“You used GitHub but not Jira. This JD mentions Jira. Is that a gap?”

Strong answer:
- Yes, Jira specifically is a gap.
- The transferable skill is Agile delivery: backlog, milestones, risk log, decision log, issues/PRs/commits.
- I would ramp quickly on Jira workflow conventions.
- Do not pretend GitHub Projects equals Jira experience if it is not true.

Push:
“What would you show leadership weekly?”

Recovery:
“Milestone status, risk register, eval trend, blockers, next decision, owner/dependency list, and quality gates passed/failed.”

---

### 5.21 AutoGen/CrewAI contrast

Question:
“The JD mentions LangGraph, AutoGen, and CrewAI. You built only LangGraph. Why?”

Strong answer:
- The goal was depth and defensible production mechanics, not shallow demos in three frameworks.
- LangGraph is strong for explicit stateful workflows, cycles, persistence, observability, and human-in-the-loop.
- CrewAI is often ergonomic for role-based collaborative agents.
- AutoGen is useful for conversational multi-agent patterns and experimentation.
- In enterprise, choose based on control, auditability, integration surface, and team skill.

Push:
“Would you reject AutoGen/CrewAI?”

Recovery:
“No. I would evaluate them against requirements. For this project’s durable, auditable workflow, LangGraph was the best fit.”

---

### 5.22 Documentation drift trap

Question:
“I see README status, handoff, and decisions may not agree about what is done. How do you handle that?”

Strong answer:
- Call it out directly.
- README status appears stale relative to DECISIONS/HANDOFF.
- Source of truth should be clarified: DECISIONS for design outcomes, PLAN for scope, README for public status, handoff for session delta.
- Fix by updating README checkboxes and adding dates/results to milestone entries.
- In an interview, do not overclaim; say what was built, run, measured, and what remains pending.

Push:
“Isn’t doc drift a leadership failure?”

Recovery:
“It is a process smell. The right response is not defensiveness; it is to assign source-of-truth ownership and close the loop immediately because leadership depends on trustworthy status.”

---

## 6. Stress scenarios for the rehearsal

Use these when the candidate gives a polished but shallow answer.

### Scenario A: The grader false-positive

“You shipped the agent. A customer reports a hallucinated answer that passed groundedness. What do you do in the next 24 hours?”

Expected answer:
- Pull trace: question, retrieved docs, rerank scores, generated answer, grader output.
- Label as grader false positive, retrieval issue, prompt issue, or source ambiguity.
- Add case to regression set.
- Check human-labeled anchor agreement.
- Adjust grader prompt or route; consider stricter constrained generation/refusal.
- Report incident and mitigation without overclaiming.

### Scenario B: Cost explosion

“Map-reduce synthesis over hundreds of reports suddenly costs 10x more.”

Expected answer:
- Bound fan-out, chunk budget, and model choice.
- Cache retrieval/summary outputs.
- Use smaller model for map stage if quality holds.
- Add budget guardrails and per-node cost tracing.
- Degrade gracefully: ask user to narrow scope or return sampled/partial synthesis.

### Scenario C: Security incident

“The agent generated a Mongo query that exposed more data than the user should see.”

Expected answer:
- Treat tool as capability grant.
- Enforce authz outside the model.
- Use scoped tools/views, row/field-level filters, schema allowlist.
- Add audit, alerts, test cases, and HITL for exports.
- Never rely only on prompt instructions.

### Scenario D: Leadership escalation

“Leadership asks why Snowflake is not done.”

Expected answer:
- Explain tiering: core role risk was LangGraph/evals/security; Snowflake was Tier-3 designed/cost-modeled.
- Provide integration plan, cost/risk estimate, and what decision is needed.
- Be honest about hands-on gap.

---

## 7. Fast drills, 30 seconds each

1. Define reducer.
2. Define super-step.
3. Explain why ReAct agent node runs twice.
4. Runtime grader vs offline eval.
5. RAGAS faithfulness in one sentence.
6. Why RAGAS is still LLM-as-judge.
7. Why BLEU is weak for grounded QA.
8. What trajectory eval catches that outcome eval misses.
9. What checkpoint stores.
10. Where to place interrupt.
11. Why tool = capability grant.
12. MCP vs direct tool binding.
13. Why raft-3B is reader, not orchestrator.
14. What to do with aggregate questions.
15. How to frame Snowflake gap.

---

## 8. Candidate “must-say” phrases

These are not memorized scripts; they are conceptual anchors.

- “The path is the bug in agent systems, not just the final answer.”
- “Runtime graders are control signals; offline evals are quality measurement.”
- “RAGAS reduces judge variance; it does not remove judge dependency.”
- “Top-k retrieval is not a corpus aggregate.”
- “A tool is a capability grant, not just a function call.”
- “The fine-tune is the reader, not the orchestrator.”
- “I will not fake Snowflake or Jira. I’ll frame transfer and ramp plan honestly.”
- “I prefer working + measured over gold-plated + unmeasured.”
- “Edge functions should be pure; side effects belong in nodes.”

---

## 9. Final interviewer decision template

At the end, fill this quickly:

- Architecture clarity: __ / 20
- LangGraph mechanics: __ / 20
- Evals/observability: __ / 20
- Enterprise security/integration: __ / 15
- Model/data/RAG judgment: __ / 15
- Leadership/gap handling: __ / 10
- Total: __ / 100

Decision:
- Strong pass / pass / pass with concerns / no pass

Evidence:
- Strongest signal:
- Weakest signal:
- One risk to remediate before real interview:
- One story to sharpen:

