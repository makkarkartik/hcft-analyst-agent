# JD.md — target role (source of truth for the coverage map)

**Role:** Senior Technical Lead for Autonomous Agentic Frameworks and System Integration (Cisco CX)

> Transcribed verbatim from the posting image (2026-06-13). The intro paragraph is partial in the
> source (begins mid-sentence); captured as shown.

## Verbatim posting

> ...constantly evolving market. Pioneers in our field, CX enables customers to maximize value with
> a simple, unified Cisco experience, driving accelerated time to impact. We deliver on time with
> high quality & urgency, and we drive decisions with data and transparency while leading with
> ownership & accountability.

**Your Impact**

We are seeking a motivated and detail-oriented Software Engineering Technical Lead to join our
growing Agent Harness Engineering team. In this role, you will not only help build persistent,
goal-oriented agent systems that "think" and "act" autonomously but also lead the strategic
planning and execution of the agentic roadmap. You will bridge the gap between technical innovation
and project delivery, ensuring that our autonomous systems are built on time, align with
cross-functional milestones, and integrate seamlessly into the broader ecosystem.

Key responsibilities include designing, building, and deploying autonomous agents that operate
within dynamic environments, making independent decisions to achieve goals. You will lead the
end-to-end lifecycle of agentic projects, defining project scopes, breaking down complex technical
requirements into actionable tasks, and managing the team's backlog to ensure high-velocity
delivery. Implementing multi-agent, collaborative systems and designing complex agent workflows
using frameworks like LangGraph, AutoGen, or CrewAI will be part of your role. You will act as the
primary technical point of contact to coordinate schedules and dependencies across multiple teams,
ensuring that agentic capabilities are synchronized with organizational priorities. Developing and
integrating tools, APIs, and databases for agents to interact with, enabling real-world actions, is
essential. You will optimize large language model (LLM) interactions, including prompting,
fine-tuning, and context management to enhance reasoning and reduce hallucinations. Providing
regular status updates and risk assessments to leadership regarding project timelines and technical
blockers is expected. Additionally, you will collaborate with engineering teams to integrate
autonomous capabilities into existing software development life cycles (SDLC), championing Agile
methodologies and best practices for task tracking and documentation.

**Minimum Qualifications:**
- Bachelors + 8yrs of related experience OR Masters + 6 years of related experience.
- Experience in designing, developing, and deploying AI agents, agentic systems and/or agentic orchestration patterns.
- Experience with evals driven development and observability for non-deterministic systems.
- Experience developing code in Python.
- Experience with frameworks such as LangChain or LangGraph.
- Experience with cloud-based data platforms like SnowFlake.
- Experience with asynchronous programming (multi-turn agents or multi-step agents or similar).
- MCP or tool-integration patterns for enterprise systems.
- Experience using Jira and GitHub.

**Preferred Qualifications:**
- Demonstrated leadership skills with the ability to collaborate across teams.
- Ability to lead projects, organize and manage deliverables.
- Experience working through new features and prioritizing execution.
- Strong communication skills with both technical and non-technical stakeholders.
- Proactive "Go Getters" comfortable in fast-paced environments.
- Experience building applications with frontier models and open-source LLM Ecosystems.
- Strong understanding of large language models (LLMs), including how they work and how to train and focus them on specific areas.
- Experience with natural language processing and prompt engineering.
- Experience with enterprise security and agentic access control patterns.
- Experience with using AI Development Tools like Claude Code, Codex, Snowflake Cortex or similar.

## Coverage reconciliation (verbatim → project)

**Minimum quals:**
| JD requirement | Covered by | Status |
|---|---|---|
| Design/develop/deploy AI agents & orchestration patterns | M2–M4 | ✅ built |
| Evals-driven development + observability for non-deterministic systems | M5 (tracing) + M6 (trajectory/RAGAS/regression gate) | ⬆️ elevated to primary |
| Python | throughout | ✅ |
| LangChain / LangGraph | throughout | ✅ |
| Cloud data platform (Snowflake) | bridge from Databricks (SLM stage-02) | ⚠️ Tier-3 designed & cost-modeled, NOT operated |
| Asynchronous programming (multi-step agents) | async graphs from M2 | ⬆️ |
| MCP / tool-integration for enterprise | M8 | ✅ built (basic) |
| Jira & GitHub | GitHub used; Agile/SDLC framed in delivery docs | ⚠️ Jira not used — concede or frame |

**Preferred quals:**
| JD requirement | Covered by | Status |
|---|---|---|
| Leadership / collaborate across teams | PLAN=roadmap, DECISIONS=risk log | ✅ framed |
| Lead projects, manage deliverables | milestone delivery + handoffs | ✅ framed |
| Prioritize execution / new features | honesty-ladder tiering in PLAN | ✅ |
| Communicate to technical + non-technical stakeholders | stakeholder summary artifact (planned) | ⚠️ to produce |
| Fast-paced "go getter" | n/a (soft) | — |
| Frontier + open-source LLM ecosystems | M9 frontier reader vs raft-3b | ✅ built |
| Understand LLMs / train / focus on areas | SLM RAFT fine-tune project | ✅ strong evidence |
| NLP & prompt engineering | per-node system prompts (M2+) | ✅ |
| Enterprise security & agentic access control | M7 (tool perm scoping, AST allowlist + sandbox, HITL approval) | ⬆️ built |
| AI dev tools (Claude Code, Codex, Snowflake Cortex) | Cursor used to build; Cortex in Tier-3 design | ✅ / ⚠️ |

## Gaps to address or honestly concede
1. **Snowflake** — real gap. Mitigation: transferable Databricks medallion ETL experience +
   architected Snowflake/Cortex path with a cost model. Frame as "platform-transferable," don't fake.
2. **Jira** — not used (GitHub only). Concede plainly; emphasize Agile/SDLC + backlog/risk framing.
3. **Stakeholder communication artifact** — produce a one-page non-technical status/risk summary
   to evidence the "communicate to non-technical stakeholders" + "status updates to leadership" lines.
4. **AutoGen / CrewAI** — JD lists them alongside LangGraph; we build LangGraph only. Be ready to
   contrast the three verbally (when to pick which) rather than building in all three.
5. **Years of experience** (8yr/BS or 6yr/MS) — fixed bar; out of scope for the project to change.
