# HCFT Analyst Agent

A LangGraph multi-agent system over ~6,000 public U.S. healthcare reports (519,555 chunks),
built as the orchestration + serving layer for a QLoRA fine-tuned Llama-3.2-3B reader from the
sibling **SLM_Fine_Tuning** (HCFT) project.

**What it does** (target state):

1. **Grounded Q&A with self-correction** — retrieve (Pinecone + BGE rerank), grade the evidence,
   rewrite the query and retry on a miss, generate a cited answer, refuse cleanly when the
   corpus doesn't contain it.
2. **Analytics questions** a vector index can't answer — routed to an agent that runs MongoDB
   aggregations over report metadata (counts by state/hospital/year, etc.).
3. **Multi-report synthesis briefs** — map-reduce summarization across reports, with a
   human-in-the-loop approval gate before anything is written to disk.

Threads are persistent (MongoDB checkpointer), responses stream token-by-token, and the
generation ("reader") slot is swappable behind an OpenAI-compatible interface: public frontier
model first, the fine-tuned `raft-3b-r64-v2_2` adapter (served via Ollama) later — the headline
eval compares the two in the same slot on groundedness, refusal accuracy, latency, and cost.

## Status

- [x] Local MongoDB (Docker) + chunk migration — **519,555 docs** in `hcft.chunks`, indexed for
      retrieval hydration (`_id` = chunk_id) and analytics (`state+hospital+year`, `document_type`, `split`)
- [ ] M0 hello graph (state, reducers, conditional edge)
- [ ] M1 ReAct loop from scratch + retriever tool
- [ ] M2 self-corrective RAG subgraph
- [ ] M3 persistence (Sqlite → Mongo checkpointer), interrupts, time travel
- [ ] M4 supervisor multi-agent + Send map-reduce
- [ ] M5 streaming + observability
- [ ] M6 eval + hardening
- [ ] raft-3b reader swap (merge → GGUF → Ollama) + comparison eval

## Stack

| Component | Choice | Notes |
|---|---|---|
| Orchestration | LangGraph | all graph code hand-written (see `DECISIONS.md`) |
| Vectors | Pinecone `hcft` (768-dim, cosine) | reused from HCFT stage 02c; vectors only, no text |
| Rerank | BAAI/bge-reranker-v2-m3 | reused from HCFT stage 06 |
| Text + metadata + checkpoints | MongoDB 7 (Docker, `hcft-mongo` :27017) | replaces HCFT's sqlite text store in this repo |
| Reader (phase 1) | public model via OpenAI-compatible API | Fireworks / gpt-4o-mini |
| Reader (phase 2) | `raft-3b-r64-v2_2` merged + GGUF via Ollama | see lineage note below |

## Quickstart

```powershell
docker compose up -d                                  # MongoDB on 127.0.0.1:27017
.venv\Scripts\python.exe scripts\migrate_chunks.py --drop   # load chunks from ../SLM_Fine_Tuning
.venv\Scripts\python.exe scripts\verify_mongo.py            # sanity checks
```

Requires the sibling `SLM_Fine_Tuning` repo for `data/chunks/*.jsonl` (migration source) and
Pinecone credentials in `.env` for live retrieval.

## Reader model lineage (read before serving the fine-tune)

The reader adapter is **`raft-3b-r64-v2_2`** — QLoRA (4-bit NF4) on Llama-3.2-3B-Instruct,
trained in the HCFT repo (`src/04_train_qlora.py`, frozen 2026-06-02).

**rsLoRA conditionality — important for any merge math.** The HCFT trainer enables rsLoRA
*conditionally*: `config.yaml` sets `use_rslora: true`, but the code applies it **only at
rank ≥ 16** (`use_rslora = bool(cfg.lora.use_rslora and rank >= 16)`). Consequences:

- The deliverable r=64 adapter **was trained with rsLoRA**, i.e. update scale = α/√r = 16/√64
  = **2.0**, not the vanilla α/r = 16/64 = 0.25 — an **8× difference**.
- The r=8 ablation arm was vanilla LoRA, so the rank ablation is not a pure rank comparison
  across the r=8 ↔ r≥16 boundary (scaling scheme changes too).
- **When merging for GGUF/Ollama, use PEFT's `merge_and_unload()`**, which reads `use_rslora`
  from `adapter_config.json` and applies the correct scaling. Hand-rolled merge math that
  assumes α/r will under-scale the adapter by 8× and silently degrade the model.
- Verify after merge: same prompt template as training, and spot-check refusal behavior on
  known distractor questions before trusting the endpoint.

## Relationship to the HCFT (SLM_Fine_Tuning) project

This repo executes the "LangGraph graph" item from HCFT's designed-not-built Phase-2 list and
implements HCFT's designed fix for its known model-level refusal weakness (an **external
retrieval-confidence gate** — the document-grading and groundedness nodes here). Serving via
Ollama is a deliberate divergence from the original vLLM LoRA hot-load plan: vLLM doesn't run
natively on Windows, and the architecturally relevant property (OpenAI-compatible boundary in
front of the fine-tune) is preserved. vLLM remains architected/cost-modeled, not operated.
