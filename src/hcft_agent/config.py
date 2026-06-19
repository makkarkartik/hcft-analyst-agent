"""Central settings. Everything configurable lives here; nothing reads os.environ elsewhere.

The reader/generation model is deliberately just (base_url, model, api_key) -- an
OpenAI-compatible endpoint. Swapping frontier -> local raft-3b (Ollama) is a .env change.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    # Data layer
    mongo_uri: str = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
    mongo_db: str = os.getenv("MONGO_DB", "hcft")
    chunks_collection: str = "chunks"
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "hcft")

    # Orchestrator / grader model (tool-capable, public)
    orchestrator_base_url: str = os.getenv("ORCHESTRATOR_BASE_URL", "https://api.openai.com/v1")
    orchestrator_model: str = os.getenv("ORCHESTRATOR_MODEL", "gpt-4o-mini")
    orchestrator_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Reader model (swappable slot: frontier now, raft-3b-r64-v2_2 via Ollama in M7)
    reader_base_url: str = os.getenv("READER_BASE_URL") or "https://api.openai.com/v1"
    reader_model: str = os.getenv("READER_MODEL") or "gpt-4o-mini"
    reader_api_key: str = os.getenv("READER_API_KEY") or os.getenv("OPENAI_API_KEY", "")

    # Retrieval stack (stage-06 parity — query embed must match the index that built `hcft`)
    embed_model: str = "Qwen/Qwen3-Embedding-4B"
    embed_dim: int = 768                # Matryoshka truncation; MUST match the Pinecone index
    embed_dtype: str = "bfloat16"
    embed_max_seq_length: int = 768
    embed_normalize: bool = True
    query_instruction: str = "Given a question, retrieve passages that answer it"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_max_length: int = 512
    dense_top_k: int = 50               # candidates pulled from Pinecone before rerank
    final_top_k: int = 10               # kept after rerank
    context_top_k: int = 5              # concatenated into the LLM prompt
    context_char_cap: int = 1800        # per-chunk char cap when assembling context

    # --- RAG chat agent: deterministic gates (no gold at inference) ---
    # grade gate: top reranked candidate must clear this score to be "answerable".
    # PLACEHOLDER until calibrated empirically from the gold-hit vs non-gold score
    # distribution (`eval.retrieval --calibrate`). BGE-reranker-v2-m3 emits raw logits.
    grade_min_rerank_score: float = 0.0
    max_rewrites: int = 2               # query-reformulation attempts before refusing
    gen_temperature: float = 0.0        # deterministic generation

    # --- output groundedness guard (inline, fast, deterministic) ---
    # Vectara HHEM-2.1-open cross-encoder: P(answer is grounded in context), 0..1.
    hhem_model: str = "vectara/hallucination_evaluation_model"
    grounded_min_score: float = 0.5     # below -> refuse rather than risk fabrication


settings = Settings()
