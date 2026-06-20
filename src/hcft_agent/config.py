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

    # Retrieval mode: "dense" (Pinecone only) or "hybrid" (dense + Mongo $text BM25, fused by RRF).
    # Hybrid catches exact terms (figures, named entities) that dense embeddings miss. Default
    # hybrid — measured A/B on v3 (scripts/retrieval_ab.py): recall@50 +0.10, hit@5 +0.10.
    retrieval_mode: str = os.getenv("RETRIEVAL_MODE") or "hybrid"
    sparse_top_k: int = 50              # candidates from the lexical (BM25) arm before fusion
    rrf_k: int = 60                     # Reciprocal Rank Fusion constant (standard default)

    # --- RAG chat agent: deterministic gates (no gold at inference) ---
    # grade gate: top reranked candidate must clear this score to be "answerable".
    # Calibrated 2026-06-19 (`eval.retrieval --calibrate`) -> rerank_score is a WEAK gate, two
    # reasons: (1) BGE sigmoid scores SATURATE near 1.0 (gold median 1.00 vs unanswerable-top
    # 0.96, overlap); (2) calibration positives are synthetic QA authored FROM the gold chunk,
    # so they score ~1.0 while a realistic free-text query's top chunk scores ~0.24 -> a 0.5
    # floor over-refuses real queries. Decision: CATASTROPHIC-ONLY floor here (reject empty /
    # garbage retrieval); the real refuse decision belongs to the HHEM output guard + generator
    # refusal, which see the actual answer-vs-context. (See ARCHITECTURE.md grade-gate note.)
    grade_min_rerank_score: float = 0.05
    max_rewrites: int = 2               # query-reformulation attempts before refusing
    gen_temperature: float = 0.0        # deterministic generation

    # --- output groundedness guard (inline, fast, deterministic) ---
    # Vectara HHEM-2.1-open cross-encoder: P(answer is grounded in context), 0..1.
    hhem_model: str = "vectara/hallucination_evaluation_model"
    grounded_min_score: float = 0.5     # below -> refuse rather than risk fabrication

    # --- input ring: prompt-injection classifier ---
    # Open, non-gated default. Swap to meta-llama/Llama-Prompt-Guard-2-86M by setting
    # INJECTION_MODEL + accepting the Meta license + HF_TOKEN (that model is gated).
    injection_model: str = os.getenv("INJECTION_MODEL") or "protectai/deberta-v3-base-prompt-injection-v2"
    injection_threshold: float = 0.5

    # --- context ring: indirect-injection scan on RETRIEVED chunks (untrusted input) ---
    # Retrieved chunks can carry "ignore your instructions ..." into the reader. We scan each chunk
    # that could enter the context window with the SAME injection classifier and QUARANTINE any
    # that clear this threshold (kept higher than the input threshold: document prose triggers more
    # false positives than a short user query, and the eval tracks that FP rate).
    context_injection_threshold: float = float(os.getenv("CONTEXT_INJECTION_THRESHOLD") or 0.8)

    # --- input ring: PII detection + redaction (Microsoft Presidio) ---
    # Redact only UNAMBIGUOUS identifiers — NOT PERSON/LOCATION/DATE, which are often legitimate
    # query content (e.g. "hospitals in California") and whose redaction would wreck retrieval.
    pii_score_threshold: float = 0.5
    pii_redact_entities: tuple = (
        "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD", "US_BANK_NUMBER",
        "IBAN_CODE", "US_DRIVER_LICENSE", "MEDICAL_LICENSE", "IP_ADDRESS", "CRYPTO",
    )

    # === Offline LLM-judge eval stack (DeepEval G-Eval + RAGAS cross-family + κ validation) ===
    # Two judges on PURPOSE-different model families so a same-model rubber-stamp can't pass twice:
    #   * G-Eval (DeepEval)  -> OpenAI gpt-4o-mini — SAME family as the reader, so it's the strict
    #     "gate" judge and the one we then VALIDATE (a same-family judge is the circularity risk).
    #   * RAGAS faithfulness/answer-relevancy -> Fireworks Llama-3.3-70B — a DIFFERENT family, the
    #     cross-family second opinion that catches what a same-family judge would wave through.
    # The actual circularity-breaker is κ against a DETERMINISTIC anchor (gold, non-LLM) below.
    geval_judge_model: str = os.getenv("GEVAL_JUDGE_MODEL") or "gpt-4o-mini"
    # RAGAS judge = the only NON-CHINESE serverless LLM this Fireworks key reaches: OpenAI's open-
    # weight gpt-oss-120b (Apache-2.0 MoE). It's a DIFFERENT model + training recipe from the
    # gpt-4o-mini reader/G-Eval (cross-MODEL, served off a different stack), though same vendor
    # lineage — so weaker than a true cross-vendor judge (Gemma/Nemotron aren't on this key). The
    # κ deterministic anchor is what actually breaks circularity; this adds a structured
    # claim-decomposition+NLI second opinion. Verified: supported→1.0, hallucinated→0.0.
    ragas_judge_model: str = (
        os.getenv("RAGAS_JUDGE_MODEL") or "accounts/fireworks/models/gpt-oss-120b"
    )
    ragas_judge_base_url: str = os.getenv("RAGAS_JUDGE_BASE_URL") or "https://api.fireworks.ai/inference/v1"
    ragas_judge_api_key: str = os.getenv("RAGAS_JUDGE_API_KEY") or os.getenv("FIREWORKS_API_KEY", "")
    # Answer-relevancy needs an embedder; OpenAI's small model keeps it cheap + dependency-free.
    ragas_embed_model: str = os.getenv("RAGAS_EMBED_MODEL") or "text-embedding-3-small"
    geval_threshold: float = 0.5        # G-Eval score ≥ this -> judge says "appropriate"
    # κ judge-validation: deterministic anchor by default. Drop a human-labeled JSONL here
    # (one {"qa_id":..., "appropriate": bool} per line) to upgrade to a true HUMAN κ.
    kappa_human_labels: str = os.getenv("KAPPA_HUMAN_LABELS") or ""


settings = Settings()
