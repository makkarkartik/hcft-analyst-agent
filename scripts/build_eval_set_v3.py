"""Generate a diverse, leakage-safe QA eval set (~2000 pairs) from the HCFT corpus.

Bigger AND better than qa_v2 (which was single-chunk, synthetic-from-chunk, only 12
unanswerable). This set:
  * draws ONLY from split='test' + document_type='STRONG_hcft' chunks -> leakage-safe for the
    raft-3b fine-tune (trained on 'train') and clean for the gpt-4o-mini reader;
  * mixes single-hop / multi-hop / unanswerable so retrieval depth AND refusal are both testable;
  * runs a verification pass (a 2nd LLM call) — grounded answers must be entailed by their
    context; "unanswerable" must be genuinely unanswerable — and drops failures;
  * logs gen_model so the eval can judge with a DIFFERENT family (anti-circularity).

Concurrent (ThreadPoolExecutor) + resumable (append-checkpoint to JSONL; reruns skip done keys).

    python scripts/build_eval_set_v3.py                      # full run, defaults
    python scripts/build_eval_set_v3.py --single 20 --multi 6 --unanswerable 6 --workers 8   # smoke
"""
from __future__ import annotations

import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from pymongo import MongoClient

from hcft_agent.config import settings

OUT_DEFAULT = Path(
    r"C:\Users\kartik\OMSCS\Personal Projects\SLM_Fine_Tuning\data\qa_eval_v3.jsonl"
)
POOL_FILTER = {"split": "test", "document_type": "STRONG_hcft", "n_tokens": {"$gte": 120}}
GEN_MODEL = settings.orchestrator_model           # gpt-4o-mini
CHUNK_CAP = 2200                                   # chars of chunk text fed to the generator

_client = OpenAI(api_key=settings.orchestrator_api_key, base_url=settings.orchestrator_base_url)
_write_lock = threading.Lock()


# --------------------------------------------------------------------- LLM helpers
def _chat_json(system: str, user: str) -> dict | None:
    """One JSON-mode completion; returns parsed dict or None on failure."""
    try:
        r = _client.chat.completions.create(
            model=GEN_MODEL, temperature=0.7, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return json.loads(r.choices[0].message.content)
    except Exception:
        return None


_GEN_SINGLE = (
    "You write reading-comprehension QA from an excerpt of a U.S. community health / hospital "
    "report. Produce ONE specific question answerable SOLELY from the excerpt, plus a concise, "
    "accurate answer grounded ONLY in it. No yes/no questions; prefer questions needing a real "
    'detail. Reply JSON: {"question": str, "answer": str, "difficulty": "easy|medium|hard"}.'
)
_GEN_MULTI = (
    "You write a multi-hop QA pair from TWO excerpts of the SAME report. Produce ONE question "
    "that genuinely REQUIRES information from BOTH excerpts (not answerable from either alone), "
    'plus a concise answer grounded in both. Reply JSON: {"question": str, "answer": str, '
    '"difficulty": "easy|medium|hard"}.'
)
_GEN_UNANS = (
    "You write an UNANSWERABLE test question. Given an excerpt, produce ONE realistic, on-topic "
    "question a reader might plausibly ask BUT whose answer is NOT present in the excerpt (the "
    'excerpt lacks the needed information). Reply JSON: {"question": str}.'
)
_VERIFY_GROUNDED = (
    "Judge a QA pair against its source excerpt. Is the QUESTION answerable from the excerpt AND "
    'is the ANSWER fully supported by it (no outside facts)? Reply JSON: {"ok": true|false}.'
)
_VERIFY_UNANS = (
    "Judge whether a question is genuinely NOT answerable from the excerpt (the excerpt lacks the "
    'needed information). Reply JSON: {"unanswerable": true|false}.'
)


# --------------------------------------------------------------------- per-item work
def _make_single(chunk: dict) -> dict | None:
    text = (chunk["text"] or "")[:CHUNK_CAP]
    g = _chat_json(_GEN_SINGLE, f"Excerpt:\n{text}")
    if not g or not g.get("question") or not g.get("answer"):
        return None
    v = _chat_json(_VERIFY_GROUNDED, f"Excerpt:\n{text}\n\nQ: {g['question']}\nA: {g['answer']}")
    if not v or not v.get("ok"):
        return None
    return _row(g["question"], g["answer"], [chunk["chunk_id"]], "grounded_rag", "single",
                chunk, difficulty=g.get("difficulty"))


def _make_multi(pair: tuple[dict, dict]) -> dict | None:
    a, b = pair
    ta, tb = (a["text"] or "")[:CHUNK_CAP], (b["text"] or "")[:CHUNK_CAP]
    g = _chat_json(_GEN_MULTI, f"Excerpt 1:\n{ta}\n\nExcerpt 2:\n{tb}")
    if not g or not g.get("question") or not g.get("answer"):
        return None
    v = _chat_json(_VERIFY_GROUNDED,
                   f"Excerpt:\n{ta}\n\n{tb}\n\nQ: {g['question']}\nA: {g['answer']}")
    if not v or not v.get("ok"):
        return None
    return _row(g["question"], g["answer"], [a["chunk_id"], b["chunk_id"]], "grounded_rag",
                "multi", a, difficulty=g.get("difficulty"))


def _make_unans(chunk: dict) -> dict | None:
    text = (chunk["text"] or "")[:CHUNK_CAP]
    g = _chat_json(_GEN_UNANS, f"Excerpt:\n{text}")
    if not g or not g.get("question"):
        return None
    v = _chat_json(_VERIFY_UNANS, f"Excerpt:\n{text}\n\nQ: {g['question']}")
    if not v or not v.get("unanswerable"):
        return None
    return _row(g["question"], "", [chunk["chunk_id"]], "unanswerable_rag", "none", chunk)


def _row(question, answer, chunk_ids, eval_kind, hop_type, chunk, difficulty=None) -> dict:
    return {
        "qa_id": f"v3_{chunk_ids[0]}_{hop_type}",
        "question": question, "answer": answer,
        "source_chunk_id": chunk_ids[0], "source_chunk_ids": chunk_ids,
        "eval_kind": eval_kind, "hop_type": hop_type, "difficulty": difficulty,
        "gen_model": GEN_MODEL, "doc_id": chunk.get("doc_id"),
        "state": chunk.get("state"), "document_type": chunk.get("document_type"),
    }


def _append(out_path: Path, row: dict) -> None:
    with _write_lock:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------- orchestration
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", type=int, default=1350)
    ap.add_argument("--multi", type=int, default=450)
    ap.add_argument("--unanswerable", type=int, default=480)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # resume: keys already produced
    done: set[str] = set()
    if args.out.exists():
        for line in open(args.out, encoding="utf-8"):
            try:
                done.add(json.loads(line)["qa_id"])
            except Exception:
                pass
    print(f"[build] resuming — {len(done)} rows already present")

    coll = MongoClient(settings.mongo_uri)[settings.mongo_db][settings.chunks_collection]
    pool = list(coll.find(POOL_FILTER, {"chunk_id": 1, "doc_id": 1, "state": 1, "document_type": 1}))
    random.seed(args.seed)
    random.shuffle(pool)
    print(f"[build] eligible pool: {len(pool)} chunks")

    # partition the pool: single | unanswerable | multi (pairs within a doc)
    single_src = pool[: args.single]
    unans_src = pool[args.single: args.single + args.unanswerable]
    rest = pool[args.single + args.unanswerable:]
    by_doc: dict[str, list] = {}
    for c in rest:
        by_doc.setdefault(c["doc_id"], []).append(c)
    multi_pairs = []
    for chunks in by_doc.values():
        for i in range(0, len(chunks) - 1, 2):
            multi_pairs.append((chunks[i], chunks[i + 1]))
            if len(multi_pairs) >= args.multi:
                break
        if len(multi_pairs) >= args.multi:
            break

    # hydrate text for every selected chunk
    need = {c["chunk_id"] for c in single_src + unans_src}
    for a, b in multi_pairs:
        need.add(a["chunk_id"]); need.add(b["chunk_id"])
    text_by_id = {d["_id"]: d.get("text", "")
                  for d in coll.find({"_id": {"$in": list(need)}}, {"text": 1})}
    for c in pool:
        c["text"] = text_by_id.get(c["chunk_id"], "")

    # build the task list, skipping resumed keys
    tasks = []
    for c in single_src:
        if f"v3_{c['chunk_id']}_single" not in done:
            tasks.append(("single", _make_single, c))
    for c in unans_src:
        if f"v3_{c['chunk_id']}_none" not in done:
            tasks.append(("unans", _make_unans, c))
    for a, b in multi_pairs:
        if f"v3_{a['chunk_id']}_multi" not in done:
            tasks.append(("multi", _make_multi, (a, b)))
    print(f"[build] {len(tasks)} new tasks "
          f"(single={args.single} unans={args.unanswerable} multi={len(multi_pairs)}) · {args.workers} workers")

    kept = {"single": 0, "multi": 0, "unans": 0}
    dropped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fn, item): kind for kind, fn, item in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            kind = futs[fut]
            row = fut.result()
            if row:
                _append(args.out, row)
                kept[kind] += 1
            else:
                dropped += 1
            if i % 50 == 0:
                tot = sum(kept.values())
                print(f"  ...{i}/{len(tasks)} done · kept {tot} (S{kept['single']} "
                      f"M{kept['multi']} U{kept['unans']}) · dropped {dropped}")

    total = sum(kept.values()) + len(done)
    print(f"\n[build] DONE — kept this run {sum(kept.values())} "
          f"(single {kept['single']}, multi {kept['multi']}, unans {kept['unans']}), "
          f"dropped {dropped}. Total in file: {total} -> {args.out}")


if __name__ == "__main__":
    main()
