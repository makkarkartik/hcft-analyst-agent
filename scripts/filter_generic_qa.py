"""Flag (and optionally drop) over-generic questions in the v3 eval set.

Synthetic QA generated from a chunk sometimes asks something the chunk can't *uniquely* answer —
"What is discussed in this section?", "What are the main findings?". Those are retrieval-ambiguous
(many chunks 'answer' them), so they punish retrieval metrics for a data problem, not a model one.
We strip them so hit@k / faithfulness reflect real questions.

Conservative on purpose — a question is dropped only if it (a) matches a generic TEMPLATE, or
(b) carries NO specific anchor (no number, year, proper noun, %, $, or quoted term) AND is short.
Unanswerable rows are NEVER dropped (vagueness is the point there). Dry-run by default; pass
``--write`` to emit the filtered set + a dropped sidecar with reasons.

    python scripts/filter_generic_qa.py                       # dry-run stats
    python scripts/filter_generic_qa.py --write               # write *_filtered.jsonl + *_dropped.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

V3 = Path(r"C:\Users\kartik\OMSCS\Personal Projects\SLM_Fine_Tuning\data\qa_eval_v3.jsonl")

# Generic-question templates — phrased to retrieve "anything on the topic", not a specific fact.
# Narrow on purpose: the topic/areas clause requires a discuss/cover verb so that a SPECIFIC
# "what areas are excluded from REMSA's service area" is NOT swept up.
GENERIC_RE = re.compile(
    r"\b(what (is|are) (discussed|covered|mentioned|included|presented|shown)"
    r"|what (does|do) (the|this) (report|document|section|profile|table|chapter|text) (say|cover|contain|discuss|describe|present|include)"
    r"|what (is|are) the (main|key|general|overall) (findings|points|topics|themes|takeaways|ideas)"
    r"|(give|provide) (a|an) (overview|summary)|summari[sz]e|describe (the|this) (section|document|report|profile)"
    r"|what (topics|subjects|areas) (are|is) (discussed|covered|included|mentioned|presented|addressed))\b",
    re.IGNORECASE,
)
# Quantitative INTENT — a concrete factual target even when no digit appears in the question.
QUANT_RE = re.compile(
    r"\b(what percentage|what (is|was) the (percentage|proportion|share|rate|number|count|total|population)"
    r"|how (many|much)|what (is|was) the population of)\b",
    re.IGNORECASE,
)
# A "specific anchor" — a number, symbol, year, quote, or proper noun / acronym.
ANCHOR_RE = re.compile(r"(\d|%|\$|\bpercent\b|\b(19|20)\d{2}\b|\"[^\"]+\"|'[^']+')")
PROPER_RE = re.compile(r"^([A-Z][a-z]{2,}|[A-Z]{2,})$")   # a single capitalized/acronym token
SHORT_TOKENS = 9   # questions at/under this many words with NO anchor at all are "thin"


def _has_anchor(q: str) -> bool:
    if ANCHOR_RE.search(q) or QUANT_RE.search(q):
        return True
    # any capitalized token AFTER the first word is a proper-noun anchor (Middletown, REMSA, Brooklyn)
    return any(PROPER_RE.match(t.strip(".,?'\"")) for t in q.split()[1:])


def classify(q: str) -> str | None:
    """Return a drop-reason string, or None to keep."""
    if GENERIC_RE.search(q):
        return "generic_template"
    if not _has_anchor(q) and len(q.split()) <= SHORT_TOKENS:
        return "no_anchor_and_short"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=Path, default=V3)
    ap.add_argument("--write", action="store_true", help="write filtered + dropped files (default dry-run)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.infile, encoding="utf-8")]
    kept, dropped = [], []
    for r in rows:
        # never drop unanswerable rows — vagueness is intentional there
        reason = None if r.get("eval_kind") == "unanswerable_rag" else classify(r["question"])
        (dropped if reason else kept).append((r, reason))

    by_reason: dict[str, int] = {}
    for _, reason in dropped:
        by_reason[reason] = by_reason.get(reason, 0) + 1

    print(f"[filter] {len(rows)} rows -> keep {len(kept)} · drop {len(dropped)}")
    for reason, n in sorted(by_reason.items(), key=lambda x: -x[1]):
        print(f"   drop[{reason}] = {n}")
    print("\n  sample drops:")
    for r, reason in dropped[:12]:
        print(f"   [{reason}] {r['question'][:88]}")

    if not args.write:
        print("\n  (dry-run; pass --write to emit files)")
        return

    keep_path = args.infile.with_name(args.infile.stem + "_filtered.jsonl")
    drop_path = args.infile.with_name(args.infile.stem + "_dropped.jsonl")
    with open(keep_path, "w", encoding="utf-8") as f:
        for r, _ in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(drop_path, "w", encoding="utf-8") as f:
        for r, reason in dropped:
            f.write(json.dumps({**r, "drop_reason": reason}, ensure_ascii=False) + "\n")
    print(f"\n  wrote {len(kept)} -> {keep_path}")
    print(f"  wrote {len(dropped)} -> {drop_path}")


if __name__ == "__main__":
    main()
