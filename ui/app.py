"""Streamlit test harness for the RAG chat agent — an X-ray of the pipeline.

Shows each stage the way the agent sees it, then streams the answer:
  input guard  ->  dense retrieval  ->  rerank (with movement)  ->  context window  ->  streamed generation  ->  groundedness

Reuses the SAME components as ``agents/rag_chat.py`` (retriever singleton, input ring, the
shared generation prompt, the HHEM guard) — what you see is what the agent does, not a second
code path. Run:  ``streamlit run ui/app.py``
"""
from __future__ import annotations

import html
import time

import streamlit as st

from hcft_agent import generate as gen
from hcft_agent.agents import rag_chat
from hcft_agent.config import settings
from hcft_agent.guards import input_ring
from hcft_agent.obs.telemetry import flush, init_telemetry, trace_block

st.set_page_config(page_title="HCFT RAG — pipeline X-ray", layout="wide", page_icon="🔎")
init_telemetry("hcft-agent")  # idempotent; exports UI runs (and their sub-runs) to LangSmith

# --------------------------------------------------------------------------- styling
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');
      html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea {
        font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif !important;
        -webkit-font-smoothing: antialiased;
      }
      .block-container { padding-top: 2.2rem; max-width: 1200px; }
      h1, h2, h3 { letter-spacing: -0.01em; }

      .hero-title { font-size: 1.7rem; font-weight: 700; color: #141a22; margin-bottom: .15rem; }
      .hero-sub   { color: #5b6573; font-size: .95rem; margin-bottom: 1.4rem; }

      .stage { font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em;
               color: #8a94a3; margin: 1.5rem 0 .5rem; border-bottom: 1px solid #eef1f6; padding-bottom: .35rem; }

      /* badges */
      .badge { display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: .8rem;
               font-weight: 600; margin-right: 6px; }
      .b-ok   { background: #e7f6ec; color: #1a7f43; }
      .b-warn { background: #fdf3e3; color: #a86a12; }
      .b-stop { background: #fdeaea; color: #b62828; }
      .b-info { background: #eef1f6; color: #4a5568; }

      /* chunk cards */
      .chunk { background: #ffffff; border: 1px solid #e7ebf2; border-radius: 12px;
               padding: 14px 16px; margin-bottom: 12px; box-shadow: 0 1px 2px rgba(20,30,50,.04); }
      .chunk.win { border-left: 4px solid #3b6fe0; }
      .chunk-head { display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
                    font-size: .82rem; color: #5b6573; margin-bottom: 8px; }
      .chunk-src { font-weight: 700; color: #3b6fe0; }
      .chunk-meta { background: #f4f6fb; border-radius: 6px; padding: 1px 8px; color: #51607a; font-size: .78rem; }
      .chunk-body { font-size: .92rem; line-height: 1.6; color: #2b333f; }

      /* answer */
      .answer-box { max-width: 760px; font-family: 'Source Serif 4', Georgia, serif !important;
                    font-size: 1.1rem; line-height: 1.8; color: #1d242e; background: #fbfcfe;
                    border: 1px solid #e7ebf2; border-radius: 14px; padding: 22px 26px; }
      .answer-box .src { color: #3b6fe0; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading retriever (embedder + reranker)…")
def get_retriever():
    r = rag_chat._retriever()
    try:  # warm up: TLS handshake to Pinecone + CUDA kernel JIT, off the first real query
        r.candidates("warmup")
    except Exception:
        pass
    return r


@st.cache_resource(show_spinner="Loading groundedness guard (HHEM)…")
def get_guard():
    g = rag_chat._guard()
    try:
        g.score("warmup context", "warmup answer")
    except Exception:
        pass
    return g


def _short(cid: str, n: int = 18) -> str:
    return cid if len(cid) <= n else cid[:n] + "…"


def _meta_bits(c: dict) -> str:
    bits = []
    for key in ("hospital", "state", "year"):
        v = c.get(key)
        if v:
            bits.append(f"<span class='chunk-meta'>{html.escape(str(v))}</span>")
    if c.get("page_num") is not None:
        bits.append(f"<span class='chunk-meta'>p.{c['page_num']}</span>")
    return " ".join(bits)


def chunk_card(idx: int, c: dict, in_window: bool, score_label: str) -> str:
    body = html.escape((c.get("text") or "")[: settings.context_char_cap])
    star = "★ " if in_window else ""
    return (
        f"<div class='chunk{' win' if in_window else ''}'>"
        f"<div class='chunk-head'><span class='chunk-src'>{star}Source {idx}</span>"
        f"{_meta_bits(c)}<span class='chunk-meta'>{score_label}</span>"
        f"<span class='chunk-meta'>{_short(c['chunk_id'], 34)}</span></div>"
        f"<div class='chunk-body'>{body}</div></div>"
    )


# --------------------------------------------------------------------------- header
st.markdown("<div class='hero-title'>HCFT RAG · pipeline X-ray</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-sub'>Retrieve → rerank → filter → stream, with input &amp; output guards — "
    "the agent's own view of each step.</div>",
    unsafe_allow_html=True,
)

q = st.text_input(
    "Ask the corpus",
    "What is the purpose of Hannibal Regional Healthcare System's 2019 feedback survey?",
    label_visibility="collapsed",
)
go = st.button("Run", type="primary")

# --------------------------------------------------------------------------- run
if go and q.strip():
    _cm = trace_block("ui.rag_run", run_type="chain", inputs={"question": q})
    _cm.__enter__()  # sub-runs (retriever stages, ChatOpenAI, HHEM) nest under this run

    # 1. input ring -------------------------------------------------------
    flags = input_ring.scan(q)
    st.markdown("<div class='stage'>Input guard</div>", unsafe_allow_html=True)
    if "injection" in flags:
        st.markdown(f"<span class='badge b-stop'>🛑 injection blocked</span>"
                    f"<span class='badge b-info'>{flags}</span>", unsafe_allow_html=True)
        st.info("Fail-closed: refused before retrieval (prompt-injection detected).")
        _cm.__exit__(None, None, None); flush()
        st.stop()
    elif flags:
        st.markdown(f"<span class='badge b-warn'>⚠️ {', '.join(flags)}</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='badge b-ok'>✓ clean</span>", unsafe_allow_html=True)

    r, guard = get_retriever(), get_guard()
    with st.spinner("Retrieving + reranking…"):
        cands = r.candidates(q)

    dense_rank = {c["chunk_id"]: i + 1 for i, c in enumerate(cands)}     # dense order = as returned
    rerank_order = sorted(cands, key=lambda c: c.get("rerank_score", 0.0), reverse=True)
    window = rerank_order[: settings.context_top_k]
    window_ids = {c["chunk_id"] for c in window}

    # 2/3. rerank movement table -----------------------------------------
    st.markdown("<div class='stage'>Retrieval · reranked order (and how each chunk moved)</div>",
                unsafe_allow_html=True)
    rows = []
    for i, c in enumerate(rerank_order[: settings.final_top_k], 1):
        dr = dense_rank[c["chunk_id"]]
        rows.append({
            "in context": c["chunk_id"] in window_ids,
            "rerank #": i,
            "dense #": dr,
            "moved": dr - i,                       # +ve = promoted by the reranker
            "rerank score": float(c.get("rerank_score", 0.0)),
            "dense score": float(c["dense_score"]),
            "hospital": c.get("hospital") or "—",
            "preview": (c.get("text") or "").replace("\n", " ")[:90],
        })
    st.dataframe(
        rows, hide_index=True, use_container_width=True,
        column_config={
            "in context": st.column_config.CheckboxColumn("ctx", help="filtered into the LLM context window", width="small"),
            "rerank #": st.column_config.NumberColumn(width="small"),
            "dense #": st.column_config.NumberColumn(width="small"),
            "moved": st.column_config.NumberColumn("Δrank", help="dense rank − rerank rank (+ = promoted)", format="%+d", width="small"),
            "rerank score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
            "dense score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
            "preview": st.column_config.TextColumn(width="large"),
        },
    )

    top_rr = window[0].get("rerank_score", 0.0) if window else 0.0
    relevant = top_rr >= settings.grade_min_rerank_score
    badge = "b-ok" if relevant else "b-stop"
    st.markdown(
        f"<span class='badge {badge}'>grade gate: top rerank {top_rr:.3f} "
        f"{'≥' if relevant else '<'} floor {settings.grade_min_rerank_score} · "
        f"{'answerable' if relevant else 'would rewrite/refuse'}</span>",
        unsafe_allow_html=True,
    )

    # context window cards (accordion + internal scroll, so it stays compact) ----------
    st.markdown("<div class='stage'>Context window · what the model reads</div>", unsafe_allow_html=True)
    with st.expander(f"{len(window)} sources in the context window", expanded=True):
        with st.container(height=340):
            for i, c in enumerate(window, 1):
                st.markdown(chunk_card(i, c, True, f"rerank {c.get('rerank_score', 0.0):.3f}"),
                            unsafe_allow_html=True)

    n_all = len(rerank_order[: settings.final_top_k])
    with st.expander(f"Browse all {n_all} reranked candidates", expanded=False):
        with st.container(height=400):
            for i, c in enumerate(rerank_order[: settings.final_top_k], 1):
                st.markdown(chunk_card(i, c, c["chunk_id"] in window_ids,
                                       f"rerank {c.get('rerank_score', 0.0):.3f} · dense {c['dense_score']:.3f}"),
                            unsafe_allow_html=True)

    # 4. streamed generation (st.write_stream — the standard incremental primitive) -------
    st.markdown("<div class='stage'>Generation · streamed</div>", unsafe_allow_html=True)
    _gt: dict = {}

    def _timed(it):
        t0 = time.perf_counter()
        for i, d in enumerate(it):
            if i == 0:
                _gt["ttft"] = time.perf_counter() - t0   # time to FIRST token
            yield d
        _gt["total"] = time.perf_counter() - t0

    with st.container(border=True):
        acc = st.write_stream(_timed(gen.stream(q, window)))
    if _gt:
        st.caption(f"⏱ TTFT {_gt.get('ttft', 0):.2f}s · full generation {_gt.get('total', 0):.2f}s")

    # 5. output groundedness guard ---------------------------------------
    context, _, id_by_num = gen.build_context(q, window)
    result = gen.finalize(acc, id_by_num, context)
    grounded, score = guard.is_grounded(context, acc)

    st.markdown("<div class='stage'>Output guard · HHEM groundedness</div>", unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    g1.metric("groundedness", f"{score:.3f}", help=f"threshold {settings.grounded_min_score}")
    g2.metric("verdict", "grounded ✓" if grounded else "ungrounded ✗")
    g3.metric("citations", str(len(result["cited_ids"])) or "0")
    if result["is_refusal"]:
        st.markdown("<span class='badge b-info'>honest refusal — corpus lacks the answer</span>",
                    unsafe_allow_html=True)
    elif not grounded:
        st.markdown("<span class='badge b-stop'>below floor → agent would refuse rather than serve this</span>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<span class='badge b-ok'>✓ grounded in retrieved context</span>", unsafe_allow_html=True)

    # close the trace -> exports ui.rag_run + all nested sub-runs to LangSmith
    _cm.__exit__(None, None, None); flush()
    st.caption("✓ traced to LangSmith · project `hcft-agent` · run `ui.rag_run`")
