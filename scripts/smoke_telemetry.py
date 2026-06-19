"""Smoke test for P0 telemetry.

Runs one traced ``gpt-4o-mini`` call inside a manual ``smoke.rag_chat`` span carrying a few
``hcft.*`` attributes, then flushes. Success = it prints "telemetry ok" AND a trace named
``smoke.rag_chat`` (with a child LLM span) shows up in the LangSmith ``hcft-agent`` project.

    python scripts/smoke_telemetry.py
"""
from hcft_agent.config import settings
from hcft_agent.obs import attributes as A
from hcft_agent.obs.telemetry import flush, get_tracer, init_telemetry

exporting = init_telemetry()
print(f"[telemetry] exporting to LangSmith: {exporting}")

from langchain_openai import ChatOpenAI  # noqa: E402  (import after instrumentation)

tracer = get_tracer()
with tracer.start_as_current_span("smoke.rag_chat") as span:
    A.set_attrs(span, {A.ROUTE: "rag_chat", A.RETRIES: 0, A.IS_REFUSAL: False})
    llm = ChatOpenAI(
        model=settings.orchestrator_model,
        api_key=settings.orchestrator_api_key,
        base_url=settings.orchestrator_base_url,
        temperature=0,
    )
    resp = llm.invoke("Reply with exactly: telemetry ok")
    A.set_attrs(span, {A.TERMINAL: "answer"})
    print(f"[llm] {resp.content!r}")

flush()
print("[telemetry] flushed — check the LangSmith 'hcft-agent' project for trace 'smoke.rag_chat'")
