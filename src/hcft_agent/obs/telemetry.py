"""OpenTelemetry + OpenInference wiring, exporting to LangSmith over OTLP/HTTP.

One idempotent call to :func:`init_telemetry` sets up:
  * an OTel ``TracerProvider``;
  * the OTLP/HTTP exporter pointed at LangSmith's OTel endpoint — swap the endpoint for any
    OTel backend (Phoenix/Langfuse/…) by env var and nothing else changes (vendor-neutral);
  * OpenInference's LangChain instrumentation, which auto-creates spans for every LLM / tool /
    retriever / chain step (LangGraph is built on LangChain runnables, so it's covered too),
    with token + cost attributes following the GenAI semantic conventions.

Env (.env): ``LANGSMITH_API_KEY`` (required to export), ``LANGSMITH_PROJECT`` (optional).
With no key, telemetry stays local and never blocks the app — observability fails *open*.
"""
from __future__ import annotations

import os
from functools import lru_cache

# LangSmith's OTLP/HTTP traces endpoint (self-hosted: <instance>/api/v1/otel/v1/traces).
LANGSMITH_OTLP_TRACES_ENDPOINT = "https://api.smith.langchain.com/otel/v1/traces"


@lru_cache(maxsize=1)
def init_telemetry(service_name: str = "hcft-agent") -> bool:
    """Idempotent. Returns ``True`` if spans are being exported to LangSmith, ``False`` if
    no API key is set (app still runs — obs fails open)."""
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from openinference.instrumentation.langchain import LangChainInstrumentor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))

    exporting = False
    api_key = os.getenv("LANGSMITH_API_KEY")
    if api_key:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        headers = {"x-api-key": api_key}
        project = os.getenv("LANGSMITH_PROJECT")
        if project:
            headers["Langsmith-Project"] = project
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=LANGSMITH_OTLP_TRACES_ENDPOINT, headers=headers)
            )
        )
        exporting = True

    trace.set_tracer_provider(provider)
    LangChainInstrumentor().instrument(tracer_provider=provider)
    return exporting


def get_tracer(name: str = "hcft-agent"):
    from opentelemetry import trace

    return trace.get_tracer(name)


def current_span():
    """The span in the active context — attach ``hcft.*`` attributes to it."""
    from opentelemetry import trace

    return trace.get_current_span()


def flush() -> None:
    """Force-export buffered spans (BatchSpanProcessor is async; call before a short script
    exits so traces actually reach LangSmith)."""
    from opentelemetry import trace

    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()
