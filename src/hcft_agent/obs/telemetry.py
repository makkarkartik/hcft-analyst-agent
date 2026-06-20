"""Native LangSmith tracing (LangChain / LangGraph first-party).

We switched OFF the OTel / OpenInference OTLP path. Why: LangGraph runs each node in a worker
thread, and OpenTelemetry's "current span" lives in a thread-local context that did NOT
propagate into those threads — so every manual span started inside a node orphaned into its
own root trace (an unreadable flat list). LangChain's native tracer threads its run-tree
context through its own callback machinery across those worker threads, so node spans nest
correctly with zero effort. Trade-off: this is LangSmith-specific — we give up the
vendor-neutral "swap the OTel endpoint for Phoenix/Langfuse" property. Acceptable here.

Enable by env (.env): ``LANGSMITH_API_KEY`` (required to export), ``LANGSMITH_PROJECT``
(optional, defaults to the service name). With no key, tracing stays off and the app still
runs — observability fails *open*.

Helpers:
  * :func:`init_telemetry` — idempotent; flips ``LANGSMITH_TRACING`` on.
  * :func:`trace_block`    — context manager that nests a child run under the current run tree
                             (no-op when tracing is off); used for non-LangChain sub-steps
                             (retriever stages, the HHEM guard, the UI run).
  * :func:`tag`            — attach metadata (our ``hcft.*`` verdicts) to the current run.
  * :func:`flush`          — wait for background tracers before a short-lived process exits.
"""
from __future__ import annotations

import os
from contextlib import nullcontext
from functools import lru_cache


@lru_cache(maxsize=1)
def init_telemetry(service_name: str = "hcft-agent") -> bool:
    """Idempotent. Returns ``True`` if LangSmith tracing is enabled, ``False`` if no API key."""
    if not os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_TRACING"] = "false"
        return False
    os.environ.setdefault("LANGSMITH_PROJECT", service_name)
    os.environ["LANGSMITH_TRACING"] = "true"
    return True


def _tracing_on() -> bool:
    return os.getenv("LANGSMITH_TRACING") == "true"


def trace_block(name: str, run_type: str = "chain", **kwargs):
    """Context manager nesting a child run (``name``) under the active run tree. No-op when
    tracing is off or langsmith is unavailable — so call sites stay clean and fail open."""
    if _tracing_on():
        try:
            from langsmith import trace as _trace

            return _trace(name=name, run_type=run_type, **kwargs)
        except Exception:
            pass
    return nullcontext()


def tag(**metadata) -> None:
    """Attach ``hcft.*`` verdicts (or any metadata) to the current run, so they're queryable on
    the trace. No-op when there's no active run."""
    if not _tracing_on():
        return
    try:
        from langsmith import get_current_run_tree

        rt = get_current_run_tree()
        if rt is not None:
            rt.add_metadata({k: v for k, v in metadata.items() if v is not None})
    except Exception:
        pass


def flush() -> None:
    """Block until queued traces are sent (call before a short script exits)."""
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
    except Exception:
        pass
