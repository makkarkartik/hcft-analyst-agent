"""Domain-specific OpenTelemetry span attribute keys — the ``hcft.*`` namespace.

OpenInference/OTel already standardize the *generic* run record (LLM / retriever / tool
spans, prompts, token counts, latency). These keys carry the *domain verdicts* our eval and
guardrails read off the trace — routing, refusal, groundedness, degradation — that no library
predefines. We attach them as custom attributes on the active span (the standard's extension
point), so we add zero parallel infrastructure.
"""
from __future__ import annotations

import json

NS = "hcft"

# --- routing / trajectory ---
ROUTE = f"{NS}.route"                     # skill/agent the router chose
ROUTE_EXPECTED = f"{NS}.route.expected"   # gold route (eval only)
RETRIES = f"{NS}.retries"
TERMINAL = f"{NS}.terminal"               # "answer" | "refuse" | "error"

# --- grounding / outcome ---
IS_REFUSAL = f"{NS}.is_refusal"
GROUNDED = f"{NS}.grounded"
CITED_IDS = f"{NS}.cited_ids"
RETRIEVED_IDS = f"{NS}.retrieved_ids"

# --- guardrail verdicts ---
INPUT_FLAGS = f"{NS}.guard.input_flags"          # e.g. ["injection","pii"]
OUTPUT_GROUNDED = f"{NS}.guard.output_grounded"
ACTION_ALLOWED = f"{NS}.guard.action_allowed"
SANDBOX_RESULT = f"{NS}.guard.sandbox_result"

# --- reliability ---
DEGRADED = f"{NS}.degraded"                # produced via a fallback path?
DEGRADED_REASON = f"{NS}.degraded_reason"


def set_attrs(span, attrs: dict) -> None:
    """Set ``hcft.*`` attributes on a span. Skips ``None``; JSON-encodes lists/dicts
    (OTel attributes must be scalars or scalar sequences)."""
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, (list, dict, tuple)):
            value = json.dumps(value)
        span.set_attribute(key, value)
