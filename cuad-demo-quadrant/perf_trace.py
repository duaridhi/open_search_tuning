"""
perf_trace.py
-------------
Per-request latency spans for the /search and /chat dev loop.

Gated on PERF_TRACE=1 — when unset, the `span()` context manager is a no-op
and there is no measurable overhead.

Spans are accumulated in a `contextvars.ContextVar` so they survive across
`asyncio.to_thread(...)` boundaries (Python copies the context into the thread).
Repeated spans with the same name within one request are summed, so
`highlight_text` being called once per result still yields a single
`rerank` + `highlight_assemble` total per request.
"""

import json
import logging
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

logger = logging.getLogger(__name__)

PERF_TRACE_ENABLED = os.getenv("PERF_TRACE", "").lower() in ("1", "true", "yes")

_spans: ContextVar[Optional[dict]] = ContextVar("perf_spans", default=None)


def start_trace() -> None:
    if PERF_TRACE_ENABLED:
        _spans.set({})


def get_spans() -> Optional[dict]:
    return _spans.get()


@contextmanager
def span(name: str):
    if not PERF_TRACE_ENABLED:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        record_span(name, elapsed_ms)


def record_span(name: str, elapsed_ms: float) -> None:
    """Add `elapsed_ms` to the named span (sums on repeated names)."""
    if not PERF_TRACE_ENABLED:
        return
    store = _spans.get()
    if store is not None:
        store[name] = store.get(name, 0.0) + elapsed_ms


def spans_header_value() -> Optional[str]:
    s = get_spans()
    if not s:
        return None
    return json.dumps({k: round(v, 2) for k, v in s.items()})
