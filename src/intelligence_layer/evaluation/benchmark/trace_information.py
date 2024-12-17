from collections.abc import Sequence
from datetime import timedelta
from typing import cast

from aleph_alpha_client import CompletionResponse

from intelligence_layer.core import ExportedSpan
from intelligence_layer.core.model import _Complete
from intelligence_layer.core.tracer.tracer import SpanType


def _get_root(trace: Sequence[ExportedSpan]) -> ExportedSpan | None:
    root_spans = [span for span in trace if span.parent_id is None]
    if len(root_spans) != 1:
        return None
    return root_spans[0]


def extract_latency_from_trace(trace: Sequence[ExportedSpan]) -> int:
    """Extract the total duration of a given trace based on its root trace.

    Args:
        trace: trace to analyze

    Returns:
        The duration of the trace in microseconds
    """
    root_span = _get_root(trace)
    if root_span is None:
        raise ValueError("No root span found in the trace")
    latency = (root_span.end_time - root_span.start_time) / timedelta(microseconds=1)
    return int(latency)


def _is_complete_request(span: ExportedSpan) -> bool:
    # Assuming that LLM requests have a specific name or attribute
    return span.name == _Complete.__name__


def _extract_tokens_from_complete_request(span: ExportedSpan) -> int:
    if not hasattr(span.attributes, "output"):
        raise ValueError(
            "Function expects a complete span with attributes.output. Output was not present."
        )
    completion_output = cast(CompletionResponse, span.attributes.output)
    return completion_output.num_tokens_generated


def extract_token_count_from_trace(trace: Sequence[ExportedSpan]) -> int:
    """Extract the number of tokens generated in a trace based on its completion requests.

    Note: Does not support traces of streamed responses.

    Args:
        trace: trace to analyze.

    Returns:
        The sum of newly generated tokens across all spans in the given trace.
    """
    token_count = 0
    for span in trace:
        if span.attributes.type != SpanType.TASK_SPAN:
            continue
        if _is_complete_request(span):
            token_count += _extract_tokens_from_complete_request(span)
    return token_count
