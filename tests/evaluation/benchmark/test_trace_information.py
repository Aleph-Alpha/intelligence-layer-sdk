from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import pytest
from aleph_alpha_client import CompletionRequest, CompletionResponse
from aleph_alpha_client.completion import CompletionResult

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
)
from intelligence_layer.core import (
    Context,
    ExportedSpan,
    SpanAttributes,
    SpanStatus,
    TaskSpanAttributes,
)
from intelligence_layer.core.model import CompleteInput, Message, Pharia1ChatModel
from intelligence_layer.core.tracer.in_memory_tracer import InMemoryTracer
from intelligence_layer.evaluation.benchmark.trace_information import (
    extract_latency_from_trace,
    extract_token_count_from_trace,
)


@pytest.fixture
def root_span() -> ExportedSpan:
    trace_id = uuid4()
    return ExportedSpan(
        context=Context(
            trace_id=trace_id,
            span_id=trace_id,
        ),
        name="root",
        parent_id=None,
        start_time=datetime(2022, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2022, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        attributes=SpanAttributes(),
        events=[],
        status=SpanStatus.OK,
    )


@pytest.fixture
def child_span(root_span: ExportedSpan) -> ExportedSpan:
    return ExportedSpan(
        context=Context(
            trace_id=root_span.context.trace_id,
            span_id=uuid4(),
        ),
        name="child",
        parent_id=root_span.context.span_id,
        start_time=datetime(2022, 1, 1, 12, 0, 0, 500000, tzinfo=timezone.utc),
        end_time=datetime(2022, 1, 1, 12, 0, 1, 500000, tzinfo=timezone.utc),
        attributes=TaskSpanAttributes(
            input="input",
            output="output",
        ),
        events=[],
        status=SpanStatus.OK,
    )


SECOND_IN_MICROSECOND = 10**6


def test_extract_latency_from_trace_root_span(root_span: ExportedSpan):
    latency = extract_latency_from_trace([root_span])
    assert latency == SECOND_IN_MICROSECOND


def test_extract_latency_from_trace_root_span_can_do_ns_differences(
    root_span: ExportedSpan,
):
    root_span.end_time = root_span.start_time + timedelta(microseconds=1)
    latency = extract_latency_from_trace([root_span])
    assert latency == 1


def test_extract_latency_from_trace_root_span_with_child(
    root_span: ExportedSpan, child_span: ExportedSpan
):
    latency = extract_latency_from_trace([root_span, child_span])
    assert latency == SECOND_IN_MICROSECOND

    latency = extract_latency_from_trace([child_span, root_span])
    assert latency == SECOND_IN_MICROSECOND


def test_extract_latency_from_trace_no_root_span(child_span: ExportedSpan):
    # Call the function with the child span
    with pytest.raises(ValueError):
        extract_latency_from_trace([child_span])


class MockClient(AlephAlphaClientProtocol):
    def __init__(self, generated_tokens: int):
        self.generated_tokens = generated_tokens

    def complete(
        self,
        request: CompletionRequest,
        model: str,
    ) -> CompletionResponse:
        return CompletionResponse(
            model_version="---",
            completions=[CompletionResult()],
            num_tokens_generated=self.generated_tokens,
            num_tokens_prompt_total=20,
        )

    def models(self) -> Sequence[Mapping[str, Any]]:
        return [{"name": "pharia-1-llm-7b-control"}]


def test_extract_token_count_from_trace_works_without_llm_spans(
    root_span: ExportedSpan,
):
    result = extract_token_count_from_trace([root_span])
    assert result == 0


def test_extract_token_count_from_trace_works_with_complete_spans():
    tokens_to_generate = 10
    model_str = "pharia-1-llm-7b-control"
    model = Pharia1ChatModel(
        name=model_str, client=MockClient(generated_tokens=tokens_to_generate)
    )
    tracer = InMemoryTracer()
    with tracer.span("root") as root:
        model.complete(CompleteInput(prompt=model.to_instruct_prompt("test")), root)
        model.complete(CompleteInput(prompt=model.to_instruct_prompt("test")), root)

    complete_trace = tracer.export_for_viewing()

    result = extract_token_count_from_trace(complete_trace)
    assert result == tokens_to_generate * 2


def test_extract_token_count_from_trace_works_chat():
    tokens_to_generate = 10
    model_str = "pharia-1-llm-7b-control"
    model = Pharia1ChatModel(
        name=model_str, client=MockClient(generated_tokens=tokens_to_generate)
    )
    tracer = InMemoryTracer()
    with tracer.span("root") as root:
        model.generate_chat(
            messages=[Message(role="user", content="dummy")],
            response_prefix=None,
            tracer=root,
        )
    complete_trace = tracer.export_for_viewing()

    result = extract_token_count_from_trace(complete_trace)
    assert result == tokens_to_generate
