from intelligence_layer.classify import (
    SingleLabelClassify,
    ClassifyInput,
    ClassifyOutput,
)
from intelligence_layer._task import DebugLog


def test_classify() -> None:
    """Test that basic seniment analysis is working"""
    classify = SingleLabelClassify()
    classify_input = ClassifyInput(text="This is good", labels={"positive", "negative"})
    classify_output = classify.run(classify_input)
    assert isinstance(classify_output, ClassifyOutput)
    assert classify_input.labels == set(r.label for r in classify_output.results)
    assert isinstance(classify_output.debug_log, DebugLog)
    # assert positive > negative
