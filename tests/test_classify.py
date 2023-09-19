from intelligence_layer.classify import Classify, ClassifyInput, ClassifyOutput
from intelligence_layer._output import AuditTrail


def test_classify():
    classify = Classify()
    classify_input = ClassifyInput(text="This is good", labels=["positive", "negative"])
    classify_output = classify.run(classify_input)
    assert isinstance(classify_output, ClassifyOutput)
    assert classify_input.labels == set(r.label for r in classify_output.results)
    assert isinstance(classify_output.audit_trail, AuditTrail)
    # assert positive > negative
