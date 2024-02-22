from fastapi import FastAPI

from intelligence_layer.core import IntelligenceApp
from intelligence_layer.use_cases.classify.classify import ClassifyInput
from intelligence_layer.use_cases.classify.prompt_based_classify import (
    PromptBasedClassify,
)
from intelligence_layer.use_cases.qa.long_context_qa import (
    LongContextQa,
    LongContextQaInput,
)
from intelligence_layer.use_cases.summarize.steerable_long_context_summarize import (
    SteerableLongContextSummarize,
)
from intelligence_layer.use_cases.summarize.summarize import LongContextSummarizeInput


class IntelligenceStarterApp(IntelligenceApp):
    def __init__(self, fast_api_app: FastAPI) -> None:
        super().__init__(fast_api_app)
        prompt_based_classify = PromptBasedClassify()
        self.register_task(prompt_based_classify, ClassifyInput, "/classify")
        long_chunk_qa = LongContextQa()
        self.register_task(long_chunk_qa, LongContextQaInput, "/qa")
        summarize = SteerableLongContextSummarize(
            max_generated_tokens=512, max_tokens_per_chunk=1024
        )
        self.register_task(summarize, LongContextSummarizeInput, "/summarize")


def main() -> None:
    fast_api = FastAPI()
    app = IntelligenceStarterApp(fast_api)
    app.serve()


if __name__ == "__main__":
    main()
