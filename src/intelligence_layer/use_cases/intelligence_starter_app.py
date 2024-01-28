import os

from aleph_alpha_client import Client
from dotenv import load_dotenv
from fastapi import FastAPI

from intelligence_layer.connectors import (
    AlephAlphaClientProtocol,
    LimitedConcurrencyClient,
)
from intelligence_layer.core import IntelligenceApp
from intelligence_layer.use_cases.classify.classify import ClassifyInput
from intelligence_layer.use_cases.classify.prompt_based_classify import (
    PromptBasedClassify,
)
from intelligence_layer.use_cases.qa.long_context_qa import (
    LongContextQa,
    LongContextQaInput,
)
from intelligence_layer.use_cases.summarize.long_context_high_compression_summarize import (
    LongContextHighCompressionSummarize,
)
from intelligence_layer.use_cases.summarize.summarize import LongContextSummarizeInput


class IntelligenceStarterApp(IntelligenceApp):
    def __init__(self, fast_api_app: FastAPI, client: AlephAlphaClientProtocol) -> None:
        super().__init__(fast_api_app)
        prompt_based_classify = PromptBasedClassify(client)
        self.register_task(prompt_based_classify, ClassifyInput, "/classify")
        long_chunk_qa = LongContextQa(client)
        self.register_task(long_chunk_qa, LongContextQaInput, "/qa")
        summarize = LongContextHighCompressionSummarize(client)
        self.register_task(summarize, LongContextSummarizeInput, "/summarize")


def main() -> None:
    load_dotenv()
    aa_token = os.getenv("AA_TOKEN")
    assert aa_token
    aa_client = Client(aa_token)
    client = LimitedConcurrencyClient(aa_client)
    fast_api = FastAPI()
    app = IntelligenceStarterApp(fast_api, client)
    app.serve()


if __name__ == "__main__":
    main()
