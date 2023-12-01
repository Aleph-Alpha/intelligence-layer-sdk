import os

from aleph_alpha_client import Client
from dotenv import load_dotenv
from fastapi import FastAPI

from intelligence_layer.core import IntelligenceApp
from intelligence_layer.use_cases.classify.prompt_based_classify import (
    PromptBasedClassify,
)
from intelligence_layer.use_cases.qa.long_context_qa import LongContextQa


def intelligence_starter_app(
    fast_api: FastAPI, aleph_alpha_client: Client
) -> IntelligenceApp:
    app = IntelligenceApp(fast_api)
    prompt_based_classify = PromptBasedClassify(aleph_alpha_client)
    app.register_task(prompt_based_classify, "/classify")
    long_chunk_qa = LongContextQa(aleph_alpha_client)
    app.register_task(long_chunk_qa, "/qa")
    return app


def main() -> None:
    load_dotenv()
    aa_token = os.getenv("AA_TOKEN")
    assert aa_token
    client = Client(aa_token)
    fast_api = FastAPI()
    app = intelligence_starter_app(fast_api, client)
    app.serve()


if __name__ == "__main__":
    main()
