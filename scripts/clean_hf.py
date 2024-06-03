import os
import warnings

from dotenv import load_dotenv
from huggingface_hub import HfApi  # type: ignore


def clean_up_dangling_hf_repos(hugging_face_token: str) -> None:
    api = HfApi(token=hugging_face_token)
    datasets = list(
        api.list_datasets(author="Aleph-Alpha", dataset_name="IL-temp-tests")
    )
    if len(datasets) > 0:
        warnings.warn("dangling hf datasets found, attempting to delete")
    for dataset in datasets:
        api.delete_repo(dataset.id, repo_type="dataset", missing_ok=True)


if __name__ == "__main__":
    load_dotenv()
    token = os.getenv("HUGGING_FACE_TOKEN")
    assert isinstance(token, str)
    clean_up_dangling_hf_repos(token)
