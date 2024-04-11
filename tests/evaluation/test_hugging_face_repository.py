from uuid import uuid4
import huggingface_hub  # type: ignore

from intelligence_layer.evaluation.infrastructure.hugging_face_repository import HuggingFaceRepository


def test_hugging_face_repository_can_create_and_delete_a_repository(
    hugging_face_token: str,
) -> None:
    repository_id = f"Aleph-Alpha/test-{uuid4()}"

    assert not huggingface_hub.repo_exists(
        repo_id=repository_id,
        token=hugging_face_token,
        repo_type="dataset",
    ), f"This is very unlikely but it seems that the repository with the ID {repository_id} already exists."

    created_repository = HuggingFaceRepository(
        repository_id=repository_id,
        token=hugging_face_token,
        private=True,
    )

    try:
        assert huggingface_hub.repo_exists(
            repo_id=repository_id,
            token=hugging_face_token,
            repo_type="dataset",
        )
        created_repository.delete_repository()
        assert not huggingface_hub.repo_exists(
            repo_id=repository_id,
            token=hugging_face_token,
            repo_type="dataset",
        )
    finally:
        huggingface_hub.delete_repo(
            repo_id=repository_id,
            token=hugging_face_token,
            repo_type="dataset",
            missing_ok=True,
        )
