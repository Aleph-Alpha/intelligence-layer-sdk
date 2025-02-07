import huggingface_hub

from intelligence_layer.evaluation.infrastructure.hugging_face_repository import (
    HuggingFaceRepository,
)


def test_hugging_face_repository_can_create_and_delete_a_repository(
    hugging_face_token: str, hugging_face_test_repository_id: str
) -> None:
    repository_id = hugging_face_test_repository_id + "unused-suffix"

    assert not huggingface_hub.repo_exists(
        repo_id=repository_id,
        token=hugging_face_token,
        repo_type="dataset",
    ), f"The repository with the ID {repository_id} already exists. Try to run the clean_hf script."

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
