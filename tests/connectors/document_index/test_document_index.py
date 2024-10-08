import random
import re
import string
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from functools import wraps
from http import HTTPStatus
from time import sleep
from typing import ParamSpec, TypeVar, get_args, overload

import pytest
from pydantic import ValidationError
from pytest import fixture, raises

from intelligence_layer.connectors.base.json_serializable import JsonSerializable
from intelligence_layer.connectors.document_index.document_index import (
    CollectionPath,
    DocumentContents,
    DocumentFilterQueryParams,
    DocumentIndexClient,
    DocumentPath,
    EmbeddingType,
    FilterField,
    FilterOps,
    Filters,
    HybridIndex,
    IndexConfiguration,
    IndexPath,
    InvalidInput,
    ResourceNotFound,
    SearchQuery,
)

P = ParamSpec("P")
R = TypeVar("R")


@overload
def retry(
    func: None = None, max_retries: int = 3, secondy_delay: float = 0.0
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


@overload
def retry(
    func: Callable[P, R], max_retries: int = 3, secondy_delay: float = 0.0
) -> Callable[P, R]: ...


def retry(
    func: Callable[P, R] | None = None,
    max_retries: int = 25,
    secondy_delay: float = 0.2,
) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for _ in range(1 + max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    sleep(secondy_delay)

            raise last_exception

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


def random_identifier() -> str:
    name = "".join(random.choices(string.ascii_letters + string.digits, k=20))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"ci-il-{name}-{timestamp}"


def is_outdated_identifier(identifier: str, timestamp_threshold: datetime) -> bool:
    # match the format that is defined in random_identifier()
    matched = re.match(
        r"^ci-il-[a-zA-Z0-9]{20}-(?P<timestamp>\d{8}T\d{6})$", identifier
    )
    if matched is None:
        return False

    timestamp = datetime.strptime(matched["timestamp"], "%Y%m%dT%H%M%S").replace(
        tzinfo=timezone.utc
    )
    if timestamp > timestamp_threshold:
        return False

    return True


@fixture(scope="session")
def document_index_namespace() -> str:
    return "team-document-index"


@fixture(scope="session", autouse=True)
def _teardown(token: str, document_index_namespace: str) -> Iterator[None]:
    yield

    # Cleanup leftover resources from previous runs.
    timestamp_threshold = datetime.now(timezone.utc) - timedelta(hours=1)

    document_index = DocumentIndexClient(token)
    collections = document_index.list_collections(document_index_namespace)
    for collection_path in collections:
        if is_outdated_identifier(collection_path.collection, timestamp_threshold):
            document_index.delete_collection(collection_path)

    indexes = document_index.list_indexes(document_index_namespace)
    for index_path in indexes:
        if is_outdated_identifier(index_path.index, timestamp_threshold):
            document_index.delete_index(index_path)


@fixture(scope="session")
def filter_index_config() -> dict[str, dict[str, str]]:
    return {
        "test-string-filter": {
            "field-name": "string-field",
            "field-type": "string",
        },
        "test-integer-filter": {
            "field-name": "integer-field",
            "field-type": "integer",
        },
        "test-float-filter": {
            "field-name": "float-field",
            "field-type": "float",
        },
        "test-boolean-filter": {
            "field-name": "boolean-field",
            "field-type": "boolean",
        },
        "test-date-filter": {
            "field-name": "date-field",
            "field-type": "date_time",
        },
    }


@fixture
def random_collection_path(
    document_index: DocumentIndexClient,
    document_index_namespace: str,
) -> Iterator[CollectionPath]:
    name = random_identifier()
    collection_path = CollectionPath(
        namespace=document_index_namespace, collection=name
    )
    try:
        document_index.create_collection(collection_path)

        yield collection_path
    finally:
        document_index.delete_collection(collection_path)


@fixture(scope="session")
def read_only_collection_path(
    token: str,
    document_index_namespace: str,
    document_contents_with_metadata: list[DocumentContents],
    filter_index_config: dict[str, dict[str, str]],
) -> Iterator[CollectionPath]:
    document_index = DocumentIndexClient(token)

    name = random_identifier()
    collection_path = CollectionPath(
        namespace=document_index_namespace, collection=name
    )
    try:
        document_index.create_collection(collection_path)

        # Add 3 documents
        for i, content in enumerate(document_contents_with_metadata):
            document_index.add_document(
                DocumentPath(
                    collection_path=collection_path,
                    document_name=f"document-metadata-{i}",
                ),
                content,
            )

        # Assign index
        document_index.assign_index_to_collection(
            collection_path, "ci-intelligence-layer"
        )

        # Assign filter indexes
        for filter_index in filter_index_config:
            document_index.assign_filter_index_to_search_index(
                collection_path=collection_path,
                index_name="ci-intelligence-layer",
                filter_index_name=filter_index,
            )

        yield collection_path
    finally:
        document_index.delete_collection(collection_path)


@fixture
def random_index(
    document_index: DocumentIndexClient, document_index_namespace: str
) -> Iterator[tuple[IndexPath, IndexConfiguration]]:
    name = random_identifier()
    chunk_size, chunk_overlap = sorted(
        random.sample([0, 32, 64, 128, 256, 512, 1024], 2), reverse=True
    )
    embedding_type = random.choice(get_args(EmbeddingType))
    hybrid_index_choices: list[HybridIndex] = ["bm25", None]
    hybrid_index = random.choice(hybrid_index_choices)

    index = IndexPath(namespace=document_index_namespace, index=name)
    index_configuration = IndexConfiguration(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_type=embedding_type,
        hybrid_index=hybrid_index,
    )
    try:
        document_index.create_index(index, index_configuration)
        yield index, index_configuration
    finally:
        document_index.delete_index(index)


@fixture
def document_contents() -> DocumentContents:
    text = """John Stith Pemberton, the inventor of the world-renowned beverage Coca-Cola, was a figure whose life was marked by creativity, entrepreneurial spirit, and the turbulent backdrop of 19th-century America. Born on January 8, 1831, in Knoxville, Georgia, Pemberton grew up in an era of profound transformation and change.

Pemberton began his professional journey by studying medicine and pharmacy. After earning a degree in pharmacy, he started his career as a druggist in Columbus, Georgia. He was known for his keen interest in creating medicinal concoctions and was well-respected in his community. His early creations included various medicines and tonics, which were typical of the times when pharmacists often concocted their own remedies.

Pemberton's life took a significant turn during the American Civil War. He served as a lieutenant colonel in the Confederate Army, and it was during this period that he sustained a wound that led him to become dependent on morphine. This personal struggle with addiction likely influenced his later work in seeking out alternatives and remedies for pain relief.

In the post-war years, Pemberton relocated to Atlanta, Georgia, where he continued to experiment with various medicinal syrups and tonics. It was during this time, in the late 19th century, that he developed a beverage he initially called "Pemberton's French Wine Coca." This concoction was inspired by Vin Mariani, a popular French tonic wine that contained coca leaves. Pemberton's beverage was intended to serve not just as a refreshing drink but also as a remedy for various ailments, including morphine addiction, indigestion, and headaches.

However, in 1886, when Atlanta introduced prohibition legislation, Pemberton was compelled to create a non-alcoholic version of his beverage. He experimented with a combination of carbonated water, coca leaf extract, kola nut, and other ingredients, eventually perfecting the formula for what would soon become Coca-Cola. The name was suggested by his bookkeeper, Frank Robinson, who also created the distinctive cursive logo that is still in use today.

Pemberton advertised his new creation as a "brain tonic" and "temperance drink," asserting that it could alleviate headaches and fatigue. However, due to his declining health and financial difficulties, Pemberton was eventually compelled to sell portions of his business to various partners. Shortly before his death in 1888, he sold his remaining stake in Coca-Cola to Asa G. Candler, a fellow pharmacist and businessman.

Under Candler's leadership, Coca-Cola transformed from a pharmacist's concoction into a mass-produced and marketed beverage that became a staple of American culture and a global icon. Despite the changes and the immense growth of the brand, the legacy of John Stith Pemberton as the inventor of Coca-Cola remains an integral part of the beverage's history.

Pemberton's life story is a testament to the spirit of innovation and resilience. His creation, borne out of personal struggles and the context of his times, went on to transcend its origins and become a symbol recognized across the globe. Today, when we think of Coca-Cola, we are reminded of Pemberton's journey from a small-town pharmacist to the creator of one of the world's most enduring and beloved brands."""
    return DocumentContents(contents=[text], metadata={"Some": "Metadata"})


@fixture(scope="session")
def document_contents_with_metadata() -> list[DocumentContents]:
    text_1 = """John Stith Pemberton, the inventor of the world-renowned beverage Coca-Cola, was a figure whose life was marked by creativity, entrepreneurial spirit, and the turbulent backdrop of 19th-century America. Born on January 8, 1831, in Knoxville, Georgia, Pemberton grew up in an era of profound transformation and change."""
    text_2 = """Pemberton began his professional journey by studying medicine and pharmacy. After earning a degree in pharmacy, he started his career as a druggist in Columbus, Georgia. He was known for his keen interest in creating medicinal concoctions and was well-respected in his community. His early creations included various medicines and tonics, which were typical of the times when pharmacists often concocted their own remedies."""
    text_3 = """Pemberton's life took a significant turn during the American Civil War. He served as a lieutenant colonel in the Confederate Army, and it was during this period that he sustained a wound that led him to become dependent on morphine. This personal struggle with addiction likely influenced his later work in seeking out alternatives and remedies for pain relief."""

    metadata_1: JsonSerializable = {
        "string-field": "example_string_1",
        "integer-field": 123,
        "float-field": 123.45,
        "boolean-field": True,
        "date-field": datetime(2022, 1, 1, tzinfo=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    }

    metadata_2: JsonSerializable = {
        "string-field": "example_string_2",
        "integer-field": 456,
        "float-field": 678.90,
        "boolean-field": False,
        "date-field": datetime(2023, 1, 1, tzinfo=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    }

    metadata_3: JsonSerializable = {
        "string-field": "example_string_3",
        "integer-field": 789,
        "float-field": 101112.13,
        "boolean-field": True,
        "date-field": datetime(2024, 1, 1, tzinfo=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    }

    return [
        DocumentContents(contents=[text_1], metadata=metadata_1),
        DocumentContents(contents=[text_2], metadata=metadata_2),
        DocumentContents(contents=[text_3], metadata=metadata_3),
    ]


@pytest.mark.internal
def test_document_index_sets_authorization_header_for_given_token() -> None:
    token = "some-token"

    document_index = DocumentIndexClient(token)

    assert document_index.headers["Authorization"] == f"Bearer {token}"


@pytest.mark.internal
def test_document_index_sets_no_authorization_header_when_token_is_none() -> None:
    document_index = DocumentIndexClient(None)

    assert "Authorization" not in document_index.headers


@pytest.mark.internal
def test_document_index_lists_namespaces(document_index: DocumentIndexClient) -> None:
    namespaces = document_index.list_namespaces()

    assert "aleph-alpha" in namespaces


@pytest.mark.internal
def test_document_index_gets_collection(
    document_index: DocumentIndexClient, random_collection_path: CollectionPath
) -> None:
    collections = document_index.list_collections(random_collection_path.namespace)

    assert random_collection_path in collections


@pytest.mark.internal
@pytest.mark.parametrize(
    "document_name",
    [
        "Example Document",
        "!@#$%^&*()-_+={}[]\\|;:'\"<>,.?/~`",
    ],
)
def test_document_index_adds_document(
    document_index: DocumentIndexClient,
    random_collection_path: CollectionPath,
    document_contents: DocumentContents,
    document_name: str,
) -> None:
    document_path = DocumentPath(
        collection_path=random_collection_path,
        document_name=document_name,
    )
    document_index.add_document(document_path, document_contents)

    assert any(
        d.document_path == document_path
        for d in document_index.documents(random_collection_path)
    )
    assert document_contents == document_index.document(document_path)


@pytest.mark.internal
def test_document_index_searches_asymmetrically(
    document_index: DocumentIndexClient, random_collection_path: CollectionPath
) -> None:
    document_path = DocumentPath(
        collection_path=random_collection_path,
        document_name="test_document_index_searches_asymmetrically",
    )
    document_contents = DocumentContents.from_text("Mark likes pizza.")
    document_index.add_document(document_path, document_contents)

    document_index.assign_index_to_collection(
        collection_path=random_collection_path, index_name="ci-intelligence-layer"
    )

    search_query = SearchQuery(query="Who likes pizza?", max_results=1, min_score=0.0)

    @retry
    def search() -> None:
        search_result = document_index.search(
            document_path.collection_path, "ci-intelligence-layer", search_query
        )

        assert "Mark" in search_result[0].section

    search()


def test_document_index_hybrid_search_combines_semantic_and_keyword_search(
    document_index: DocumentIndexClient, random_collection_path: CollectionPath
) -> None:
    document_index.assign_index_to_collection(
        random_collection_path, "ci-intelligence-layer-hybrid"
    )

    document_path = DocumentPath(
        collection_path=random_collection_path,
        document_name="test_document_index_hybrid_search_combines_semantic_and_keyword_search",
    )
    document_contents = DocumentContents(
        contents=[
            "Infant and baby are synonyms. Baby is also an informal term for a lover or spouse.",
            "The infant was crying because it was hungry.",
            "People cry when they are sad or hurt.",
        ],
    )
    document_index.add_document(document_path, document_contents)

    search_query = SearchQuery(
        query="Why is the baby crying?",
        max_results=3,
        min_score=0.0,
    )

    @retry
    def search() -> None:
        search_results = document_index.search(
            document_path.collection_path, "ci-intelligence-layer-hybrid", search_query
        )

        assert "The infant was crying because" in search_results[0].section
        assert "Infant and baby are synonyms" in search_results[1].section
        assert "People cry" in search_results[2].section

    search()


@pytest.mark.internal
@pytest.mark.parametrize(
    "document_name",
    [
        "Document to be deleted",
        "Document to be deleted !@#$%^&*()-_+={}[]\\|;:'\"<>,.?/~`",
    ],
)
def test_document_index_deletes_document(
    document_index: DocumentIndexClient,
    random_collection_path: CollectionPath,
    document_name: str,
) -> None:
    document_path = DocumentPath(
        collection_path=random_collection_path, document_name=document_name
    )
    document_contents = DocumentContents.from_text("Some text...")

    document_index.add_document(document_path, document_contents)
    document_index.delete_document(document_path)
    document_paths = document_index.documents(document_path.collection_path)

    assert not any(d.document_path == document_path for d in document_paths)


def test_document_index_raises_on_getting_non_existing_document(
    document_index: DocumentIndexClient,
) -> None:
    non_existing_document = DocumentPath(
        collection_path=CollectionPath(namespace="does", collection="not"),
        document_name="exist",
    )
    with raises(ResourceNotFound) as exception_info:
        document_index.document(non_existing_document)
    assert exception_info.value.status_code == HTTPStatus.NOT_FOUND
    assert (
        non_existing_document.collection_path.namespace in exception_info.value.message
    )


@pytest.mark.parametrize(
    "slash_separated_str,expected_document_path",
    [
        (
            "a/b/c",
            DocumentPath(
                collection_path=CollectionPath(namespace="a", collection="b"),
                document_name="c",
            ),
        ),
        (
            "a/b/c/d",
            DocumentPath(
                collection_path=CollectionPath(namespace="a", collection="b"),
                document_name="c/d",
            ),
        ),
    ],
)
def test_document_path_from_string(
    slash_separated_str: str, expected_document_path: DocumentPath
) -> None:
    actual_document_path = DocumentPath.from_slash_separated_str(slash_separated_str)
    assert actual_document_path == expected_document_path
    with raises(AssertionError):
        DocumentPath.from_slash_separated_str("a/c")


def test_document_list_all_documents(
    document_index: DocumentIndexClient, read_only_collection_path: CollectionPath
) -> None:
    filter_result = document_index.documents(read_only_collection_path)

    assert len(filter_result) == 3


def test_document_list_max_n_documents(
    document_index: DocumentIndexClient, read_only_collection_path: CollectionPath
) -> None:
    filter_query_params = DocumentFilterQueryParams(max_documents=1, starts_with=None)

    filter_result = document_index.documents(
        read_only_collection_path, filter_query_params
    )

    assert len(filter_result) == 1


def test_document_list_documents_with_matching_prefix(
    document_index: DocumentIndexClient, random_collection_path: CollectionPath
) -> None:
    document_index.add_document(
        document_path=DocumentPath(
            collection_path=random_collection_path, document_name="Example document"
        ),
        contents=DocumentContents.from_text("Document with matching prefix"),
    )
    document_index.add_document(
        document_path=DocumentPath(
            collection_path=random_collection_path, document_name="Another document"
        ),
        contents=DocumentContents.from_text("Document without matching prefix"),
    )
    prefix = "Example"
    filter_query_params = DocumentFilterQueryParams(
        max_documents=None, starts_with=prefix
    )

    filter_result = document_index.documents(
        random_collection_path, filter_query_params
    )

    assert len(filter_result) == 1
    assert filter_result[0].document_path.document_name.startswith(prefix)


def test_document_path_is_immutable() -> None:
    path = DocumentPath(
        collection_path=CollectionPath(namespace="1", collection="2"), document_name="3"
    )
    dictionary = {}
    dictionary[path] = 1

    assert dictionary[path] == 1


def test_index_configuration_rejects_invalid_chunk_overlap() -> None:
    try:
        IndexConfiguration(
            chunk_size=128, chunk_overlap=128, embedding_type="asymmetric"
        )
    except ValidationError as e:
        assert "chunk_overlap must be less than chunk_size" in str(e)
    else:
        raise AssertionError("ValidationError was not raised")


def test_indexes_in_namespace_are_returned(
    document_index: DocumentIndexClient,
    random_index: tuple[IndexPath, IndexConfiguration],
) -> None:
    index_path, index_configuration = random_index
    retrieved_index_configuration = document_index.index_configuration(index_path)

    assert retrieved_index_configuration == index_configuration


def test_indexes_for_collection_are_returned(
    document_index: DocumentIndexClient, read_only_collection_path: CollectionPath
) -> None:
    index_names = document_index.list_assigned_index_names(read_only_collection_path)
    assert "ci-intelligence-layer" in index_names


def test_create_filter_indexes_in_namespace(
    document_index: DocumentIndexClient,
    document_index_namespace: str,
    filter_index_config: dict[str, dict[str, str]],
) -> None:
    for index_name, index_config in filter_index_config.items():
        document_index.create_filter_index_in_namespace(
            namespace=document_index_namespace,
            filter_index_name=index_name,
            field_name=index_config["field-name"],
            field_type=index_config["field-type"],  # type:ignore[arg-type]
        )

    assert all(
        filter_index
        in document_index.list_filter_indexes_in_namespace(document_index_namespace)
        for filter_index in filter_index_config
    )


def test_create_filter_index_invalid_name(
    document_index: DocumentIndexClient, document_index_namespace: str
) -> None:
    with pytest.raises(ValueError) as context:
        document_index.create_filter_index_in_namespace(
            document_index_namespace, "invalid index!", "field_name", "string"
        )
        assert (
            str(context.value)
            == "Filter index name can only contain alphanumeric characters (a-z, A-Z, -, . and 0-9)."
        )


def test_create_filter_index_name_too_long(
    document_index: DocumentIndexClient, document_index_namespace: str
) -> None:
    with pytest.raises(ValueError) as context:
        document_index.create_filter_index_in_namespace(
            document_index_namespace, "a" * 51, "field_name", "string"
        )
    assert (
        str(context.value) == "Filter index name cannot be longer than 50 characters."
    )


def test_assign_filter_indexes_to_collection(
    document_index: DocumentIndexClient,
    random_collection_path: CollectionPath,
    filter_index_config: dict[str, dict[str, str]],
) -> None:
    document_index.assign_index_to_collection(
        collection_path=random_collection_path, index_name="ci-intelligence-layer"
    )

    for index_name in filter_index_config:
        document_index.assign_filter_index_to_search_index(
            collection_path=random_collection_path,
            filter_index_name=index_name,
            index_name="ci-intelligence-layer",
        )

    assert all(
        filter_index
        in document_index.list_assigned_filter_index_names(
            random_collection_path, "ci-intelligence-layer"
        )
        for filter_index in filter_index_config
    )


def test_document_index_adds_documents_with_metadata(
    document_index: DocumentIndexClient,
    random_collection_path: CollectionPath,
    document_contents_with_metadata: list[DocumentContents],
) -> None:
    for i, doc_content in enumerate(document_contents_with_metadata):
        document_path = DocumentPath(
            collection_path=random_collection_path,
            document_name=f"document-metadata-{i}",
        )
        document_index.add_document(document_path, doc_content)

        assert any(
            d.document_path == document_path
            for d in document_index.documents(random_collection_path)
        )
        assert doc_content == document_index.document(document_path)


def test_search_with_string_filter(
    document_index: DocumentIndexClient,
    read_only_collection_path: CollectionPath,
) -> None:
    search_query = SearchQuery(
        query="Coca-Cola",
        max_results=10,
        min_score=0.5,
        filters=[
            Filters(
                filter_type="with",
                fields=[
                    FilterField(
                        field_name="string-field",
                        field_value="example_string_1",
                        criteria=FilterOps.EQUAL_TO,
                    )
                ],
            )
        ],
    )

    @retry
    def search() -> None:
        results = document_index.search(
            read_only_collection_path, "ci-intelligence-layer", search_query
        )
        assert results[0].document_path.document_name == "document-metadata-0"

    search()


def test_search_with_integer_filter(
    document_index: DocumentIndexClient,
    read_only_collection_path: CollectionPath,
) -> None:
    search_query = SearchQuery(
        query="Coca-Cola",
        max_results=10,
        min_score=0.5,
        filters=[
            Filters(
                filter_type="with",
                fields=[
                    FilterField(
                        field_name="integer-field",
                        field_value=123,
                        criteria=FilterOps.EQUAL_TO,
                    )
                ],
            )
        ],
    )

    @retry
    def search() -> None:
        results = document_index.search(
            read_only_collection_path, "ci-intelligence-layer", search_query
        )
        assert len(results) == 1
        assert results[0].document_path.document_name == "document-metadata-0"

    search()


def test_search_with_float_filter(
    document_index: DocumentIndexClient,
    read_only_collection_path: CollectionPath,
) -> None:
    search_query = SearchQuery(
        query="Coca-Cola",
        max_results=10,
        min_score=0.1,
        filters=[
            Filters(
                filter_type="with",
                fields=[
                    FilterField(
                        field_name="float-field",
                        field_value=123.45,
                        criteria=FilterOps.GREATER_THAN,
                    )
                ],
            )
        ],
    )

    @retry
    def search() -> None:
        results = document_index.search(
            read_only_collection_path, "ci-intelligence-layer", search_query
        )
        assert len(results) == 2
        assert results[0].document_path.document_name == "document-metadata-1"
        assert results[1].document_path.document_name == "document-metadata-2"

    search()


def test_search_with_boolean_filter(
    document_index: DocumentIndexClient,
    read_only_collection_path: CollectionPath,
) -> None:
    search_query = SearchQuery(
        query="Coca-Cola",
        max_results=10,
        min_score=0.5,
        filters=[
            Filters(
                filter_type="with",
                fields=[
                    FilterField(
                        field_name="boolean-field",
                        field_value=True,
                        criteria=FilterOps.EQUAL_TO,
                    )
                ],
            )
        ],
    )

    @retry
    def search() -> None:
        results = document_index.search(
            read_only_collection_path, "ci-intelligence-layer", search_query
        )
        assert len(results) == 1
        assert results[0].document_path.document_name == "document-metadata-0"

    search()


def test_search_with_datetime_filter(
    document_index: DocumentIndexClient,
    read_only_collection_path: CollectionPath,
) -> None:
    search_query = SearchQuery(
        query="Coca-Cola",
        max_results=10,
        min_score=0.1,
        filters=[
            Filters(
                filter_type="with",
                fields=[
                    FilterField(
                        field_name="date-field",
                        field_value=datetime(2023, 1, 1, tzinfo=timezone.utc),
                        criteria=FilterOps.BEFORE,
                    )
                ],
            )
        ],
    )

    @retry
    def search() -> None:
        results = document_index.search(
            read_only_collection_path, "ci-intelligence-layer", search_query
        )
        assert len(results) == 1
        assert results[0].document_path.document_name == "document-metadata-0"

    search()


def test_search_with_invalid_datetime_filter(
    document_index: DocumentIndexClient,
    read_only_collection_path: CollectionPath,
) -> None:
    search_query = SearchQuery(
        query="Coca-Cola",
        max_results=10,
        min_score=0.1,
        filters=[
            Filters(
                filter_type="with",
                fields=[
                    FilterField(
                        field_name="date-field",
                        field_value="2023-01-01T12:00:00",
                        criteria=FilterOps.BEFORE,
                    )
                ],
            )
        ],
    )
    with raises(InvalidInput):
        document_index.search(
            read_only_collection_path, "ci-intelligence-layer", search_query
        )


def test_search_with_multiple_filters(
    document_index: DocumentIndexClient,
    read_only_collection_path: CollectionPath,
) -> None:
    search_query = SearchQuery(
        query="Coca-Cola",
        max_results=10,
        min_score=0.1,
        filters=[
            Filters(
                filter_type="with",
                fields=[
                    FilterField(
                        field_name="integer-field",
                        field_value=123,
                        criteria=FilterOps.EQUAL_TO,
                    ),
                    FilterField(
                        field_name="boolean-field",
                        field_value=True,
                        criteria=FilterOps.EQUAL_TO,
                    ),
                ],
            )
        ],
    )

    @retry
    def search() -> None:
        results = document_index.search(
            read_only_collection_path, "ci-intelligence-layer", search_query
        )
        assert len(results) == 1
        assert results[0].document_path.document_name == "document-metadata-0"

    search()


def test_search_with_filter_type_without(
    document_index: DocumentIndexClient,
    read_only_collection_path: CollectionPath,
) -> None:
    search_query = SearchQuery(
        query="Coca-Cola",
        max_results=10,
        min_score=0.1,
        filters=[
            Filters(
                filter_type="without",
                fields=[
                    FilterField(
                        field_name="integer-field",
                        field_value=456,
                        criteria=FilterOps.EQUAL_TO,
                    )
                ],
            )
        ],
    )

    @retry
    def search() -> None:
        results = document_index.search(
            read_only_collection_path, "ci-intelligence-layer", search_query
        )
        assert len(results) == 2

    search()


def test_search_with_filter_type_without_and_with(
    document_index: DocumentIndexClient,
    read_only_collection_path: CollectionPath,
) -> None:
    search_query = SearchQuery(
        query="Coca-Cola",
        max_results=10,
        min_score=0.1,
        filters=[
            Filters(
                filter_type="without",
                fields=[
                    FilterField(
                        field_name="integer-field",
                        field_value=456,
                        criteria=FilterOps.EQUAL_TO,
                    )
                ],
            ),
            Filters(
                filter_type="with",
                fields=[
                    FilterField(
                        field_name="boolean-field",
                        field_value=True,
                        criteria=FilterOps.EQUAL_TO,
                    )
                ],
            ),
        ],
    )

    @retry
    def search() -> None:
        results = document_index.search(
            read_only_collection_path, "ci-intelligence-layer", search_query
        )
        assert len(results) == 2
        assert results[0].document_path.document_name == "document-metadata-0"
        assert results[1].document_path.document_name == "document-metadata-2"

    search()


def test_search_with_filter_type_with_one_of(
    document_index: DocumentIndexClient,
    read_only_collection_path: CollectionPath,
) -> None:
    search_query = SearchQuery(
        query="Coca-Cola",
        max_results=10,
        min_score=0.1,
        filters=[
            Filters(
                filter_type="with_one_of",
                fields=[
                    FilterField(
                        field_name="integer-field",
                        field_value=456,
                        criteria=FilterOps.EQUAL_TO,
                    )
                ],
            ),
            Filters(
                filter_type="with_one_of",
                fields=[
                    FilterField(
                        field_name="integer-field",
                        field_value=789,
                        criteria=FilterOps.EQUAL_TO,
                    )
                ],
            ),
        ],
    )

    @retry
    def search() -> None:
        results = document_index.search(
            read_only_collection_path, "ci-intelligence-layer", search_query
        )
        assert len(results) == 2
        assert results[0].document_path.document_name == "document-metadata-1"
        assert results[1].document_path.document_name == "document-metadata-2"

    search()


def test_document_indexes_works(
    document_index: DocumentIndexClient, random_collection_path: CollectionPath
) -> None:
    document_index.progress(random_collection_path)
