from datetime import datetime, timezone
from http import HTTPStatus

import pytest
from pydantic import ValidationError
from pytest import fixture, raises

from intelligence_layer.connectors.document_index.document_index import (
    CollectionPath,
    DocumentContents,
    DocumentFilterQueryParams,
    DocumentIndexClient,
    DocumentPath,
    FilterField,
    FilterOps,
    Filters,
    IndexConfiguration,
    IndexPath,
    InvalidInput,
    ResourceNotFound,
    SearchQuery,
)


@fixture
def aleph_alpha_namespace() -> str:
    return "aleph-alpha"


@fixture
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
def collection_path(aleph_alpha_namespace: str) -> CollectionPath:
    return CollectionPath(
        namespace=aleph_alpha_namespace, collection="intelligence-layer-sdk-ci"
    )


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


@fixture
def document_contents_with_metadata() -> list[DocumentContents]:
    text_1 = """John Stith Pemberton, the inventor of the world-renowned beverage Coca-Cola, was a figure whose life was marked by creativity, entrepreneurial spirit, and the turbulent backdrop of 19th-century America. Born on January 8, 1831, in Knoxville, Georgia, Pemberton grew up in an era of profound transformation and change."""
    text_2 = """Pemberton began his professional journey by studying medicine and pharmacy. After earning a degree in pharmacy, he started his career as a druggist in Columbus, Georgia. He was known for his keen interest in creating medicinal concoctions and was well-respected in his community. His early creations included various medicines and tonics, which were typical of the times when pharmacists often concocted their own remedies."""
    text_3 = """Pemberton's life took a significant turn during the American Civil War. He served as a lieutenant colonel in the Confederate Army, and it was during this period that he sustained a wound that led him to become dependent on morphine. This personal struggle with addiction likely influenced his later work in seeking out alternatives and remedies for pain relief."""

    metadata_1 = {
        "string-field": "example_string_1",
        "integer-field": 123,
        "float-field": 123.45,
        "boolean-field": True,
        "date-field": datetime(2022, 1, 1, tzinfo=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    }

    metadata_2 = {
        "string-field": "example_string_2",
        "integer-field": 456,
        "float-field": 678.90,
        "boolean-field": False,
        "date-field": datetime(2023, 1, 1, tzinfo=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    }

    metadata_3 = {
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
def test_document_index_creates_collection(
    document_index: DocumentIndexClient, collection_path: CollectionPath
) -> None:
    document_index.create_collection(collection_path)
    collections = document_index.list_collections(collection_path.namespace)

    assert collection_path in collections


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
    collection_path: CollectionPath,
    document_contents: DocumentContents,
    document_name: str,
) -> None:
    document_path = DocumentPath(
        collection_path=collection_path,
        document_name=document_name,
    )
    document_index.add_document(document_path, document_contents)

    assert any(
        d.document_path == document_path
        for d in document_index.documents(collection_path)
    )
    assert document_contents == document_index.document(document_path)


@pytest.mark.internal
def test_document_index_searches_asymmetrically(
    document_index: DocumentIndexClient, collection_path: CollectionPath
) -> None:
    document_path = DocumentPath(
        collection_path=collection_path,
        document_name="test_document_index_searches_asymmetrically",  # is always there
    )
    search_query = SearchQuery(query="Who likes pizza?", max_results=1, min_score=0.0)
    search_result = document_index.search(
        document_path.collection_path, "asymmetric", search_query
    )

    assert "Mark" in search_result[0].section


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
    collection_path: CollectionPath,
    document_name: str,
) -> None:
    document_path = DocumentPath(
        collection_path=collection_path, document_name=document_name
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
    document_index: DocumentIndexClient, collection_path: CollectionPath
) -> None:
    filter_result = document_index.documents(collection_path)

    assert len(filter_result) == 6


def test_document_list_max_n_documents(
    document_index: DocumentIndexClient, collection_path: CollectionPath
) -> None:
    filter_query_params = DocumentFilterQueryParams(max_documents=1, starts_with=None)

    filter_result = document_index.documents(collection_path, filter_query_params)

    assert len(filter_result) == 1


def test_document_list_documents_with_matching_prefix(
    document_index: DocumentIndexClient, collection_path: CollectionPath
) -> None:
    prefix = "Example"
    filter_query_params = DocumentFilterQueryParams(
        max_documents=None, starts_with=prefix
    )

    filter_result = document_index.documents(collection_path, filter_query_params)

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


def test_document_indexes_are_returned(
    document_index: DocumentIndexClient, collection_path: CollectionPath
) -> None:
    index_names = document_index.list_assigned_index_names(collection_path)
    index_name = index_names[0]
    index_configuration = document_index.index_configuration(
        IndexPath(namespace=collection_path.namespace, index=index_name)
    )

    assert index_configuration.embedding_type == "asymmetric"
    assert index_configuration.chunk_overlap == 0
    assert index_configuration.chunk_size == 512


def test_create_filter_indexes_in_namespace(
    document_index: DocumentIndexClient,
    aleph_alpha_namespace: str,
    filter_index_config: dict[str, dict[str, str]],
) -> None:
    for index_name, index_config in filter_index_config.items():
        document_index.create_filter_index_in_namespace(
            namespace=aleph_alpha_namespace,
            filter_index_name=index_name,
            field_name=index_config["field-name"],
            field_type=index_config["field-type"],
        )

    assert all(
        filter_index
        in document_index.list_filter_indexes_in_namespace(aleph_alpha_namespace)
        for filter_index in filter_index_config
    )


def test_create_filter_index_invalid_name(
    document_index: DocumentIndexClient, aleph_alpha_namespace: str
) -> None:
    with pytest.raises(ValueError) as context:
        document_index.create_filter_index_in_namespace(
            aleph_alpha_namespace, "invalid index!", "field_name", "string"
        )
        assert (
            str(context.value)
            == "Filter index name can only contain alphanumeric characters (a-z, A-Z, -, . and 0-9)."
        )


def test_create_filter_index_name_too_long(
    document_index: DocumentIndexClient, aleph_alpha_namespace: str
) -> None:
    with pytest.raises(ValueError) as context:
        document_index.create_filter_index_in_namespace(
            aleph_alpha_namespace, "a" * 51, "field_name", "string"
        )
    assert (
        str(context.value) == "Filter index name cannot be longer than 50 characters."
    )


def test_assign_filter_indexes_to_collection(
    document_index: DocumentIndexClient,
    collection_path: CollectionPath,
    filter_index_config: dict[str, dict[str, str]],
) -> None:
    for index_name in filter_index_config:
        document_index.assign_filter_index_to_search_index(
            collection_path=collection_path,
            filter_index_name=index_name,
            index_name="asymmetric",
        )

    assert all(
        filter_index
        in document_index.list_assigned_filter_index_names(
            collection_path, "asymmetric"
        )
        for filter_index in filter_index_config
    )


def test_document_index_adds_documents_with_metadata(
    document_index: DocumentIndexClient,
    collection_path: CollectionPath,
    document_contents_with_metadata: list[DocumentContents],
) -> None:
    for i, doc_content in enumerate(document_contents_with_metadata):
        document_path = DocumentPath(
            collection_path=collection_path,
            document_name=f"document-metadata-{i}",
        )
        document_index.add_document(document_path, doc_content)

        assert any(
            d.document_path == document_path
            for d in document_index.documents(collection_path)
        )
        assert doc_content == document_index.document(document_path)


def test_search_with_string_filter(
    document_index: DocumentIndexClient,
    collection_path: CollectionPath,
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
    results = document_index.search(collection_path, "asymmetric", search_query)
    assert results[0].document_path.document_name == "document-metadata-0"


def test_search_with_integer_filter(
    document_index: DocumentIndexClient,
    collection_path: CollectionPath,
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
    results = document_index.search(collection_path, "asymmetric", search_query)
    assert len(results) == 1
    assert results[0].document_path.document_name == "document-metadata-0"


def test_search_with_float_filter(
    document_index: DocumentIndexClient,
    collection_path: CollectionPath,
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
    results = document_index.search(collection_path, "asymmetric", search_query)
    assert len(results) == 2
    assert results[0].document_path.document_name == "document-metadata-1"
    assert results[1].document_path.document_name == "document-metadata-2"


def test_search_with_boolean_filter(
    document_index: DocumentIndexClient,
    collection_path: CollectionPath,
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
    results = document_index.search(collection_path, "asymmetric", search_query)
    assert len(results) == 1
    assert results[0].document_path.document_name == "document-metadata-0"


def test_search_with_datetime_filter(
    document_index: DocumentIndexClient,
    collection_path: CollectionPath,
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
    results = document_index.search(collection_path, "asymmetric", search_query)
    assert len(results) == 1
    assert results[0].document_path.document_name == "document-metadata-0"


def test_search_with_invalid_datetime_filter(
    document_index: DocumentIndexClient,
    collection_path: CollectionPath,
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
        document_index.search(collection_path, "asymmetric", search_query)


def test_search_with_multiple_filters(
    document_index: DocumentIndexClient,
    collection_path: CollectionPath,
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
    results = document_index.search(collection_path, "asymmetric", search_query)
    assert len(results) == 1
    assert results[0].document_path.document_name == "document-metadata-0"


def test_search_with_filter_type_without(
    document_index: DocumentIndexClient,
    collection_path: CollectionPath,
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
    results = document_index.search(collection_path, "asymmetric", search_query)
    assert len(results) == 7


def test_search_with_filter_type_without_and_with(
    document_index: DocumentIndexClient,
    collection_path: CollectionPath,
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
    results = document_index.search(collection_path, "asymmetric", search_query)
    assert len(results) == 2
    assert results[0].document_path.document_name == "document-metadata-0"
    assert results[1].document_path.document_name == "document-metadata-2"


def test_search_with_filter_type_with_one_of(
    document_index: DocumentIndexClient,
    collection_path: CollectionPath,
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
    results = document_index.search(collection_path, "asymmetric", search_query)
    assert len(results) == 2
    assert results[0].document_path.document_name == "document-metadata-1"
    assert results[1].document_path.document_name == "document-metadata-2"
