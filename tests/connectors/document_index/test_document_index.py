from datetime import datetime, timezone
from http import HTTPStatus

import pytest
from pydantic import ValidationError
from pytest import raises

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
    SemanticEmbed,
)
from tests.conftest_document_index import (
    random_embedding_config,
    random_identifier,
    retry,
)

pytestmark = pytest.mark.document_index


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
def test_document_index_lists_namespaces(
    document_index: DocumentIndexClient,
    document_index_namespace: str,
) -> None:
    namespaces = document_index.list_namespaces()

    assert document_index_namespace in namespaces


@pytest.mark.internal
def test_document_index_gets_collection(
    document_index: DocumentIndexClient, random_collection: CollectionPath
) -> None:
    collections = document_index.list_collections(random_collection.namespace)

    assert random_collection in collections


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
    random_collection: CollectionPath,
    document_contents: DocumentContents,
    document_name: str,
) -> None:
    document_path = DocumentPath(
        collection_path=random_collection,
        document_name=document_name,
    )
    document_index.add_document(document_path, document_contents)

    assert any(
        d.document_path == document_path
        for d in document_index.documents(random_collection)
    )
    assert document_contents == document_index.document(document_path)


@pytest.mark.internal
def test_document_index_searches(
    document_index: DocumentIndexClient,
    read_only_populated_collection: tuple[CollectionPath, IndexPath],
) -> None:
    collection, index = read_only_populated_collection
    search_query = SearchQuery(
        query="Pemberton began his professional journey by studying medicine and pharmacy.",
        max_results=1,
        min_score=0.0,
    )

    @retry
    def search() -> None:
        search_result = document_index.search(
            collection,
            index.index,
            search_query,
        )

        assert search_query.query in search_result[0].section

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
    random_collection: CollectionPath,
    document_contents: DocumentContents,
    document_name: str,
) -> None:
    document_path = DocumentPath(
        collection_path=random_collection, document_name=document_name
    )

    document_index.add_document(document_path, document_contents)
    document_index.delete_document(document_path)
    document_paths = document_index.documents(document_path.collection_path)

    assert not any(d.document_path == document_path for d in document_paths)


def test_document_index_raises_on_getting_non_existing_document(
    document_index: DocumentIndexClient, document_index_namespace: str
) -> None:
    non_existing_document = DocumentPath(
        collection_path=CollectionPath(
            namespace=document_index_namespace, collection="not"
        ),
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
    document_index: DocumentIndexClient,
    read_only_populated_collection: tuple[CollectionPath, IndexPath],
) -> None:
    filter_result = document_index.documents(read_only_populated_collection[0])

    assert len(filter_result) == 3


def test_document_list_max_n_documents(
    document_index: DocumentIndexClient,
    read_only_populated_collection: tuple[CollectionPath, IndexPath],
) -> None:
    filter_query_params = DocumentFilterQueryParams(max_documents=1, starts_with=None)

    filter_result = document_index.documents(
        read_only_populated_collection[0], filter_query_params
    )

    assert len(filter_result) == 1


def test_document_list_documents_with_matching_prefix(
    document_index: DocumentIndexClient, random_collection: CollectionPath
) -> None:
    document_index.add_document(
        document_path=DocumentPath(
            collection_path=random_collection, document_name="Example document"
        ),
        contents=DocumentContents.from_text("Document with matching prefix"),
    )
    document_index.add_document(
        document_path=DocumentPath(
            collection_path=random_collection, document_name="Another document"
        ),
        contents=DocumentContents.from_text("Document without matching prefix"),
    )
    prefix = "Example"
    filter_query_params = DocumentFilterQueryParams(
        max_documents=None, starts_with=prefix
    )

    filter_result = document_index.documents(random_collection, filter_query_params)

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
            chunk_size=128,
            chunk_overlap=128,
            embedding=random_embedding_config(),
        )
    except ValidationError as e:
        assert "chunk_overlap must be less than chunk_size" in str(e)
    else:
        raise AssertionError("ValidationError was not raised")


def test_semantic_indexes_in_namespace_are_returned(
    document_index: DocumentIndexClient,
    random_semantic_index: tuple[IndexPath, IndexConfiguration],
) -> None:
    index_path, index_configuration = random_semantic_index
    retrieved_index_configuration = document_index.index_configuration(index_path)

    assert retrieved_index_configuration == index_configuration


def test_instructable_indexes_in_namespace_are_returned(
    document_index: DocumentIndexClient,
    random_instructable_index: tuple[IndexPath, IndexConfiguration],
) -> None:
    index_path, index_configuration = random_instructable_index
    retrieved_index_configuration = document_index.index_configuration(index_path)

    assert retrieved_index_configuration == index_configuration


def test_indexes_for_collection_are_returned(
    document_index: DocumentIndexClient,
    read_only_populated_collection: tuple[CollectionPath, IndexPath],
) -> None:
    index_names = document_index.list_assigned_index_names(
        read_only_populated_collection[0]
    )
    assert read_only_populated_collection[1].index in index_names


def test_create_filter_indexes_in_namespace(
    document_index: DocumentIndexClient,
    document_index_namespace: str,
    filter_index_configs: dict[str, dict[str, str]],
) -> None:
    for index_name, index_config in filter_index_configs.items():
        document_index.create_filter_index_in_namespace(
            namespace=document_index_namespace,
            filter_index_name=index_name,
            field_name=index_config["field-name"],
            field_type=index_config["field-type"],  # type:ignore[arg-type]
        )

    assert all(
        filter_index
        in document_index.list_filter_indexes_in_namespace(document_index_namespace)
        for filter_index in filter_index_configs
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
    random_searchable_collection: tuple[CollectionPath, IndexPath],
    filter_index_configs: dict[str, dict[str, str]],
) -> None:
    collection_path, index_path = random_searchable_collection
    index_name = index_path.index

    for filter_index_name in filter_index_configs:
        document_index.assign_filter_index_to_search_index(
            collection_path=collection_path,
            index_name=index_name,
            filter_index_name=filter_index_name,
        )

    assigned_indexes = document_index.list_assigned_filter_index_names(
        collection_path, index_name
    )
    assert all(
        filter_index in assigned_indexes for filter_index in filter_index_configs
    )


def test_document_index_adds_documents_with_metadata(
    document_index: DocumentIndexClient,
    random_collection: CollectionPath,
    document_contents_with_metadata: list[DocumentContents],
) -> None:
    for i, doc_content in enumerate(document_contents_with_metadata):
        document_path = DocumentPath(
            collection_path=random_collection,
            document_name=f"document-metadata-{i}",
        )
        document_index.add_document(document_path, doc_content)

        assert any(
            d.document_path == document_path
            for d in document_index.documents(random_collection)
        )
        assert doc_content == document_index.document(document_path)


def test_search_with_string_filter(
    document_index: DocumentIndexClient,
    read_only_populated_collection: tuple[CollectionPath, IndexPath],
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
        collection_path, index_path = read_only_populated_collection
        results = document_index.search(
            collection_path,
            index_path.index,
            search_query,
        )
        assert results[0].document_path.document_name == "document-0"

    search()


def test_search_with_null_filter(
    document_index: DocumentIndexClient,
    read_only_populated_collection: tuple[CollectionPath, IndexPath],
) -> None:
    search_query = SearchQuery(
        query="Pemberton",
        max_results=10,
        min_score=0.5,
        filters=[
            Filters(
                filter_type="with",
                fields=[
                    FilterField(
                        field_name="option-field",
                        field_value=True,
                        criteria=FilterOps.IS_NULL,
                    )
                ],
            )
        ],
    )

    @retry
    def search() -> None:
        collection_path, index_path = read_only_populated_collection
        results = document_index.search(
            collection_path,
            index_path.index,
            search_query,
        )
        assert len(results) == 1
        assert results[0].document_path.document_name == "document-0"

    search()


def test_search_with_null_filter_without(
    document_index: DocumentIndexClient,
    read_only_populated_collection: tuple[CollectionPath, IndexPath],
) -> None:
    search_query = SearchQuery(
        query="Pemberton",
        max_results=10,
        min_score=0.5,
        filters=[
            Filters(
                filter_type="without",
                fields=[
                    FilterField(
                        field_name="option-field",
                        field_value=True,
                        criteria=FilterOps.IS_NULL,
                    )
                ],
            )
        ],
    )

    @retry
    def search() -> None:
        collection_path, index_path = read_only_populated_collection
        results = document_index.search(
            collection_path,
            index_path.index,
            search_query,
        )
        assert len(results) == 2
        assert {r.document_path.document_name for r in results} == {
            "document-1",
            "document-2",
        }

    search()


def test_search_with_integer_filter(
    document_index: DocumentIndexClient,
    read_only_populated_collection: tuple[CollectionPath, IndexPath],
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
        collection_path, index_path = read_only_populated_collection
        results = document_index.search(
            collection_path,
            index_path.index,
            search_query,
        )
        assert len(results) == 1
        assert results[0].document_path.document_name == "document-0"

    search()


def test_search_with_float_filter(
    document_index: DocumentIndexClient,
    read_only_populated_collection: tuple[CollectionPath, IndexPath],
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
        collection_path, index_path = read_only_populated_collection
        results = document_index.search(
            collection_path,
            index_path.index,
            search_query,
        )
        assert len(results) == 2
        assert results[0].document_path.document_name == "document-1"
        assert results[1].document_path.document_name == "document-2"

    search()


def test_search_with_boolean_filter(
    document_index: DocumentIndexClient,
    read_only_populated_collection: tuple[CollectionPath, IndexPath],
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
        collection_path, index_path = read_only_populated_collection
        results = document_index.search(
            collection_path,
            index_path.index,
            search_query,
        )
        assert len(results) == 1
        assert results[0].document_path.document_name == "document-0"

    search()


def test_search_with_datetime_filter(
    document_index: DocumentIndexClient,
    read_only_populated_collection: tuple[CollectionPath, IndexPath],
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
        collection_path, index_path = read_only_populated_collection
        results = document_index.search(
            collection_path,
            index_path.index,
            search_query,
        )
        assert len(results) == 1
        assert results[0].document_path.document_name == "document-0"

    search()


def test_search_with_invalid_datetime_filter(
    document_index: DocumentIndexClient,
    read_only_populated_collection: tuple[CollectionPath, IndexPath],
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
        collection_path, index_path = read_only_populated_collection
        document_index.search(
            collection_path,
            index_path.index,
            search_query,
        )


def test_search_with_multiple_filters(
    document_index: DocumentIndexClient,
    read_only_populated_collection: tuple[CollectionPath, IndexPath],
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
        collection_path, index_path = read_only_populated_collection
        results = document_index.search(
            collection_path,
            index_path.index,
            search_query,
        )
        assert len(results) == 1
        assert results[0].document_path.document_name == "document-0"

    search()


def test_search_with_filter_type_without(
    document_index: DocumentIndexClient,
    read_only_populated_collection: tuple[CollectionPath, IndexPath],
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
        collection_path, index_path = read_only_populated_collection
        results = document_index.search(
            collection_path,
            index_path.index,
            search_query,
        )
        assert len(results) == 2

    search()


def test_search_with_filter_type_without_and_with(
    document_index: DocumentIndexClient,
    read_only_populated_collection: tuple[CollectionPath, IndexPath],
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
        collection_path, index_path = read_only_populated_collection
        results = document_index.search(
            collection_path,
            index_path.index,
            search_query,
        )
        assert len(results) == 2
        assert results[0].document_path.document_name == "document-0"
        assert results[1].document_path.document_name == "document-2"

    search()


def test_search_with_filter_type_with_one_of(
    document_index: DocumentIndexClient,
    read_only_populated_collection: tuple[CollectionPath, IndexPath],
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
        collection_path, index_path = read_only_populated_collection
        results = document_index.search(
            collection_path,
            index_path.index,
            search_query,
        )
        assert len(results) == 2
        assert results[0].document_path.document_name == "document-1"
        assert results[1].document_path.document_name == "document-2"

    search()


def test_document_indexes_works(
    document_index: DocumentIndexClient, random_collection: CollectionPath
) -> None:
    document_index.progress(random_collection)


def test_retrieve_chunks(
    document_index: DocumentIndexClient,
    random_collection: CollectionPath,
    document_index_namespace: str,
) -> None:
    index_name = random_identifier()
    index_path = IndexPath(namespace=document_index_namespace, index=index_name)
    index_configuration = IndexConfiguration(
        chunk_size=512,
        chunk_overlap=0,
        embedding=SemanticEmbed(
            representation="asymmetric",
            model_name="luminous-base",
        ),
    )
    document_index.create_index(index_path, index_configuration)
    document_index.assign_index_to_collection(random_collection, index_name)

    document_path = DocumentPath(
        collection_path=random_collection,
        document_name="document-with-chunks",
    )
    document_contents = DocumentContents(
        contents=[
            # because chunk size is 512, this item will be split into 2 chunks
            " token" * 750,
            "final chunk",
        ],
    )
    document_index.add_document(document_path, document_contents)

    @retry
    def chunks() -> None:
        chunks = document_index.chunks(document_path, index_name)
        assert len(chunks) == 3

    chunks()
