from datetime import datetime, timezone
from http import HTTPStatus

import pytest
from pydantic import ValidationError
from pytest import raises

from intelligence_layer.connectors.document_index.document_index import (
    AsyncDocumentIndexClient,
    CollectionPath,
    DocumentContents,
    DocumentFilterQueryParams,
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
from tests.conftest_document_index import random_embedding_config, random_identifier, retry


@pytest.mark.internal
def test_document_index_sets_authorization_header_for_given_token() -> None:
    token = "some-token"
    async_document_index = AsyncDocumentIndexClient(token)
    assert async_document_index.headers["Authorization"] == f"Bearer {token}"


@pytest.mark.internal
def test_document_index_sets_no_authorization_header_when_token_is_none() -> None:
    async_document_index = AsyncDocumentIndexClient(None)
    assert "Authorization" not in async_document_index.headers


@pytest.mark.asyncio
@pytest.mark.internal
async def test_document_index_lists_namespaces_async(
    async_document_index: AsyncDocumentIndexClient,
    async_document_index_namespace: str,
) -> None:
    async with async_document_index as client:
        namespaces = await client.list_namespaces()
        assert async_document_index_namespace in namespaces


@pytest.mark.asyncio
@pytest.mark.internal
async def test_document_index_gets_collection_async(
    async_document_index: AsyncDocumentIndexClient, random_collection: CollectionPath
) -> None:
    async with async_document_index as client:
        collections = await client.list_collections(random_collection.namespace)
        assert random_collection in collections


@pytest.mark.asyncio
@pytest.mark.internal
@pytest.mark.parametrize(
    "document_name",
    ["Example Document", "!@#$%^&*()-_+={}[]\\|;:'\"<>,.?/~`"],
)
async def test_document_index_adds_document_async(
    async_document_index: AsyncDocumentIndexClient,
    random_collection: CollectionPath,
    document_contents: DocumentContents,
    document_name: str,
) -> None:
    async with async_document_index as client:
        document_path = DocumentPath(
            collection_path=random_collection,
            document_name=document_name,
        )

        await client.add_document(document_path, document_contents)

        assert any(
            d.document_path == document_path for d in await client.documents(random_collection)
        )
        assert document_contents == await client.document(document_path)


@pytest.mark.asyncio
@pytest.mark.internal
async def test_document_index_searches_async(
    async_document_index: AsyncDocumentIndexClient,
    read_only_populated_collection: tuple[CollectionPath, IndexPath],
) -> None:
    collection, index = read_only_populated_collection
    search_query = SearchQuery(
        query="Pemberton began his professional journey by studying medicine and pharmacy.",
        max_results=1,
        min_score=0.0,
    )

    @retry
    async def search() -> None:
        async with async_document_index as client:
            search_result = await client.search(
                collection, index.index, search_query
            )

        assert search_query.query in search_result[0].section

    await search()


@pytest.mark.asyncio
@pytest.mark.internal
@pytest.mark.parametrize(
    "document_name",
    [
        "Document to be deleted",
        "Document to be deleted !@#$%^&*()-_+={}[]\\|;:'\"<>,.?/~`",
    ],
)
async def test_document_index_deletes_document_async(
    async_document_index: AsyncDocumentIndexClient,
    random_collection: CollectionPath,
    document_contents: DocumentContents,
    document_name: str,
) -> None:
    async with async_document_index as client:
        document_path = DocumentPath(
            collection_path=random_collection, document_name=document_name
        )

        await client.add_document(document_path, document_contents)
        await client.delete_document(document_path)
        document_paths = await client.documents(document_path.collection_path)

    assert not any(d.document_path == document_path for d in document_paths)


@pytest.mark.asyncio
@pytest.mark.internal
async def test_document_index_raises_on_getting_non_existing_document_async(
    async_document_index: AsyncDocumentIndexClient, async_document_index_namespace: str
) -> None:
    async with async_document_index as client:
        non_existing_document = DocumentPath(
            collection_path=CollectionPath(
                namespace=async_document_index_namespace, collection="not"
            ),
            document_name="exist",
        )
        with raises(ResourceNotFound) as exception_info:
            await client.document(non_existing_document)
        assert exception_info.value.status_code == HTTPStatus.NOT_FOUND
        assert (
            non_existing_document.collection_path.namespace
            in exception_info.value.message
        )


@pytest.mark.asyncio
@pytest.mark.internal
async def test_document_list_all_documents_async(
    async_document_index: AsyncDocumentIndexClient,
    read_only_populated_collection: tuple[CollectionPath, IndexPath],
) -> None:
    async with async_document_index as client:
        filter_result = await client.documents(read_only_populated_collection[0])

        assert len(filter_result) == 3


@pytest.mark.asyncio
@pytest.mark.internal
async def test_document_list_max_n_documents_async(
    async_document_index: AsyncDocumentIndexClient,
    read_only_populated_collection: tuple[CollectionPath, IndexPath],
) -> None:
    async with async_document_index as client:
        filter_query_params = DocumentFilterQueryParams(max_documents=1, starts_with=None)

        filter_result = await client.documents(
            read_only_populated_collection[0], filter_query_params
        )

        assert len(filter_result) == 1


@pytest.mark.asyncio
@pytest.mark.internal
async def test_document_list_documents_with_matching_prefix_async(
    async_document_index: AsyncDocumentIndexClient, random_collection: CollectionPath
) -> None:
    async with async_document_index as client:
        await client.add_document(
            document_path=DocumentPath(
                collection_path=random_collection, document_name="Example document"
            ),
            contents=DocumentContents.from_text("Document with matching prefix"),
        )
        await client.add_document(
            document_path=DocumentPath(
                collection_path=random_collection, document_name="Another document"
            ),
            contents=DocumentContents.from_text("Document without matching prefix"),
        )
        prefix = "Example"
        filter_query_params = DocumentFilterQueryParams(
            max_documents=None, starts_with=prefix
        )

        filter_result = await client.documents(random_collection, filter_query_params)

        assert len(filter_result) == 1
        assert filter_result[0].document_path.document_name.startswith(prefix)


@pytest.mark.asyncio
@pytest.mark.internal
async def test_semantic_indexes_in_namespace_are_returned_async(
    async_document_index: AsyncDocumentIndexClient,
    random_semantic_index: tuple[IndexPath, IndexConfiguration],
) -> None:
    async with async_document_index as client:
        index_path, index_configuration = random_semantic_index
        retrieved_index_configuration = await client.index_configuration(index_path)

        assert retrieved_index_configuration == index_configuration


@pytest.mark.asyncio
@pytest.mark.internal
async def test_instructable_indexes_in_namespace_are_returned_async(
    async_document_index: AsyncDocumentIndexClient,
    random_instructable_index: tuple[IndexPath, IndexConfiguration],
) -> None:
    async with async_document_index as client:
        index_path, index_configuration = random_instructable_index
        retrieved_index_configuration = await client.index_configuration(index_path)

        assert retrieved_index_configuration == index_configuration


@pytest.mark.asyncio
@pytest.mark.internal
async def test_indexes_for_collection_are_returned_async(
    async_document_index: AsyncDocumentIndexClient,
    read_only_populated_collection: tuple[CollectionPath, IndexPath],
) -> None:
    async with async_document_index as client:
        index_names = await client.list_assigned_index_names(
            read_only_populated_collection[0]
        )
        assert read_only_populated_collection[1].index in index_names


@pytest.mark.asyncio
@pytest.mark.internal
async def test_create_filter_indexes_in_namespace_async(
    async_document_index: AsyncDocumentIndexClient,
    document_index_namespace: str,
    filter_index_configs: dict[str, dict[str, str]],
) -> None:
    async with async_document_index as client:
        for index_name, index_config in filter_index_configs.items():
            await client.create_filter_index_in_namespace(
                namespace=document_index_namespace,
                filter_index_name=index_name,
                field_name=index_config["field-name"],
                field_type=index_config["field-type"],  # type:ignore[arg-type]
            )

        indexes = await client.list_filter_indexes_in_namespace(document_index_namespace)
        assert all(filter_index in indexes for filter_index in filter_index_configs)


@pytest.mark.asyncio
@pytest.mark.internal
async def test_create_filter_index_invalid_name_async(
    async_document_index: AsyncDocumentIndexClient, async_document_index_namespace: str
) -> None:
    async with async_document_index as client:
        with pytest.raises(ValueError) as context:
            await client.create_filter_index_in_namespace(
                async_document_index_namespace, "invalid index!", "field_name", "string"
            )
            assert (
                str(context.value)
                == "Filter index name can only contain alphanumeric characters (a-z, A-Z, -, . and 0-9)."
            )


@pytest.mark.asyncio
@pytest.mark.internal
async def test_create_filter_index_name_too_long_async(
    async_document_index: AsyncDocumentIndexClient, async_document_index_namespace: str
) -> None:
    async with async_document_index as client:
        with pytest.raises(ValueError) as context:
            await client.create_filter_index_in_namespace(
                async_document_index_namespace, "a" * 51, "field_name", "string"
            )
        assert (
            str(context.value) == "Filter index name cannot be longer than 50 characters."
        )


@pytest.mark.asyncio
@pytest.mark.internal
async def test_assign_filter_indexes_to_collection_async(
    async_document_index: AsyncDocumentIndexClient,
    random_searchable_collection: tuple[CollectionPath, IndexPath],
    filter_index_configs: dict[str, dict[str, str]],
) -> None:
    async with async_document_index as client:
        collection_path, index_path = random_searchable_collection
        index_name = index_path.index

        for filter_index_name in filter_index_configs:
            await client.assign_filter_index_to_search_index(
                collection_path=collection_path,
                index_name=index_name,
                filter_index_name=filter_index_name,
            )

        assigned_indexes = await client.list_assigned_filter_index_names(
            collection_path, index_name
        )
        assert all(
            filter_index in assigned_indexes for filter_index in filter_index_configs
        )


@pytest.mark.asyncio
@pytest.mark.internal
async def test_document_index_adds_documents_with_metadata_async(
    async_document_index: AsyncDocumentIndexClient,
    random_collection: CollectionPath,
    document_contents_with_metadata: list[DocumentContents],
) -> None:
    async with async_document_index as client:
        for i, doc_content in enumerate(document_contents_with_metadata):
            document_path = DocumentPath(
                collection_path=random_collection,
                document_name=f"document-metadata-{i}",
            )
            await client.add_document(document_path, doc_content)

            assert any(
                d.document_path == document_path
                for d in await client.documents(random_collection)
            )
            assert doc_content == await client.document(document_path)


@pytest.mark.asyncio
@pytest.mark.internal
async def test_search_with_string_filter_async(
    async_document_index: AsyncDocumentIndexClient,
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
    async def search() -> None:
        async with async_document_index as client:
            collection_path, index_path = read_only_populated_collection
            results = await client.search(
                collection_path, index_path.index, search_query
            )
        assert results[0].document_path.document_name == "document-0"

    await search()


@pytest.mark.asyncio
@pytest.mark.internal
async def test_search_with_null_filter_async(
    async_document_index: AsyncDocumentIndexClient,
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
    async def search() -> None:
        async with async_document_index as client:
            collection_path, index_path = read_only_populated_collection
            results = await client.search(
                collection_path, index_path.index, search_query
            )
        assert len(results) == 1
        assert results[0].document_path.document_name == "document-0"

    await search()


@pytest.mark.asyncio
@pytest.mark.internal
async def test_search_with_null_filter_without_async(
    async_document_index: AsyncDocumentIndexClient,
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
    async def search() -> None:
        async with async_document_index as client:
            collection_path, index_path = read_only_populated_collection
            results = await client.search(
                collection_path, index_path.index, search_query
            )
        assert len(results) == 2
        assert {r.document_path.document_name for r in results} == {
            "document-1",
            "document-2",
        }

    await search()


@pytest.mark.asyncio
@pytest.mark.internal
async def test_search_with_integer_filter_async(
    async_document_index: AsyncDocumentIndexClient,
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
    async def search() -> None:
        async with async_document_index as client:
            collection_path, index_path = read_only_populated_collection
            results = await client.search(
                collection_path, index_path.index, search_query
            )
        assert len(results) == 1
        assert results[0].document_path.document_name == "document-0"

    await search()


@pytest.mark.asyncio
@pytest.mark.internal
async def test_search_with_float_filter_async(
    async_document_index: AsyncDocumentIndexClient,
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
    async def search() -> None:
        async with async_document_index as client:
            collection_path, index_path = read_only_populated_collection
            results = await client.search(
                collection_path, index_path.index, search_query
            )
        assert len(results) == 2
        assert results[0].document_path.document_name == "document-1"
        assert results[1].document_path.document_name == "document-2"

    await search()


@pytest.mark.asyncio
@pytest.mark.internal
async def test_search_with_boolean_filter_async(
    async_document_index: AsyncDocumentIndexClient,
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
    async def search() -> None:
        async with async_document_index as client:
            collection_path, index_path = read_only_populated_collection
            results = await client.search(
                collection_path, index_path.index, search_query
            )
        assert len(results) == 1
        assert results[0].document_path.document_name == "document-0"

    await search()


@pytest.mark.asyncio
@pytest.mark.internal
async def test_search_with_datetime_filter_async(
    async_document_index: AsyncDocumentIndexClient,
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
    async def search() -> None:
        async with async_document_index as client:
            collection_path, index_path = read_only_populated_collection
            results = await client.search(
                collection_path, index_path.index, search_query
            )
        assert len(results) == 1
        assert results[0].document_path.document_name == "document-0"

    await search()


@pytest.mark.asyncio
@pytest.mark.internal
async def test_search_with_invalid_datetime_filter_async(
    async_document_index: AsyncDocumentIndexClient,
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
        async with async_document_index as client:
            collection_path, index_path = read_only_populated_collection
            await client.search(
                collection_path, index_path.index, search_query
            )


@pytest.mark.asyncio
@pytest.mark.internal
async def test_search_with_multiple_filters_async(
    async_document_index: AsyncDocumentIndexClient,
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
    async def search() -> None:
        async with async_document_index as client:
            collection_path, index_path = read_only_populated_collection
            results = await client.search(
                collection_path, index_path.index, search_query
            )
        assert len(results) == 1
        assert results[0].document_path.document_name == "document-0"

    await search()


@pytest.mark.asyncio
@pytest.mark.internal
async def test_search_with_filter_type_without_async(
    async_document_index: AsyncDocumentIndexClient,
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
    async def search() -> None:
        async with async_document_index as client:
            collection_path, index_path = read_only_populated_collection
            results = await client.search(
                collection_path, index_path.index, search_query
            )
        assert len(results) == 2

    await search()


@pytest.mark.asyncio
@pytest.mark.internal
async def test_search_with_filter_type_without_and_with_async(
    async_document_index: AsyncDocumentIndexClient,
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
    async def search() -> None:
        async with async_document_index as client:
            collection_path, index_path = read_only_populated_collection
            results = await client.search(
                collection_path, index_path.index, search_query
            )
        assert len(results) == 2
        assert results[0].document_path.document_name == "document-0"
        assert results[1].document_path.document_name == "document-2"

    await search()


@pytest.mark.asyncio
@pytest.mark.internal
async def test_search_with_filter_type_with_one_of_async(
    async_document_index: AsyncDocumentIndexClient,
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
    async def search() -> None:
        async with async_document_index as client:
            collection_path, index_path = read_only_populated_collection
            results = await client.search(
                collection_path, index_path.index, search_query
            )
        assert len(results) == 2
        assert results[0].document_path.document_name == "document-1"
        assert results[1].document_path.document_name == "document-2"

    await search()


@pytest.mark.asyncio
@pytest.mark.internal
async def test_document_indexes_works_async(
    async_document_index: AsyncDocumentIndexClient, random_collection: CollectionPath
) -> None:
    async with async_document_index as client:
        await client.progress(random_collection)


@pytest.mark.asyncio
@pytest.mark.internal
async def test_retrieve_chunks_async(
    async_document_index: AsyncDocumentIndexClient,
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
    async with async_document_index as client:
        await client.create_index(index_path, index_configuration)
        await client.assign_index_to_collection(random_collection, index_name)

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
        await client.add_document(document_path, document_contents)

        @retry
        async def chunks() -> None:
            async with async_document_index as client:
                chunks = await client.chunks(document_path, index_name)
                assert len(chunks) == 3

        await chunks()