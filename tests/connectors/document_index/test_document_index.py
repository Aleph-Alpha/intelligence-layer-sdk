from http import HTTPStatus

import pytest
from pytest import fixture, raises

from intelligence_layer.connectors.document_index.document_index import (
    CollectionPath,
    DocumentContents,
    DocumentIndexClient,
    DocumentPath,
    ResourceNotFound,
    SearchQuery,
)


@fixture
def collection_path() -> CollectionPath:
    return CollectionPath(namespace="aleph-alpha", collection="ci-collection")


@fixture
def document_path(
    document_index: DocumentIndexClient, collection_path: CollectionPath
) -> DocumentPath:
    document_index.create_collection(collection_path)
    return DocumentPath(
        collection_path=collection_path, document_name="Example Document"
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
def test_document_index_adds_document(
    document_index: DocumentIndexClient,
    document_path: DocumentPath,
    document_contents: DocumentContents,
) -> None:
    document_index.add_document(document_path, document_contents)
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
    search_result = document_index.asymmetric_search(
        document_path.collection_path, search_query
    )

    assert "Mark" in search_result[0].section


@pytest.mark.internal
def test_document_index_deletes_document(
    document_index: DocumentIndexClient, collection_path: CollectionPath
) -> None:
    document_path = DocumentPath(
        collection_path=collection_path, document_name="Document to be deleted"
    )
    document_contents = DocumentContents.from_text("Some text...")

    document_index.add_document(document_path, document_contents)
    document_index.delete_document(document_path)
    document_paths = document_index.list_documents(document_path.collection_path)

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


def test_document_path_from_string() -> None:
    abc = DocumentPath.from_slash_separated_str("a/b/c")
    assert abc == DocumentPath(
        collection_path=CollectionPath(namespace="a", collection="b"), document_name="c"
    )
    with raises(AssertionError):
        DocumentPath.from_slash_separated_str("a/c")
