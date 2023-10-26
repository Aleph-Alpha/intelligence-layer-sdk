import json
from typing import Any, Sequence

import requests

from intelligence_layer.connectors.retrievers.base_retriever import (
    BaseRetriever,
    SearchResult,
)


class DocumentIndex:
    """Client for the Document Index allowing handling documents and search.

    Document Index is a tool for managing collections of documents, enabling operations such as creation, deletion, listing, and searching.
    Documents can be stored either in the cloud or in a local deployment.

    Args:
        token: A valid token for the document index API.
        base_document_index_url: The url of the document index' API.

    Example:
        >>> document_index = DocumentIndex(os.getenv("AA_TOKEN"))
        >>> document_index.create_collection(namespace="my_namespace", collection="germany_facts_collection")
        >>> document_index.add_document(
                namespace="my_namespace",
                collection="germany_facts_collection",
                name="Fun facts about Germany",
                content="Germany is a country located in ...")
        >>> documents = document_index.search(
        >>>     namespace="my_namespace",
        >>>     collection="germany_facts_collection",
        >>>     query: "What is the capital of Germany",
        >>>     max_results=4,
        >>>     min_score: 0.5
        >>> )
    """

    def __init__(
        self,
        token: str,
        base_document_index_url: str = "https://knowledge.aleph-alpha.com",
    ) -> None:
        self._base_document_index_url = base_document_index_url
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }

    def create_collection(self, namespace: str, collection: str) -> None:
        url = f"{self._base_document_index_url}/collections/{namespace}/{collection}"
        response = requests.put(url, headers=self.headers)
        response.raise_for_status()

    def delete_collection(self, namespace: str, collection: str) -> None:
        url = f"{self._base_document_index_url}/collections/{namespace}/{collection}"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()

    def add_document(
        self,
        namespace: str,
        collection: str,
        name: str,
        content: str,
    ) -> None:
        url = f"{self._base_document_index_url}/collections/{namespace}/{collection}/docs/{name}"
        data = {
            "schema_version": "V1",
            "contents": [{"modality": "text", "text": content}],
        }
        response = requests.put(url, data=json.dumps(data), headers=self.headers)
        response.raise_for_status()

    def delete_document(self, namespace: str, collection: str, name: str) -> None:
        url = f"{self._base_document_index_url}/collections/{namespace}/{collection}/docs/{name}"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()

    def get_document(
        self, namespace: str, collection: str, name: str, get_chunks: bool = False
    ) -> Any:
        if not get_chunks:
            url = f"{self._base_document_index_url}/collections/{namespace}/{collection}/docs/{name}"
        else:
            url = f"{self._base_document_index_url}/collections/{namespace}/{collection}/docs/{name}/chunks"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def list_documents(self, namespace: str, collection: str) -> Any:
        url = (
            f"{self._base_document_index_url}/collections/{namespace}/{collection}/docs"
        )
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def search(
        self,
        namespace: str,
        collection: str,
        query: str,
        max_results: int,
        min_score: float,
    ) -> Any:
        url = f"{self._base_document_index_url}/collections/{namespace}/{collection}/search"
        data = {
            "query": [{"modality": "text", "text": query}],
            "max_results": max_results,
            "min_score": min_score,
            "filter": [{"with": [{"modality": "text"}]}],
        }
        response = requests.post(url, data=json.dumps(data), headers=self.headers)
        response.raise_for_status()
        return response.json()
