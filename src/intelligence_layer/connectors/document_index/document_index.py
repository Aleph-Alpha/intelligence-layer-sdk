from datetime import datetime
import json
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, Field
import requests


class DocumentContents(BaseModel):
    contents: Sequence[str]

    @classmethod
    def from_text(cls, text: str) -> "DocumentContents":
        return cls(contents=[text])

    @classmethod
    def _from_modalities_json(
        cls, modalities_json: Mapping[str, Any]
    ) -> "DocumentContents":
        contents = []
        for m in modalities_json.get("contents", []):
            if m["modality"] == "text":
                contents.append(m["text"])
        return cls(contents=contents)

    def _to_modalities_json(self) -> Sequence[Mapping[str, str]]:
        text_contents = []
        for c in self.contents:
            if not isinstance(c, str):
                raise TypeError("Currently, only str modality is supported.")
            text_contents.append({"modality": "text", "text": c})
        return text_contents


class CollectionPath(BaseModel):
    namespace: str
    collection: str


class DocumentPath(BaseModel):
    collection_path: CollectionPath
    document_name: str

    @classmethod
    def _from_json(cls, document_path_json: Mapping[str, str]) -> "DocumentPath":
        return cls(
            collection_path=CollectionPath(
                namespace=document_path_json["namespace"],
                collection=document_path_json["collection"],
            ),
            document_name=document_path_json["name"],
        )


class DocumentInfo(BaseModel):
    document_path: DocumentPath
    created: datetime
    version: int

    @classmethod
    def _from_list_documents_response(
        cls, list_documents_response: Mapping[str, Any]
    ) -> "DocumentInfo":
        return cls(
            document_path=DocumentPath._from_json(list_documents_response["path"]),
            created=datetime.strptime(
                list_documents_response["created_timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            version=list_documents_response["version"],
        )


class SearchQuery(BaseModel):
    query: str
    max_results: int = Field(..., ge=0)
    min_score: float = Field(..., ge=0.0, le=1.0)


class DocumentSearchResult(BaseModel):
    document_path: DocumentPath
    section: str
    score: float

    @classmethod
    def _from_search_response(
        cls, search_response: Mapping[str, Any]
    ) -> "DocumentSearchResult":
        return cls(
            document_path=DocumentPath._from_json(search_response["document_path"]),
            section=search_response["section"][0]["text"],
            score=search_response["score"],
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
        >>>     document_path=CollectionPath(
        >>>         namespace="my_namespace",
        >>>         collection="germany_facts_collection",
        >>>         document_name="Fun facts about Germany",
        >>>     )
        >>>     content=DocumentContents.from_text("Germany is a country located in ...")
        >>> )
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

    def create_collection(self, collection_path: CollectionPath) -> None:
        url = f"{self._base_document_index_url}/collections/{collection_path.namespace}/{collection_path.collection}"
        response = requests.put(url, headers=self.headers)
        response.raise_for_status()

    def delete_collection(self, collection_path: CollectionPath) -> None:
        url = f"{self._base_document_index_url}/collections/{collection_path.namespace}/{collection_path.collection}"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()

    def list_collections(self, namespace: str) -> Sequence[str]:
        url = f"{self._base_document_index_url}/collections/{namespace}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        collections: Sequence[str] = response.json()
        return collections

    def add_document(
        self,
        document_path: DocumentPath,
        contents: DocumentContents,
    ) -> None:
        url = f"{self._base_document_index_url}/collections/{document_path.collection_path.namespace}/{document_path.collection_path.collection}/docs/{document_path.document_name}"
        data = {
            "schema_version": "V1",
            "contents": contents._to_modalities_json(),
        }
        response = requests.put(url, data=json.dumps(data), headers=self.headers)
        response.raise_for_status()

    def delete_document(self, document_path: DocumentPath) -> None:
        url = f"{self._base_document_index_url}/collections/{document_path.collection_path.namespace}/{document_path.collection_path.collection}/docs/{document_path.document_name}"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()

    def document(self, document_path: DocumentPath) -> DocumentContents:
        url = f"{self._base_document_index_url}/collections/{document_path.collection_path.namespace}/{document_path.collection_path.collection}/docs/{document_path.document_name}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return DocumentContents._from_modalities_json(response.json())

    def list_documents(self, collection_path: CollectionPath) -> Sequence[DocumentInfo]:
        url = f"{self._base_document_index_url}/collections/{collection_path.namespace}/{collection_path.collection}/docs"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return [DocumentInfo._from_list_documents_response(r) for r in response.json()]

    def search(
        self,
        collection_path: CollectionPath,
        index: str,
        search_query: SearchQuery,
    ) -> Sequence[DocumentSearchResult]:
        url = f"{self._base_document_index_url}/collections/{collection_path.namespace}/{collection_path.collection}/indexes/{index}/search"
        data = {
            "query": [{"modality": "text", "text": search_query.query}],
            "max_results": search_query.max_results,
            "min_score": search_query.min_score,
            "filter": [{"with": [{"modality": "text"}]}],
        }
        response = requests.post(url, data=json.dumps(data), headers=self.headers)
        response.raise_for_status()
        return [DocumentSearchResult._from_search_response(r) for r in response.json()]

    def asymmetric_search(
        self,
        collection_path: CollectionPath,
        search_query: SearchQuery,
    ) -> Sequence[DocumentSearchResult]:
        return self.search(collection_path, "asymmetric", search_query)
