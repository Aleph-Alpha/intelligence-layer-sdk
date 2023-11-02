from datetime import datetime
import json
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, Field
import requests


class DocumentContents(BaseModel):
    """Actual content of a document.

    Note:
        Currently only supports text-only documents.

    Args:
        contents: List of text items.
    """

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
    """Path to a collection.

    Args:
        namespace: Holds collections.
        collection: Holds documents.
            Unique within a namespace.
    """

    namespace: str
    collection: str


class DocumentPath(BaseModel):
    """Path to a document.

    Args:
        collection_path: Path to a collection.
        document_name: Points to a document.
            Unique within a collection.
    """

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
    """Presents an overview of a document.

    Args:
        document_path: Path to a document.
        created: When this version of the document was created.
            Equivalent to when it was last updated.
        version: How many times the document was updated.
    """

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
    """Query to search through a collection with.

    Args:
        query: Actual text to be searched with.
        max_results: Max number of search results to be retrieved by the query.
            Must be larger than 0.
        min_score: Min score needed for a search result to be returned.
            Must be between 0 and 1.
    """

    query: str
    max_results: int = Field(..., ge=0)
    min_score: float = Field(..., ge=0.0, le=1.0)


class DocumentSearchResult(BaseModel):
    """Result of a search query for one individual section.

    Args:
        document_path: Path to the document that the section originates from.
        section: Actual section of the document that was found as a match to the query.
        score: Actual search score of the section found.
            Generally, higher scores correspond to better matches.
            Will be between 0 and 1.
    """

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


class DocumentIndexError(Exception):
    """Raised in case of any `DocumentIndexClient`-related errors.

    Attributes:
        message: The error message as returned by the Document Index.
        status_code: The http error code.
    """

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class DocumentIndexClient:
    """Client for the Document Index allowing handling documents and search.

    Document Index is a tool for managing collections of documents, enabling operations such as creation, deletion, listing, and searching.
    Documents can be stored either in the cloud or in a local deployment.

    Args:
        token: A valid token for the document index API.
        base_document_index_url: The url of the document index' API.

    Example:
        >>> document_index = DocumentIndex(os.getenv("AA_TOKEN"))
        >>> collection_path = CollectionPath(
        >>>     namespace="my_namespace",
        >>>     collection="germany_facts_collection"
        >>> )
        >>> document_index.create_collection(collection_path)
        >>> document_index.add_document(
        >>>     document_path=DocumentPath(
        >>>         collection_path=collection_path,
        >>>         document_name="Fun facts about Germany"
        >>>     ),
        >>>     contents=DocumentContents.from_text("Germany is a country located in ...")
        >>> )
        >>> search_result = document_index.asymmetric_search(
        >>>     collection_path=collection_path,
        >>>     search_query=SearchQuery(
        >>>         query="What is the capital of Germany",
        >>>         max_results=4,
        >>>         min_score=0.5
        >>>     )
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

    def _raise_for_status(self, response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except:
            raise DocumentIndexError(response.text, response.status_code)

    def create_collection(self, collection_path: CollectionPath) -> None:
        """Creates a collection at the path.

        Note:
            Collection's name must be unique within a namespace.

        Args:
            collection_path: Path to the collection of interest.
        """

        url = f"{self._base_document_index_url}/collections/{collection_path.namespace}/{collection_path.collection}"
        response = requests.put(url, headers=self.headers)
        self._raise_for_status(response)

    def delete_collection(self, collection_path: CollectionPath) -> None:
        """Deletes the collection at the path.

        Args:
            collection_path: Path to the collection of interest.
        """

        url = f"{self._base_document_index_url}/collections/{collection_path.namespace}/{collection_path.collection}"
        response = requests.delete(url, headers=self.headers)
        self._raise_for_status(response)

    def list_collections(self, namespace: str) -> Sequence[str]:
        """Lists all collections within a namespace.

        Args:
            namespace: For a collection of documents.
                Typically corresponds to an organization.

        Returns:
            List of all collections' names.
        """

        url = f"{self._base_document_index_url}/collections/{namespace}"
        response = requests.get(url, headers=self.headers)
        self._raise_for_status(response)
        collections: Sequence[str] = response.json()
        return collections

    def add_document(
        self,
        document_path: DocumentPath,
        contents: DocumentContents,
    ) -> None:
        """Add a document to a collection.

        Note:
            If a document with the same `document_path` exists, it will be updated with the new `contents`.

        Args:
            document_path: Consists of `collection_path` and name of document to be created.
            contents: Actual content of the document.
                Currently only supports text.
        """

        url = f"{self._base_document_index_url}/collections/{document_path.collection_path.namespace}/{document_path.collection_path.collection}/docs/{document_path.document_name}"
        data = {
            "schema_version": "V1",
            "contents": contents._to_modalities_json(),
        }
        response = requests.put(url, data=json.dumps(data), headers=self.headers)
        self._raise_for_status(response)

    def delete_document(self, document_path: DocumentPath) -> None:
        """Delete a document from a collection.

        Args:
            document_path: Consists of `collection_path` and name of document to be deleted.
        """

        url = f"{self._base_document_index_url}/collections/{document_path.collection_path.namespace}/{document_path.collection_path.collection}/docs/{document_path.document_name}"
        response = requests.delete(url, headers=self.headers)
        self._raise_for_status(response)

    def document(self, document_path: DocumentPath) -> DocumentContents:
        """Retrieve a document from a collection.

        Args:
            document_path: Consists of `collection_path` and name of document to be retrieved.

        Returns:
            Content of the retrieved document.
        """

        url = f"{self._base_document_index_url}/collections/{document_path.collection_path.namespace}/{document_path.collection_path.collection}/docs/{document_path.document_name}"
        response = requests.get(url, headers=self.headers)
        self._raise_for_status(response)
        return DocumentContents._from_modalities_json(response.json())

    def list_documents(self, collection_path: CollectionPath) -> Sequence[DocumentInfo]:
        """List all documents within a collection.

        Note:
            Does not return each document's content.

        Args:
            collection_path: Path to the collection of interest.

        Returns:
            Overview of all documents within the collection.
        """

        url = f"{self._base_document_index_url}/collections/{collection_path.namespace}/{collection_path.collection}/docs"
        response = requests.get(url, headers=self.headers)
        self._raise_for_status(response)
        return [DocumentInfo._from_list_documents_response(r) for r in response.json()]

    def search(
        self,
        collection_path: CollectionPath,
        index: str,
        search_query: SearchQuery,
    ) -> Sequence[DocumentSearchResult]:
        """Search through a collection with a `search_query`.

        Args:
            collection_path: Path to the collection of interest.
            index: Name of the search configuration.
                Currently only supports "asymmetric".
            search_query: The query to search with.

        Returns:
            Result of the search operation. Will be empty if nothing was retrieved.
        """

        url = f"{self._base_document_index_url}/collections/{collection_path.namespace}/{collection_path.collection}/indexes/{index}/search"
        data = {
            "query": [{"modality": "text", "text": search_query.query}],
            "max_results": search_query.max_results,
            "min_score": search_query.min_score,
            "filter": [{"with": [{"modality": "text"}]}],
        }
        response = requests.post(url, data=json.dumps(data), headers=self.headers)
        self._raise_for_status(response)
        return [DocumentSearchResult._from_search_response(r) for r in response.json()]

    def asymmetric_search(
        self,
        collection_path: CollectionPath,
        search_query: SearchQuery,
    ) -> Sequence[DocumentSearchResult]:
        """Search through a collection with a `search_query` using the asymmetric search configuration.

        Args:
            collection_path: Path to the collection of interest.
            search_query: The query to search with.

        Returns:
            Result of the search operation. Will be empty if nothing was retrieved.
        """

        return self.search(collection_path, "asymmetric", search_query)
