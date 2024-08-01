import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum
from http import HTTPStatus
from json import dumps
from typing import Annotated, Any, Literal, Optional, Union
from urllib.parse import quote

import requests
from pydantic import BaseModel, Field
from pydantic.types import StringConstraints
from requests import HTTPError

from intelligence_layer.connectors.base.json_serializable import JsonSerializable


class IndexPath(BaseModel, frozen=True):
    """Path to an index.

    Args:
        namespace: Holds collections.
        index: The name of the index, holds a config.
    """

    namespace: str
    index: str


class IndexConfiguration(BaseModel):
    """Configuration of an index.

    Args:
        embedding_type: "symmetric" or "asymmetric" embedding type.
        chunk_size: The maximum size of the chunks in tokens to be used for the index.
    """

    embedding_type: Literal["symmetric", "asymmetric"]
    chunk_size: int


class DocumentContents(BaseModel):
    """Actual content of a document.

    Note:
        Currently only supports text-only documents.

    Args:
        contents: List of text items.
        metadata: Any metadata that is kept along with the document. This could contain things like
            author, creation-data, references to external systems. The content must be serializable using
            `json.dumps`. The document-index leaves it unchanged.
    """

    contents: Sequence[str]
    metadata: JsonSerializable = None

    @classmethod
    def from_text(cls, text: str) -> "DocumentContents":
        return cls(contents=[text])

    @classmethod
    def _from_modalities_json(
        cls, modalities_json: Mapping[str, Any]
    ) -> "DocumentContents":
        return cls(
            contents=[
                modality["text"]
                for modality in modalities_json.get("contents", [])
                if modality["modality"] == "text"
            ],
            metadata=modalities_json.get("metadata"),
        )

    def _to_modalities_json(self) -> Mapping[str, Any]:
        return {
            "schema_version": "V1",
            "contents": [{"modality": "text", "text": c} for c in self.contents],
            "metadata": self.metadata,
        }


class CollectionPath(BaseModel, frozen=True):
    """Path to a collection.

    Args:
        namespace: Holds collections.
        collection: Holds documents.
            Unique within a namespace.
    """

    namespace: str
    collection: str


class DocumentPath(BaseModel, frozen=True):
    """Path to a document.

    Args:
        collection_path: Path to a collection.
        document_name: Points to a document.
            Unique within a collection.
    """

    collection_path: CollectionPath
    document_name: str

    def encoded_document_name(self) -> str:
        return quote(self.document_name, safe="")

    @classmethod
    def from_json(cls, document_path_json: Mapping[str, str]) -> "DocumentPath":
        return cls(
            collection_path=CollectionPath(
                namespace=document_path_json["namespace"],
                collection=document_path_json["collection"],
            ),
            document_name=document_path_json["name"],
        )

    def to_slash_separated_str(self) -> str:
        return f"{self.collection_path.namespace}/{self.collection_path.collection}/{self.document_name}"

    @classmethod
    def from_slash_separated_str(cls, path: str) -> "DocumentPath":
        split = path.split("/", 2)
        assert len(split) == 3
        return cls(
            collection_path=CollectionPath(
                namespace=split[0],
                collection=split[1],
            ),
            document_name=split[2],
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
    def from_list_documents_response(
        cls, list_documents_response: Mapping[str, Any]
    ) -> "DocumentInfo":
        return cls(
            document_path=DocumentPath.from_json(list_documents_response["path"]),
            created=datetime.strptime(
                list_documents_response["created_timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            version=list_documents_response["version"],
        )


class FilterOps(Enum):
    """Enumeration of possible filter operations."""

    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL_TO = "greater_than_or_equal_to"
    LESS_THAN = "less_than"
    LESS_THAN_OR_EQUAL_TO = "less_than_or_equal_to"
    AFTER = "after"
    AT_OR_AFTER = "at_or_after"
    BEFORE = "before"
    AT_OR_BEFORE = "at_or_before"
    EQUAL_TO = "equal_to"


class FilterField(BaseModel):
    """Represents a field to filter on in the DocumentIndex metadata."""

    field_name: Annotated[
        str, StringConstraints(max_length=1000, pattern=r"^[\w-]+(\.\d{0,5})?[\w-]*$")
    ] = Field(
        ...,
        description="The name of the field present in DocumentIndex collection metadata.",
    )
    field_value: Union[str, int, float, bool, datetime] = Field(
        ..., description="The value to filter on in the DocumentIndex metadata."
    )
    criteria: FilterOps = Field(..., description="The criteria to apply for filtering.")


class Filters(BaseModel):
    """Represents a set of filters to apply to a search query."""

    filter_type: Literal["with", "without", "with_one_of"] = Field(
        ..., description="The type of filter to apply."
    )
    fields: list[FilterField] = Field(
        ..., description="The list of fields to filter on."
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
    filters: Optional[list[Filters]] = None


class DocumentFilterQueryParams(BaseModel):
    """Query to filter documents by.

    Args:
        max_documents: Maximum number of documents to display.
        starts_with: Document title prefix/substring to search by.
    """

    max_documents: Optional[Annotated[int, Field(default=25, ge=0)]]
    starts_with: Optional[str]


class DocumentTextPosition(BaseModel):
    """Position of a document chunk within a document item.

    Args:
        item: Which item in the document the chunk belongs to.
        start_position: Start index of the chunk within the item.
        end_position: End index of the chunk within the item.
    """

    item: int
    start_position: int
    end_position: int


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
    chunk_position: DocumentTextPosition

    @classmethod
    def _from_search_response(
        cls, search_response: Mapping[str, Any]
    ) -> "DocumentSearchResult":
        assert search_response["start"]["item"] == search_response["end"]["item"]
        return cls(
            document_path=DocumentPath.from_json(search_response["document_path"]),
            section=search_response["section"][0]["text"],
            score=search_response["score"],
            chunk_position=DocumentTextPosition(
                item=search_response["start"]["item"],
                start_position=search_response["start"]["position"],
                end_position=search_response["end"]["position"],
            ),
        )


class DocumentIndexError(RuntimeError):
    """Raised in case of any `DocumentIndexClient`-related errors.

    Attributes:
        message: The error message as returned by the Document Index.
        status_code: The http error code.
    """

    def __init__(self, message: str, status_code: HTTPStatus) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ExternalServiceUnavailable(DocumentIndexError):
    """Raised in case external service is unavailable when the request is executed."""

    pass


class ResourceNotFound(DocumentIndexError):
    """Raised when a resource like a namespace or a document cannot be found.

    Note that this can also mean that the user executing the request does not have
    permission to access the resource.
    """

    pass


class InvalidInput(DocumentIndexError):
    """Raised when the user-input could not be processed as it violates pre-conditions."""

    pass


class ConstraintViolation(DocumentIndexError):
    """Raised when the request cannot be processed as it would lead to an inconsistent state."""

    pass


_status_code_to_exception = {
    HTTPStatus.SERVICE_UNAVAILABLE: ExternalServiceUnavailable,
    HTTPStatus.NOT_FOUND: ResourceNotFound,
    HTTPStatus.UNPROCESSABLE_ENTITY: InvalidInput,
    HTTPStatus.CONFLICT: ConstraintViolation,
}


class InternalError(DocumentIndexError):
    """Raised in case of unexpected errors."""

    pass


class DocumentIndexClient:
    """Client for the Document Index allowing handling documents and search.

    Document Index is a tool for managing collections of documents, enabling operations such as creation, deletion, listing, and searching.
    Documents can be stored either in the cloud or in a local deployment.

    Args:
        token: A valid token for the document index API.
        base_document_index_url: The url of the document index' API.

    Example:
        >>> import os

        >>> from intelligence_layer.connectors import (
        ...     CollectionPath,
        ...     DocumentContents,
        ...     DocumentIndexClient,
        ...     DocumentPath,
        ...     SearchQuery,
        ... )

        >>> document_index = DocumentIndexClient(os.getenv("AA_TOKEN"))
        >>> collection_path = CollectionPath(
        ...     namespace="aleph-alpha", collection="wikipedia-de"
        ... )
        >>> document_index.create_collection(collection_path)
        >>> document_index.add_document(
        ...     document_path=DocumentPath(
        ...         collection_path=collection_path, document_name="Fun facts about Germany"
        ...     ),
        ...     contents=DocumentContents.from_text("Germany is a country located in ..."),
        ... )
        >>> search_result = document_index.search(
        ...     collection_path=collection_path,
        ...     index_name="asymmetric",
        ...     search_query=SearchQuery(
        ...         query="What is the capital of Germany", max_results=4, min_score=0.5
        ...     ),
        ... )
    """

    def __init__(
        self,
        token: str | None,
        base_document_index_url: str = "https://document-index.aleph-alpha.com",
    ) -> None:
        self._base_document_index_url = base_document_index_url
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **({"Authorization": f"Bearer {token}"} if token is not None else {}),
        }

    def list_namespaces(self) -> Sequence[str]:
        """Lists all available namespaces.

        Returns:
            List of all available namespaces.
        """
        url = f"{self._base_document_index_url}/namespaces"
        response = requests.get(url, headers=self.headers)
        self._raise_for_status(response)
        return [str(namespace) for namespace in response.json()]

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

    def list_collections(self, namespace: str) -> Sequence[CollectionPath]:
        """Lists all collections within a namespace.

        Args:
            namespace: For a collection of documents.
                Typically corresponds to an organization.

        Returns:
            List of all `CollectionPath` instances in the given namespace.
        """
        url = f"{self._base_document_index_url}/collections/{namespace}"
        response = requests.get(url, headers=self.headers)
        self._raise_for_status(response)
        return [
            CollectionPath(namespace=namespace, collection=collection)
            for collection in response.json()
        ]

    def create_index(
        self, index_path: IndexPath, index_configuration: IndexConfiguration
    ) -> None:
        """Creates an index in a namespace.

        Args:
            index_path: Path to the index.
            index_configuration: Configuration of the index to be created.
        """
        url = f"{self._base_document_index_url}/indexes/{index_path.namespace}/{index_path.index}"

        data = {
            "chunk_size": index_configuration.chunk_size,
            "embedding_type": index_configuration.embedding_type,
        }
        response = requests.put(url, data=dumps(data), headers=self.headers)
        self._raise_for_status(response)

    def create_filter_index_in_namespace(
        self,
        namespace: str,
        filter_index_name: str,
        field_name: str,
        field_type: Literal["string", "integer", "float", "boolean", "datetime"],
    ) -> None:
        """Create a filter index in a specified namespace.

        Args:
            namespace (str): The namespace in which to create the filter index.
            filter_index_name (str): The name of the filter index to create.
            field_name (str): The name of the field to index.
            field_type (Literal["string", "integer", "float", "boolean", "datetime"]): The type of the field to index.

        Returns:
            None
        """
        if not re.match(r"^[a-zA-Z0-9\-.]+$", filter_index_name):
            raise ValueError(
                "Filter index name can only contain alphanumeric characters (a-z, A-Z, -, . and 0-9)."
            )
        if len(filter_index_name) > 50:
            raise ValueError("Filter index name cannot be longer than 50 characters.")

        url = f"{self._base_document_index_url}/filter_indexes/{namespace}/{filter_index_name}"
        data = {"field_name": field_name, "field_type": field_type}
        response = requests.put(url, data=dumps(data), headers=self.headers)
        self._raise_for_status(response)

    def index_configuration(self, index_path: IndexPath) -> IndexConfiguration:
        """Retrieve the configuration of an index in a namespace given its name.

        Args:
            index_path: Path to the index.

        Returns:
            Configuration of the index.
        """
        url = f"{self._base_document_index_url}/indexes/{index_path.namespace}/{index_path.index}"
        response = requests.get(url, headers=self.headers)
        self._raise_for_status(response)
        response_json: Mapping[str, Any] = response.json()
        return IndexConfiguration(
            embedding_type=response_json["embedding_type"],
            chunk_size=response_json["chunk_size"],
        )

    def assign_index_to_collection(
        self, collection_path: CollectionPath, index_name: str
    ) -> None:
        """Assign an index to a collection.

        Args:
            collection_path: Path to the collection of interest.
            index_name: Name of the index.
        """
        url = f"{self._base_document_index_url}/collections/{collection_path.namespace}/{collection_path.collection}/indexes/{index_name}"
        response = requests.put(url, headers=self.headers)
        self._raise_for_status(response)

    def assign_filter_index_to_search_index(
        self, collection_path: CollectionPath, index_name: str, filter_index_name: str
    ) -> None:
        """Assign an existing filter index to an assigned search index.

        Args:
            collection_path: Path to the collection of interest.
            index_name: Name of the index to assign the filter index to.
            filter_index_name: Name of the filter index.
        """
        url = f"{self._base_document_index_url}/collections/{collection_path.namespace}/{collection_path.collection}/indexes/{index_name}/filter_indexes/{filter_index_name}"
        response = requests.put(url, headers=self.headers)
        self._raise_for_status(response)

    def unassign_filter_index_from_search_index(
        self, collection_path: CollectionPath, index_name: str, filter_index_name: str
    ) -> None:
        """Unassign a filter index from an assigned search index.

        Args:
            collection_path: Path to the collection of interest.
            index_name: Name of the index to unassign the filter index from.
            filter_index_name: Name of the filter index.
        """
        url = f"{self._base_document_index_url}/collections/{collection_path.namespace}/{collection_path.collection}/indexes/{index_name}/filter_indexes/{filter_index_name}"
        response = requests.delete(url, headers=self.headers)
        self._raise_for_status(response)

    def delete_index_from_collection(
        self, collection_path: CollectionPath, index_name: str
    ) -> None:
        """Delete an index from a collection.

        Args:
            index_name: Name of the index.
            collection_path: Path to the collection of interest.
        """
        url = f"{self._base_document_index_url}/collections/{collection_path.namespace}/{collection_path.collection}/indexes/{index_name}"
        response = requests.delete(url, headers=self.headers)
        self._raise_for_status(response)

    def list_assigned_index_names(
        self, collection_path: CollectionPath
    ) -> Sequence[str]:
        """List all indexes assigned to a collection.

        Args:
            collection_path: Path to the collection of interest.

        Returns:
            List of all indexes that are assigned to the collection.
        """
        url = f"{self._base_document_index_url}/collections/{collection_path.namespace}/{collection_path.collection}/indexes"
        response = requests.get(url, headers=self.headers)
        self._raise_for_status(response)
        return [str(index_name) for index_name in response.json()]

    def list_assigned_filter_index_names(
        self, collection_path: CollectionPath, index_name: str
    ) -> Sequence[str]:
        """List all filter-indexes assigned to a search index in a collection.

        Args:
            collection_path: Path to the collection of interest.
            index_name: Search index to check.

        Returns:
            List of all filter-indexes that are assigned to the collection.
        """
        url = f"{self._base_document_index_url}/collections/{collection_path.namespace}/{collection_path.collection}/indexes/{index_name}/filter_indexes"
        response = requests.get(url, headers=self.headers)
        self._raise_for_status(response)
        return [str(filter_index_name) for filter_index_name in response.json()]

    def list_filter_indexes_in_namespace(self, namespace: str) -> Sequence[str]:
        """List all filter indexes in a namespace.

        Args:
            namespace: The namespace to list filter indexes in.

        Returns:
            List of all filter indexes in the namespace.
        """
        url = f"{self._base_document_index_url}/filter_indexes/{namespace}"
        response = requests.get(url, headers=self.headers)
        self._raise_for_status(response)
        return [str(filter_index_name) for filter_index_name in response.json()]

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
        url = f"{self._base_document_index_url}/collections/{document_path.collection_path.namespace}/{document_path.collection_path.collection}/docs/{document_path.encoded_document_name()}"
        response = requests.put(
            url, data=dumps(contents._to_modalities_json()), headers=self.headers
        )
        self._raise_for_status(response)

    def delete_document(self, document_path: DocumentPath) -> None:
        """Delete a document from a collection.

        Args:
            document_path: Consists of `collection_path` and name of document to be deleted.
        """
        url = f"{self._base_document_index_url}/collections/{document_path.collection_path.namespace}/{document_path.collection_path.collection}/docs/{document_path.encoded_document_name()}"
        response = requests.delete(url, headers=self.headers)
        self._raise_for_status(response)

    def document(self, document_path: DocumentPath) -> DocumentContents:
        """Retrieve a document from a collection.

        Args:
            document_path: Consists of `collection_path` and name of document to be retrieved.

        Returns:
            Content of the retrieved document.
        """
        url = f"{self._base_document_index_url}/collections/{document_path.collection_path.namespace}/{document_path.collection_path.collection}/docs/{document_path.encoded_document_name()}"
        response = requests.get(url, headers=self.headers)
        self._raise_for_status(response)
        return DocumentContents._from_modalities_json(response.json())

    def documents(
        self,
        collection_path: CollectionPath,
        filter_query_params: Optional[DocumentFilterQueryParams] = None,
    ) -> Sequence[DocumentInfo]:
        """List all documents within a collection.

        Note:
            Does not return each document's content.

        Args:
            collection_path: Path to the collection of interest.
            filter_query_params: Query parameters to filter the results.

        Returns:
            Overview of all documents within the collection.
        """
        if filter_query_params is None:
            filter_query_params = DocumentFilterQueryParams(
                max_documents=None, starts_with=None
            )

        url = f"{self._base_document_index_url}/collections/{collection_path.namespace}/{collection_path.collection}/docs"

        query_params = {}
        if filter_query_params.max_documents:
            query_params["max_documents"] = str(filter_query_params.max_documents)
        if filter_query_params.starts_with:
            query_params["starts_with"] = filter_query_params.starts_with

        response = requests.get(url=url, params=query_params, headers=self.headers)
        self._raise_for_status(response)
        return [DocumentInfo.from_list_documents_response(r) for r in response.json()]

    def search(
        self,
        collection_path: CollectionPath,
        index_name: str,
        search_query: SearchQuery,
    ) -> Sequence[DocumentSearchResult]:
        """Search through a collection with a `search_query`.

        Args:
            collection_path: Path to the collection of interest.
            index_name: Name of the index to search with.
            search_query: The query to search with.

        Returns:
            Result of the search operation. Will be empty if nothing was retrieved.
        """
        url = f"{self._base_document_index_url}/collections/{collection_path.namespace}/{collection_path.collection}/indexes/{index_name}/search"

        filters: list[dict[str, Any]] = [{"with": [{"modality": "text"}]}]
        if search_query.filters:
            for metadata_filter in search_query.filters:
                filters.append(
                    {
                        f"{metadata_filter.filter_type}": [
                            {
                                "metadata": {
                                    "field": filter_field.field_name,
                                    f"{filter_field.criteria.value}": filter_field.field_value,
                                }
                            }
                            for filter_field in metadata_filter.fields
                        ]
                    }
                )

        data = {
            "query": [{"modality": "text", "text": search_query.query}],
            "max_results": search_query.max_results,
            "min_score": search_query.min_score,
            "filters": filters,
        }

        response = requests.post(url, data=dumps(data), headers=self.headers)
        self._raise_for_status(response)
        return [DocumentSearchResult._from_search_response(r) for r in response.json()]

    def _raise_for_status(self, response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except HTTPError as e:
            exception_factory = _status_code_to_exception.get(
                HTTPStatus(response.status_code), InternalError
            )
            raise exception_factory(
                response.text, HTTPStatus(response.status_code)
            ) from e
