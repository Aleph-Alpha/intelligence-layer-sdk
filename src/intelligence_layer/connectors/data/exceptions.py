class DataError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, *args: object) -> None:
        default_message = getattr(self, "DEFAULT_MESSAGE", "")
        super().__init__(default_message, *args)


class DataInternalError(DataError):
    """Exception raised when an internal error occurs."""

    DEFAULT_MESSAGE = "Internal error: An unexpected error occurred. "


class DataResourceNotFound(DataError):
    """Exception raised when a resource is not found."""

    DEFAULT_MESSAGE = "Resource not found: The requested resource was not found. "


class DataInvalidInput(DataError):
    """Exception raised when the input is invalid."""

    DEFAULT_MESSAGE = "Invalid input: The input provided is invalid. "


class DataExternalServiceUnavailable(DataError):
    """Exception raised when an external service is unavailable."""

    DEFAULT_MESSAGE = (
        "External service unavailable: The external service is unavailable. "
    )


class DataForbiddenError(DataError):
    """Exception raised when a forbidden error occurs."""

    DEFAULT_MESSAGE = (
        "Forbidden error: Client does not have permission to access the resource. "
    )
