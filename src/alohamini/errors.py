"""Errors reported by AlohaMini clients."""


class AlohaMiniError(Exception):
    """Base class for recoverable interface errors."""


class ProtocolError(AlohaMiniError):
    """A response does not satisfy the supported protocol."""


class ModelMismatchError(ProtocolError):
    """The connected robot does not match the requested model."""


class ResponseTimeoutError(AlohaMiniError):
    """No matching response arrived within the request deadline."""


class ConnectionError(AlohaMiniError):
    """The transport could not complete the request."""


class CommandRejectedError(AlohaMiniError):
    """Client preconditions do not permit queueing a command."""
