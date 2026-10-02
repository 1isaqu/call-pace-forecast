"""Exception hierarchy.

The split matters for the serving layer: `InvalidRequestError` is the caller's fault and is
always re-raised, while every other failure while reading today's data degrades the response
instead of breaking a wall display.
"""

from __future__ import annotations


class CallPaceError(Exception):
    """Base class for every error raised on purpose by this library."""


class ConfigError(CallPaceError, ValueError):
    """A configuration value is out of range or inconsistent."""


class ContractError(CallPaceError, ValueError):
    """Input data does not satisfy the documented data contract."""


class InsufficientDataError(CallPaceError):
    """There is not enough history to fit or evaluate what was requested."""


class DataSourceError(CallPaceError):
    """A loader could not read from its source (connection, timeout, query failure)."""


class ArtifactError(CallPaceError):
    """A fitted artifact is missing, unreadable or inconsistent with the configuration."""


class InvalidRequestError(CallPaceError, ValueError):
    """The caller asked for something that does not exist, such as an unknown bucket."""
