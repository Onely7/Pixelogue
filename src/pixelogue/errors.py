"""Application errors and stable process exit codes."""

from enum import IntEnum


class ExitCode(IntEnum):
    """Stable CLI exit codes."""

    SUCCESS = 0
    CONFIGURATION = 2
    EXTERNAL_INPUT = 3
    CAPABILITY = 4
    SHORTFALL = 5
    SOLVER_UNKNOWN = 6
    EXECUTION = 7
    AUDIT = 8


class PixelogueError(Exception):
    """Base error carrying a stable reason and process exit code."""

    def __init__(self, reason: str, message: str, exit_code: ExitCode) -> None:
        """Initialize an application error.

        Args:
            reason: Stable machine-readable reason code.
            message: Concise diagnostic for an operator.
            exit_code: Process status associated with the failure category.
        """
        super().__init__(message)
        self.reason = reason
        self.exit_code = exit_code


class ConfigurationError(PixelogueError):
    """Report invalid or contradictory configuration."""

    def __init__(self, reason: str, message: str) -> None:
        """Initialize a configuration failure."""
        super().__init__(reason, message, ExitCode.CONFIGURATION)


class ExternalInputError(PixelogueError):
    """Report a missing or invalid external input."""

    def __init__(self, reason: str, message: str) -> None:
        """Initialize an external-input failure."""
        super().__init__(reason, message, ExitCode.EXTERNAL_INPUT)


class ExecutionError(PixelogueError):
    """Report a runtime or inference failure."""

    def __init__(self, reason: str, message: str) -> None:
        """Initialize an execution failure."""
        super().__init__(reason, message, ExitCode.EXECUTION)


class AuditError(PixelogueError):
    """Report a final artifact or selection mismatch."""

    def __init__(self, reason: str, message: str) -> None:
        """Initialize an audit failure."""
        super().__init__(reason, message, ExitCode.AUDIT)


class CapabilityError(PixelogueError):
    """Report unavailable hardware, model locks, or serving features."""

    def __init__(self, reason: str, message: str) -> None:
        """Initialize a capability failure."""
        super().__init__(reason, message, ExitCode.CAPABILITY)


class ShortfallError(PixelogueError):
    """Report that valid inputs cannot meet a requested count or quota."""

    def __init__(self, reason: str, message: str) -> None:
        """Initialize a data or selection shortfall."""
        super().__init__(reason, message, ExitCode.SHORTFALL)


class SolverUnknownError(PixelogueError):
    """Report that constrained selection ended without a conclusion."""

    def __init__(self, reason: str, message: str) -> None:
        """Initialize an inconclusive solver result."""
        super().__init__(reason, message, ExitCode.SOLVER_UNKNOWN)
