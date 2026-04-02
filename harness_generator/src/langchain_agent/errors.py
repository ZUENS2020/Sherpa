"""Sherpa domain-specific exceptions."""


class SherpaError(RuntimeError):
    """Base exception for all Sherpa-specific errors."""


class BuildError(SherpaError):
    """Harness compilation or build script failure."""


class RunError(SherpaError):
    """Fuzzer execution failure."""


class TriageError(SherpaError):
    """Crash triage or analysis failure."""


class ConfigError(SherpaError):
    """Configuration validation or loading failure."""


class K8sJobError(SherpaError):
    """Kubernetes job submission, execution, or timeout failure."""
