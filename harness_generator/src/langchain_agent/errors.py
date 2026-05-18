"""TianHeng domain-specific exceptions."""


class TianHengError(RuntimeError):
    """Base exception for all TianHeng-specific errors."""


class BuildError(TianHengError):
    """Harness compilation or build script failure."""


class RunError(TianHengError):
    """Fuzzer execution failure."""


class TriageError(TianHengError):
    """Crash triage or analysis failure."""


class ConfigError(TianHengError):
    """Configuration validation or loading failure."""


class K8sJobError(TianHengError):
    """Kubernetes job submission, execution, or timeout failure."""
