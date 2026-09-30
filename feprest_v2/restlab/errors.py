"""Exception types used across the pipeline."""


class PipelineError(RuntimeError):
    """A fatal error during pipeline execution."""


class ConfigError(PipelineError):
    """Invalid or missing configuration."""
