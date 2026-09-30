"""Exception types used across the pipeline."""


class PipelineError(RuntimeError):
    """A fatal error during pipeline execution."""


class ConfigError(PipelineError):
    """Invalid or missing configuration."""


class RlistExceededError(PipelineError):
    """mdrun failed because an intra-ligand pair crossed the pair-list cutoff.

    Raised by the mdrun retry loop when the log carries the GROMACS
    'perturbed non-bonded pair interactions beyond the pair-list cutoff'
    error.  The pipeline can catch this, enlarge rlist in the offending
    mdp(s) and retry once instead of failing the whole run (large-ligand
    support).
    """

    def __init__(self, message: str, log_path=None):
        super().__init__(message)
        self.log_path = log_path
