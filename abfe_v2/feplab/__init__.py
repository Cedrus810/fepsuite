"""feplab: refactored FEP-ABFE pipeline (abfe_v2).

This package is a ground-up reorganization of the `abfe` pipeline of
FEP-suite.  The scientific algorithm (Aldeghi et al. 2016 style thermodynamic
cycle with Boresch restraints, computed with GROMACS) is kept identical to
the original pipeline, while the implementation is restructured as:

* named pipeline stages (``setup`` ... ``analysis``) instead of numbered ones,
* a Python package for all pipeline logic, with a thin zsh adapter
  (``pipeline.zsh``) that keeps the ``controller.zsh`` / ``submit_scripts``
  job-submission ecosystem working unchanged,
* one shared place for MDP handling, topology parsing, lambda schedules and
  the thermodynamic-cycle bookkeeping.
"""

__version__ = "2.0.0a1"
