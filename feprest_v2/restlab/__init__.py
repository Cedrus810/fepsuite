"""restlab: refactored FEP/REST2 relative-FEP pipeline (feprest_v2).

This package is a ground-up reorganization of the `feprest` pipeline of
FEP-suite (REST2-enhanced Hamiltonian replica exchange with a
single/dual topology FEP, evaluated with a patched GROMACS 2020).  The
scientific algorithm is kept identical while the implementation is
restructured as:

* named pipeline stages (``minimize`` ... ``prodrun``) instead of
  numbered ones,
* a Python package for all pipeline logic, with a thin zsh adapter
  (``pipeline.zsh``) that keeps the ``controller.zsh`` /
  ``submit_scripts`` job-submission ecosystem working unchanged,
* the previously scattered helper scripts (add_underline, turn-heavy,
  neutralize, recover-water, bar_deltae, rest2py, replica_optimizer)
  become importable, unit-testable modules.

The same engine serves both relative-FEP applications of FEP-suite:
protein-mutation stability changes (FEP) and ligand transformations
(RBFE); they differ only in system preparation (FASPR/fepgen on the
input side), not in the run pipeline.
"""

__version__ = "2.0.0a1"
