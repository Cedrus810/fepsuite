#!/usr/bin/zsh
# FEP-ABFE v2 run script, copied into the calculation directory.
# See abfe_v2/README.md for the full usage.

# Root directory of FEP-suite
FEPSUITE_ROOT=/path/to/fepsuite

# Root GROMACS directory. Note ABFE v2 requires GROMACS >= 2022.5
# and Python 3.11+ (with numpy, mdtraj, pyedr).
GROMACS_DIR=/path/to/gromacs-2022.5
# uncomment these lines to manually specify the GROMACS binary
#GMX=$GROMACS_DIR/bin/gmx_mpi
#GMX_MPI=$GROMACS_DIR/bin/gmx_mpi

# Specify jobtype: abfe_v2 (the refactored ABFE pipeline)
JOBTYPE=abfe_v2

# Specify jobsystem (see submit_scripts/)
JOBSYSTEM=none
# If jobsystem need additional information
#SUBSYSTEM=

# Run actual controller
source $FEPSUITE_ROOT/controller.zsh $0 $@
