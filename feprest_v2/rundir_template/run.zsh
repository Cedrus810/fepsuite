#!/usr/bin/zsh
# FEP/REST2 v2 run script, copied into the calculation directory.
# See feprest_v2/README.md for the full usage.

# Root directory of FEP-suite
FEPSUITE_ROOT=/path/to/fepsuite

# Root GROMACS directory. Note FEP/REST needs the HREX-patched
# GROMACS 2020 (see feprest/README.md "Preparation 4") and Python 3
# with numpy, mdtraj and pymbar 3.0.3.
GROMACS_DIR=$HOME/opt/gromacs-2020-hrexpatch
# uncomment these lines to manually specify the GROMACS binary
#GMX=$GROMACS_DIR/bin/gmx_mpi
#GMX_MPI=$GROMACS_DIR/bin/gmx_mpi

# Specify jobtype: feprest_v2 (the refactored FEP/REST2 pipeline)
JOBTYPE=feprest_v2

# Specify jobsystem (see submit_scripts/)
JOBSYSTEM=none
# If jobsystem need additional information
#SUBSYSTEM=

# Run actual controller
source $FEPSUITE_ROOT/controller.zsh $0 $@
