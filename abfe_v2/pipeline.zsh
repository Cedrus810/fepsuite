#!/bin/zsh
# FEP-suite ABFE pipeline v2 — thin adapter for controller.zsh.
#
# All the real logic lives in the feplab Python package (feplab/stages.py);
# this file only forwards the controller protocol and exports the context
# variables the Python side and jobbridge.zsh need.  controller.zsh either
# sources this script ("run" mode, with the job context as shell variables)
# or executes it as a command ("query" mode).

reqstate=$1
stateno=$2
if [[ -z $stateno ]]; then
    echo "This file should be called from controller.zsh" 2>&1
    exit 1
fi

# controller.zsh sets <JOBTYPE:u>_ROOT (= ABFE_V2_ROOT for JOBTYPE=abfe_v2)
# before using us; fall back to this file's location (works both when the
# script is sourced and when it is executed).
: ${ABFE_V2_ROOT:=${0:A:h}}

export PYTHONPATH=$ABFE_V2_ROOT${PYTHONPATH:+:$PYTHONPATH}

if [[ $reqstate == query ]]; then
    ${PYTHON3:-python3} -m feplab.cli query $stateno
    exit $?
fi

# run mode: sourced inside controller.zsh, so all the job context (ID,
# STEPNO, PROCS, PPN, TPP, GPP, GMX, ...) is visible as shell variables.
# The submit scripts may set some of them as plain shell variables, so
# re-export the ones the Python pipeline and jobbridge.zsh rely on.
typeset -a _FEPLAB_EXPORTS
_FEPLAB_EXPORTS=(FEPSUITE_ROOT JOBSYSTEM JOBTYPE ID STEPNO PROCS PPN GPP TPP OMP_NUM_THREADS GMX GMX_MPI PYTHON3 FEPLAB_PARA_CONF APBS)
for _v in $_FEPLAB_EXPORTS; do
    if [[ -n ${(P)_v} ]]; then
        export $_v
    fi
done

# NOTE: this file is SOURCED in run mode, so we must `return`, not `exit`,
# or controller.zsh would terminate before recording the job bookkeeping.
${PYTHON3:-python3} -m feplab.cli run $stateno
unset _FEPLAB_EXPORTS _v
return $?
