#!/bin/zsh
# Bridge between the restlab Python pipeline and the FEP-suite job-system
# wrappers.  The Python side launches GROMACS through this script so that
# the system-specific job_mpirun / job_singlerun implementations in
# $FEPSUITE_ROOT/submit_scripts/$JOBSYSTEM.zsh keep working unchanged.
#
# usage: jobbridge.zsh singlerun -- CMD [ARGS...]
#        jobbridge.zsh mpirun N -- CMD [ARGS...]
#
# FEPSUITE_ROOT and JOBSYSTEM must be set in the environment.

if [[ -z $FEPSUITE_ROOT || -z $JOBSYSTEM ]]; then
    print -u2 "jobbridge.zsh: FEPSUITE_ROOT and JOBSYSTEM must be set"
    exit 1
fi

JOB_SCRIPT=$FEPSUITE_ROOT/submit_scripts/$JOBSYSTEM.zsh
if [[ ! -e $JOB_SCRIPT ]]; then
    print -u2 "jobbridge.zsh: no such job system: $JOB_SCRIPT"
    exit 1
fi
source $JOB_SCRIPT

mode=$1
shift

np=""
if [[ $mode == mpirun ]]; then
    np=$1
    shift
fi

if [[ $1 != "--" ]]; then
    print -u2 "jobbridge.zsh: expected -- separator after mode"
    exit 1
fi
shift

if [[ $mode == mpirun ]]; then
    job_mpirun $np "$@"
elif [[ $mode == singlerun ]]; then
    job_singlerun "$@"
else
    print -u2 "jobbridge.zsh: unknown mode: $mode"
    exit 1
fi

exit $?
