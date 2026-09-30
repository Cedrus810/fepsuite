# Put this file at either the parent directory or its child.
# The file on the parent directory is sourced by the pipeline, then the file on the child directory is sourced, so the child one will overwrite the variables in the parent.
#
# Values must be literal numbers/strings: this file is parsed by both zsh
# (controller.zsh) and Python (restlab), so shell expansions like the v1
# "REFINIT=$BASECONF" are not allowed — write the filename literally.

# initial structure and topology
BASECONF=conf_ionized.pdb
BASETOP=topol_ionized.top

# Allowed values are "auto", "no", "posonly"
# "auto" randomly pick water (which is not close to non-solvent molecules) and convert to ions
# "no" uses unneutralized perturbation which typically causes artifacts (currently research purpose only)
# "posonly" randomly pick either water or positive ion, then convert it to vice versa. This is useful if you want to use more than one reference state.
CHARGE=auto

# Force field types to use. Supported force field types are listed under water_ion_models.
# to support new ff types see water_ion_models/readme.txt
FF=amber

# Number of replicas to use. Recommended: 32-36 except the cases below
# 48-64 for mutations involving charge change and/or large residue such as F/Y/W
NREP=32

# Number of MPI processes per replica. For GPU, recommended = 1.
PARA=8

# Number of threads per process. Recommended: for CPU: 1, for GPU: 2-6.
TPP=1

# Number of tuning cycles for fep parameters
NTUNE=5

# Simulation time of one chunk in ps, recommended total time is 4 ns for a good trade-off.
# You can extend the sampling by re-running the "prodrun" stage.
SIMLENGTH=4000

# replica exchange interval, recommended: 1000
REPLICA_INTERVAL=1000

# Energy sampling interval, recommended: 100
SAMPLING_INTERVAL=100

# Increase if you are using strange topology file. If you made the reference state by nucfepgen you will need at least 1 for the lack of reference state position restraint.
BASEWARN=1

# reference coordinate upon initialization (must be a literal filename now;
# unset it by making the value empty if you do not want the restraints)
REFINIT=conf_ionized.pdb

# Set if you want extra restraints during FEP (useful for protein complex)
REFCRD=

# Because dummy atoms in state A does not interact with those in state B, there are exclusions between these atoms,
# which results in relatively long interaction between the two. Option -rdd on mdrun is thus required with this number
DOMAIN_SHRINK=0.6

# REST2 configurations
# Atoms within ($REST2_REGION_DISTANCE) nm, and same residues as these atoms, are considered "hot" region
REST2_REGION_DISTANCE=0.4

# REST2 hot-region temperature.
REST2_TEMP=1200

#*************************
# v2 additions (defaults shown; all optional)
#*************************

# Run the per-replica grompp/convert-tpr of a stage concurrently (grompp of
# large systems is slow and can dominate the stage wall time).
RUN_TPR_PARALLEL=yes

# Moleculetypes ignored when searching perturbed atoms / the hot region
#NON_PERTURBED_MOLECULETYPE=SOL SOL2pos SOL2neg

# mdtraj selection of molecules considered when growing the hot region
#TARGET_MOLECULE=not (resname HOH SOL NA CL Na Cl K SOD CLA)

# Moleculetype restored to the flexible water of the force field
#WATER_MOLTYPE=SOL

# Base temperature of the REST2 ladder (K)
#REST2_TEMP0=300

# grompp/convert-tpr appendixes of the tuning stage (v1 environment
# overrides, now ordinary configuration keys)
#EXTEND_RUN_LENGTH=50
#REPOPT_REPLEX_INTERVAL=100

# Optional appendix appended to every mdrun command line (as in v1)
#NSTLIST_CMD=
