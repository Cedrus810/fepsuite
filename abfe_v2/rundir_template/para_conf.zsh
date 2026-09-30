#*************************
# Long-range correction
#*************************
# Hard limit on |LRC term| in kcal/mol; the analysis aborts with a
# diagnostic instead of writing a poisoned result.txt.
LRC_SANITY_LIMIT=50

# Skip the LRC re-evaluation of the decoupled annihilation endpoints
# (lr-annihilation-lig/complex).  The lambda=1 state has no
# ligand-environment dispersion to correct (~0.1 kcal/mol, nearly
# cancelling in the cycle).  Workaround for the GROMACS 2026 soft-core +
# PME-LJ breakage; see docs/issue-lrc-gmx2026.md.
SKIP_ANNIHILATION_LRC=no

# Override the EDR energy-term names summed for the LRC (comma
# separated).  Matching normalizes spaces/hyphens/parentheses, so
# "LJ (SR)" and "LJ-(SR)" both match; only set this if your GROMACS
# renames terms beyond that.
#LRC_ENERGY_TERMS=LJ (SR),LJ-14,LJ recip.,LJC-14 q,LJC Pairs NB,Disper. corr.

#*************************
# Resource auto-tuning
#*************************
# Detect the GPUs actually allocated by the job system (OpenPBS
# qstat/PBS_GPUFILE, CUDA_VISIBLE_DEVICES, nvidia-smi, in this order) at
# mdrun launch time, and derive the MPI layout: every replica gets at
# least one rank, GPUs are shared when scarce and added as extra ranks
# per replica when abundant.  Falls back to the static *_PARA values
# below when nothing can be detected.
AUTO_RESOURCE=yes

# Run the pre-equilibration chain (steep/cg/nvt/npt and the ligand
# diameter sampling) without MPI: a single non-MPI rank with OMP
# threads.  The corresponding stages then request only one rank.
# NOTE: set this key in the base para_conf.zsh (not per-ID), so the
# submit-time resource request and the runtime agree.
PREP_NONMPI=yes

# GROMACS binary for the non-MPI pre-equilibration runs (MPI-only
# installs can point this at a thread build).
#GMX_NOMPI=gmx

#*************************
# GROMACS parallelization settings
#*************************
# N* parameters are the number of replicas in Hamiltonian replica exchange.
# *_PARA parameters are the number of MPI processes for each replica.
# Thus, number of total MPI processes is e.g. NRESTR * COMPLEX_PARA for the restraint replica run.


# Prot-lig restraint relaxation. Recommended: 4
NRESTR=4

# Charge-discharge. Recommended: 8-12. For large molecules & strongly charged molecules we recommend greater values.
NCHARGE=12

# Annihilation. Recommended:12. For large molecules we recommend greater values, e.g. for ligand with 60 heavy atoms NANNIH=20 will be a good starting point.
NANNIH=12

# Number of MPI processes used when calculating ligands (LIG_PARA) and complexes (COMPLEX_PARA).
# Recommended number of _PARA is a divisor of the number of cores / CPU.
# LIG_PARA needs to be a small number (due to its small system size).
# If you use GPU, we recommend 1 for both variables.
LIG_PARA=2
COMPLEX_PARA=12

# Threads per process. Recommended values are: 1 (CPU) or 2-4 (GPU).
TPP=1

# first RUN_PROD ps will be ignored as equilibration
RUN_PROD=2000

#*************************
# System setups
#*************************

# Gromacs [ moleculetype ] name of the ligand.
LIG_GMX="MOL"

# Receptor selection string (used to determine restraints). Python "mdtraj" library's selection syntax.
# Note in the mdtraj library's syntax, "residue" or "resSeq" is PDB-based residue ID (probably 1-origin), while "resid" is 0-origin residue number from the beginning.
RECEPTOR_MDTRAJ="protein"

# Solvent moleculetype name, used by gmx genion and the flexible-solvent
# topology switch during minimization.
SOLVENT="SOL"

#*************************
# Thresholding
#*************************
# If ligand RMSD during equilibration exceeds this value, calculation stops (unit: nm)
# RMSD is measured by best-fitting the receptor, but only ligand is used in the RMSD calculation
# The final snapshot of the equilibration will be used as the reference structure.
EQ_RMSD_CUTOFF=0.4

# change this value to max possible ligand diameter (nm). This is usually unnecessary as the proper values are set automatically.
# Large-ligand note: rlist is sized from the diameter measured in the
# restraints stage; if a production mdrun still hits the pair-list
# cutoff, the pipeline automatically enlarges rlist (from the distance
# reported in the mdrun log, or +20% otherwise) and retries before
# giving up and asking for this value.
LIGAND_DIAMETER=0

#*************************
# Parameter optimization
#*************************
# Annihilation lambda parameters are optimized during pre-run phase, ANNIH_LAMBDA_OPT_LENGTH ps and ANNIH_LAMBDA_OPT times
ANNIH_LAMBDA_OPT=5
ANNIH_LAMBDA_OPT_LENGTH=50

#*************************
# Solvation and ionization
#*************************
# water structure file, tip3p & spc & spc-e -> spc216, tip4p -> tip4p
WATER_STRUCTURE=spc216

# water thickness used in ligand system
WATER_THICKNESS=1.0

# ionic strength in M
IONIC_STRENGTH=0.150

# positive ion name (CHARMM uses SOD)
ION_POSITIVE=NA
# negative ion name (CHARMM uses CLA)
ION_NEGATIVE=CL

#*************************
# Charge correction for charged ligands
#*************************

# APBS binary. Only used when the ligand is charged.
APBS=apbs

# Number of sampling used in charge correction. Set to 0 to disable the
# charge correction entirely.
CHARGE_CORRECTION_NSAMP=5
