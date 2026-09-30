"""Declarative tables of the ABFE pipeline.

This module replaces the numbered stages (``run,1`` ... ``run,12``) of the
v1 ``abfe/pipeline.zsh`` with named stages, and collects the thermodynamic
cycle bookkeeping (which BAR/LRC/charge-correction term contributes to
which subtotal, with which sign) in a single place.

The stage DAG is identical to v1's dependency declarations:

    setup -> equilibrate -> restraints -+-> prep-charging-lig -> charging-lig
                                        |      -> annihilation-lig -> lrc-lig
                                        +-> charging-complex -> annihilation-complex
                                        +-> restraint-decouple
                                        (annihilation-complex, restraint-decouple) -> lrc-complex
    (everything) -> analysis
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Phase:
    """One FEP phase of the thermodynamic cycle (a replex product run)."""

    key: str                 # "charging-lig" etc.
    system: str              # "ligand" | "complex"
    template: str            # mdp addenda file under abfe_v2/template/
    nstates_key: str         # config key: number of lambda states
    para_key: str | None     # config key: MPI ranks per replica (None -> 1)
    lambda_kind: str | None  # lambda schedule kind
    pull_mdp: str | None     # restraint pull mdp inside the run ID dir
    traj_keep: int | str | None  # replica keeping its trajectory: None / int / "last"
    maxwarn: int
    needs_rlist: bool
    lr_template: str | None  # mdp template of the PME re-evaluation run


PHASES: dict[str, Phase] = {
    "charging-lig": Phase(
        key="charging-lig", system="ligand", template="charging-lig.mdp",
        nstates_key="NCHARGE", para_key="LIG_PARA", lambda_kind="charging",
        pull_mdp=None, traj_keep=0,  # replica 0 trajectory feeds lr-lig
        maxwarn=0, needs_rlist=True, lr_template="lr-lig.mdp",
    ),
    "charging-complex": Phase(
        key="charging-complex", system="complex", template="charging.mdp",
        nstates_key="NCHARGE", para_key="COMPLEX_PARA", lambda_kind="charging",
        pull_mdp="restr_pull.mdp", traj_keep=None,
        maxwarn=0, needs_rlist=True, lr_template=None,
    ),
    "annihilation-lig": Phase(
        key="annihilation-lig", system="ligand", template="annihilation.mdp",
        nstates_key="NANNIH", para_key="LIG_PARA", lambda_kind="annihilation",
        pull_mdp=None, traj_keep="last",
        maxwarn=0, needs_rlist=True, lr_template="lr-annihilation-lig.mdp",
    ),
    "annihilation-complex": Phase(
        key="annihilation-complex", system="complex", template="annihilation.mdp",
        nstates_key="NANNIH", para_key="COMPLEX_PARA", lambda_kind="annihilation",
        pull_mdp="restr_pull.mdp", traj_keep="last",
        maxwarn=0, needs_rlist=True, lr_template="lr-annihilation-complex.mdp",
    ),
    "restraint": Phase(
        key="restraint", system="complex", template="restraint.mdp",
        nstates_key="NRESTR", para_key="COMPLEX_PARA", lambda_kind="restraint",
        pull_mdp="restr_pull_decouple.mdp", traj_keep="last",
        maxwarn=1, needs_rlist=False, lr_template="lr-complex.mdp",
    ),
}


# Thermodynamic-cycle bookkeeping: term -> (subtotal name, coefficient).
# Signs are exactly those of v1 calc_bar_replex.py cycle_contribution.
BAR_TERMS: dict[str, tuple[str, float]] = {
    "charging-lig": ("charging", 1.0),
    "charging-complex": ("charging", -1.0),
    "annihilation-lig": ("annihilation", 1.0),
    "annihilation-complex": ("annihilation", -1.0),
    "restraint": ("restraint", 1.0),
}

LRC_TERMS: dict[str, tuple[str, float]] = {
    "lr-lig": ("long-range-correction", -1.0),
    "lr-annihilation-lig": ("long-range-correction", 1.0),
    "lr-complex": ("long-range-correction", 1.0),
    "lr-annihilation-complex": ("long-range-correction", -1.0),
}

# Charge-correction subtotals are folded into "charging" by the report
# writer (v1 behavior); they never appear as their own subtotal.
CC_TERMS: dict[str, tuple[str, float]] = {
    "charge-correction-complex": ("charge-correction", -1.0),
    "charge-correction-ligand": ("charge-correction", 1.0),
}

# Which phase/replica each long-range re-evaluation reads.
LRC_SOURCES: dict[str, tuple[str, str]] = {
    "lr-lig": ("charging-lig", "first"),
    "lr-annihilation-lig": ("annihilation-lig", "last"),
    "lr-complex": ("restraint", "last"),
    "lr-annihilation-complex": ("annihilation-complex", "last"),
}

# The lambda=1 (decoupled) endpoints: their ligand-environment dispersion
# is identically zero, so their LRC re-evaluation can be skipped
# (SKIP_ANNIHILATION_LRC) — see docs/issue-lrc-gmx2026.md.
ANNIHILATION_LRC_TERMS = ("lr-annihilation-lig", "lr-annihilation-complex")


@dataclass(frozen=True)
class Stage:
    """A pipeline stage: what it depends on and what resources it needs."""

    key: str
    deps: tuple[str, ...]
    ppm_key: str | None = None     # config key for ranks/replica, None -> 1
    multi_key: str | None = None   # config key for replica count, None -> 1
    cpu_only: bool = False
    prep_like: bool = False        # runs only pre-equilibration-class mdruns
    description: str = ""


STAGES: list[Stage] = [
    Stage("setup", (), ppm_key="COMPLEX_PARA", prep_like=True,
          description="GROMACS version gate, preprocessing, complex index, restrained minimization/eq chain"),
    Stage("equilibrate", ("setup",), ppm_key="COMPLEX_PARA",
          description="unrestrained NPT equilibration of the complex"),
    Stage("restraints", ("equilibrate",), prep_like=True,
          description="PBC-fixed trajectory analysis, Boresch restraint search, ligand extraction and solvation"),
    Stage("prep-charging-lig", ("restraints",), ppm_key="LIG_PARA", prep_like=True,
          description="minimization/eq chain of the ligand-only system"),
    Stage("charging-lig", ("prep-charging-lig",), ppm_key="LIG_PARA", multi_key="NCHARGE",
          description="ligand charge-discharge replex run"),
    Stage("annihilation-lig", ("charging-lig",), ppm_key="LIG_PARA", multi_key="NANNIH",
          description="ligand annihilation replex run with lambda optimization preruns"),
    Stage("lrc-lig", ("charging-lig", "annihilation-lig"), ppm_key="LIG_PARA",
          description="PME long-range re-evaluation runs for the ligand system"),
    Stage("charging-complex", ("restraints",), ppm_key="COMPLEX_PARA", multi_key="NCHARGE",
          description="complex charge-discharge replex run"),
    Stage("annihilation-complex", ("annihilation-lig", "charging-complex"),
          ppm_key="COMPLEX_PARA", multi_key="NANNIH",
          description="complex annihilation replex run with lambda optimization preruns"),
    Stage("restraint-decouple", ("restraints",), ppm_key="COMPLEX_PARA", multi_key="NRESTR",
          description="Boresch restraint decoupling replex run"),
    Stage("lrc-complex", ("annihilation-complex", "restraint-decouple"), ppm_key="COMPLEX_PARA",
          description="PME long-range re-evaluation runs for the complex system"),
    Stage("analysis", ("setup", "equilibrate", "restraints", "prep-charging-lig",
                       "charging-lig", "annihilation-lig", "lrc-lig",
                       "charging-complex", "annihilation-complex", "restraint-decouple",
                       "lrc-complex"),
          cpu_only=True,
          description="BAR analysis, long-range corrections, charge correction, final report"),
]


def stage_index(key: str) -> int:
    for i, stage in enumerate(STAGES):
        if stage.key == key:
            return i
    raise KeyError(f"unknown stage: {key}")


def query_line(key: str, config=None) -> str:
    """Render the controller.zsh protocol line for a stage.

    controller.zsh evaluates this line, so PPM/MULTI must be literal shell
    variable references (or 1), exactly as in v1's ``query,N`` outputs.

    When a ``Config`` is given, values that are knowable without the job
    allocation are resolved to concrete numbers: with ``PREP_NONMPI=yes``
    the prep-like stages run their mdruns on a single non-MPI rank, so
    they only need PPM=1.  The GPU-dependent stages keep their literal
    ``$KEY`` references because the GPU count is only known inside the
    job (see feplab.resources for the runtime side).
    """
    stage = STAGES[stage_index(key)]
    if stage.prep_like and config is not None and config.prep_nonmpi:
        ppm = "1"
    else:
        ppm = f"${stage.ppm_key}" if stage.ppm_key else "1"
    multi = f"${stage.multi_key}" if stage.multi_key else "1"
    deps = " ".join(stage.deps)
    line = f"DEPENDS=({deps}); PPM={ppm} ; MULTI={multi}"
    if stage.cpu_only:
        line += " ; CPU_ONLY_STAGE=yes"
    return line
