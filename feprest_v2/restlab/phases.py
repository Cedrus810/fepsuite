"""Declarative stage table of the FEP/REST pipeline.

This module replaces the numbered stages (``run,1`` ... ``run,8``,
``run,999``) of the v1 ``feprest/pipeline.zsh`` with named stages.  The
mapping is one-to-one:

    v1 run,1   -> minimize      (steep + cg of state A)
    v1 run,2   -> nvt           (state-A NVT with hydrogen-massed topology)
    v1 run,3   -> npt           (state-A NPT, emits fep_pp.top)
    v1 run,4   -> rest2-setup   (hot-region underline, neutralization,
                                 per-replica REST2 topologies and minimization)
    v1 run,5   -> tune          (replex probability tuning cycles)
    v1 run,6   -> npt-replex    (multi-replica NPT)
    v1 run,7   -> prodrun-init  (first production chunk + checkpoint)
    v1 run,8+  -> prodrun       (extension chunks; re-run the stage to extend)
    v1 run,999 -> trajectory    (centered state A/B trajectories)

The stage order is a linear chain.  Unlike the ABFE pipeline there is no
DAG branching: every stage depends on the previous one.  The ``trajectory``
stage is auxiliary (excluded from ``all``, as in v1 where ``all`` meant
``{1..8}``).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    """One pipeline stage.

    ``multi_key`` names the config key holding the replica count for the
    replex stages (controller.zsh computes PROCS = MULTI * PPM); the
    single-replica preparation stages keep MULTI=1.  ``in_all`` selects
    whether the stage belongs to the ``all`` expansion.
    """

    key: str
    deps: tuple[str, ...]
    multi_key: str | None = None   # config key for replica count, None -> 1
    ppm_key: str | None = "PARA"   # config key for ranks/replica, None -> 1
    in_all: bool = True
    description: str = ""


# All replex stages run PPM = $PARA (ranks per replica) exactly like v1.
STAGES: list[Stage] = [
    Stage("minimize", (),
          description="state-A steepest-descent + cg minimization"),
    Stage("nvt", ("minimize",),
          description="state-A NVT with hydrogen-massed topology (emits fep_pp.top)"),
    Stage("npt", ("nvt",),
          description="state-A NPT equilibration (10 ns)"),
    Stage("rest2-setup", ("npt",),
          description="REST2 hot-region underline, neutralization, per-replica REST2 "
                      "topologies and minimization"),
    Stage("tune", ("rest2-setup",), multi_key="NREP",
          description="replica-exchange probability tuning cycles (NTUNE times)"),
    Stage("npt-replex", ("tune",), multi_key="NREP",
          description="multi-replica NPT equilibration"),
    Stage("prodrun-init", ("npt-replex",), multi_key="NREP",
          description="first production chunk (50 ps) with deltae sampling"),
    Stage("prodrun", ("prodrun-init",), multi_key="NREP",
          description="production extension chunk (SIMLENGTH ps) + BAR analysis; "
                      "re-run to extend the sampling"),
    Stage("trajectory", ("prodrun",), ppm_key=None, in_all=False,
          description="centered state-A/B trajectories for inspection"),
]


def stage_index(key: str) -> int:
    for i, stage in enumerate(STAGES):
        if stage.key == key:
            return i
    raise KeyError(f"unknown stage: {key}")


def query_line(key: str) -> str:
    """Render the controller.zsh protocol line for a stage.

    controller.zsh evaluates this line, so PPM/MULTI must be literal shell
    variable references (or 1), exactly as in v1's ``query,N`` outputs.
    Every replex-capable stage runs PPM=$PARA (v1: ``(( PPM = PARA ))``);
    the auxiliary trajectory stage runs a single rank (v1 run,999:
    ``PPM=1; MULTI=1``).
    """
    stage = STAGES[stage_index(key)]
    ppm = f"${stage.ppm_key}" if stage.ppm_key else "1"
    multi = f"${stage.multi_key}" if stage.multi_key else "1"
    deps = " ".join(stage.deps)
    return f"DEPENDS=({deps}); PPM={ppm} ; MULTI={multi}"


def all_stages() -> list[str]:
    """Stage names submitted by ``run.zsh <ID> all`` (v1: stages 1..8)."""
    return [s.key for s in STAGES if s.in_all]
