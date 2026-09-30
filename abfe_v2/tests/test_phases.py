import pytest

from feplab.errors import PipelineError
from feplab.phases import (BAR_TERMS, LRC_SOURCES, LRC_TERMS, PHASES, STAGES,
                           query_line, stage_index)
from feplab.cli import do_query
import io
import contextlib


def test_stages_consistent():
    keys = [s.key for s in STAGES]
    assert len(keys) == len(set(keys))
    seen = set()
    for stage in STAGES:
        for dep in stage.deps:
            assert dep in seen, f"{stage.key} depends on not-yet-declared {dep}"
        seen.add(stage.key)
    assert STAGES[-1].key == "analysis"
    assert STAGES[-1].cpu_only
    assert stage_index("setup") == 0


def test_query_lines():
    assert do_query_capture("all") == " ".join(s.key for s in STAGES)
    line = do_query_capture("charging-lig")
    assert line == "DEPENDS=(prep-charging-lig); PPM=$LIG_PARA ; MULTI=$NCHARGE"
    line = do_query_capture("restraints")
    assert line == "DEPENDS=(equilibrate); PPM=1 ; MULTI=1"
    line = do_query_capture("analysis")
    assert line.endswith(" ; CPU_ONLY_STAGE=yes")
    with pytest.raises(KeyError):
        stage_index("nope")


def do_query_capture(stage: str) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        do_query(stage)
    return buf.getvalue().strip()


def test_phase_table():
    assert set(PHASES) == {"charging-lig", "charging-complex", "annihilation-lig",
                           "annihilation-complex", "restraint"}
    cl = PHASES["charging-lig"]
    assert cl.traj_keep == 0 and cl.system == "ligand"
    assert cl.lr_template == "lr-lig.mdp" and cl.needs_rlist
    rst = PHASES["restraint"]
    assert rst.maxwarn == 1 and not rst.needs_rlist
    assert PHASES["charging-complex"].lr_template is None
    for phase in PHASES.values():
        assert phase.nstates_key in ("NCHARGE", "NANNIH", "NRESTR")


def test_cycle_signs_match_v1():
    # Signs must reproduce v1 calc_bar_replex cycle_contribution exactly.
    assert BAR_TERMS["charging-lig"] == ("charging", 1.0)
    assert BAR_TERMS["charging-complex"] == ("charging", -1.0)
    assert BAR_TERMS["annihilation-lig"] == ("annihilation", 1.0)
    assert BAR_TERMS["annihilation-complex"] == ("annihilation", -1.0)
    assert BAR_TERMS["restraint"] == ("restraint", 1.0)
    assert LRC_TERMS["lr-lig"] == ("long-range-correction", -1.0)
    assert LRC_TERMS["lr-annihilation-lig"] == ("long-range-correction", 1.0)
    assert LRC_TERMS["lr-complex"] == ("long-range-correction", 1.0)
    assert LRC_TERMS["lr-annihilation-complex"] == ("long-range-correction", -1.0)
    assert LRC_SOURCES["lr-lig"] == ("charging-lig", "first")
    assert LRC_SOURCES["lr-complex"] == ("restraint", "last")
