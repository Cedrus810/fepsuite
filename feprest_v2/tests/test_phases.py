import io
import contextlib

import pytest

from restlab.errors import PipelineError
from restlab.phases import STAGES, all_stages, query_line, stage_index
from restlab.cli import do_query


def test_stages_consistent():
    keys = [s.key for s in STAGES]
    assert len(keys) == len(set(keys))
    seen = set()
    for stage in STAGES:
        for dep in stage.deps:
            assert dep in seen, f"{stage.key} depends on not-yet-declared {dep}"
        seen.add(stage.key)
    stage_index("minimize") == 0
    with pytest.raises(KeyError):
        stage_index("nope")


def test_v1_numbering_mapping():
    # the named stages must appear in v1's numeric order (1..8, 999)
    assert [s.key for s in STAGES] == [
        "minimize", "nvt", "npt", "rest2-setup", "tune",
        "npt-replex", "prodrun-init", "prodrun", "trajectory"]


def test_all_excludes_trajectory():
    # v1 "all" meant stages {1..8}; the trjconv stage is auxiliary
    assert all_stages()[-1] == "prodrun"
    assert "trajectory" not in all_stages()


def test_query_lines_match_v1_resources():
    # v1: stages 1-4 -> MULTI=1, stages 5-8 -> MULTI=NREP, PPM=PARA everywhere
    assert do_query_capture("all") == " ".join(all_stages())
    assert do_query_capture("minimize") == "DEPENDS=(); PPM=$PARA ; MULTI=1"
    assert do_query_capture("nvt") == "DEPENDS=(minimize); PPM=$PARA ; MULTI=1"
    assert do_query_capture("npt") == "DEPENDS=(nvt); PPM=$PARA ; MULTI=1"
    assert do_query_capture("rest2-setup") == "DEPENDS=(npt); PPM=$PARA ; MULTI=1"
    assert do_query_capture("tune") == "DEPENDS=(rest2-setup); PPM=$PARA ; MULTI=$NREP"
    assert do_query_capture("npt-replex") == "DEPENDS=(tune); PPM=$PARA ; MULTI=$NREP"
    assert do_query_capture("prodrun-init") == "DEPENDS=(npt-replex); PPM=$PARA ; MULTI=$NREP"
    assert do_query_capture("prodrun") == "DEPENDS=(prodrun-init); PPM=$PARA ; MULTI=$NREP"
    # v1 run,999: DEPENDS=(); PPM=1; MULTI=1
    assert do_query_capture("trajectory") == "DEPENDS=(prodrun); PPM=1 ; MULTI=1"


def do_query_capture(stage: str) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        do_query(stage)
    return buf.getvalue().strip()


def test_query_errors_are_pipeline_errors():
    # controller.zsh evals the query output; a bad stage must not traceback
    with pytest.raises(KeyError):
        do_query("unknown-stage")
