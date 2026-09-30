from pathlib import Path

import pytest

from restlab.runner import RunContext, Runner


def make_runner(tmp_path):
    ctx = RunContext(rundir=tmp_path, run_id="mol1", feprest_v2_root=tmp_path,
                     fepsuite_root=None, jobsystem=None, gmx="gmx",
                     gmx_mpi="gmx_mpi", omp_threads=None, python3="python3")
    return Runner(ctx)


def test_mdrun_log_paths_single():
    assert Runner._mdrun_log_paths(["-deffnm", "mol1/steepA"]) == \
        [Path("mol1/steepA.log")]


def test_mdrun_log_paths_multidir():
    paths = Runner._mdrun_log_paths(
        ["-deffnm", "nvt", "-multidir", "nvt1/rep0", "nvt1/rep1"])
    assert paths == [Path("nvt1/rep0/nvt.log"), Path("nvt1/rep1/nvt.log")]


def test_mdrun_log_paths_log_flag():
    assert Runner._mdrun_log_paths(["-l", "foo", "-multidir", "d"]) == \
        [Path("d/foo.log")]
    assert Runner._mdrun_log_paths(["-deffnm", "x.log"]) == [Path("x.log")]
    assert Runner._mdrun_log_paths(["-multidir", "d"]) == []


def test_is_domain_error(tmp_path):
    runner = make_runner(tmp_path)
    (tmp_path / "rep0").mkdir()
    (tmp_path / "rep0" / "nvt.log").write_text(
        "ok\n" * 30 + "Fatal error: domain decomposition grid\n")
    assert runner._is_domain_error(
        ["-deffnm", "nvt", "-multidir", "rep0"])

    # a listed multidir without any log -> startup failure, not silently
    # classified as domain-unrelated (v1 fix: per-replica logs inspected)
    (tmp_path / "rep1").mkdir()
    with pytest.raises(Exception, match="startup"):
        runner._is_domain_error(["-deffnm", "nvt", "-multidir", "rep1"])


def test_is_domain_error_unrelated(tmp_path):
    runner = make_runner(tmp_path)
    (tmp_path / "mol1").mkdir()
    (tmp_path / "mol1" / "steepA.log").write_text(
        "ok\n" * 30 + "Fatal error: unknown\n")
    assert not runner._is_domain_error(["-deffnm", "mol1/steepA"])
