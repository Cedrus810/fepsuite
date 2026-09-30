"""Large-ligand rlist auto-retry tests."""

import pytest

from feplab.config import Config
from feplab.errors import PipelineError, RlistExceededError
from feplab.stages import _next_rlist


LOG_WITH_DISTANCE = """Step 100
There are perturbed non-bonded pair interactions beyond the pair-list cutoff
which is not supported with domain decomposition
At least one pair-wise interaction distance exceeds rlist (1.200)
2 17 1.417 nm - max distance known to be within rlist
Fatal error, exiting
"""


def test_next_rlist_from_reported_distance(tmp_path):
    log = tmp_path / "md.log"
    log.write_text(LOG_WITH_DISTANCE)
    # the log reports pairs up to 1.417 nm -> retry just above that
    assert _next_rlist(log, current=1.2) == pytest.approx(1.417 * 1.1)
    # reported values below current (weird log) -> fall back to the pad
    assert _next_rlist(log, current=2.0) == pytest.approx(2.4)


def test_next_rlist_no_distance(tmp_path):
    log = tmp_path / "md.log"
    log.write_text("There are perturbed non-bonded pair interactions beyond the"
                   " pair-list cutoff\n(no distance given)\n")
    assert _next_rlist(log, current=1.0) == pytest.approx(1.2)
    # missing log file -> same fallback
    assert _next_rlist(tmp_path / "gone.log", current=1.0) == pytest.approx(1.2)


def test_rlist_exceeded_error_carries_log(tmp_path):
    e = RlistExceededError("boom", log_path=str(tmp_path / "md.log"))
    assert e.log_path == str(tmp_path / "md.log")
    assert isinstance(e, PipelineError)


def test_product_runs_rlist_retry(tmp_path, monkeypatch):
    """First mdrun fails with a pair-list cutoff error; the retry rewrites
    the mdp with a larger rlist and succeeds."""
    from feplab import stages
    from feplab.phases import PHASES
    from feplab.runner import RunContext, Runner

    rundir = tmp_path
    run_id = "mol"
    (rundir / "mdp").mkdir(parents=True)
    (rundir / run_id).mkdir()
    template_dir = rundir / "template"
    template_dir.mkdir()
    (template_dir / "charging-lig.mdp").write_text(
        "free_energy = yes\nfep_lambdas = {lambdas_formatted}\n"
        "init_lambda_state = {lambda_state}\ncouple_moltype = {group_mol}\n")
    (rundir / "mdp/run.mdp").write_text(
        "integrator = sd\ndt = 0.002\nnsteps = 100\nrvdw = 1.0\nrcoulomb = 1.0\n"
        "ref_t = 300\nnstxout-compressed = 100\n")
    # a diameter file with a safe value
    (rundir / run_id / "diameter.txt").write_text("avg 1.0\nmax 1.2\nsafe 1.5\n")

    ctx = RunContext(rundir=rundir, run_id=run_id, abfe_v2_root=rundir,
                     fepsuite_root=None, jobsystem=None, gmx="gmx",
                     gmx_mpi="gmx_mpi", gmx_nompi="gmx", omp_threads=None,
                     python3="python3")
    cfg = Config({"NCHARGE": "2", "LIG_PARA": "1", "ANNIH_LAMBDA_OPT": "1"},
                 sources=[])

    calls = {"grompp": 0, "mdrun": 0}

    class FakeRunner:
        def __init__(self):
            self._np_saved = None

        def grompp(self, **kwargs):
            calls["grompp"] += 1
            # touch the tpr so nothing else trips
            Path = type(rundir / "x")
            tpr = rundir / kwargs["out_tpr"]
            tpr.parent.mkdir(parents=True, exist_ok=True)
            tpr.touch()
            for key in ("conf", "cpt"):
                p = rundir / (kwargs[key] + ".pdb" if key == "conf" else kwargs[key] + ".cpt")
                p.parent.mkdir(parents=True, exist_ok=True)
                if not p.exists():
                    p.touch()

        def mpirun_mdrun(self, np_, args, *, least_unit=1, stdout_path=None,
                         stderr_path=None, rlist_retry=False):
            calls["mdrun"] += 1
            if calls["mdrun"] == 1:
                # simulate the GROMACS pair-list failure incl. its log
                log = rundir / run_id / "charging-lig.0" / "charging-lig.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(
                    "domain decomposition error\n"
                    "There are perturbed non-bonded pair interactions beyond the "
                    "pair-list cutoff\nMaximum distance 2.5 nm\n")
                raise RlistExceededError("rlist exceeded", log_path=log)
            # second attempt succeeds
            return None

        def check_replica_probs(self, logfile, threshold=0.03):
            pass

    pipeline = stages.Pipeline(config=cfg, ctx=ctx, runner=FakeRunner())
    monkeypatch.setattr(stages.Pipeline, "_auto_np", lambda self, r: 1)
    pipeline.product_runs(PHASES["charging-lig"], topol="t.top", prev="prev")

    assert calls["mdrun"] == 2
    # after the retry the mdp must carry the enlarged rlist (2.5*1.1)
    mdp_text = (rundir / "mdp" / "charging-lig-0.mdp").read_text()
    assert "rlist = 2.75" in mdp_text


def test_product_runs_rlist_retry_gives_up(tmp_path, monkeypatch):
    """A pathological log demanding >10 nm must abort with a clear error."""
    from feplab import stages
    from feplab.errors import RlistExceededError
    from feplab.phases import PHASES
    from feplab.runner import RunContext

    rundir = tmp_path
    (rundir / "mdp").mkdir(parents=True)
    (rundir / "mol").mkdir()
    template_dir = rundir / "template"
    template_dir.mkdir()
    (template_dir / "charging-lig.mdp").write_text(
        "free_energy = yes\nfep_lambdas = {lambdas_formatted}\n"
        "init_lambda_state = {lambda_state}\ncouple_moltype = {group_mol}\n")
    (rundir / "mdp/run.mdp").write_text(
        "integrator = sd\ndt = 0.002\nnsteps = 100\nrvdw = 1.0\nrcoulomb = 1.0\nref_t = 300\n")
    ctx = RunContext(rundir=rundir, run_id="mol", abfe_v2_root=rundir,
                     fepsuite_root=None, jobsystem=None, gmx="gmx",
                     gmx_mpi="gmx_mpi", gmx_nompi="gmx", omp_threads=None,
                     python3="python3")
    cfg = Config({"NCHARGE": "2", "LIG_PARA": "1"}, sources=[])

    class FakeRunner:
        _np_saved = None

        def grompp(self, **kwargs):
            p = rundir / kwargs["out_tpr"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()

        def mpirun_mdrun(self, np_, args, **kw):
            log = rundir / "mol" / "charging-lig.0" / "charging-lig.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("perturbed non-bonded pair interactions beyond the "
                           "pair-list cutoff\nMaximum distance 11.0 nm\n")
            raise RlistExceededError("rlist exceeded", log_path=log)

        def check_replica_probs(self, logfile, threshold=0.03):
            pass

    pipeline = stages.Pipeline(config=cfg, ctx=ctx, runner=FakeRunner())
    monkeypatch.setattr(stages.Pipeline, "_auto_np", lambda self, r: 1)
    with pytest.raises(PipelineError, match="LIGAND_DIAMETER"):
        pipeline.product_runs(PHASES["charging-lig"], topol="t.top", prev="prev")
