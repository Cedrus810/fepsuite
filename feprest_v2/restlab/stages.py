"""The pipeline stages: a port of v1 ``feprest/pipeline.zsh``.

Mapping of the v1 numbered stages to the named stages (identical
content, reorganized):

    v1 run,1   -> stage_minimize
    v1 run,2   -> stage_nvt
    v1 run,3   -> stage_npt
    v1 run,4   -> stage_rest2_setup
    v1 run,5   -> stage_tune
    v1 run,6   -> stage_npt_replex
    v1 run,7   -> stage_prodrun_init
    v1 run,8+  -> stage_prodrun   (re-running the stage extends by SIMLENGTH)
    v1 run,999 -> stage_trajectory

All paths passed to GROMACS are relative to the run directory (v1 used
``$ID/...`` with the working directory at the run root, and so does
this port).

Behavioral notes carried over from v1 (and small fixes):

* every mdrun goes through the job-system MPI wrapper with the
  NP-search retry loop (``runner.Runner.mpirun_mdrun``), exactly like
  v1's ``mdrun_find_possible_np``;
* v1's ``RUN_TPR_PARALLEL`` (grompp/convert-tpr of all replicas
  concurrently — they dominate wall time otherwise) is now the default;
* v1 clamped the tuning blend step to 1 (``NINITIALTUNE`` equals
  ``NTUNE``), i.e. every tuning cycle blends the optimized and previous
  coordinates 50/50 — kept, formula included;
* the ``checkpoint_N`` directories of v1 become ``checkpoint_init`` /
  ``checkpoint_ph<N>`` (they were named after stage numbers);
* the BAR analysis failure stays non-fatal with v1's message, so a
  chunk can simply be extended to improve convergence.
"""

from __future__ import annotations

import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path

from . import bar, mdp, neutralize
from .analysis_index import write_centering_ndx
from .config import Config
from .errors import PipelineError
from .heavyhydrogen import turn_heavy
from .hotregion import add_underline, write_hot_index
from .phases import stage_index
from .rest2 import replica
from .runner import RunContext, Runner
from .waterion import recover_water


class Tee:
    """Redirect_stdout target mirroring lines to the real stdout."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


class Pipeline:
    def __init__(self, config: Config, ctx: RunContext, runner: Runner):
        self.cfg = config
        self.ctx = ctx
        self.runner = runner
        self.rundir = ctx.rundir
        self.id = Path(ctx.run_id)
        self.mdp_dir = self.rundir / "mdp"

    # ---- helpers -----------------------------------------------------------
    def idpath(self, *parts) -> str:
        return str(self.id.joinpath(*parts))

    def mdppath(self, name: str) -> str:
        return str(self.mdp_dir / name)

    def _ref_init_args(self) -> list[str]:
        """v1 initref(): REFCMDINIT (-r $ID/$REFINIT) when REFINIT is set."""
        refinit = self.cfg.refinit
        return ["-r", self.idpath(refinit)] if refinit else []

    def _ref_args(self) -> list[str]:
        """v1 initref(): REFCMD (-r $ID/$REFCRD) when REFCRD is set."""
        refcrd = self.cfg.refcrd
        return ["-r", self.idpath(refcrd)] if refcrd else []

    def _read_top(self) -> str:
        return (self.id / self.cfg.basetop).read_text()

    def _strip_posres_if_absent(self, mdp_text: str, top_text: str) -> str:
        if not mdp.topology_defines_posres(top_text):
            mdp_text = mdp.remove_posres_define(mdp_text)
        return mdp_text

    def _itp_symlink(self, target: Path, link: Path) -> None:
        try:
            os.symlink(str(target), str(link))
        except (FileExistsError, OSError):
            pass  # v1: "|| true"

    def _run_parallel(self, jobs: list[tuple[str, Path, list[str]]]) -> None:
        """Run grompp/convert-tpr jobs, concurrently when enabled.

        ``jobs`` is a list of (error label, stdout log path, command args).
        Mirrors v1 parallelizable_singlerun + wait_if_needed: on failure
        the offending replica is named and the stage aborts.
        """
        if self.cfg.run_tpr_parallel and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                futures = [(label, log, pool.submit(
                    self.runner.singlerun, args, stdout_path=log, check=True))
                           for (label, log, args) in jobs]
                failures = []
                for label, _, future in futures:
                    try:
                        future.result()
                    except Exception as e:  # noqa: BLE001 — reported below
                        failures.append(f"{label}: {e}")
                if failures:
                    raise PipelineError("Error: " + "; ".join(failures))
        else:
            for (label, log, args) in jobs:
                try:
                    self.runner.singlerun(args, stdout_path=log, check=True)
                except Exception as e:
                    raise PipelineError(f"Error: {label}: {e}") from None

    def _grompp_args(self, *, mdp_path, conf, topol, out_tpr, out_mdp,
                     maxwarn, cpt=None, restr=None, ndx=None, pp=None) -> list[str]:
        args = [self.runner.ctx.gmx, "grompp", "-f", mdp_path,
                "-p", topol, "-c", conf]
        if cpt:
            args += ["-t", cpt]
        if restr:
            args += ["-r", restr]
        if ndx:
            args += ["-n", ndx]
        if pp:
            args += ["-pp", pp]
        args += ["-o", out_tpr, "-po", out_mdp, "-maxwarn", str(maxwarn)]
        return args

    def _mdrun_multidir_args(self, deffnm: str, reps: list[str], extra: list[str]) -> list[str]:
        args = ["mdrun", "-deffnm", deffnm, "-multidir", *reps, "-rdd",
                str(self.cfg.domain_shrink)]
        return args + extra

    def _checkpoint(self, source_paths: list[str], dirname: str) -> None:
        ckdir = self.id / dirname
        for src in source_paths:
            dest = ckdir / src
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    # ---- stages -------------------------------------------------------------
    def stage_minimize(self) -> None:
        # min for state A
        top_text = self._read_top()
        min_mdp = self._strip_posres_if_absent(
            mdp.strip_fep((self.mdp_dir / "cginit.mdp").read_text()), top_text)
        (self.id / "minA.mdp").write_text(min_mdp)
        (self.id / "steepA.mdp").write_text(
            mdp.set_integrator(min_mdp, "steep"))
        refinit = self._ref_init_args()
        self.runner.grompp(
            mdp=self.idpath("steepA.mdp"), topol=self.idpath(self.cfg.basetop),
            conf=self.idpath(self.cfg.baseconf), out_tpr=self.idpath("steepA.tpr"),
            out_mdp=self.idpath("steepA.mdout"),
            maxwarn=self.cfg.basewarn + 0, restr=refinit or None)
        self.runner.mpirun_mdrun(
            self.cfg.para, ["-deffnm", self.idpath("steepA")],
            least_unit=1, nstlist_cmd=self.cfg.nstlist_cmd)
        self.runner.grompp(
            mdp=self.idpath("minA.mdp"), topol=self.idpath(self.cfg.basetop),
            conf=self.idpath("steepA.gro"), out_tpr=self.idpath("minA.tpr"),
            out_mdp=self.idpath("minA.mdout"),
            maxwarn=self.cfg.basewarn + 0, restr=refinit or None)
        self.runner.mpirun_mdrun(
            self.cfg.para, ["-deffnm", self.idpath("minA")],
            least_unit=1, nstlist_cmd=self.cfg.nstlist_cmd)

    def stage_nvt(self) -> None:
        # NVT run for state A
        top_text = self._read_top()
        nvt_mdp = self._strip_posres_if_absent(
            mdp.strip_fep((self.mdp_dir / "nvtinit.mdp").read_text()), top_text)
        (self.id / "nvtA.mdp").write_text(nvt_mdp)
        # v1 turn-heavy.py
        turn_heavy(self.idpath(self.cfg.basetop), self.idpath("heavy.top"))
        self.runner.grompp(
            mdp=self.idpath("nvtA.mdp"), topol=self.idpath("heavy.top"),
            conf=self.idpath("minA.gro"), out_tpr=self.idpath("nvtA.tpr"),
            out_mdp=self.idpath("nvtA.mdout"),
            maxwarn=self.cfg.basewarn + 1,
            pp=self.idpath("fep_pp.top"),
            restr=self._ref_init_args() or None)
        self.runner.mpirun_mdrun(
            self.cfg.para, ["-deffnm", self.idpath("nvtA")],
            least_unit=1, nstlist_cmd=self.cfg.nstlist_cmd)

    def stage_npt(self) -> None:
        # NPT run (10 ns); produces the preprocessed topology for the FEP legs
        shutil.copyfile(self.mdp_dir / "nptinit.mdp", self.id / "nptA.mdp")
        self.runner.grompp(
            mdp=self.idpath("nptA.mdp"), topol=self.idpath("heavy.top"),
            conf=self.idpath("nvtA.gro"), out_tpr=self.idpath("nptA.tpr"),
            out_mdp=self.idpath("nptA.mdout"),
            maxwarn=self.cfg.basewarn + 1,
            pp=self.idpath("fep_pp.top"),
            restr=self._ref_args() or None)
        self.runner.mpirun_mdrun(
            self.cfg.para, ["-deffnm", self.idpath("nptA")],
            least_unit=1, nstlist_cmd=self.cfg.nstlist_cmd)

    def stage_rest2_setup(self) -> None:
        # initialize the REST2 region and the per-replica states
        add_underline(
            structure=self.idpath(self.cfg.baseconf),
            topology=self.idpath("fep_pp.top"),
            output=self.idpath("fep_underlined.top"),
            distance=self.cfg.rest2_region_distance,
            target_molecule=self.cfg.target_molecule,
            non_perturbed_moleculetype=self.cfg.non_perturbed_moleculetype)
        prev = self.idpath("nptA.gro")
        top = self.idpath("fep_underlined.top")
        if self.cfg.charge_mode != "no":
            neutralize(topology=self.idpath("fep_underlined.top"),
                       gro=self.idpath("nptA.gro"),
                       output_topology=self.idpath("fep_underlined_neut.top"),
                       output_gro=self.idpath("nptA_neut.gro"),
                       mode=self.cfg.charge_mode, ff=self.cfg.ff)
            prev = self.idpath("nptA_neut.gro")
            top = self.idpath("fep_underlined_neut.top")
        write_hot_index(top, self.idpath("for_rest.ndx"))
        replica.do_init(self.cfg.nrep, "feprest", basedir=self.id)
        genmdps = self.id / "genmdps"
        gentops = self.id / "gentops"
        genmdps.mkdir(exist_ok=True)
        gentops.mkdir(exist_ok=True)
        replica.update_mdps(self.mdppath("cg.mdp"),
                            self.idpath("genmdps/cg%d.mdp"),
                            basedir=self.id, temp0=self.cfg.rest2_temp0,
                            temp=self.cfg.rest2_temp)
        replica.update_topologies(top, self.idpath("gentops/fep_%d.top"),
                                  basedir=self.id, temp0=self.cfg.rest2_temp0,
                                  temp=self.cfg.rest2_temp)
        # enable #include "foo.itp" / "../foo.itp" in the topologies
        # (v1 FIXME hack, kept)
        for itp in (self.ctx.feprest_v2_root / "itp_addenda").glob("*.itp"):
            self._itp_symlink(itp, self.id / itp.name)
        for itp in (self.rundir / self.id).glob("*.itp"):
            self._itp_symlink(itp.resolve(), gentops / itp.name)
        for itp in self.rundir.glob("*.itp"):
            self._itp_symlink(itp.resolve(), self.id / itp.name)

        nrep = self.cfg.nrep
        for i in range(nrep):
            work = self.id / f"min{i}"
            work.mkdir(exist_ok=True)
            (work / f"steep{i}.mdp").write_text(
                mdp.cg_to_steep_minimization((genmdps / f"cg{i}.mdp").read_text()))
            # water moleculetype restored to the flexible force-field version,
            # then hydrogens turned heavy for the minimization
            recover_water(gentops / f"fep_{i}.top",
                          gentops / f"fep_tip3p_{i}_light.top", ff=self.cfg.ff,
                          water_moltype=self.cfg.water_moltype)
            turn_heavy(gentops / f"fep_tip3p_{i}_light.top",
                       gentops / f"fep_tip3p_{i}.top")
            self.runner.grompp(
                mdp=str(work / f"steep{i}.mdp"), conf=prev,
                topol=self.idpath(f"gentops/fep_tip3p_{i}.top"),
                out_tpr=str(work / f"steep{i}.tpr"),
                out_mdp=str(work / f"steep.mdout.{i}"),
                maxwarn=self.cfg.basewarn + 1,
                restr=self._ref_args() or None)
            self.runner.mpirun_mdrun(
                self.cfg.para, ["-deffnm", str(work / f"steep{i}"),
                                "-rdd", str(self.cfg.domain_shrink)],
                least_unit=1, nstlist_cmd=self.cfg.nstlist_cmd)
            self.runner.grompp(
                mdp=str(genmdps / f"cg{i}.mdp"), conf=str(work / f"steep{i}.gro"),
                topol=self.idpath(f"gentops/fep_tip3p_{i}.top"),
                out_tpr=str(work / f"min{i}.tpr"),
                out_mdp=str(work / f"min.mdout.{i}"),
                maxwarn=self.cfg.basewarn + 1,
                restr=self._ref_args() or None)
            self.runner.mpirun_mdrun(
                self.cfg.para, ["-deffnm", str(work / f"min{i}"),
                                "-rdd", str(self.cfg.domain_shrink)],
                least_unit=1, nstlist_cmd=self.cfg.nstlist_cmd)
            prev = str(work / f"min{i}")

    def stage_tune(self) -> None:
        # tune the replica exchange parameters (NTUNE cycles)
        top = self.idpath("fep_underlined.top")
        if self.cfg.charge_mode != "no":
            top = self.idpath("fep_underlined_neut.top")
        nrep = self.cfg.nrep
        prevgro = [self.idpath(f"min{i}/min{i}.gro") for i in range(nrep)]
        # v1: NINITIALTUNE=${NTUNE:-0}; STEPCOUNT = clamp(p - NINITIALTUNE, 1, ..)
        # so every cycle blends with step 1 (documented in the module docstring)
        ntune_initial = self.cfg.ntune
        for p in range(1, self.cfg.ntune + 1):
            work = self.id / f"nvt{p}"
            work.mkdir(exist_ok=True)
            replica.update_mdps(self.mdppath("nvt.mdp"),
                                str(work / f"nvt{p}_%d.mdp"),
                                basedir=self.id, temp0=self.cfg.rest2_temp0,
                                temp=self.cfg.rest2_temp)
            replica.update_topologies(top, str(work / "fep_%d.top"),
                                      basedir=self.id, temp0=self.cfg.rest2_temp0,
                                      temp=self.cfg.rest2_temp)
            reps = [str(work / f"rep{i}") for i in range(nrep)]
            jobs = []
            for i in range(nrep):
                mrundir = work / f"rep{i}"
                mrundir.mkdir(exist_ok=True)
                recover_water(work / f"fep_{i}.top",
                              work / f"fep_tip3p_{i}_light.top", ff=self.cfg.ff,
                              water_moltype=self.cfg.water_moltype)
                turn_heavy(work / f"fep_tip3p_{i}_light.top",
                           work / f"fep_tip3p_{i}.top")
                with open(work / f"nvt{p}_{i}.mdp", "a") as fh:
                    fh.write("energygrps = hot\nuserint1 = 1\n")
                jobs.append((f"failed to grompp on tuning cycle {p} replica {i}",
                             mrundir / "grompp.log",
                             self._grompp_args(
                                 mdp_path=str(work / f"nvt{p}_{i}.mdp"),
                                 conf=prevgro[i],
                                 topol=str(work / f"fep_tip3p_{i}.top"),
                                 out_tpr=str(mrundir / "nvt.tpr"),
                                 out_mdp=str(mrundir / "nvt.mdout"),
                                 maxwarn=self.cfg.basewarn + 1,
                                 ndx=self.idpath("for_rest.ndx"),
                                 restr=self._ref_args() or None)))
            self._run_parallel(jobs)

            # Actually run MD in parallel
            self.runner.mpirun_mdrun(
                nrep, self._mdrun_multidir_args("nvt", reps, []),
                least_unit=nrep, nstlist_cmd=self.cfg.nstlist_cmd)

            # Extend the run to do replica exchange
            jobs = []
            for i in range(nrep):
                mrundir = work / f"rep{i}"
                jobs.append((f"failed to run convert-tpr on tuning cycle {p} "
                             f"replica {i}", mrundir / "extend_run.log",
                             [self.runner.ctx.gmx, "convert-tpr",
                              "-s", str(mrundir / "nvt.tpr"),
                              "-o", str(mrundir / "nvt_c.tpr"),
                              "-extend", str(self.cfg.repopt_extend_run_length)]))
            self._run_parallel(jobs)
            # -bonded cpu is needed because of the current patch's restriction
            extra = ["-s", "nvt_c", "-cpi", "nvt", "-hrex",
                     "-replex", str(self.cfg.repopt_replex_interval)]
            if self.cfg.repopt_replex_interval <= 50:
                # Add nstlist if replex interval is too short
                extra += ["-nstlist", str(self.cfg.repopt_replex_interval)]
            extra += ["-bonded", "cpu"]
            self.runner.mpirun_mdrun(
                nrep, self._mdrun_multidir_args("nvt", reps, extra),
                least_unit=nrep, nstlist_cmd=self.cfg.nstlist_cmd)
            step = p - ntune_initial
            if step < 1:
                step = 1
            replica.do_optimize_step(work / f"rep0" / "nvt.log",
                                     basedir=self.id, nstep=step)
            print((self.id / "replica_states").read_text(), end="")
            prevgro = [str(work / f"rep{i}" / "nvt.gro") for i in range(nrep)]
        for i in range(nrep):
            shutil.copyfile(work / f"fep_tip3p_{i}.top",
                            self.id / "gentops" / f"fep_tip3p_{i}.top")

    def stage_npt_replex(self) -> None:
        # NPT run of all replicas
        work = self.id / "npt"
        work.mkdir(exist_ok=True)
        replica.update_mdps(self.mdppath("npt.mdp"),
                            str(work / "npt%d.mdp"),
                            basedir=self.id, temp0=self.cfg.rest2_temp0,
                            temp=self.cfg.rest2_temp)
        nrep = self.cfg.nrep
        reps = [str(work / f"rep{i}") for i in range(nrep)]
        jobs = []
        for i in range(nrep):
            mrundir = work / f"rep{i}"
            mrundir.mkdir(exist_ok=True)
            jobs.append((f"failed to grompp on final npt run replica {i}",
                         mrundir / "npt_grompp.log",
                         self._grompp_args(
                             mdp_path=str(work / f"npt{i}.mdp"),
                             conf=self.idpath(f"nvt{self.cfg.ntune}/rep{i}/nvt.gro"),
                             topol=self.idpath(f"gentops/fep_tip3p_{i}.top"),
                             out_tpr=str(mrundir / "npt.tpr"),
                             out_mdp=str(mrundir / "npt.mdout"),
                             cpt=self.idpath(f"nvt{self.cfg.ntune}/rep{i}/nvt.cpt"),
                             maxwarn=self.cfg.basewarn + 1,
                             restr=self._ref_args() or None)))
        self._run_parallel(jobs)
        self.runner.mpirun_mdrun(
            nrep, self._mdrun_multidir_args("npt", reps, []),
            least_unit=nrep, nstlist_cmd=self.cfg.nstlist_cmd)

    def stage_prodrun_init(self) -> None:
        # 50 ps initialization of the production run
        runmdps = self.id / "runmdps"
        runmdps.mkdir(exist_ok=True)
        (self.id / "run.mdout").mkdir(exist_ok=True)  # v1 artifact, kept
        replica.update_mdps(self.mdppath("run.mdp"),
                            self.idpath("runmdps/run%d.mdp"),
                            basedir=self.id, temp0=self.cfg.rest2_temp0,
                            temp=self.cfg.rest2_temp)
        nrep = self.cfg.nrep
        work = self.id / "prodrun"
        work.mkdir(exist_ok=True)
        reps = [str(work / f"rep{i}") for i in range(nrep)]
        jobs = []
        for i in range(nrep):
            mrundir = work / f"rep{i}"
            mrundir.mkdir(exist_ok=True)
            with open(runmdps / f"run{i}.mdp", "a") as fh:
                fh.write("energygrps = hot\nuserint1 = 1\n")
            deltae = mrundir / "deltae.xvg"
            if deltae.exists():
                deltae.rename(mrundir / "deltae.xvg.bak")
            jobs.append((f"failed to grompp on final production run replica {i}",
                         mrundir / "prodrun_grompp.log",
                         self._grompp_args(
                             mdp_path=str(runmdps / f"run{i}.mdp"),
                             conf=self.idpath(f"npt/rep{i}/npt.gro"),
                             topol=self.idpath(f"gentops/fep_tip3p_{i}.top"),
                             out_tpr=str(mrundir / "prodrun.tpr"),
                             out_mdp=str(mrundir / "run.mdout"),
                             cpt=self.idpath(f"npt/rep{i}/npt.cpt"),
                             maxwarn=self.cfg.basewarn + 1,
                             ndx=self.idpath("for_rest.ndx"),
                             restr=self._ref_args() or None)))
        self._run_parallel(jobs)
        self.runner.mpirun_mdrun(
            nrep, self._mdrun_multidir_args("prodrun", reps, []),
            least_unit=nrep, nstlist_cmd=self.cfg.nstlist_cmd)
        for i in range(nrep):
            self._checkpoint([f"prodrun/rep{i}/prodrun.cpt"], "checkpoint_init")
            shutil.copyfile(work / f"rep{i}" / "prodrun.tpr",
                            work / f"rep{i}" / "prodrun_ph0.tpr")

    def _next_prodrun_phase(self) -> int:
        rep0 = self.id / "prodrun" / "rep0"
        existing = [int(p.name[len("prodrun_ph"):-len(".tpr")])
                    for p in rep0.glob("prodrun_ph*.tpr")]
        if not existing:
            raise PipelineError(
                "no prodrun_ph*.tpr found: run the prodrun-init stage first")
        return max(existing) + 1

    def stage_prodrun(self) -> None:
        # one production extension chunk + BAR analysis (re-run to extend)
        phase = self._next_prodrun_phase()
        temp = mdp.production_temperature((self.mdp_dir / "run.mdp").read_text())
        nrep = self.cfg.nrep
        work = self.id / "prodrun"
        reps = [str(work / f"rep{i}") for i in range(nrep)]
        jobs = []
        for i in range(nrep):
            mrundir = work / f"rep{i}"
            jobs.append((f"failed to extend on production chunk {phase} "
                         f"replica {i}", mrundir / f"extend{phase}.log",
                         [self.runner.ctx.gmx, "convert-tpr",
                          "-s", str(mrundir / f"prodrun_ph{phase - 1}.tpr"),
                          "-o", str(mrundir / f"prodrun_ph{phase}.tpr"),
                          "-extend", str(self.cfg.simlength)]))
        self._run_parallel(jobs)
        # -bonded cpu is needed because of the current patch's restriction
        extra = ["-s", f"prodrun_ph{phase}", "-cpi", "prodrun", "-cpt", "60",
                 "-hrex", "-othersim", "deltae",
                 "-othersiminterval", str(self.cfg.sampling_interval),
                 "-replex", str(self.cfg.replica_interval),
                 "-bonded", "cpu"]
        self.runner.mpirun_mdrun(
            nrep, self._mdrun_multidir_args("prodrun", reps, extra),
            least_unit=nrep, nstlist_cmd=self.cfg.nstlist_cmd)

        for i in range(nrep):
            self._checkpoint([f"prodrun/rep{i}/prodrun.cpt"],
                             f"checkpoint_ph{phase}")
        self.do_bar(temp)

    def do_bar(self, temp: float) -> None:
        """BAR analysis of the finished chunk (non-fatal, as in v1)."""
        bar_dir = self.id / "bar"
        bar_dir.mkdir(exist_ok=True)
        log_path = self.id / "bar1.log"
        try:
            with open(log_path, "w") as log_fh:
                with redirect_stdout(Tee(sys.stdout, log_fh)):
                    bar.run_analysis(self.idpath("prodrun/rep%sim/deltae.xvg"),
                                     self.cfg.nrep, temp=temp,
                                     save_dir=str(bar_dir))
        except Exception:  # noqa: BLE001 — v1: `|| echo ...` (non-fatal)
            print("BAR failed due to bad convergence, "
                  "please continue the run to get it fixed")

    def stage_trajectory(self) -> None:
        # centered state A / state B trajectories for inspection
        nrep = self.cfg.nrep
        for state, repno in (("A", 0), ("B", nrep - 1)):
            ndxfile = self.id / "prodrun" / f"fepbase_{state}.ndx"
            if not ndxfile.exists():
                write_centering_ndx(self.idpath(f"fepbase_{state}.pdb"),
                                    str(ndxfile))
            sourcefile = self.id / "prodrun" / f"rep{repno}" / "prodrun.trr"
            destfile = self.id / "prodrun" / f"state{state}.xtc"
            if not sourcefile.exists():
                break
            if destfile.exists() and \
                    destfile.stat().st_mtime > sourcefile.stat().st_mtime:
                continue
            self.runner.gmx("trjconv",
                            "-s", self.idpath(f"prodrun/rep{repno}/prodrun.tpr"),
                            "-f", str(sourcefile), "-o", str(destfile),
                            "-pbc", "atom", "-ur", "compact", "-center",
                            "-n", str(ndxfile),
                            stdin="centering\noutput\n")

    # ---- dispatch -----------------------------------------------------------
    def run_stage(self, key: str) -> None:
        stage_index(key)  # validate the name
        methods = {
            "minimize": self.stage_minimize,
            "nvt": self.stage_nvt,
            "npt": self.stage_npt,
            "rest2-setup": self.stage_rest2_setup,
            "tune": self.stage_tune,
            "npt-replex": self.stage_npt_replex,
            "prodrun-init": self.stage_prodrun_init,
            "prodrun": self.stage_prodrun,
            "trajectory": self.stage_trajectory,
        }
        methods[key]()
