"""The pipeline stages: a port of v1 ``abfe/pipeline.zsh``.

Mapping of the v1 numbered stages to the named stages (identical
content, reorganized):

    v1 run,1  -> stage_setup
    v1 run,2  -> stage_equilibrate
    v1 run,3  -> stage_restraints
    v1 run,4  -> stage_prep_charging_lig
    v1 run,5  -> stage_charging_lig
    v1 run,6  -> stage_annihilation_lig
    v1 run,7  -> stage_lrc_lig
    v1 run,8  -> stage_charging_complex
    v1 run,9  -> stage_annihilation_complex
    v1 run,10 -> stage_restraint_decouple
    v1 run,11 -> stage_lrc_complex
    v1 run,12 -> stage_analysis

All paths passed to GROMACS are relative to the run directory (v1 used
``$ID/...`` with the working directory at the run root, and so does
this port).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import charge_correction, lrc, restraints
from .analysis import build_report
from .config import Config
from .errors import PipelineError, RlistExceededError
from .mdp import (Mdp, apply_rlist, assemble_product_mdp, compute_rlist,
                  set_nsteps)
from .ligand import (extract_ligand, ligand_diameter, make_ndx, read_safe_diameter,
                     resurrect_flexible, write_diameter_txt)
from .lambdas import schedule, update_lambda
from .phases import (ANNIHILATION_LRC_TERMS, LRC_SOURCES, PHASES, STAGES,
                     Stage)
from .resources import detect_gpus, ranks_for_replicas
from .runner import RunContext, Runner


# GROMACS prints the offending pair with its distance somewhere in the
# error context ("... 2.500 nm ... beyond rlist ..."); the exact wording
# differs between versions, so just look for the largest nm value in the
# tail of the log and fall back to padding when nothing parses.
_NM_VALUE_PAT = re.compile(r"\b([0-9]{1,2}\.[0-9]{1,4})\s*nm\b")


def _next_rlist(log_path, current: float) -> float:
    """Larger rlist for the retry: the largest distance the mdrun log
    reports in nm (when any), otherwise the current value padded by 20%."""
    if log_path is not None:
        try:
            tail = "\n".join(Path(log_path).read_text(errors="replace")
                             .splitlines()[-40:])
            values = [float(m.group(1)) for m in _NM_VALUE_PAT.finditer(tail)]
            reported = max(values) if values else 0.0
            if current < reported <= 10.0:
                return reported * 1.1  # small margin over the observed maximum
        except OSError:
            pass
    return current * 1.2


class Pipeline:
    def __init__(self, config: Config, ctx: RunContext, runner: Runner):
        self.cfg = config
        self.ctx = ctx
        self.runner = runner
        self.rundir = ctx.rundir
        self.id = Path(ctx.run_id)
        self.template_dir = ctx.abfe_v2_root / "template"
        self._stage_procs = 1

    # ---- helpers -----------------------------------------------------------
    def idpath(self, *parts) -> str:
        return str(self.id.joinpath(*parts))

    def resolve_structure(self, base: str) -> str:
        """Resolve conf_ionized-style names to their .pdb/.gro file."""
        p = Path(base)
        if p.suffix in (".pdb", ".gro"):
            candidate = self.rundir / p
            if candidate.exists():
                return str(p)
            raise PipelineError(f"structure file not found: {candidate}")
        for ext in (".pdb", ".gro"):
            candidate = self.rundir / self.id / f"{base}{ext}"
            if candidate.exists():
                return self.idpath(f"{base}{ext}")
        raise PipelineError(
            f"structure {base}.pdb/.gro not found under {self.id}/")

    def _stage_resource(self, stage: Stage) -> tuple[int, int]:
        multi = self.cfg.get_int(stage.multi_key) if stage.multi_key else 1
        ppm = self.cfg.get_int(stage.ppm_key) if stage.ppm_key else 1
        return multi, ppm

    def _auto_np(self, replicas: int) -> int:
        """Initial mdrun rank count, GPU-aware when AUTO_RESOURCE is on.

        The queueing system decides the GPU count of the allocation, so
        the layout is chosen at runtime: every replica gets at least one
        rank, GPUs are shared when scarce and added as extra ranks per
        replica when abundant.  Falls back to the static MULTI x PPM
        value when nothing can be detected.
        """
        base = self._stage_procs
        if not self.cfg.auto_resource:
            return base
        detected = detect_gpus()
        if detected is None or detected.gpus <= 0:
            return base
        per_replica, total = ranks_for_replicas(detected.gpus, replicas)
        print(f"Resource auto-tuning: {detected.gpus} GPU(s) detected via "
              f"{detected.source} -> {per_replica} rank(s)/replica, NP={total} "
              f"(static value was {base})")
        return total

    def _expected_warn(self, conf: str) -> int:
        # v1 do_prep_runs: expect a warning from -DPOSRES on ligand topologies.
        # This is hacky but other solutions are equally dirty... (v1 comment)
        return 1 if Path(conf).name.startswith("ligand-") else 0

    # ---- shared run helpers (v1 do_run / do_prep_runs / do_solvate) --------
    def do_run(self, *, topol, conf, outprefix, phase, restr=None, ndx=None,
               cont=False, maxwarn=0) -> None:
        grompp_kwargs = dict(
            mdp=f"mdp/{phase}.mdp", topol=topol, conf=conf,
            out_tpr=self.idpath(f"{outprefix}.{phase}.tpr"),
            out_mdp=self.idpath(f"{outprefix}.{phase}.out.mdp"),
            maxwarn=maxwarn)
        if cont:
            cpt = Path(conf).with_suffix(".cpt")
            grompp_kwargs["cpt"] = self.idpath(cpt.name) if not cpt.is_absolute() else str(cpt)
        if restr:
            grompp_kwargs["restr"] = restr
        if ndx:
            grompp_kwargs["ndx"] = ndx
        self.runner.grompp(**grompp_kwargs)
        mdrun_args = ["-deffnm", self.idpath(f"{outprefix}.{phase}"),
                      "-c", self.idpath(f"{outprefix}.{phase}.pdb")]
        stdout_path = self.idpath(f"{outprefix}.{phase}.stdout")
        stderr_path = self.idpath(f"{outprefix}.{phase}.stderr")
        if phase in ("steep", "cg", "nvt", "npt") and self.cfg.prep_nonmpi:
            # Pre-equilibration: a single non-MPI rank with OMP threads is
            # plenty, and the job only needs to request one rank for it.
            self.runner.mdrun_threaded(mdrun_args, stdout_path=stdout_path,
                                       stderr_path=stderr_path)
        else:
            self.runner.mpirun_mdrun(self._auto_np(1), least_unit=1,
                                     args=[self.ctx.gmx_mpi, "mdrun", *mdrun_args],
                                     stdout_path=stdout_path, stderr_path=stderr_path)
        self.runner.check_rlimit(self.idpath(f"{outprefix}.{phase}.stderr"))

    def prep_runs(self, *, topol, conf, outprefix, restr, ndx) -> None:
        warn = self._expected_warn(conf)
        self.do_run(topol=topol, conf=conf, outprefix=outprefix, phase="steep",
                    restr=restr, ndx=ndx, cont=False, maxwarn=warn + 1)  # 1 for switching
        self.do_run(topol=topol, conf=self.idpath(f"{outprefix}.steep.pdb"),
                    outprefix=outprefix, phase="cg", restr=restr, ndx=ndx,
                    cont=False, maxwarn=warn + 1)  # 1 for switching
        self.do_run(topol=topol, conf=self.idpath(f"{outprefix}.cg.pdb"),
                    outprefix=outprefix, phase="nvt", restr=restr, ndx=ndx,
                    cont=False, maxwarn=warn)
        self.do_run(topol=topol, conf=self.idpath(f"{outprefix}.nvt.pdb"),
                    outprefix=outprefix, phase="npt", restr=restr, ndx=ndx,
                    cont=True, maxwarn=warn + 1)  # 1 for gen-vel

    def solvate(self, *, solute, topol, trunk) -> None:
        topol_sol = str(topol)[:-len(".top")] + "-sol.top"
        solute_sol = str(solute).rsplit(".", 1)[0] + "-sol.pdb"

        import shutil
        shutil.copy(self.rundir / topol, self.rundir / topol_sol)
        self.runner.gmx("solvate", "-cp", str(solute), "-p", topol_sol,
                        "-cs", self.cfg.water_structure, "-o", solute_sol)
        Path(self.rundir / f"{trunk}-dummy.mdp").touch()
        self.runner.grompp(mdp=f"{trunk}-dummy.mdp", topol=topol_sol,
                           conf=solute_sol, out_tpr=f"{trunk}-sol.tpr",
                           out_mdp=f"{trunk}-dummy-out.mdp", maxwarn=0)
        topol_ion = f"{trunk}-ion.top"
        shutil.copy(self.rundir / topol_sol, self.rundir / topol_ion)
        self.runner.gmx("genion", "-s", f"{trunk}-sol.tpr", "-o", f"{trunk}-ion.pdb",
                        "-p", topol_ion, "-pname", self.cfg.ion_positive,
                        "-nname", self.cfg.ion_negative,
                        "-conc", f"{self.cfg.ionic_strength:g}", "-neutral",
                        stdin=f"{self.cfg.solvent_name}\n")

    # ---- product runs (v1 do_product_runs) ---------------------------------
    def product_runs(self, phase, *, topol, prev) -> None:
        cfg = self.cfg
        n = cfg.get_int(phase.nstates_key)
        ndx = self.idpath("complex_with_pull.ndx") if phase.system == "complex" else None
        pull_text = ""
        if phase.pull_mdp:
            pull_text = (self.rundir / self.id / phase.pull_mdp).read_text()
        # In both annihilation phases we optimize the lambda schedule on
        # preruns, because solvent and protein are totally different
        # environments; annihilation-complex starts from the ligand result.
        nprerun = cfg.annih_lambda_opt if phase.key in ("annihilation-lig",
                                                        "annihilation-complex") else 0
        run_text = (self.rundir / "mdp/run.mdp").read_text()
        safe_rlist = (read_safe_diameter(self.rundir / self.id / "diameter.txt")
                      if cfg.ligand_diameter == 0 else None)
        traj_keep_index = None
        if phase.traj_keep is not None:
            traj_keep_index = (n - 1 if phase.traj_keep == "last"
                               else int(phase.traj_keep))

        rlist_pad = 1.0  # grows when mdrun reports pair-list cutoff failures

        for iprerun in range(nprerun + 1):
            is_final = iprerun == nprerun
            infix = "" if is_final else f".pre{iprerun}"

            update_log = update_nth = None
            if phase.key == "annihilation-lig" and iprerun > 0:
                update_log = self.idpath(f"annihilation-lig.0/annihilation-lig.pre{iprerun - 1}.log")
                update_nth = iprerun
            elif phase.key == "annihilation-complex":
                if iprerun == 0:
                    update_log = self.idpath(
                        f"annihilation-lig.0/annihilation-lig.pre{cfg.annih_lambda_opt - 1}.log")
                    update_nth = cfg.annih_lambda_opt
                elif iprerun > 0:
                    update_log = self.idpath(
                        f"annihilation-complex.0/annihilation-complex.pre{iprerun - 1}.log")
                    update_nth = iprerun
            if update_log is not None:
                lambdas = update_lambda(n, self.rundir / update_log, update_nth)
            else:
                lambdas = schedule(phase.lambda_kind, n)

            template_text = (self.template_dir / phase.template).read_text()
            base_addenda = template_text + pull_text
            base_addenda = base_addenda.replace(
                "{lambdas_formatted}", " ".join(f"{v:.4f}" for v in lambdas))
            base_addenda = base_addenda.replace("{group_mol}", cfg.lig_gmx)

            def _build_mdps(pad: float) -> None:
                """(Re)write the per-replica mdps at the given rlist pad."""
                mdp_dir = self.rundir / "mdp"
                mdp_dir.mkdir(parents=True, exist_ok=True)
                for i in range(n):
                    addenda = base_addenda.replace("{lambda_state}", str(i))
                    keep = None
                    if traj_keep_index is not None:
                        keep = (i == traj_keep_index)
                    mdp = Mdp.from_text(assemble_product_mdp(run_text, addenda, keep))
                    if phase.needs_rlist:
                        base = compute_rlist(mdp, cfg.ligand_diameter, safe_rlist) * pad
                        apply_rlist(mdp, base, cfg.nstlist)
                    if not is_final:
                        set_nsteps(mdp, int(cfg.annih_lambda_opt_length / mdp.get_float("dt")))
                    mdp.write(mdp_dir / f"{phase.key}-{i}{infix}.mdp")
                    prevcrd = (f"{phase.key}.{i}/{phase.key}.pre{iprerun - 1}"
                               if iprerun > 0 else prev)
                    rdir = self.idpath(f"{phase.key}.{i}")
                    (self.rundir / rdir).mkdir(parents=True, exist_ok=True)
                    grompp_kwargs = dict(
                        mdp=f"mdp/{phase.key}-{i}{infix}.mdp", topol=topol,
                        conf=f"{prevcrd}.pdb", cpt=f"{prevcrd}.cpt",
                        out_tpr=f"{rdir}/{phase.key}{infix}.tpr",
                        out_mdp=f"mdp/{phase.key}{infix}.{i}.grompp-out.mdp",
                        maxwarn=phase.maxwarn)
                    if ndx:
                        grompp_kwargs["ndx"] = ndx
                    self.runner.grompp(**grompp_kwargs)

            dirs = [self.idpath(f"{phase.key}.{i}") for i in range(n)]
            while True:
                _build_mdps(rlist_pad)
                try:
                    self.runner.mpirun_mdrun(
                        self._auto_np(n), least_unit=n,
                        args=[self.ctx.gmx_mpi, "mdrun", "-multidir", *dirs,
                              "-deffnm", f"{phase.key}{infix}",
                              "-c", f"{phase.key}{infix}.pdb", "-replex", "500"],
                        stdout_path=self.idpath(f"{phase.key}.stdout"),
                        stderr_path=self.idpath(f"{phase.key}.stderr"),
                        rlist_retry=True)
                    break
                except RlistExceededError as e:
                    current = (compute_rlist(Mdp.read(self.rundir / f"mdp/{phase.key}-0{infix}.mdp"),
                                             cfg.ligand_diameter, safe_rlist) * rlist_pad)
                    new_rlist = _next_rlist(e.log_path, current)
                    if new_rlist <= current * 1.05 or new_rlist > 10.0:
                        raise PipelineError(
                            f"rlist auto-retry gave up for phase {phase.key}: "
                            f"rlist would have to grow from {current:.3f} to "
                            f"{new_rlist:.3f} nm; set LIGAND_DIAMETER in "
                            "para_conf.zsh manually") from None
                    print(f"Pair-list cutoff exceeded (large ligand?): retrying "
                          f"{phase.key}{infix} with rlist {current:.3f} -> "
                          f"{new_rlist:.3f} nm (all {n} replicas re-grompp'ed)")
                    # current = base * rlist_pad, so the new pad follows from
                    # the new target rlist over the same unpadded base.
                    rlist_pad = new_rlist * rlist_pad / current
            if is_final:
                self.runner.check_replica_probs(self.idpath(f"{phase.key}.0/{phase.key}.log"))

    # ---- eval runs (v1 do_eval_run) ----------------------------------------
    def eval_run(self, *, phase_name, template_name, prev, ndx, topol, maxwarn=0) -> None:
        run = Mdp.read(self.rundir / "mdp/run.mdp").without({"dispcorr", "nstxtcout"})
        template_text = (self.template_dir / template_name).read_text().replace(
            "{group_mol}", self.cfg.lig_gmx)
        safe_rlist = (read_safe_diameter(self.rundir / self.id / "diameter.txt")
                      if self.cfg.ligand_diameter == 0 else None)
        mdp = run.merged(Mdp.from_text(template_text))
        apply_rlist(mdp, compute_rlist(mdp, self.cfg.ligand_diameter, safe_rlist),
                    self.cfg.nstlist)
        mdp.write(self.rundir / self.id / f"{phase_name}.mdp")
        grompp_kwargs = dict(
            mdp=self.idpath(f"{phase_name}.mdp"), topol=topol,
            conf=f"{prev}.pdb", cpt=f"{prev}.cpt",
            out_tpr=self.idpath(f"{phase_name}.tpr"),
            out_mdp=self.idpath(f"{phase_name}.out.mdp"), maxwarn=maxwarn)
        if ndx:
            grompp_kwargs["ndx"] = ndx
        self.runner.grompp(**grompp_kwargs)
        mdrun_args = [self.ctx.gmx_mpi, "mdrun", "-deffnm", self.idpath(phase_name),
                      "-rerun", f"{prev}.xtc"]
        stdout_path = self.idpath(f"{phase_name}.stdout")
        stderr_path = self.idpath(f"{phase_name}.stderr")
        while True:
            try:
                self.runner.mpirun_mdrun(self._auto_np(1), least_unit=1, args=mdrun_args,
                                         stdout_path=stdout_path,
                                         stderr_path=stderr_path, rlist_retry=True)
                break
            except RlistExceededError as e:
                current = compute_rlist(mdp, self.cfg.ligand_diameter, safe_rlist)
                new_rlist = _next_rlist(e.log_path, current)
                if new_rlist <= current * 1.05 or new_rlist > 10.0:
                    raise PipelineError(
                        f"rlist auto-retry gave up for {phase_name}: rlist would "
                        f"have to grow from {current:.3f} to {new_rlist:.3f} nm; "
                        "set LIGAND_DIAMETER in para_conf.zsh manually") from None
                print(f"Pair-list cutoff exceeded (large ligand?): retrying "
                      f"{phase_name} with rlist {current:.3f} -> {new_rlist:.3f} nm")
                apply_rlist(mdp, new_rlist, self.cfg.nstlist)
                mdp.write(self.rundir / self.id / f"{phase_name}.mdp")
                self.runner.grompp(**grompp_kwargs)
        self.runner.check_rlimit(self.idpath(f"{phase_name}.stderr"))

    def bar(self, mode: str, nrepl: int) -> None:
        xvg_files = [self.idpath(f"{mode}.{i}/{mode}.xvg") for i in range(nrepl)]
        self.runner.gmx("bar", "-b", f"{self.cfg.run_prod:g}", "-f", *xvg_files,
                        "-o", self.idpath(f"{mode}.bar_diff.xvg"),
                        stdout_path=self.idpath(f"{mode}.bar.log"))

    # ---- stages -------------------------------------------------------------
    def stage_setup(self) -> None:
        self.runner.check_gromacs_version()
        mdp_dir = self.rundir / "mdp"
        mdp_dir.mkdir(parents=True, exist_ok=True)
        (mdp_dir / "dummy.mdp").touch()
        dummy_flex = mdp_dir / "dummy_flex.mdp"
        if not dummy_flex.exists():
            dummy_flex.write_text("define = -DFLEXIBLE\n")

        conf = self.resolve_structure("conf_ionized")
        topol = self.idpath("topol_ionized.top")
        self.runner.grompp(mdp="mdp/dummy.mdp", topol=topol, conf=conf,
                           out_tpr=self.idpath("pp.tpr"), out_mdp=self.idpath("pp.mdp"),
                           pp=self.idpath("pp.top"), maxwarn=0)
        self.runner.grompp(mdp="mdp/dummy_flex.mdp", topol=topol, conf=conf,
                           out_tpr=self.idpath("pp_flex.tpr"),
                           out_mdp=self.idpath("pp_flex.mdp"),
                           pp=self.idpath("pp_flex.top"), maxwarn=0)
        make_ndx(structure=conf, topology=self.idpath("pp.top"),
                 output=self.idpath("complex.ndx"), ligand=self.cfg.lig_gmx,
                 receptor=self.cfg.receptor_sel)
        self.prep_runs(topol=topol, conf=conf, outprefix="prep", restr=conf, ndx=None)

    def stage_equilibrate(self) -> None:
        self.do_run(topol=self.idpath("topol_ionized.top"),
                    conf=self.idpath("prep.npt.pdb"), outprefix="prerun",
                    phase="run", cont=True, maxwarn=0)

    def stage_restraints(self) -> None:
        cfg = self.cfg
        pp_flex_tpr = self.idpath("pp_flex.tpr")
        complex_ndx = self.idpath("complex.ndx")

        # Compute the ligand RMSD for thresholding.  This seemingly strange
        # procedure supports systems that do not work with gmx trjconv -pbc
        # cluster: a time-evolved trajectory without replica exchange allows
        # -pbc nojump (no >0.5 box jump during the prep runs), after which
        # -pbc mol / -ur compact prevents split-ligand/split-receptor artifacts.
        self.runner.gmx("trjconv", "-s", pp_flex_tpr, "-f", self.idpath("prerun.run.xtc"),
                        "-o", self.idpath("prerun.run.nojump.xtc"),
                        "-b", f"{cfg.run_prod:g}", "-center", "-n", complex_ndx,
                        "-pbc", "nojump", stdin="Ligand+Receptor\nSystem\n")
        self.runner.gmx("trjconv", "-s", pp_flex_tpr,
                        "-f", self.idpath("prerun.run.nojump.xtc"),
                        "-o", self.idpath("prerun.run.recpbc.xtc"),
                        "-b", f"{cfg.run_prod:g}", "-pbc", "mol", "-n", complex_ndx,
                        "-ur", "compact", stdin="System\n")
        # Note: prerun.run.xtc still holds the equilibration part, so the
        # simulation length comes from it.
        last = self.runner.gmx_check_time(self.idpath("prerun.run.xtc"))
        self.runner.gmx("trjconv", "-s", pp_flex_tpr,
                        "-f", self.idpath("prerun.run.recpbc.xtc"),
                        "-o", self.idpath("prerun.run.final.pdb"), "-n", complex_ndx,
                        "-dump", f"{last:g}", stdin="System\n")

        # Remake the tpr: the atoms may not have defined masses otherwise.
        self.runner.grompp(mdp="mdp/dummy.mdp", topol=self.idpath("topol_ionized.top"),
                           conf=self.idpath("prerun.run.final.pdb"),
                           out_tpr=self.idpath("prerun.run.final.tpr"),
                           out_mdp=self.idpath("prerun_dummy.mdp"), maxwarn=0)
        self.runner.gmx("rms", "-s", self.idpath("prerun.run.final.tpr"),
                        "-f", self.idpath("prerun.run.recpbc.xtc"),
                        "-o", self.idpath("prerun.rms.fromfinal.xvg"),
                        "-n", complex_ndx, stdin="Receptor\nLigand\n")
        try:
            restraints.check_rms_average(self.idpath("prerun.rms.fromfinal.xvg"),
                                         cfg.eq_rmsd_cutoff)
        except PipelineError as e:
            raise PipelineError(
                f"RMS of ligands too large, aborting the calculation ({e})") from None

        conf = self.resolve_structure("conf_ionized")
        restraints.find_restraints(
            topology=conf, trajectory=self.idpath("prerun.run.recpbc.xtc"),
            index=complex_ndx, prot_sel=cfg.receptor_sel, lig_sel="Ligand",
            output=self.rundir / self.id / "restrinfo")
        # Restraint for annihilation and charging...
        restraints.generate_pull_restraint(
            restrinfo=self.rundir / self.id / "restrinfo",
            mdp=self.rundir / self.id / "restr_pull.mdp",
            ndx=self.rundir / self.id / "restr_pull.ndx")
        # ...and its decoupling variant used in the restraint phase.
        restraints.generate_pull_restraint(
            restrinfo=self.rundir / self.id / "restrinfo", decouple_B=True,
            mdp=self.rundir / self.id / "restr_pull_decouple.mdp",
            ndx=self.rundir / self.id / "restr_pull_decouple.ndx")
        if (Path(self.idpath("restr_pull.ndx")).read_bytes()
                != Path(self.idpath("restr_pull_decouple.ndx")).read_bytes()):
            raise PipelineError("Two pull indices are not identical")

        # Prepare the preprocessed topology: the ligand total charge is
        # computed from it.
        self.runner.grompp(mdp="mdp/run.mdp", topol=self.idpath("topol_ionized.top"),
                           conf=self.idpath("prerun.run.pdb"),
                           cpt=self.idpath("prerun.run.cpt"),
                           out_tpr=self.idpath("pp.tpr"), out_mdp=self.idpath("pp.mdp"),
                           pp=self.idpath("pp_run.top"), maxwarn=0)
        extract_ligand(topology=self.idpath("pp_run.top"), mol=cfg.lig_gmx,
                       structure=self.idpath("prerun.run.final.pdb"),
                       index=complex_ndx, ligand_group="Ligand",
                       output_ligand_structure=self.idpath("ligand.pdb"),
                       output_ligand_topology=self.idpath("ligand.top"),
                       total_charge=self.idpath("totalcharge.txt"))

        # Run the ligand-only system to determine the cutoff distance
        # (maxwarn 1: the ligand carries PME charges in an unneutralized box).
        self.runner.gmx("editconf", "-f", self.idpath("ligand.pdb"), "-d", "3.0",
                        "-o", self.idpath("ligand-bigbox.pdb"))
        self.runner.grompp(mdp="mdp/ligsample.mdp", conf=self.idpath("ligand-bigbox.pdb"),
                           topol=self.idpath("ligand.top"),
                           out_tpr=self.idpath("ligand_only.tpr"), maxwarn=1)
        # Directly without the NP search: v1 did not want NSTLIST/SAVE_* here.
        if self.cfg.prep_nonmpi:
            self.runner.mdrun_threaded(["-deffnm", self.idpath("ligand_only")])
        else:
            self.runner.mpirun_raw(1, [self.ctx.gmx_mpi, "mdrun", "-deffnm",
                                       self.idpath("ligand_only")])
        avg, mx, safe = ligand_diameter(structure=self.idpath("ligand.pdb"),
                                        trajectory=self.idpath("ligand_only.xtc"))
        write_diameter_txt(self.rundir / self.id / "diameter.txt", avg, mx, safe)

        # Solvate the ligand-only system.
        real_water_thickness = cfg.water_thickness
        if real_water_thickness < safe:
            real_water_thickness = safe + 0.1
        self.runner.gmx("editconf", "-f", self.idpath("ligand.pdb"),
                        "-d", f"{real_water_thickness:g}", "-bt", "dodecahedron",
                        "-o", self.idpath("ligand-box.pdb"))
        self.solvate(solute=self.idpath("ligand-box.pdb"), topol=self.idpath("ligand.top"),
                     trunk=self.idpath("ligand"))
        make_ndx(structure=self.idpath("ligand-ion.pdb"),
                 topology=self.idpath("ligand-ion.top"),
                 output=self.idpath("ligand.ndx"), ligand=cfg.lig_gmx, receptor=None)

        # Index for the pulled complex runs; the ligand is never pulled.
        complex_ndx_text = Path(self.idpath("complex.ndx")).read_text()
        restr_ndx_text = Path(self.idpath("restr_pull.ndx")).read_text()
        Path(self.idpath("complex_with_pull.ndx")).write_text(
            complex_ndx_text + restr_ndx_text)

        resurrect_flexible(flexible=self.idpath("pp_flex.top"),
                           topology=self.idpath("ligand-ion.top"),
                           output=self.idpath("ligand-ion-flex.top"),
                           solvent=cfg.solvent_name)

    def stage_prep_charging_lig(self) -> None:
        self.prep_runs(topol=self.idpath("ligand-ion-flex.top"),
                       conf=self.idpath("ligand-ion.pdb"), outprefix="charging-lig",
                       restr=self.idpath("ligand-ion.pdb"), ndx=self.idpath("ligand.ndx"))

    def stage_charging_lig(self) -> None:
        self.product_runs(PHASES["charging-lig"], topol=self.idpath("ligand-ion.top"),
                          prev="charging-lig.npt")

    def stage_annihilation_lig(self) -> None:
        self.product_runs(PHASES["annihilation-lig"],
                          topol=self.idpath("ligand-ion.top"),
                          prev=f"charging-lig.{self.cfg.ncharge - 1}/charging-lig")

    def stage_lrc_lig(self) -> None:
        self.eval_run(phase_name="lr-lig",
                      template_name=PHASES["charging-lig"].lr_template,
                      prev=self.idpath(f"charging-lig.0/charging-lig"),
                      ndx=self.idpath("ligand.ndx"),
                      topol=self.idpath("ligand-ion.top"))
        self.eval_run(phase_name="lr-annihilation-lig",
                      template_name=PHASES["annihilation-lig"].lr_template,
                      prev=self.idpath(f"annihilation-lig.{self.cfg.nannih - 1}/annihilation-lig"),
                      ndx=self.idpath("ligand.ndx"),
                      topol=self.idpath("ligand-ion.top"))

    def stage_charging_complex(self) -> None:
        self.product_runs(PHASES["charging-complex"],
                          topol=self.idpath("topol_ionized.top"), prev="prerun.run")

    def stage_annihilation_complex(self) -> None:
        self.product_runs(PHASES["annihilation-complex"],
                          topol=self.idpath("topol_ionized.top"),
                          prev=f"charging-complex.{self.cfg.ncharge - 1}/charging-complex")

    def stage_restraint_decouple(self) -> None:
        # Product run for the restraint decoupling (100% -> 0% restraint) FEP.
        self.product_runs(PHASES["restraint"], topol=self.idpath("topol_ionized.top"),
                          prev="prerun.run")

    def stage_lrc_complex(self) -> None:
        self.eval_run(phase_name="lr-complex",
                      template_name=PHASES["restraint"].lr_template,
                      prev=self.idpath(f"restraint.{self.cfg.nrestr - 1}/restraint"),
                      ndx=self.idpath("complex.ndx"),
                      topol=self.idpath("topol_ionized.top"))
        self.eval_run(phase_name="lr-annihilation-complex",
                      template_name=PHASES["annihilation-complex"].lr_template,
                      prev=self.idpath(
                          f"annihilation-complex.{self.cfg.nannih - 1}/annihilation-complex"),
                      ndx=self.idpath("complex.ndx"),
                      topol=self.idpath("topol_ionized.top"))

    # ---- analysis ------------------------------------------------------------
    def bar_all(self, temp_unused=None) -> None:
        self.bar("charging-lig", self.cfg.ncharge)
        self.bar("charging-complex", self.cfg.ncharge)
        self.bar("annihilation-lig", self.cfg.nannih)
        self.bar("annihilation-complex", self.cfg.nannih)
        self.bar("restraint", self.cfg.nrestr)

    def lrc_all(self, temp: float) -> set[str]:
        """Run the PME re-evaluations; returns the set of skipped terms."""
        skipped: set[str] = set()
        nstates = {"charging-lig": self.cfg.ncharge,
                   "annihilation-lig": self.cfg.nannih,
                   "restraint": self.cfg.nrestr,
                   "annihilation-complex": self.cfg.nannih}
        for lr, (src, which) in LRC_SOURCES.items():
            if (lr in ANNIHILATION_LRC_TERMS and self.cfg.skip_annihilation_lrc):
                print(f"Skipping the LRC re-evaluation of '{lr}' "
                      "(SKIP_ANNIHILATION_LRC=yes): the decoupled endpoint has no "
                      "ligand-environment dispersion to correct; its cycle "
                      "contribution is recorded as zero.")
                skipped.add(lr)
                continue
            replica = 0 if which == "first" else nstates[src] - 1
            lrc.compute_long_range_correction(
                long=self.idpath(f"{lr}.edr"),
                short=self.idpath(f"{src}.{replica}/{src}.edr"),
                temp=temp, time_begin=self.cfg.run_prod,
                output=self.rundir / self.id / f"{lr}.lrc.txt",
                terms=self.cfg.lrc_energy_terms)
        return skipped

    def _charge_correction_side(self, side: str, temp: float) -> list[dict]:
        cfg = self.cfg
        if side == "complex":
            reftpr = self.idpath("prerun.run.final.tpr")
            traj = self.idpath("prerun.run.recpbc.xtc")
            ndx = self.idpath("complex.ndx")
            top = self.idpath("pp_run.top")
            receptor_group = "Receptor"
        else:
            reftpr = self.idpath("charging-lig.0/charging-lig.tpr")
            traj = self.idpath("charging-lig.0/charging-lig.xtc")
            ndx = self.idpath("ligand.ndx")
            top = self.idpath("ligand-ion.top")
            receptor_group = None

        simlen = self.runner.gmx_check_time(traj)
        base = self.rundir / self.id / "charge-correction" / side
        records = []
        for i in range(1, cfg.charge_correction_nsamp + 1):
            if side == "complex":
                # The first RUN_PROD ps has already been removed from this
                # trajectory by the restraints stage.
                tt = cfg.run_prod + simlen * i / cfg.charge_correction_nsamp
                dump_args = []
                stdin = "System\n"
            else:
                tt = cfg.run_prod + (simlen - cfg.run_prod) * i / cfg.charge_correction_nsamp
                dump_args = ["-pbc", "mol", "-center"]
                stdin = "Ligand\nSystem\n"
            sample_dir = base / f"sample{i}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            pdb = sample_dir / "frame.pdb"
            self.runner.gmx("trjconv", "-s", reftpr, "-f", traj, "-o", str(pdb),
                            "-n", ndx, "-dump", f"{tt:.6f}", *dump_args, stdin=stdin)
            sample = charge_correction.run_sample(
                workdir=sample_dir, top=self.rundir / top,
                ndx=self.rundir / ndx, pdb=pdb,
                ligand_group="Ligand", receptor_group=receptor_group,
                temp=temp, apbs_exe=cfg.apbs)
            records.append(sample.to_dict())
        return records

    def charge_correction_gate(self, temp: float) -> None:
        """Run the charge correction unless the ligand is (effectively) neutral."""
        totalcharge = float(Path(self.idpath("totalcharge.txt")).read_text().strip())
        cc_dir = self.rundir / self.id / "charge-correction"
        cc_dir.mkdir(parents=True, exist_ok=True)
        if abs(totalcharge) < 1e-4 or self.cfg.charge_correction_nsamp == 0:
            print("No charge correction needed")
            (cc_dir / "complex.json").write_text("[]")
            (cc_dir / "ligand.json").write_text("[]")
            return
        for side in ("complex", "ligand"):
            records = self._charge_correction_side(side, temp)
            (cc_dir / f"{side}.json").write_text(json.dumps(records, indent=1))

    def stage_analysis(self) -> None:
        temp = Mdp.read(self.rundir / "mdp/run.mdp").get_float("ref_t")
        self.bar_all()
        skipped_lrc = self.lrc_all(temp)
        self.charge_correction_gate(temp)
        report = build_report(basedir=self.rundir / self.id,
                              restrinfo=self.rundir / self.id / "restrinfo",
                              temp=temp,
                              skipped_lrc=skipped_lrc,
                              sanity_limit_kcal=self.cfg.lrc_sanity_limit)
        (self.rundir / self.id / "result.txt").write_text(report)
        print(report, end="")

    # ---- dispatch -------------------------------------------------------------
    def run_stage(self, key: str) -> None:
        stage = None
        for s in STAGES:
            if s.key == key:
                stage = s
                break
        if stage is None:
            raise PipelineError(f"unknown stage: {key}")
        multi, ppm = self._stage_resource(stage)
        if stage.prep_like and self.cfg.prep_nonmpi:
            # Matches the PPM=1 the query protocol emits for prep-like
            # stages under PREP_NONMPI: their mdruns are non-MPI.
            multi, ppm = 1, 1
        self._stage_procs = multi * ppm
        if self._stage_procs == 0:
            raise PipelineError(
                f"stage {key} resolved to 0 MPI processes "
                f"(MULTI={multi}, PPM={ppm}); check para_conf.zsh")
        _STAGE_FUNCS[key](self)


_STAGE_FUNCS = {
    "setup": Pipeline.stage_setup,
    "equilibrate": Pipeline.stage_equilibrate,
    "restraints": Pipeline.stage_restraints,
    "prep-charging-lig": Pipeline.stage_prep_charging_lig,
    "charging-lig": Pipeline.stage_charging_lig,
    "annihilation-lig": Pipeline.stage_annihilation_lig,
    "lrc-lig": Pipeline.stage_lrc_lig,
    "charging-complex": Pipeline.stage_charging_complex,
    "annihilation-complex": Pipeline.stage_annihilation_complex,
    "restraint-decouple": Pipeline.stage_restraint_decouple,
    "lrc-complex": Pipeline.stage_lrc_complex,
    "analysis": Pipeline.stage_analysis,
}
