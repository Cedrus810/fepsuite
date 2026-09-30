"""GROMACS execution layer.

The pipeline never invokes GROMACS directly: it goes through the
FEP-suite job-system wrappers (``job_singlerun`` / ``job_mpirun`` from
``submit_scripts/<JOBSYSTEM>.zsh``) via a tiny zsh bridge, so all the
system-specific mpirun handling keeps working unchanged.  When
FEPSUITE_ROOT/JOBSYSTEM are not set (development, tests) the runner
falls back to plain ``mpirun -np``.

The NP-search loop (retrying mdrun with fewer ranks when domain
decomposition fails) is a faithful port of v1's
``mdrun_find_possible_np``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from .errors import PipelineError, RlistExceededError


@dataclass
class RunContext:
    rundir: Path            # contains para_conf.zsh, mdp/, and the run-ID subdirectory
    run_id: str
    abfe_v2_root: Path
    fepsuite_root: str | None
    jobsystem: str | None
    gmx: str
    gmx_mpi: str
    gmx_nompi: str
    omp_threads: str | None
    python3: str

    @classmethod
    def from_env(cls, rundir: Path | None = None) -> "RunContext":
        run_id = os.environ.get("ID")
        if not run_id:
            raise PipelineError(
                "environment variable ID is not set (must be run via controller.zsh)")
        return cls(
            rundir=Path(rundir or os.environ.get("FEPLAB_RUNDIR") or Path.cwd()).resolve(),
            run_id=run_id,
            abfe_v2_root=Path(os.environ.get("ABFE_V2_ROOT")
                              or Path(__file__).resolve().parent.parent),
            fepsuite_root=os.environ.get("FEPSUITE_ROOT"),
            jobsystem=os.environ.get("JOBSYSTEM"),
            gmx=os.environ.get("GMX", "gmx"),
            gmx_mpi=os.environ.get("GMX_MPI", "gmx_mpi"),
            gmx_nompi=os.environ.get("GMX_NOMPI", "gmx"),
            omp_threads=os.environ.get("OMP_NUM_THREADS"),
            python3=os.environ.get("PYTHON3", "python3"),
        )


def _tee_stream(pipe, out_file, mirror) -> None:
    """Copy lines from a pipe to a file while mirroring them to a stream."""
    with pipe:
        for line in iter(pipe.readline, ""):
            out_file.write(line)
            out_file.flush()
            mirror.write(line)
            mirror.flush()


class Runner:
    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self._np_saved: int | None = None

    # ---- command plumbing ------------------------------------------------
    @property
    def _bridge_mode(self) -> bool:
        if not (self.ctx.fepsuite_root and self.ctx.jobsystem):
            return False
        script = (Path(self.ctx.fepsuite_root) / "submit_scripts"
                  / f"{self.ctx.jobsystem}.zsh")
        return script.exists()

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("GMX", self.ctx.gmx)
        env.setdefault("GMX_MPI", self.ctx.gmx_mpi)
        if self.ctx.omp_threads:
            env.setdefault("OMP_NUM_THREADS", self.ctx.omp_threads)
        return env

    def _singlerun_cmd(self, args: list[str]) -> list[str]:
        if self._bridge_mode:
            return ["zsh", str(self.ctx.abfe_v2_root / "jobbridge.zsh"),
                    "singlerun", "--", *args]
        return list(args)

    def _mpirun_cmd(self, np: int, args: list[str]) -> list[str]:
        if self._bridge_mode:
            return ["zsh", str(self.ctx.abfe_v2_root / "jobbridge.zsh"),
                    "mpirun", str(np), "--", *args]
        return ["mpirun", "-np", str(np), *args]

    def singlerun(self, args: list[str], *, stdin: str | None = None,
                  stdout_path: str | Path | None = None,
                  stderr_path: str | Path | None = None,
                  capture: bool = False, check: bool = True,
                  cwd: Path | None = None) -> subprocess.CompletedProcess:
        """Run one command; the default cwd is the run directory."""
        cmd = self._singlerun_cmd(args)
        cwd = cwd or self.ctx.rundir
        env = self._env()
        if stdout_path is not None or stderr_path is not None:
            out_fh = open(stdout_path, "w") if stdout_path else None
            err_fh = open(stderr_path, "w") if stderr_path else None
            try:
                proc = subprocess.Popen(
                    cmd, cwd=cwd, env=env,
                    stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                    stdout=subprocess.PIPE if out_fh else (subprocess.DEVNULL if err_fh else None),
                    stderr=subprocess.PIPE if err_fh else (subprocess.DEVNULL if out_fh else None),
                    text=True)
                if stdin is not None and proc.stdin:
                    proc.stdin.write(stdin)
                    proc.stdin.close()
                threads = []
                if out_fh is not None and proc.stdout:
                    threads.append(threading.Thread(
                        target=_tee_stream, args=(proc.stdout, out_fh, sys.stdout),
                        daemon=True))
                if err_fh is not None and proc.stderr:
                    threads.append(threading.Thread(
                        target=_tee_stream, args=(proc.stderr, err_fh, sys.stderr),
                        daemon=True))
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                returncode = proc.wait()
                completed = subprocess.CompletedProcess(cmd, returncode)
            finally:
                if out_fh:
                    out_fh.close()
                if err_fh:
                    err_fh.close()
            if check and completed.returncode != 0:
                raise PipelineError(f"command failed with exit code "
                                    f"{completed.returncode}: {' '.join(cmd)}")
            return completed

        if capture or stdin is not None:
            completed = subprocess.run(
                cmd, cwd=cwd, env=env, input=stdin, capture_output=True, text=True)
        else:
            completed = subprocess.run(cmd, cwd=cwd, env=env)
        if check and completed.returncode != 0:
            tail = ""
            if capture and completed.stderr:
                tail = "\n".join(completed.stderr.splitlines()[-30:])
                tail = "\n" + tail
            raise PipelineError(f"command failed with exit code "
                                f"{completed.returncode}: {' '.join(cmd)}{tail}")
        return completed

    # ---- GROMACS conveniences --------------------------------------------
    def gmx(self, *args, **kwargs) -> subprocess.CompletedProcess:
        return self.singlerun([self.ctx.gmx, *args], **kwargs)

    def gmx_mpi(self, *args, **kwargs) -> subprocess.CompletedProcess:
        return self.singlerun([self.ctx.gmx_mpi, *args], **kwargs)

    def grompp(self, *, mdp, topol, conf, out_tpr, out_mdp, cpt=None, restr=None,
               ndx=None, pp=None, maxwarn: int = 0) -> None:
        args = ["grompp", "-f", str(mdp), "-p", str(topol), "-c", str(conf)]
        if cpt:
            args += ["-t", str(cpt)]
        if restr:
            args += ["-r", str(restr)]
        if ndx:
            args += ["-n", str(ndx)]
        if pp:
            args += ["-pp", str(pp)]
        args += ["-o", str(out_tpr), "-po", str(out_mdp), "-maxwarn", str(maxwarn)]
        self.gmx(*args)

    def mpirun_mdrun(self, np_initial: int, args: list[str], *, least_unit: int = 1,
                     stdout_path=None, stderr_path=None,
                     rlist_retry: bool = False) -> None:
        """Run mdrun, retrying with fewer ranks on domain-decomposition failures.

        Port of v1 mdrun_find_possible_np.  ``least_unit`` is the granularity
        with which the rank count may be reduced (number of replicas for
        multidir runs, 1 otherwise).

        When ``rlist_retry`` is set, a pair-list cutoff failure raises
        :class:`RlistExceededError` instead of aborting: the caller can
        then enlarge rlist in the mdp and try again (large-ligand
        support).
        """
        np = self._np_saved if self._np_saved is not None else np_initial
        while True:
            print(f"Trying with NP={np}")
            cmd = self._mpirun_cmd(np, args)
            completed = self.singlerun(cmd, stdout_path=stdout_path,
                                       stderr_path=stderr_path, check=False)
            if completed.returncode != 0:
                log_path = self._mdrun_log_path(args)
                if log_path is None or not log_path.exists():
                    raise PipelineError(
                        "Error: failed during mpirun-mdrun startup. This typically "
                        "happens when there are a problem with mpirun command.")
                tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-20:])
                if not re.search(r"(domain|prime)", tail, re.I):
                    raise PipelineError(
                        "Error: mdrun stopped with errors unrelated to domain size")
                if ("There are perturbed non-bonded pair interactions beyond the "
                        "pair-list cutoff" in tail):
                    if rlist_retry:
                        raise RlistExceededError(
                            "Error: mdrun stopped because ligand atom-atom distance "
                            "exceeded automatically determined rlist value. Try "
                            "specifying LIGAND_DIAMETER in para_conf.zsh manually.",
                            log_path=log_path)
                    raise PipelineError(
                        "Error: mdrun stopped because ligand atom-atom distance "
                        "exceeded automatically determined rlist value. Try "
                        "specifying LIGAND_DIAMETER in para_conf.zsh manually.")
                prev = np
                np = (np - 1) // least_unit * least_unit
                if np == 0 or np == prev:
                    raise PipelineError(
                        "Error: unable to find proper parallel processes")
            else:
                print("Normal termination of mdrun")
                self._np_saved = np
                return

    def mpirun_raw(self, np: int, args: list[str], *, check: bool = True,
                   stdout_path=None, stderr_path=None) -> subprocess.CompletedProcess:
        """Run a command under the job-system mpirun wrapper, no NP search."""
        cmd = self._mpirun_cmd(np, args)
        return self.singlerun(cmd, check=check, stdout_path=stdout_path,
                              stderr_path=stderr_path)

    def mdrun_threaded(self, args: list[str], *, stdout_path=None,
                       stderr_path=None) -> subprocess.CompletedProcess:
        """Run mdrun without MPI (single rank + OMP threads).

        Used for the pre-equilibration chain, where a full MPI layout
        would be wasted.  ``GMX_NOMPI`` selects the binary so MPI-only
        installs can still point at a rank-less build.
        """
        completed = self.singlerun([self.ctx.gmx_nompi, "mdrun", *args],
                                   stdout_path=stdout_path,
                                   stderr_path=stderr_path, check=False)
        if completed.returncode != 0:
            raise PipelineError(
                f"mdrun (non-MPI) failed with exit code {completed.returncode}: "
                f"{self.ctx.gmx_nompi} mdrun {' '.join(args)}")
        return completed

    @staticmethod
    def _mdrun_log_path(args: list[str]) -> Path | None:
        """Locate the log file of a (multi)dir mdrun from its command line."""
        log_base = None
        dir_base = "."
        pending = None
        for arg in args:
            if pending == "deffnm" or pending == "log":
                log_base = arg
                pending = None
            elif pending == "multidir":
                dir_base = arg
                pending = None
            elif arg in ("-deffnm", "-l"):
                pending = "deffnm" if arg == "-deffnm" else "log"
            elif arg == "-multidir":
                pending = "multidir"
        if log_base is None:
            return None
        return Path(dir_base) / f"{log_base}.log"

    # ---- sanity checks ----------------------------------------------------
    def check_gromacs_version(self) -> None:
        completed = self.gmx("--version", capture=True)
        version = None
        for line in completed.stdout.splitlines():
            if "GROMACS version:" in line:
                version = line.split()[-1]
                break
        if version is None:
            raise PipelineError("could not determine the GROMACS version")
        if (version.startswith(("2016", "2017", "2018", "2019", "2020", "2021"))
                or version == "2022"
                or re.match(r"2022\.[1-4]\b", version)):
            raise PipelineError(
                f"GROMACS version {version} is not supported because it cannot "
                "detect error cases with excluded interactions properly")
        if re.match(r"2022\.[5-9]", version):
            print(f"GROMACS version {version} (supported version)")
        else:
            print(f"GROMACS version {version}. We are not sure the program works "
                  "on this version; we will continue anyway.")

    def gmx_check_time(self, traj) -> float:
        """(nframes - 1) * timestep of a trajectory, from ``gmx check``."""
        completed = self.gmx("check", "-f", str(traj), capture=True)
        combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
        time_lines = [l for l in combined.splitlines() if l.startswith("Time")]
        if not time_lines:
            raise PipelineError(f"could not parse simulation length from gmx check of {traj}")
        tokens = time_lines[-1].split()
        try:
            nframes = float(tokens[1])
            timestep = float(tokens[2])
        except (IndexError, ValueError):
            raise PipelineError(
                f"could not parse the 'Time' line of gmx check: {time_lines[-1]!r}") from None
        return (nframes - 1) * timestep

    def check_rlimit(self, stderr_path) -> None:
        try:
            text = Path(stderr_path).read_text(errors="replace")
        except OSError:
            return
        if "This is likely either a 1,4 interaction," in text:
            raise PipelineError(
                "Error: 1,4- interaction distance exceeds the predefined table "
                "distance. Set table-extension= in mdps to larger values. (Must "
                "be the possible diameter of the ligand minus initial ligand radius)")

    def check_replica_probs(self, logfile, threshold: float = 0.03) -> None:
        lines = Path(logfile).read_text(errors="replace").splitlines()
        for i, line in enumerate(lines):
            if "average probabilities:" in line:
                if i + 2 >= len(lines):
                    break
                tokens = [t for t in lines[i + 2].split() if t != "Repl"]
                probs: list[float] = []
                for t in tokens:
                    try:
                        probs.append(float(t))
                    except ValueError:
                        continue
                if any(p < threshold for p in probs):
                    raise PipelineError(
                        "Error: replica exchange probabilities are too low. You "
                        "need to increase $NRESTR, $NCHARGE or $NANNIH.")
                return
        raise PipelineError(f"could not find replica exchange probabilities in {logfile}")
