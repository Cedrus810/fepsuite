"""GROMACS execution layer.

The pipeline never invokes GROMACS directly: it goes through the
FEP-suite job-system wrappers (``job_singlerun`` / ``job_mpirun`` from
``submit_scripts/<JOBSYSTEM>.zsh``) via a tiny zsh bridge, so all the
system-specific mpirun handling keeps working unchanged.  When
FEPSUITE_ROOT/JOBSYSTEM are not set (development, tests) the runner
falls back to plain ``mpirun -np``.

The NP-search loop (retrying mdrun with fewer ranks when domain
decomposition fails) is a port of v1's ``mdrun_find_possible_np``,
including the ``-ntomp 1`` nudge and the optional ``NSTLIST_CMD``
appendix.  One deliberate fix: the v1 domain-error check tailed
``$log_basename.log`` relative to the run root even for multidir runs
(the loop variable was unused), so the check read a nonexistent path;
here the log of *each* replica directory is inspected, as intended.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from .errors import PipelineError


@dataclass
class RunContext:
    rundir: Path            # contains para_conf.zsh, mdp/, and the run-ID subdirectory
    run_id: str
    feprest_v2_root: Path
    fepsuite_root: str | None
    jobsystem: str | None
    gmx: str
    gmx_mpi: str
    omp_threads: str | None
    python3: str

    @classmethod
    def from_env(cls, rundir: Path | None = None) -> "RunContext":
        run_id = os.environ.get("ID")
        if not run_id:
            raise PipelineError(
                "environment variable ID is not set (must be run via controller.zsh)")
        return cls(
            rundir=Path(rundir or os.environ.get("RESTLAB_RUNDIR") or Path.cwd()).resolve(),
            run_id=run_id,
            feprest_v2_root=Path(os.environ.get("FEPREST_V2_ROOT")
                                 or Path(__file__).resolve().parent.parent),
            fepsuite_root=os.environ.get("FEPSUITE_ROOT"),
            jobsystem=os.environ.get("JOBSYSTEM"),
            gmx=os.environ.get("GMX", "gmx"),
            gmx_mpi=os.environ.get("GMX_MPI", "gmx_mpi"),
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
    def __init__(self, ctx: RunContext):
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
            return ["zsh", str(self.ctx.feprest_v2_root / "jobbridge.zsh"),
                    "singlerun", "--", *args]
        return list(args)

    def _mpirun_cmd(self, np: int, args: list[str]) -> list[str]:
        if self._bridge_mode:
            return ["zsh", str(self.ctx.feprest_v2_root / "jobbridge.zsh"),
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
                tail = "\n" + "\n".join(completed.stderr.splitlines()[-30:])
            raise PipelineError(f"command failed with exit code "
                                f"{completed.returncode}: {' '.join(cmd)}{tail}")
        return completed

    # ---- GROMACS conveniences --------------------------------------------
    def gmx(self, *args, **kwargs) -> subprocess.CompletedProcess:
        return self.singlerun([self.ctx.gmx, *args], **kwargs)

    def gmx_mpi(self, *args, **kwargs) -> subprocess.CompletedProcess:
        return self.singlerun([self.ctx.gmx_mpi, *args], **kwargs)

    def grompp(self, *, mdp, topol, conf, out_tpr, out_mdp, cpt=None, restr=None,
               ndx=None, pp=None, maxwarn: int = 0,
               stdout_path=None) -> None:
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
        self.gmx(*args, stdout_path=stdout_path)

    def convert_tpr(self, *, source, dest, extend, stdout_path=None) -> None:
        self.gmx_mpi("convert-tpr", "-s", str(source), "-o", str(dest),
                     "-extend", str(extend), stdout_path=stdout_path)

    def mpirun_mdrun(self, np_initial: int, args: list[str], *,
                     least_unit: int = 1,
                     stdout_path=None, stderr_path=None,
                     nstlist_cmd: str = "") -> None:
        """Run mdrun under the job-system MPI wrapper, retrying with fewer
        ranks on domain-decomposition failures.

        Port of v1 mdrun_find_possible_np.  ``least_unit`` is the rank
        granularity (number of replicas for multidir runs, 1 otherwise).
        As in v1, ``-ntomp 1`` is appended when OMP_NUM_THREADS is unset
        or 1 ("this is required for forcing GROMACS to run"), and the
        optional ``NSTLIST_CMD`` appendix is forwarded.
        """
        np = self._np_saved if self._np_saved is not None else np_initial
        while True:
            print(f"Trying with NP={np}")
            cmd = self._mpirun_cmd(np, [self.ctx.gmx_mpi, "mdrun", *args])
            if nstlist_cmd:
                cmd.append(nstlist_cmd)
            if not self.ctx.omp_threads or self.ctx.omp_threads == "1":
                cmd += ["-ntomp", "1"]
            completed = self.singlerun(cmd, stdout_path=stdout_path,
                                       stderr_path=stderr_path, check=False)
            if completed.returncode != 0:
                if not self._is_domain_error(args):
                    raise PipelineError("Error: domain-unrelated error")
                prev = np
                np = (np - 1) // least_unit * least_unit
                if np == 0 or np == prev:
                    raise PipelineError(
                        "Error: unable to find proper parallel processes")
            else:
                print("Normal termination of mdrun")
                self._np_saved = np
                return

    @staticmethod
    def _mdrun_log_paths(args: list[str]) -> list[Path]:
        """Log file(s) of a (multi)dir mdrun, from its command line.

        Mirrors v1's argument scan: positional arguments after ``-multidir``
        are all replica directories, until the next flag.
        """
        log_base = None
        dirs = []
        pending = None
        for arg in args:
            if arg in ("-deffnm", "-l"):
                pending = "deffnm" if arg == "-deffnm" else "log"
            elif arg == "-multidir":
                pending = "multidir"
            elif arg.startswith("-"):
                pending = None
            elif pending in ("deffnm", "log"):
                log_base = arg
                pending = None
            elif pending == "multidir":
                dirs.append(arg)
        if not dirs:
            dirs = ["."]  # v1: an empty multidir list means the run root
        if log_base is None:
            return []
        if log_base.endswith(".log"):
            log_base = log_base[:-len(".log")]
        return [Path(d) / f"{log_base}.log" for d in dirs]

    def _is_domain_error(self, args: list[str]) -> bool:
        """Does the mdrun log tail indicate a domain-decomposition issue?

        Port of v1's ``tail -20 ... | grep -i "\\(domain\\|prime\\)"``,
        fixed to look inside each multidir replica directory.
        """
        logs = self._mdrun_log_paths(args)
        if not logs:
            return False
        domain_error = False
        seen_any = False
        for log in logs:
            if not log.is_absolute():
                log = self.ctx.rundir / log
            try:
                text = log.read_text(errors="replace")
            except OSError:
                continue
            seen_any = True
            if re.search(r"(domain|prime)", "\n".join(
                    text.splitlines()[-20:]), re.I):
                domain_error = True
        if not seen_any:
            raise PipelineError(
                "Error: failed during mpirun-mdrun startup. This typically "
                "happens when there are a problem with mpirun command.")
        return domain_error
