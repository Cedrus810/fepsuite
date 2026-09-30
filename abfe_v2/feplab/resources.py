"""Runtime resource detection inside the job allocation.

The pipeline adapts its mdrun rank layout to what the queueing system
actually gave the job, instead of relying only on the static
``para_conf.zsh`` values.  GPU detection is OpenPBS-first and falls
back through several signals:

1. ``qstat -F json -f $PBS_JOBID`` — ``resources_allocated.ngpus``
   (PBS Pro; most reliable when $PBS_JOBID is set),
2. ``$PBS_GPUFILE`` — one GPU per line (OpenPBS),
3. ``$CUDA_VISIBLE_DEVICES`` / ``$NVIDIA_VISIBLE_DEVICES`` —
   comma-separated device list (works for most schedulers),
4. ``nvidia-smi -L`` — GPUs visible on the node.

When nothing can be detected the caller falls back to the static
configuration, so this module never breaks a run that used to work.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Callable, Mapping


@dataclass
class DetectedResources:
    gpus: int
    source: str


def _count_env_devices(value: str) -> int:
    items = [x for x in value.split(",") if x.strip()]
    return len(items)


def detect_gpus(env: Mapping[str, str] | None = None,
                run_cmd: Callable | None = None) -> DetectedResources | None:
    """Detect how many GPUs the current job allocation provides.

    ``run_cmd`` is injectable for testing (defaults to subprocess.run
    with text output).  Returns None when no signal is available.
    """
    env = dict(os.environ if env is None else env)
    run_cmd = run_cmd or (lambda cmd: subprocess.run(cmd, capture_output=True, text=True))

    jobid = env.get("PBS_JOBID", "")
    if jobid:
        try:
            completed = run_cmd(["qstat", "-F", "json", "-f", jobid])
            if completed.returncode == 0 and completed.stdout.strip():
                data = json.loads(completed.stdout)
                jobs = data.get("Jobs") or data.get("Job") or {}
                job = jobs.get(jobid) or (next(iter(jobs.values())) if jobs else None)
                alloc = (job or {}).get("resources_allocated",
                                        {}) or (job or {}).get("Resource_List", {})
                for key in ("ngpus", "ngpu", "gpu"):
                    if key in alloc:
                        try:
                            gpus = int(float(alloc[key]))
                        except (TypeError, ValueError):
                            continue
                        return DetectedResources(gpus=gpus, source="qstat")
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    gpufile = env.get("PBS_GPUFILE", "")
    if gpufile:
        try:
            with open(gpufile) as fh:
                lines = [l for l in fh.read().splitlines() if l.strip()]
            if lines:
                return DetectedResources(gpus=len(lines), source="PBS_GPUFILE")
        except OSError:
            pass

    for var in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        value = env.get(var, "").strip()
        if not value:
            continue
        if value.lower() in ("all", "none", "void"):
            continue
        return DetectedResources(gpus=_count_env_devices(value), source=var)

    try:
        completed = run_cmd(["nvidia-smi", "-L"])
        if completed.returncode == 0:
            gpus = sum(1 for l in completed.stdout.splitlines()
                       if l.strip().startswith("GPU"))
            if gpus > 0:
                return DetectedResources(gpus=gpus, source="nvidia-smi")
    except OSError:
        pass

    return None


def ranks_for_replicas(gpus: int, replicas: int) -> tuple[int, int]:
    """Split ``gpus`` GPUs over ``replicas`` replicas.

    Returns (ranks_per_replica, total_ranks).  Every replica gets at
    least one rank; GPUs are shared when there are fewer GPUs than
    replicas, and additional ranks (each pinned to a GPU) are added when
    there are more GPUs than replicas.
    """
    per_replica = max(1, gpus // replicas)
    return per_replica, replicas * per_replica
