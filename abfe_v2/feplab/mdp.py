"""GROMACS mdp file handling.

One place to read, merge, and write mdp files, including the pieces of
assembling logic that were scattered over the v1 pipeline shell code:

* merging ``mdp/run.mdp`` with per-phase addenda,
* the ``LRCONLY`` marker section that decides which replicas write
  trajectories (needed for the PME re-evaluation runs),
* the ``rlist`` bookkeeping required by ``couple-intramol = no`` decoupling.
"""

from __future__ import annotations

from pathlib import Path

from .errors import PipelineError


class Mdp(dict):
    """An ordered key->value view of an mdp file (keys lowercased)."""

    @classmethod
    def from_text(cls, text: str) -> "Mdp":
        mdp = cls()
        for raw in text.splitlines():
            line = raw.split(";", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            mdp[key.strip().lower()] = value.strip()
        return mdp

    @classmethod
    def read(cls, path) -> "Mdp":
        return cls.from_text(Path(path).read_text())

    def merged(self, other: "Mdp") -> "Mdp":
        out = Mdp(self)
        out.update(other)
        return out

    def without(self, keys) -> "Mdp":
        out = Mdp(self)
        for key in keys:
            out.pop(key.lower(), None)
        return out

    def get_float(self, key: str) -> float:
        try:
            return float(self[key.lower()])
        except KeyError:
            raise PipelineError(f"mdp key {key} is missing") from None
        except ValueError:
            raise PipelineError(f"mdp key {key} is not a number: {self[key]!r}") from None

    def set_float(self, key: str, value: float) -> None:
        self[key.lower()] = f"{value:g}"

    def render(self, header: str | None = None) -> str:
        lines = []
        if header:
            lines.append("; " + header)
        for key, value in self.items():
            lines.append(f"{key} = {value}")
        return "\n".join(lines) + "\n"

    def write(self, path, header: str | None = None) -> None:
        Path(path).write_text(self.render(header))


def read_mdp_text(text: str) -> Mdp:
    return Mdp.from_text(text)


# ---------------------------------------------------------------------------
# LRCONLY section handling (port of the v1 sed postprocessing)
# ---------------------------------------------------------------------------

def strip_for_traj(text: str, keep_traj: bool) -> str:
    """Postprocess a concatenated mdp text for trajectory output policy.

    The per-phase templates carry a ``;LRCONLY_BEGIN ... ;LRCONLY_END``
    block that enables compressed trajectory output for the PME
    re-evaluation runs.  When only one replica's trajectory is needed:

    * ``compressed`` keys from ``mdp/run.mdp`` are removed (they appear
      before the first marker), and
    * for the non-kept replicas the whole marked block is removed.

    This reproduces the v1 sed expressions exactly.
    """
    out: list[str] = []
    seen_begin = False
    in_block = False
    for line in text.splitlines():
        if not seen_begin and "LRCONLY_BEGIN" in line:
            seen_begin = True
            if keep_traj:
                out.append(line)
            else:
                in_block = True
            continue
        if in_block:
            if "LRCONLY_END" in line:
                in_block = False
            continue
        if not seen_begin and "compressed" in line:
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def assemble_product_mdp(run_text: str, addenda_text: str,
                         traj_keep: bool | None) -> str:
    """Concatenate run.mdp with phase addenda, applying the traj policy.

    ``traj_keep`` of None means "keep everything unchanged" (no replica
    filtering requested).
    """
    combined = run_text + "\n" + addenda_text
    if traj_keep is None:
        return combined
    return strip_for_traj(combined, traj_keep)


# ---------------------------------------------------------------------------
# rlist handling
# ---------------------------------------------------------------------------

def compute_rlist(mdp: Mdp, ligand_diameter: float, safe_rlist: float | None) -> float:
    """Determine the pair-list cutoff needed for decoupling.

    With ``couple-intramol = no`` the intra-ligand exclusions make GROMACS
    unable to estimate the buffer automatically, hence the explicit rlist.
    """
    if ligand_diameter != 0:
        return ligand_diameter
    rlist = 0.0
    for key in ("rvdw", "rcoulomb"):
        if key in mdp:
            rlist = max(rlist, mdp.get_float(key) * 1.2)  # 1.2 for buffering
    if safe_rlist is not None and safe_rlist > rlist:
        rlist = safe_rlist
    return rlist


def apply_rlist(mdp: Mdp, rlist: float, nstlist: int | None) -> None:
    mdp["rlist"] = f"{rlist:g}"
    mdp["verlet-buffer-tolerance"] = "-1"
    integrator = mdp.get("integrator", "")
    if nstlist is not None and integrator not in ("cg", "steep"):
        mdp["nstlist"] = str(nstlist)


def set_nsteps(mdp: Mdp, nsteps: int) -> None:
    mdp["nsteps"] = str(int(nsteps))
