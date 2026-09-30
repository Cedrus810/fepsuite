"""MDP assembly: ports of the v1 ``sed`` pipelines.

The v1 pipeline.zsh builds every initial-run mdp by piping the templates
through ``sed``:

    sed -e "/%LAMBDA%/d;/%VDWLAMBDA%/d;/%STATE%/d;/free-energy/c free-energy = no"
    sed -e "/integrator/c integrator = steep"
    sed -e "s/-DPOSRES/ /"        (only when the topology has no POSRES #ifdef)
    sed -e "s/cg/steep/;/nsteps/s/5000/500/;"

The REST2 replica mdps are rendered by ``replica_optimizer.py`` replacing
``%LAMBDA%`` / ``%VDWLAMBDA%`` with the per-replica values.  All of these
are pure string transformations here, so they can be unit-tested against
the real templates in ``rundir_template/mdp/``.
"""

from __future__ import annotations

import re
from pathlib import Path

from .errors import PipelineError

# The FEP placeholder tokens of the REST2 templates; any line carrying one
# is dropped by the "FEP-off" transformation.
FEP_TOKENS = ("%LAMBDA%", "%VDWLAMBDA%", "%STATE%")


def strip_fep(text: str) -> str:
    """Turn a REST2 template into a plain (non-free-energy) mdp.

    Port of ``sed -e "/%LAMBDA%/d;/%VDWLAMBDA%/d;/%STATE%/d;/free-energy/c
    free-energy = no"``: lines with a placeholder token are deleted and
    every line containing ``free-energy`` is replaced by
    ``free-energy = no`` (the sed pattern is unanchored, so comment lines
    mentioning free-energy are replaced too — kept as-is on purpose).
    """
    out = []
    for line in text.splitlines(keepends=True):
        if any(token in line for token in FEP_TOKENS):
            continue
        if "free-energy" in line:
            eol = "\r\n" if line.endswith("\r\n") else "\n"
            out.append("free-energy = no" + eol)
        else:
            out.append(line)
    return "".join(out)


def set_integrator(text: str, integrator: str) -> str:
    """Port of ``sed -e "/integrator/c integrator = steep"``."""
    out = []
    for line in text.splitlines(keepends=True):
        if "integrator" in line:
            eol = "\r\n" if line.endswith("\r\n") else "\n"
            out.append(f"integrator = {integrator}" + eol)
        else:
            out.append(line)
    return "".join(out)


def remove_posres_define(text: str) -> str:
    """Port of ``sed -e "s/-DPOSRES/ /"`` (first occurrence per line)."""
    return "\n".join(line.replace("-DPOSRES", " ", 1)
                     for line in text.splitlines()) + "\n"


def topology_defines_posres(topology_text: str) -> bool:
    """v1 gate: ``grep -q POSRES $ID/$BASETOP``."""
    return "POSRES" in topology_text


def cg_to_steep_minimization(cg_mdp_text: str) -> str:
    """Per-replica minimization mdp of the rest2-setup stage.

    Port of ``sed -e "s/cg/steep/;/nsteps/s/5000/500/;"``: the first
    ``cg`` on each line becomes ``steep`` and the first ``5000`` on the
    nsteps line becomes ``500`` (sed applies s/// once per line).
    """
    out = []
    for line in cg_mdp_text.splitlines(keepends=True):
        eol = "\r\n" if line.endswith("\r\n") else ""
        content = line[:-len(eol)] if eol else line
        content = content.replace("cg", "steep", 1)
        if content.lstrip().startswith("nsteps"):
            content = content.replace("5000", "500", 1)
        out.append(content + eol)
    return "".join(out)


def append_hot_group(text: str) -> str:
    """Tuning/production mdps track the REST2 hot group for the HREX patch.

    Port of ``{ echo "energygrps = hot"; echo "userint1 = 1" } >> file``.
    """
    if not text.endswith("\n"):
        text += "\n"
    return text + "energygrps = hot\nuserint1 = 1\n"


def render_lambda_template(text: str, general_lambda: float,
                           vdw_lambda: float) -> str:
    """Port of replica_optimizer.replace_mdp_template (same %.7f format)."""
    return (text
            .replace("%LAMBDA%", "%.7f" % general_lambda)
            .replace("%VDWLAMBDA%", "%.7f" % vdw_lambda))


_REF_T_RE = re.compile(r"^\s*ref[_|-]t\s*=\s*([^;]+)")


def production_temperature(run_mdp_text: str) -> float:
    r"""Reference temperature of the production mdp.

    Port of v1 do_bar's ``grep '^\s*ref[_|-]t' mdp/run.mdp | cut -d '='
    -f2 | cut -d ';' -f1``.
    """
    for line in run_mdp_text.splitlines():
        m = _REF_T_RE.match(line)
        if m:
            return float(m.group(1).strip())
    raise PipelineError("could not find ref-t/ref_t in the production mdp")


def read_mdp(path: str | Path) -> str:
    return Path(path).read_text()
