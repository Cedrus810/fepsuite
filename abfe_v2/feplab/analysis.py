"""Final free-energy analysis: the ``result.txt`` report.

Port of ``abfe/calc_bar_replex.py``, whose stdout the v1 pipeline
redirected into ``<rundir>/<ID>/result.txt``.  The math and the output
format are kept exactly:

* one ``name mean std`` line per individual term, in kcal/mol with three
  decimals, terms sorted by name,
* a ``----`` separator, then the same lines for the cycle subtotals
  (annihilation, charging, long-range-correction, restraint), sorted,
* another ``----`` separator and a final ``total mean std`` line, where
  the total is the sum over the *individual* terms (v1 sums individuals,
  not subtotals).

The per-phase terms and their coefficients are declared in
:mod:`feplab.phases`; the analytical Boresch restraint correction lives
in :mod:`feplab.restraints`, and the charge correction follows
Chen et al., JCTC 14, 6346 (2018).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from .errors import PipelineError
from .phases import BAR_TERMS, CC_TERMS, LRC_TERMS
from .restraints import (
    SPRING_ANGLE,
    SPRING_DIHEDRAL,
    SPRING_DISTANCE,
    analytical_restraint_free_energy,
    read_restrinfo,
)
from .units import EPS0, KCAL_PER_KJ

_CITATION = """If you use this program please read and cite:
Accurate Calculation of Relative Binding Free Energies between Ligands with Different Net Charges.
Wei Chen, Yuqing Deng, Ellery Russell, Yujie Wu, Robert Abel, and Lingle Wang
Journal of Chemical Theory and Computation, 14, 6346 (2018)."""


def read_bar_log(path) -> tuple[float, float]:
    """Parse a ``gmx bar`` log file into (energy, error) in kJ/mol.

    Among the lines whose first token is ``total`` (the log contains one
    per data block), the LAST one wins; the energy is token 5 and the
    statistical error token 7 (v1 convention).
    """
    energy: float | None = None
    error: float | None = None
    with open(path) as fh:
        for line in fh:
            ls = line.split()
            if len(ls) == 0 or ls[0] != "total":
                continue
            energy = float(ls[5])
            error = float(ls[7])
    if energy is None:
        raise PipelineError(f"{path}: no 'total' line found in bar log")
    return (energy, error)


def read_lrc_result(path) -> tuple[float, float]:
    """Read ``(mean, std)`` from the last line of an lrc result file."""
    with open(path) as fh:
        ls = fh.readlines()[-1].split()
    return (float(ls[0]), float(ls[1]))


def combine_charge_samples(samples: list[dict]) -> tuple[float, float]:
    """Combine per-sample charge-correction contributions, in kJ/mol.

    Port of v1 ``calc_charge_correction_impl`` over JSON-serialized
    ChargeSample dicts with keys ``lattice, xi_LS, eps_s, q_s, q_l,
    nsol, gamma_s, ip, il, volume``.  Returns ``(mean, variance)`` with
    the sample variance (ddof=1); an empty list yields (0.0, 0.0).  The
    per-sample breakdown is printed to stderr, as v1 did.
    """
    if len(samples) == 0:
        return (0.0, 0.0)

    retvals: list[float] = []
    for isample, values in enumerate(samples):
        latticea = values["lattice"]
        xi_ls = values["xi_LS"]
        eps_s = values["eps_s"]

        # Note Rocklin corrected the charging FE while we need to correct
        # the discharging FE (on the complex state).
        q_initial = values["q_s"]
        q_final = q_initial - values["q_l"]
        # latticea is used because xi_LS is scaled by this number.
        dgnet_usv = (-xi_ls / (8.0 * math.pi * EPS0 * eps_s)
                     * (q_final ** 2 - q_initial ** 2) / latticea)

        ip_v = values["ip"] / values["volume"]
        il_v = values["il"] / values["volume"]

        dgrip = ip_v * q_final - (ip_v + il_v) * q_initial

        nsol = values["nsol"]
        gammas = values["gamma_s"]
        volume = values["volume"]

        dgdsc = -gammas * nsol / (6.0 * EPS0 * volume) * (q_final - q_initial)

        dgtot = dgnet_usv + dgrip + dgdsc
        print(f" CC {isample} dgnet_usv {dgnet_usv:10.5f}  dgrip {dgrip:10.5f}"
              f"  dgdsc {dgdsc:10.5f}  dgtot {dgtot:10.5f}", file=sys.stderr)

        retvals.append(dgtot)

    mean = sum(retvals) / len(retvals)
    if len(retvals) > 1:
        devs = sum([(x - mean) ** 2 for x in retvals])
        var = devs / (len(retvals) - 1)
    else:
        var = 0.0  # a single sample carries no error estimate

    return (mean, var)


def read_charge_samples(path) -> list[dict]:
    """Load the JSON list of charge-correction sample dicts.

    A missing file means the correction was not computed; an empty list
    is returned so the correction contributes exactly zero.
    """
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as fh:
        return json.load(fh)


def build_report(*, basedir, restrinfo, temp: float,
                 distance_spring: float | None = None,
                 angle_spring: float | None = None,
                 dihedral_spring: float | None = None,
                 skipped_lrc: set[str] | None = None,
                 sanity_limit_kcal: float | None = None) -> str:
    """Assemble the final report for one transformation.

    ``basedir`` holds the per-phase analysis outputs (``<phase>.bar.log``,
    ``<lr-phase>.lrc.txt`` and ``charge-correction/{complex,ligand}.json``);
    ``restrinfo`` is the restraint file with the Boresch average
    coordinates.  The spring constants default to the module constants of
    :mod:`feplab.restraints` when not given.

    ``skipped_lrc`` lists LRC terms that were intentionally not computed
    (SKIP_ANNIHILATION_LRC); they contribute exactly zero.  Any other LRC
    term whose magnitude exceeds ``sanity_limit_kcal`` aborts the report:
    poisoned LRC values (garbage PME-LJ re-evaluations) must not silently
    enter result.txt.

    Returns the report text in kcal/mol; write it to ``result.txt``
    unchanged.
    """
    basedir = Path(basedir)
    skipped_lrc = skipped_lrc or set()
    limit = 50.0 if sanity_limit_kcal is None else sanity_limit_kcal
    individual: dict[str, tuple[float, float]] = {}
    subtotals: dict[str, tuple[float, float]] = {}

    def accumulate(name: str, energy: float, var: float) -> None:
        curene, curvar = subtotals.get(name, (0.0, 0.0))
        subtotals[name] = (curene + energy, curvar + var)

    # BAR free energies of the charging / restraint / annihilation legs.
    for key, (name, coeff) in BAR_TERMS.items():
        energy, error = read_bar_log(basedir / f"{key}.bar.log")
        individual[key] = (coeff * energy, error ** 2)
        accumulate(name, *individual[key])

    # Long-range dispersion corrections, post-processed outside GROMACS.
    for key, (name, coeff) in LRC_TERMS.items():
        if key in skipped_lrc:
            individual[key] = (0.0, 0.0)
            accumulate(name, 0.0, 0.0)
            continue
        mean, std = read_lrc_result(basedir / f"{key}.lrc.txt")
        individual[key] = (coeff * mean, std ** 2)
        if abs(coeff * mean) * KCAL_PER_KJ > limit:
            raise PipelineError(
                f"the long-range correction term '{key}' is {coeff * mean * KCAL_PER_KJ:.1f}"
                f" kcal/mol, far beyond the sanity limit of {limit:g} kcal/mol —"
                " result.txt would be poisoned.  Known cause: the PME re-evaluation"
                " of the soft-core decoupled annihilation endpoints produces garbage"
                " 'LJ recip.' energies on newer GROMACS builds (see"
                " abfe_v2/docs/issue-lrc-gmx2026.md).  Remedies: set"
                " SKIP_ANNIHILATION_LRC=yes in para_conf.zsh, run the lr-* stages"
                " with GROMACS 2022.5, or (at your own risk) raise LRC_SANITY_LIMIT.")
        accumulate(name, *individual[key])

    # Analytical Boresch correction.  ArVBA in Boresch 2003 corresponds to
    # P...L -> P+L, thus a NEGATIVE contribution.
    _anchors, avgs = read_restrinfo(Path(restrinfo))
    analytical_e = analytical_restraint_free_energy(
        avgs, temp=temp,
        distance_spring=SPRING_DISTANCE if distance_spring is None else distance_spring,
        angle_spring=SPRING_ANGLE if angle_spring is None else angle_spring,
        dihedral_spring=SPRING_DIHEDRAL if dihedral_spring is None else dihedral_spring)
    restre, var = subtotals["restraint"]
    subtotals["restraint"] = (restre - analytical_e, var)
    individual["restraint-analytical"] = (-analytical_e, 0.0)

    # Charge corrections (Chen et al. 2018).  They correct the charged
    # states and therefore fold into the "charging" subtotal, like v1
    # (CC_TERMS carries the per-term coefficients).
    for key, ccfile in (("charge-correction-complex",
                         basedir / "charge-correction" / "complex.json"),
                        ("charge-correction-ligand",
                         basedir / "charge-correction" / "ligand.json")):
        samples = read_charge_samples(ccfile)
        if len(samples) > 0:
            print(_CITATION, file=sys.stderr)
        mean, var = combine_charge_samples(samples)
        _name, coeff = CC_TERMS[key]
        individual[key] = (coeff * mean, coeff ** 2 * var)
        accumulate("charging", *individual[key])

    lines: list[str] = []
    total = 0.0
    totalvar = 0.0
    for mode in sorted(individual.keys()):
        ene, var = individual[mode]
        lines.append("%s %.3f %.3f"
                     % (mode, KCAL_PER_KJ * ene, KCAL_PER_KJ * math.sqrt(var)))
        total += ene
        totalvar += var
    lines.append("----")
    for mode in sorted(subtotals.keys()):
        ene, var = subtotals[mode]
        lines.append("%s %.3f %.3f"
                     % (mode, KCAL_PER_KJ * ene, KCAL_PER_KJ * math.sqrt(var)))
    lines.append("----")
    lines.append("%s %.3f %.3f"
                 % ("total", KCAL_PER_KJ * total, KCAL_PER_KJ * math.sqrt(totalvar)))
    return "\n".join(lines) + "\n"
