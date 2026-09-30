"""Long-range dispersion correction (port of v1 ``lr_exp.py``).

The re-evaluation runs (``lr-*`` phases) recompute the nonbonded
energies of the source trajectory with ``vdwtype = PME``; this module
turns the two energy files into the exponential-average free-energy
difference with block-wise error estimation.

Robustness hardening over v1:

* the two EDR files generally have DIFFERENT time grids (production
  writes energies every ``nstenergy`` of its own clock, ``mdrun -rerun``
  at the trajectory frame times, whose timestamps pass through float32
  in the xtc); frames are aligned by nearest timestamp with a small
  tolerance instead of by position,
* energy terms are matched by a normalized name (spaces, hyphens,
  parentheses and case removed), so term renames between GROMACS
  versions ("LJ (SR)" vs "LJ-(SR)") still match,
* a per-term decomposition of the perturbation is printed, so a broken
  re-evaluation (e.g. the GROMACS 2026 soft-core + PME-LJ garbage, see
  ``docs/issue-lrc-gmx2026.md``) is immediately visible in the log.
"""

from __future__ import annotations

import bisect
import re
import sys

import numpy

from .errors import PipelineError
from .units import kbt

# Energy terms summed for the dispersion comparison (v1 names; matched
# after normalization, see normalize_term_name).
DEFAULT_SUM_TERMS = ["LJ (SR)", "LJ-14", "LJ recip.", "LJC-14 q",
                     "LJC Pairs NB", "Disper. corr."]

# Timestamp matching tolerance [ps]: well below any real EDR grid spacing
# (>= nstcalcenergy*dt ~ 0.2 ps) but above the float32 noise of xtc times.
_TIME_TOL = 0.002

_NORMALIZE_RE = re.compile(r"[\s()\[\]_\-]")


def normalize_term_name(name: str) -> str:
    """Canonical form of an EDR energy-term name.

    Removes spaces, parentheses, hyphens and case so that GROMACS
    version-dependent spellings ("LJ (SR)" vs "LJ-(SR)") match.
    """
    return _NORMALIZE_RE.sub("", name).lower()


def resolve_terms(available: list[str], wanted: list[str] | None = None) -> list[str]:
    """Return the subset of ``available`` (verbatim) matching ``wanted``.

    Matching is done on normalized names.  Terms present in the EDR but
    not wanted (and vice versa) are reported by the caller's diagnostics.
    """
    wanted = wanted or DEFAULT_SUM_TERMS
    wanted_norm = {normalize_term_name(w): w for w in wanted}
    return [name for name in available
            if normalize_term_name(name) in wanted_norm]


def _exp_average(du, kbt_value: float) -> float:
    # ln<exp(X)> = logsumexp(x) - log(N); the free energy is -kT * ln<exp(-dU/kT)>
    import scipy.special
    lnexp = scipy.special.logsumexp(du) - numpy.log(len(du))
    return -1.0 / kbt_value * lnexp


def _sum_series(edr_path: str, wanted: list[str] | None):
    """(times, per-term value series, resolved term names) of an EDR file."""
    import pyedr

    reader = pyedr.pyedr.EDRFile(str(edr_path))
    available = [enx.name for enx in reader.nms]
    matched = resolve_terms(available, wanted)
    matched_set = set(matched)
    ixs = [i for i, enx in enumerate(reader.nms) if enx.name in matched_set]
    times: list[float] = []
    per_term: dict[str, list[float]] = {name: [] for name in matched}
    for frame in iter(reader):
        times.append(frame.t)
        for i in ixs:
            per_term[reader.nms[i].name].append(frame.ener[i].e)
    return times, per_term, matched


def _align_indices(short_times: list[float], long_times: list[float],
                   tol: float = _TIME_TOL) -> list[tuple[int, int]]:
    """Match long-frame indices to short-frame indices within ``tol`` ps."""
    out: list[tuple[int, int]] = []
    for j, t in enumerate(long_times):
        i = bisect.bisect_left(short_times, t)
        for cand in (i - 1, i):
            if 0 <= cand < len(short_times) and abs(short_times[cand] - t) <= tol:
                out.append((j, cand))
                break
    return out


def align_frames(orig: list[tuple[float, float]],
                 reeval: list[tuple[float, float]],
                 kbt_value: float, time_begin: float,
                 tol: float = _TIME_TOL) -> tuple[list[float], list[float]]:
    """Intersect the two time grids and compute the per-frame deltas.

    Returns (times, deltas) with ``delta = -(long - short) / kT`` for
    every re-evaluated frame that has an original frame within ``tol``
    ps, after ``time_begin``.
    """
    orig_times = [t for (t, _) in orig]
    orig_vals = [s for (_, s) in orig]
    times: list[float] = []
    deltas: list[float] = []
    for j, i in _align_indices(orig_times, [t for (t, _) in reeval], tol):
        t = reeval[j][0]
        if t <= time_begin:
            continue
        times.append(t)
        deltas.append(-(reeval[j][1] - orig_vals[i]) / kbt_value)
    return times, deltas


def compute_long_range_correction(*, long, short, temp: float, time_begin: float = 0.0,
                                  output, block: int = 5,
                                  terms: list[str] | None = None) -> float:
    """Compute the long-range correction; returns (mean, std) in kJ/mol."""
    import pyedr  # dependency check (also documents the requirement)

    kbt_value = kbt(temp)
    short_times, short_terms, short_matched = _sum_series(short, terms)
    long_times, long_terms, long_matched = _sum_series(long, terms)
    if not short_matched or not long_matched:
        raise PipelineError(
            "none of the LRC energy terms found in the EDR file(s) "
            f"({short}: {short_matched or 'none'}, {long}: {long_matched or 'none'}); "
            "check LRC_ENERGY_TERMS against `gmx energy` output of your GROMACS build")

    short_sums = [sum(vals[i] for vals in short_terms.values())
                  for i in range(len(short_times))]
    long_sums = [sum(vals[i] for vals in long_terms.values())
                 for i in range(len(long_times))]
    times, deltas = align_frames(list(zip(short_times, short_sums)),
                                 list(zip(long_times, long_sums)),
                                 kbt_value, time_begin)
    if not deltas:
        raise PipelineError(
            "the original and re-evaluated EDR files share no timestamp "
            f"(short grid starts at {short_times[:3]}, long grid at {long_times[:3]}); "
            "check nstenergy of the production runs and the trajectory output "
            "spacing of the source phase")

    # Per-term decomposition of the perturbation: a re-evaluation whose
    # Hamiltonian does not match the source (wrong couple/soft-core
    # handling, version regressions, ...) is visible here immediately.
    aligned = _align_indices(short_times, long_times)
    print("LRC per-term perturbation decomposition (long - short, kJ/mol):",
          file=sys.stderr)
    all_names = sorted(set(short_matched) | set(long_matched),
                       key=lambda n: -(numpy.mean(long_terms[n])
                                       if n in long_terms and long_terms[n] else 0.0))
    for name in all_names:
        in_short, in_long = name in short_terms, name in long_terms
        if in_short and in_long:
            d = numpy.array([long_terms[name][jl] - short_terms[name][il]
                             for (jl, il) in aligned])
            note = ""
        elif in_long:
            d = numpy.array([long_terms[name][jl] for (jl, _) in aligned])
            note = "  (only in the re-evaluated file)"
        else:
            d = numpy.array([-short_terms[name][il] for (_, il) in aligned])
            note = "  (only in the original file)"
        print(f"    {name:16s} mean {d.mean():14.3f}  std {d.std():10.3f}{note}",
              file=sys.stderr)

    deltas_arr = numpy.array(deltas)
    estimates = []
    with open(output, "w") as ofh:
        for b in range(block):
            ixb = (b + 0) * len(deltas_arr) // block
            ixe = (b + 1) * len(deltas_arr) // block
            estimates.append(_exp_average(deltas_arr[ixb:ixe], kbt_value))
            print(times[ixb], times[ixe - 1], "%.4f" % estimates[-1], file=ofh)
        estmean = float(numpy.mean(estimates))
        eststd = float(numpy.std(estimates, ddof=1))
        print("%.4f\t%.4f" % (estmean, eststd), file=ofh)
    return estmean, eststd
