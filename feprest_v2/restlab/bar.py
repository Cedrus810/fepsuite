"""BAR analysis of the HREX ``deltae.xvg`` files — library port of
``feprest/bar_deltae.py``.

Every production chunk writes per-replica energy differences
(``-othersim deltae``); the analysis pairs the dE samples of adjacent
replicas, evaluates BAR (via pymbar) on sliding windows (convergence
diagnostic) and on an equipartition of the latter half of the sampling
time (the estimate), and reports the mean and standard error in
kcal/mol.  The console output format is kept identical to v1 so
existing result-parsing habits (``tail -1 bar1.log``) keep working.

pymbar is imported lazily: the pipeline calls the analysis through
:func:`run_analysis` and survives its absence with the same v1
message ("BAR failed ..."), while tests of the parser need no pymbar.
"""

from __future__ import annotations

import collections
import os
import pickle
import sys

import numpy

SimEval = collections.namedtuple("SimEval", ["sim", "eval"])

# in kJ/mol/K (to fit GROMACS output)
gasconstant = 0.008314472
# in kcal/mol/K
gasconstant_kcal = 0.0019872036

KCAL_OF_KJ = 1. / 4.184


def parse_deltae(files, subsample: int = 1):
    """(time, eval-state, dE) samples of one simulation's xvg files.

    Broken (unterminated) last lines are skipped, timestamps must be
    strictly increasing (double counting guard, as in v1).
    """
    data = []
    tprev = -1
    for f in files:
        with open(f) as fh:
            samplecount = 0
            for l in fh:
                if len(l) == 0:
                    continue
                if l[-1] != "\n":
                    # typically broken file
                    break
                if l[0] in ['#', '@']:
                    pass
                else:
                    ls = l.split()
                    tt = float(ls[0])
                    if tprev >= tt:
                        # prevent double counting
                        continue
                    tprev = tt
                    eval_pot_pair = [(int(ls[i]), float(ls[1 + i]))
                                     for i in range(1, len(ls), 2)]
                    if samplecount % subsample == 0:
                        for (evix, evpot) in eval_pot_pair:
                            data.append((tt, evix, evpot))
                    samplecount += 1
    return data


def bar(emat, time_all, nsim: int, btime: float, etime: float,
        show_intermediate: bool = False) -> float:
    """Sum of the pairwise BAR free differences over nsim-1 adjacent pairs."""
    import pymbar

    dgtot = 0.0
    for isim in range(nsim - 1):
        basestate = SimEval(sim=isim, eval=isim)
        mask_isim = numpy.logical_and(
            time_all[basestate] > btime, time_all[basestate] <= etime)
        nmasked = numpy.sum(mask_isim)
        assert nmasked > 0
        assert len(emat[basestate]) == len(emat[SimEval(sim=isim, eval=isim + 1)])
        assert len(emat[basestate]) == len(emat[SimEval(sim=isim + 1, eval=isim + 1)])
        # uses u_kn representation
        # K: evaluation states
        # N: samples (N_K)
        u = numpy.empty((2, nmasked * 2))
        # evaluated by isim
        u[0, 0:nmasked] = emat[basestate][mask_isim]
        u[0, nmasked:nmasked * 2] = emat[SimEval(sim=isim + 1, eval=isim)][mask_isim]
        u[1, 0:nmasked] = emat[SimEval(sim=isim, eval=isim + 1)][mask_isim]
        u[1, nmasked:nmasked * 2] = emat[SimEval(sim=isim + 1, eval=isim + 1)][mask_isim]

        nk = numpy.array([nmasked, nmasked])

        mb = pymbar.MBAR(u, nk, verbose=False)
        rettuple = mb.getFreeEnergyDifferences(compute_uncertainty=False,
                                               warning_cutoff=1)
        Deltaf = rettuple[0]
        dgtot += Deltaf[0, 1]  # F[isim + 1] - F[isim]
        if show_intermediate:
            print("INT", isim, isim + 1, Deltaf[0, 1], dgtot)
    return dgtot  # F[nsim - 1] - F[0]


def run_analysis(xvgs: str, nsim: int, temp: float = 300.0,
                 save_dir: str | None = None, minpart: int | None = None,
                 maxpart: int | None = None, subsample: int = 1,
                 split: int = 10, show_intermediate: bool = False) -> float:
    """Port of bar_deltae.main(); returns the final estimate in kcal/mol.

    ``xvgs`` is a path pattern in which ``%sim`` (and ``%part``) are
    replaced by the simulation/part numbers.
    """
    if save_dir is None:
        save_dir = os.getcwd()
    beta = 1. / (gasconstant * temp)

    energies = {}
    time_all = {}

    for isim in range(nsim):
        files = []
        if minpart is not None:
            for part in range(minpart, maxpart + 1):
                f = xvgs.replace("%sim", str(isim)).replace("%part", "%04d" % part)
                files.append(f)
        else:
            files.append(xvgs.replace("%sim", str(isim)))
        data = parse_deltae(files, subsample)

        for (t, st, energy) in data:
            se = SimEval(sim=isim, eval=st)
            if se not in energies:
                energies[se] = []
                time_all[se] = []
            if len(time_all[se]) > 0 and time_all[se][-1] == t:
                # dup frame
                continue
            energies[se].append(energy)
            time_all[se].append(t)

        print("Finished loading %s with %d points %d frames" % (
            ", ".join(files), len(data),
            len(time_all[SimEval(sim=isim, eval=isim)])))
        sys.stdout.flush()

    # Since I am rewriting energies[k], just for the safety I don't use "for k in energies"
    for k in list(energies.keys()):
        energies[k] = numpy.array(energies[k]) * beta
        time_all[k] = numpy.array(time_all[k])

    # Do simple bar with 2x2 matrix
    print("Performing sliding-window (time x to 2x) sectioned BAR [kcal/mol]")

    tmin = time_all[SimEval(0, 0)][0]
    tmax = time_all[SimEval(0, 0)][-1]
    results_sliding = []
    for i in range(split):
        twidth = (tmax - tmin) / split
        tspan = (i + 1) * twidth
        t0 = tmin + 0.5 * tspan
        t1 = tmin + tspan
        barres = bar(energies, time_all, nsim, t0, t1, show_intermediate)
        print("BAR SLIDE %f-%f:" % (t0, t1), barres * gasconstant * temp * KCAL_OF_KJ)
        results_sliding.append((t0, t1, barres))

    pickle.dump(results_sliding,
                open("%s/results-sliding.pickle" % save_dir, "wb"))

    print("Performing time-split BAR (equipartition of the latter half) [kcal/mol]")
    tstart = (tmin + tmax) / 2
    twidth = (tmax - tstart) / split
    results_normalsplit = []
    for i in range(split):
        t0 = tstart + twidth * i
        t1 = tstart + twidth * (i + 1)
        barres = bar(energies, time_all, nsim, t0, t1, show_intermediate)
        print("BAR SPLIT %f-%f:" % (t0, t1), barres * gasconstant * temp * KCAL_OF_KJ)
        results_normalsplit.append((t0, t1, barres))

    pickle.dump(results_normalsplit,
                open("%s/results-normalsplit.pickle" % save_dir, "wb"))

    print("Final estimate [kcal/mol]")
    vals = [x * gasconstant * temp * KCAL_OF_KJ for (_, _, x) in results_normalsplit]
    femean = numpy.mean(vals)
    festderr = 0.
    if split > 2:
        festderr = numpy.std(vals, ddof=1) / numpy.sqrt(split - 1)
    print("BAR %.2f %.2f" % (femean, festderr))
    return femean
