"""Lambda schedule handling.

* static schedules for charging / restraint / annihilation phases,
* optimization of the annihilation schedule from replica-exchange
  probabilities observed in the prerun logs (the v1 scheme, kept as is),
* parsing of the exchange statistics out of GROMACS log files.
"""

import math
import re

import numpy


def schedule(kind: str, n: int) -> list[float]:
    """Initial lambda values for n states of the given phase kind."""
    if kind in ("charging", "restraint"):
        return [float(x) for x in numpy.linspace(0.0, 1.0, n, endpoint=True)]
    if kind == "annihilation":
        # Quadratic ease-in schedule; an SSC(2) variant (Lee et al., JCTC 16,
        # 5512 (2020)) was tested in v1 and performed considerably worse.
        x = numpy.linspace(0.0, 1.0, n, endpoint=True)
        return [float(v) for v in (1.0 - (x - 1.0) ** 2)]
    raise RuntimeError(f"Unsupported lambda schedule kind: {kind}")


def parse_repl_ex(logfile) -> tuple[list[float], list[float]]:
    """Extract (fep_lambdas, exchange probabilities) from a GROMACS log."""
    findpat_lam = re.compile(r"\s*fep-lambdas\s*=")
    findpat = re.compile(r"Repl +average probabilities:")
    ret_lambda: list[float] | None = None
    with open(logfile) as fh:
        for line in fh:
            if findpat_lam.match(line):
                lambdas = line.split("=")[1]
                if lambdas.strip() == "TRUE":
                    continue
                ret_lambda = [float(v) for v in lambdas.split()]
            if findpat.match(line):
                break
        else:
            raise RuntimeError(f"exchange statistics not found in {logfile}")
        next(fh)  # Replica #s
        line = next(fh)  # actual exchange probabilities
        exs = [float(x) for x in line.split()[1:]]
    if ret_lambda is None:
        raise RuntimeError(f"lambda does not appear in log file {logfile}")
    return ret_lambda, exs


def optimize_state_from_exprobs(n: int, prevlambda: list[float],
                                exprobs: list[float]) -> list[float]:
    """Redistribute intermediate lambda states for equal exchange rates.

    The new states are placed so that the cumulative -log(exchange
    probability) is spaced equally between the fixed endpoints 0 and 1.
    """
    if len(prevlambda) != n or len(exprobs) != n - 1:
        raise RuntimeError("lambda/probability count mismatch")
    nlexval = [-math.log(max(e, 0.01)) for e in exprobs]

    cumuval = []
    cur = 0.0
    for i in range(n - 1):
        cumuval.append(cur)
        cur += nlexval[i]
    cumuval.append(cur)

    prop = cumuval[-1] / (n - 1.0)
    result = [0.0]
    for i in range(1, n - 1):
        level = float(i) * prop
        ix = 0
        for j in range(n):
            if cumuval[j] > level:
                ix = j - 1
                break
        remain = level - cumuval[ix]
        newlam = (remain / (cumuval[ix + 1] - cumuval[ix])
                  * (prevlambda[ix + 1] - prevlambda[ix]) + prevlambda[ix])
        result.append(newlam)
    result.append(1.0)
    return result


def update_lambda(n: int, update_log, prerun_phase: int) -> list[float]:
    """Optimized (and blended) lambda values from a prerun log.

    The blend factor 0.7**nth damps successive schedule updates so the
    optimization converges instead of oscillating.
    """
    prevlambda, exprob = parse_repl_ex(update_log)
    newlambda = optimize_state_from_exprobs(n, prevlambda, exprob)
    blend = 0.7 ** prerun_phase
    return [((1.0 - blend) * old + blend * new)
            for old, new in zip(prevlambda, newlambda)]
