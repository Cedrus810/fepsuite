"""Replica-parameter state and optimization — library port of
``feprest/rest2py/replica_optimizer.py``.

The REST2 replica ladder is described by a small state file
(``replica_states``) inside the run-ID directory: a mode, the replica
count, the number of intermediate "integer" anchor points, and the
fractional coordinate of every replica on the [0, intstates] ladder.
During the ``tune`` stage, the observed exchange probabilities are
converted into per-pair free energies, whose cumulative sum gives a new
coordinate set; the new set is blended with the old one and re-anchored
to the integer points (:func:`reset_midpoints`).

The feprest mode maps each coordinate ``c`` onto REST2 parameters
(:func:`get_parameter_lists`):

    0 <= c <= 1 : charge-annihilation region  (A-state charge -> 0)
    1 <  c <  2 : vdW-annihilation region
    2 <= c <= 3 : charge-introduction region  (0 -> B-state charge)

with the temperature ladder exponentially spaced between REST2_TEMP0 and
REST2_TEMP over the first half of the replicas (mirrored for the rest).

v1 drove ``rest2py.py`` through a subprocess per replica; here the
scaling runs in-process (:func:`update_topologies`).
"""

from __future__ import annotations

import collections
import math
import os
import re

from ..mdp import render_lambda_template
from ..errors import PipelineError
from .scaling import Rest2Options, convert_topology

REPLICA_STATE_FILE = "replica_states"

State = collections.namedtuple(
    "State", ["mode", "n", "mid_integer_points", "coordinates"])
RestParams = collections.namedtuple(
    "RestParams",
    ["temp0", "temp", "general_lambda", "charge_lambda_A", "charge_lambda_B",
     "vdw_lambda", "end_restrain_A", "end_restrain_B"])


def parse_repl_ex(logfile) -> list[float]:
    """Average exchange probabilities from a HREX mdrun log."""
    findpat = re.compile("Repl  average probabilities:")
    with open(logfile) as fh:
        for l in fh:
            if findpat.match(l):
                break
        else:
            raise PipelineError(
                f"could not find 'Repl  average probabilities:' in {logfile}")
        next(fh)  # Replica #s
        l = next(fh)  # actual exchange probabilities
        exs = [float(x) for x in l.split()[1:]]
    return exs


def state_path(basedir) -> str:
    return os.path.join(basedir, REPLICA_STATE_FILE)


def load_state(basedir) -> State:
    with open(state_path(basedir)) as fh:
        l = next(fh)
        ls = l.split()
        assert ls[0] == "mode" and ls[1] == "="
        mode = ls[2]

        l = next(fh)
        ls = l.split()
        assert ls[0] == "n" and ls[1] == "="
        n = int(ls[2])

        l = next(fh)
        ls = l.split()
        assert ls[0] == "mid-points" and ls[1] == "="
        mid_points = int(ls[2])

        l = next(fh)
        ls = l.split()
        assert ls[0] == "coordinates" and ls[1] == "="
        coordinates = [float(x) for x in ls[2:]]
    return State(mode=mode, n=n, mid_integer_points=mid_points,
                 coordinates=coordinates)


def save_state(state: State, basedir) -> None:
    assert isinstance(state.mode, str)
    assert isinstance(state.n, int)
    assert isinstance(state.mid_integer_points, int)
    assert isinstance(state.coordinates, list)
    assert len(state.coordinates) == state.n
    assert abs(state.coordinates[0] - 0.0) < 1e-8
    assert abs(state.coordinates[-1] - (state.mid_integer_points + 1)) < 1e-8
    with open(state_path(basedir), 'w') as fh:
        print("mode = %s" % state.mode, file=fh)
        print("n = %d" % state.n, file=fh)
        print("mid-points = %d" % state.mid_integer_points, file=fh)
        print("coordinates = %s" % (" ".join(str(x) for x in state.coordinates)),
              file=fh)


def reset_midpoints(coords, intstates):
    # (idx, prev, new)
    anchor = [(0, 0., 0.)]
    for i in range(1, intstates):
        nearest = -1
        dist = 1e9
        for (j, x) in enumerate(coords):
            r = abs(x - i)
            if r < dist:
                dist = r
                nearest = j
        anchor.append((nearest, coords[nearest], float(i)))
        coords[nearest] = float(i)

    anchor.append((len(coords) - 1, float(intstates), float(intstates)))
    # rescale values according to anchors
    for i in range(1, intstates):
        (b, prevx, newx) = anchor[i - 1]
        (e, prevy, newy) = anchor[i]
        for j in range(b + 1, e):
            coords[j] = (coords[j] - prevx) / (prevy - prevx) * (newy - newx) + newx

    assert abs(coords[0] - 0.0) < 1e-8
    assert abs(coords[-1] - intstates) < 1e-8

    # fix to be just that value
    coords[0] = 0.0
    coords[-1] = float(intstates)
    return coords


def init_coords(nstate: int, intermeds: int) -> list[float]:
    coords = []
    for i in range(nstate):
        x = float(i) / (nstate - 1)
        coords.append(x * intermeds)
    return reset_midpoints(coords, intermeds)


def do_init(nstate: int, mode: str, basedir) -> State:
    if mode == "feprest":
        state_trans = 3
        coords = init_coords(nstate, state_trans)
        state = State(mode=mode,
                      n=nstate,
                      mid_integer_points=state_trans - 1,
                      coordinates=coords)
    else:
        raise PipelineError(f"unknown replica mode: {mode} (v1 silently "
                            "wrote nothing; v2 fails explicitly)")
    save_state(state, basedir)
    return state


def optimize_state_from_exprobs(state: State, probs) -> list[float]:
    nlexval = [-math.log(max(e, 0.01)) for e in probs]

    cumuval = []
    cur = 0.0
    for i in range(state.n - 1):
        cumuval.append(cur)
        cur += nlexval[i]
    cumuval.append(cur)

    prop = cumuval[-1] / (state.n - 1.0)
    result = [0.0]
    for i in range(1, state.n - 1):
        level = float(i) * prop
        ix = 0
        for j in range(0, state.n):
            if cumuval[j] > level:
                ix = j - 1
                break
        remain = level - cumuval[ix]
        newlam = remain / (cumuval[ix + 1] - cumuval[ix]) \
            * (state.coordinates[ix + 1] - state.coordinates[ix]) \
            + state.coordinates[ix]
        result.append(newlam)
    result.append(float(state.mid_integer_points + 1))
    return result


def update_params(oldcoord, newcoords, nstep) -> list[float]:
    blend = 1 / (1. + nstep)
    return [((1. - blend) * oldcoord[i] + blend * x)
            for (i, x) in enumerate(newcoords)]


def do_optimize_step(logfile, basedir, nstep: int) -> State:
    state = load_state(basedir)
    exprobs = parse_repl_ex(logfile)
    new_coords = optimize_state_from_exprobs(state, exprobs)
    new_coords = update_params(state.coordinates, new_coords, nstep)
    new_coords = reset_midpoints(new_coords, state.mid_integer_points + 1)

    state = state._replace(coordinates=new_coords)
    save_state(state, basedir)
    return state


def get_parameter_lists(state: State, temp0: float, temp: float
                        ) -> list[RestParams]:
    """REST2 parameter set of every replica for the current coordinates."""
    paramlist = []
    if state.mode == "feprest":
        basetemp = temp0
        maxtemp = temp
        rest_nmax = (state.n - 1) // 2  # [0 .. rest_nmax]
        for i in range(state.n):
            rest_state = i
            if rest_state > rest_nmax:
                rest_state = state.n - 1 - i
            assert rest_state <= rest_nmax
            reftemp = basetemp * math.exp(
                math.log(maxtemp / basetemp) / rest_nmax * rest_state)
            c = state.coordinates[i]
            assert state.mid_integer_points == 2
            maxlambda = 3.0
            general_lambda = c / maxlambda
            if c <= 1.0:
                cx = c
                vdw_lambda = 0.0
                charge_lambda_A = cx
                charge_lambda_B = 0.0
                end_restrain_A = 0.0
                end_restrain_B = math.sin(cx * 90.0 * math.pi / 180.0)
            elif c > 1.0 and c < 2.0:
                cx = c - 1.0
                vdw_lambda = cx
                charge_lambda_A = 1.0
                charge_lambda_B = 0.0
                end_restrain_A = 0.0
                end_restrain_B = 1.0
            else:
                cx = c - 2.0
                vdw_lambda = 1.0
                charge_lambda_A = 1.0
                charge_lambda_B = cx
                end_restrain_A = 1. - math.cos(cx * 90.0 * math.pi / 180.0)
                end_restrain_B = 1.0

            paramlist.append(RestParams(temp0=basetemp, temp=reftemp,
                                        general_lambda=general_lambda,
                                        charge_lambda_A=charge_lambda_A,
                                        charge_lambda_B=charge_lambda_B,
                                        vdw_lambda=vdw_lambda,
                                        end_restrain_A=end_restrain_A,
                                        end_restrain_B=end_restrain_B))

    return paramlist


def scaling_options(param: RestParams) -> Rest2Options:
    """rest2py options replicating v1 call_rest2py's command line."""
    return Rest2Options(
        unify_charge=True,
        charge_lambda=param.general_lambda,
        charge_lambda_A=param.charge_lambda_A,
        charge_lambda_B=param.charge_lambda_B,
        end_restrain_dihedralA=param.end_restrain_A,
        end_restrain_dihedralB=param.end_restrain_B,
        temp0=param.temp0,
        temp=param.temp,
    )


def update_topologies(top_pp, top_generate, basedir, temp0: float,
                      temp: float) -> None:
    """Generate the per-replica REST2 topologies (v1 do_update_topology)."""
    state = load_state(basedir)
    params = get_parameter_lists(state, temp0, temp)
    if state.mode == "feprest":
        for i in range(state.n):
            convert_topology(top_pp, top_generate % i, scaling_options(params[i]))


def update_mdps(mdp_template, mdp_generate, basedir, temp0: float,
                temp: float) -> None:
    """Generate the per-replica REST2 mdps (v1 do_update_mdp)."""
    state = load_state(basedir)
    params = get_parameter_lists(state, temp0, temp)
    if state.mode == "feprest":
        for i in range(state.n):
            param = params[i]
            with open(mdp_template) as fh, open(mdp_generate % i, "w") as ofh:
                ofh.write(render_lambda_template(fh.read(),
                                                 param.general_lambda,
                                                 param.vdw_lambda))
