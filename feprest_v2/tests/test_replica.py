import pytest

from restlab.errors import PipelineError
from restlab.rest2 import replica


def test_init_and_roundtrip(tmp_path):
    state = replica.do_init(4, "feprest", basedir=tmp_path)
    assert state.mode == "feprest"
    assert state.n == 4
    assert state.mid_integer_points == 2
    # linspace * 3 = [0, 1, 2, 3] (already anchored)
    assert state.coordinates == [0.0, 1.0, 2.0, 3.0]
    loaded = replica.load_state(tmp_path)
    assert loaded == state


def test_init_unknown_mode_fails(tmp_path):
    # v1 silently produced a NameError for unknown modes; v2 fails clearly
    with pytest.raises(PipelineError, match="mode"):
        replica.do_init(4, "typo", basedir=tmp_path)


def test_init_coords_anchored():
    coords = replica.init_coords(5, 3)
    assert coords[0] == 0.0 and coords[-1] == 3.0
    # integer anchors exist at 1 and 2
    assert 1.0 in coords and 2.0 in coords
    # monotonic
    assert coords == sorted(coords)


def test_get_parameter_lists_ladder():
    state = replica.State(mode="feprest", n=4, mid_integer_points=2,
                          coordinates=[0.0, 1.0, 2.0, 3.0])
    params = replica.get_parameter_lists(state, temp0=300.0, temp=1200.0)
    assert len(params) == 4
    # temperature ladder: rest_nmax = (n-1)//2 = 1, mirrored second half
    assert params[0].temp == pytest.approx(300.0)
    assert params[1].temp == pytest.approx(1200.0)
    assert params[2].temp == pytest.approx(1200.0)
    assert params[3].temp == pytest.approx(300.0)

    # coordinate regions: charge-out / vdw / charge-in
    p0, p1, p2, p3 = params
    assert p0.general_lambda == pytest.approx(0.0)
    assert p0.vdw_lambda == 0.0 and p0.charge_lambda_A == 0.0
    assert p0.end_restrain_B == pytest.approx(0.0)
    assert p1.charge_lambda_A == pytest.approx(1.0)
    assert p1.end_restrain_B == pytest.approx(1.0)
    assert p2.vdw_lambda == pytest.approx(1.0)
    assert p2.charge_lambda_A == pytest.approx(1.0)
    assert p2.charge_lambda_B == 0.0
    assert p3.vdw_lambda == pytest.approx(1.0)
    assert p3.charge_lambda_B == pytest.approx(1.0)
    assert p3.end_restrain_A == pytest.approx(1.0)
    # general lambda is c / maxlambda(=3)
    assert p3.general_lambda == pytest.approx(1.0)


def test_optimize_uniform_probs_keeps_spacing():
    state = replica.State(mode="feprest", n=4, mid_integer_points=2,
                          coordinates=[0.0, 1.0, 2.0, 3.0])
    result = replica.optimize_state_from_exprobs(state, [0.5, 0.5, 0.5])
    assert result[0] == 0.0 and result[-1] == 3.0
    assert result[1] == pytest.approx(1.0)
    assert result[2] == pytest.approx(2.0)


def test_optimize_skews_toward_low_probability():
    # pair 0-1 is hard to exchange (p=0.01): its large free-energy
    # segment absorbs both interior coordinates of the uniform ladder
    # (deterministic reparametrization of v1's optimizer)
    import math

    state = replica.State(mode="feprest", n=4, mid_integer_points=2,
                          coordinates=[0.0, 1.0, 2.0, 3.0])
    result = replica.optimize_state_from_exprobs(state, [0.01, 0.5, 0.5])
    assert result[0] == 0.0 and result[-1] == 3.0
    share = (-math.log(0.01) + 2 * -math.log(0.5)) / 3
    assert result[1] == pytest.approx(share / -math.log(0.01))
    assert result[2] == pytest.approx(2 * share / -math.log(0.01))
    assert result[1] < result[2] < 1.0


def test_update_params_blending():
    assert replica.update_params([0.0, 2.0], [1.0, 4.0], nstep=1) == \
        [0.5, 3.0]
    # higher step -> stays closer to the old coordinates
    assert replica.update_params([0.0], [1.0], nstep=3)[0] == pytest.approx(0.25)


def test_parse_repl_ex(tmp_path):
    # GROMACS prints the values through print_prob(): a "Repl" label
    # token followed by the nrepl-1 pair probabilities
    log = tmp_path / "nvt.log"
    log.write_text(
        "some header\n"
        "Repl  average probabilities:\n"
        "Repl      0   1   2   3\n"
        "Repl    0.10 0.20 0.30\n")
    assert replica.parse_repl_ex(log) == pytest.approx([0.1, 0.2, 0.3])


def test_parse_repl_ex_missing(tmp_path):
    log = tmp_path / "nvt.log"
    log.write_text("no probabilities here\n")
    with pytest.raises(PipelineError):
        replica.parse_repl_ex(log)


def test_do_optimize_step_roundtrip(tmp_path):
    replica.do_init(4, "feprest", basedir=tmp_path)
    log = tmp_path / "nvt.log"
    log.write_text(
        "Repl  average probabilities:\n"
        "Repl      0   1   2   3\n"
        "Repl    0.50 0.50 0.50\n")
    state = replica.do_optimize_step(log, basedir=tmp_path, nstep=1)
    # uniform probabilities keep the uniform ladder; the 50/50 blend
    # with the previous uniform coordinates keeps it uniform too
    assert state.coordinates == pytest.approx([0.0, 1.0, 2.0, 3.0])
