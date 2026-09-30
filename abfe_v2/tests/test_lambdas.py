import pytest

from feplab.errors import PipelineError
from feplab.lambdas import (optimize_state_from_exprobs, parse_repl_ex,
                            schedule, update_lambda)

# GROMACS log snippet in the format v1 parses (values line starts with "Repl").
LOG = """
fep-lambdas = 0.0000 0.2500 0.5000 0.7500 1.0000
some other lines
Repl  average probabilities:
Repl        0        1        2        3        4
Repl     0.4120   0.3950   0.3800   0.3600
"""


def test_schedules():
    charging = schedule("charging", 5)
    assert charging[0] == 0.0 and charging[-1] == 1.0
    assert len(charging) == 5
    assert schedule("restraint", 4) == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])
    annih = schedule("annihilation", 5)
    assert annih[0] == pytest.approx(0.0)
    assert annih[-1] == pytest.approx(1.0)
    assert all(b > a for a, b in zip(annih, annih[1:]))  # monotonic
    assert annih[2] == pytest.approx(0.75)  # 1-(0.5-1)^2
    with pytest.raises(RuntimeError):
        schedule("bogus", 3)


def test_parse_repl_ex(tmp_path):
    log = tmp_path / "md.log"
    log.write_text(LOG)
    lambdas, probs = parse_repl_ex(log)
    assert lambdas == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert probs == pytest.approx([0.412, 0.395, 0.38, 0.36])
    with pytest.raises((RuntimeError, FileNotFoundError)):
        parse_repl_ex(tmp_path / "missing.log")


def test_optimize_endpoints_and_monotonic():
    prev = [0.0, 0.25, 0.5, 0.75, 1.0]
    probs = [0.412, 0.395, 0.38, 0.36]
    new = optimize_state_from_exprobs(5, prev, probs)
    assert new[0] == 0.0 and new[-1] == 1.0
    assert all(b > a for a, b in zip(new, new[1:]))
    assert all(0.0 <= v <= 1.0 for v in new)


def test_update_lambda_blend(tmp_path):
    log = tmp_path / "md.log"
    log.write_text(LOG)
    first = update_lambda(5, log, 1)
    second = update_lambda(5, log, 2)
    assert first[0] == 0.0 and first[-1] == 1.0
    # Stronger damping at higher nth: stays closer to the previous schedule.
    prev = [0.0, 0.25, 0.5, 0.75, 1.0]
    assert abs(second[2] - prev[2]) <= abs(first[2] - prev[2])
