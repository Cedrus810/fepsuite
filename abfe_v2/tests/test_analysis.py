import json
import math

import pytest

from feplab.analysis import (build_report, combine_charge_samples, read_bar_log,
                             read_charge_samples, read_lrc_result)
from feplab.restraints import analytical_restraint_free_energy
from feplab.units import EPS0, KCAL_PER_KJ

BAR = {"charging-lig": (10.0, 0.5), "charging-complex": (-8.0, 0.4),
       "annihilation-lig": (20.0, 1.0), "annihilation-complex": (-18.0, 1.2),
       "restraint": (5.0, 0.3)}
LRC = {"lr-lig": (2.0, 0.1), "lr-annihilation-lig": (1.0, 0.05),
       "lr-complex": (3.0, 0.2), "lr-annihilation-complex": (1.5, 0.1)}

BAR_LOG = """some header
total  0  1  2  3  {energy:12.5f}  0.9  {error:8.5f}  0.02
"""

SAMPLE = {"lattice": 4.0, "xi_LS": -1.4, "volume": 64.0, "eps_s": 70.0,
          "q_s": -1.0, "q_l": 1.0, "q_p": 0.0, "nsol": 1000,
          "gamma_s": 1.5e-3, "ip": 10.0, "il": -3.0, "l_ref": 6.0, "temp": 300.0}

RESTRINFO = """# anchors
10 11 12 40 41 42
# avgs
0.450 1.5708 1.5708 0.5000 0.2000 -0.3000
# std
0.02 0.10 0.10 0.15 0.15 0.15
"""


def test_read_bar_log(tmp_path):
    p = tmp_path / "x.bar.log"
    p.write_text(BAR_LOG.format(energy=-45.678, error=1.234))
    energy, error = read_bar_log(p)
    assert energy == pytest.approx(-45.678)
    assert error == pytest.approx(1.234)
    # last 'total' line wins
    with open(p, "a") as fh:
        fh.write(BAR_LOG.format(energy=-40.0, error=2.0))
    assert read_bar_log(p)[0] == pytest.approx(-40.0)
    empty = tmp_path / "empty.log"
    empty.write_text("nothing here\n")
    with pytest.raises(Exception):
        read_bar_log(empty)


def test_read_lrc_result(tmp_path):
    p = tmp_path / "x.lrc.txt"
    p.write_text("0 1000 1.1000\n1000 2000 1.3000\n1.2345\t0.0521\n")
    mean, std = read_lrc_result(p)
    assert mean == pytest.approx(1.2345)
    assert std == pytest.approx(0.0521)


def test_combine_charge_samples_manual():
    mean, var = combine_charge_samples([SAMPLE, dict(SAMPLE, ip=12.0)])
    # manual computation for one sample
    eps0 = EPS0
    qi = SAMPLE["q_s"]
    qf = qi - SAMPLE["q_l"]
    dgnet = -SAMPLE["xi_LS"] / (8 * math.pi * eps0 * SAMPLE["eps_s"]) \
        * (qf ** 2 - qi ** 2) / SAMPLE["lattice"]
    ip_v = SAMPLE["ip"] / SAMPLE["volume"]
    il_v = SAMPLE["il"] / SAMPLE["volume"]
    dgrip = ip_v * qf - (ip_v + il_v) * qi
    dgdsc = -SAMPLE["gamma_s"] * SAMPLE["nsol"] / (6 * eps0 * SAMPLE["volume"]) * (qf - qi)
    expect1 = dgnet + dgrip + dgdsc
    assert len(combine_charge_samples([])) == 2
    assert combine_charge_samples([]) == (0.0, 0.0)
    vals = [expect1, expect1 + (2.0 / SAMPLE["volume"]) * (qf - qi)]  # ip +2 shifts dgrip
    m = sum(vals) / 2
    v = sum((x - m) ** 2 for x in vals)
    assert mean == pytest.approx(m)
    assert var == pytest.approx(v)


def _make_basedir(tmp_path, with_cc=True):
    basedir = tmp_path / "basedir"
    basedir.mkdir()
    for key, (energy, error) in BAR.items():
        (basedir / f"{key}.bar.log").write_text(BAR_LOG.format(energy=energy, error=error))
    for key, (mean, std) in LRC.items():
        (basedir / f"{key}.lrc.txt").write_text("0 100 1.0\n%.4f\t%.4f\n" % (mean, std))
    (basedir / "restrinfo").write_text(RESTRINFO)
    ccdir = basedir / "charge-correction"
    ccdir.mkdir()
    if with_cc:
        (ccdir / "complex.json").write_text(json.dumps([SAMPLE]))
        (ccdir / "ligand.json").write_text("[]")
    return basedir


def test_build_report_structure(tmp_path):
    basedir = _make_basedir(tmp_path)
    report = build_report(basedir=basedir, restrinfo=basedir / "restrinfo", temp=300.0)
    lines = report.strip().splitlines()
    # 11 individual terms + separator + 4 subtotals + separator + total
    names = [l.split()[0] for l in lines[:lines.index("----")]]
    assert set(names) == {"charging-lig", "charging-complex", "annihilation-lig",
                          "annihilation-complex", "restraint", "restraint-analytical",
                          "lr-lig", "lr-annihilation-lig", "lr-complex",
                          "lr-annihilation-complex", "charge-correction-complex",
                          "charge-correction-ligand"}
    middle = lines[lines.index("----") + 1:lines.index("----", lines.index("----") + 1)]
    assert [l.split()[0] for l in middle] == ["annihilation", "charging",
                                              "long-range-correction", "restraint"]
    assert lines[-1].startswith("total ")


def test_build_report_total(tmp_path):
    basedir = _make_basedir(tmp_path)
    report = build_report(basedir=basedir, restrinfo=basedir / "restrinfo", temp=300.0)
    lines = report.strip().splitlines()
    total = float(lines[-1].split()[1])

    # Independently recompute the expected total from the module functions.
    individual = 0.0
    individual += 10.0                      # charging-lig
    individual += 8.0                       # -(-8)
    individual += 20.0                      # annihilation-lig
    individual += 18.0                      # -(-18)
    individual += 5.0                       # restraint bar
    individual += -2.0 + 1.0 + 3.0 - 1.5    # lrc with signs
    avgs = [float(x) for x in RESTRINFO.splitlines()[3].split()]
    analytical = analytical_restraint_free_energy(avgs, temp=300.0)
    individual += -analytical               # restraint-analytical
    cc_mean, _ = combine_charge_samples([SAMPLE])
    individual += -cc_mean                  # charge-correction-complex
    assert total == pytest.approx(KCAL_PER_KJ * individual, abs=5e-4)
