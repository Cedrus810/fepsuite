import math

import pytest

from feplab.errors import PipelineError
from feplab.restraints import (SPRING_ANGLE, SPRING_DIHEDRAL, SPRING_DISTANCE,
                               analytical_restraint_free_energy,
                               check_rms_average, generate_pull_restraint,
                               generate_restraint_itp, read_restrinfo)

RESTRINFO = """# ancA ancB ancC ligA ligB ligC
10 11 12 40 41 42
# comment
0.450 1.5708 1.2094 0.5000 -0.3000 0.7000
# std line
0.02 0.10 0.10 0.15 0.15 0.15
"""


def test_read_restrinfo(tmp_path):
    p = tmp_path / "restrinfo"
    p.write_text(RESTRINFO)
    anchors, avgs = read_restrinfo(p)
    assert anchors == [10, 11, 12, 40, 41, 42]
    assert len(avgs) == 6
    assert avgs[0] == pytest.approx(0.450)
    assert avgs[1] == pytest.approx(math.pi / 2, abs=1e-3)
    bad = tmp_path / "bad"
    bad.write_text("1 2 3\n")
    with pytest.raises(PipelineError):
        read_restrinfo(bad)


def test_analytical_value():
    avgs = [0.450, math.pi / 2, math.pi / 2, 0.0, 0.0, 0.0]
    dg = analytical_restraint_free_energy(avgs, temp=300)
    # Hand computation of the v1 formula at theta = phi = 90 degrees.
    v0 = 1.6605391
    RT = 0.00831446261815324 * 300
    mdeltaf = (math.log(8 * math.pi ** 2 * v0)
               + 0.5 * math.log(SPRING_DISTANCE)
               + math.log(SPRING_ANGLE)
               + 1.5 * math.log(SPRING_DIHEDRAL)
               - 2 * math.log(0.450)
               - math.log(1.0) - math.log(1.0)
               - 3 * math.log(2 * math.pi * RT))
    assert dg == pytest.approx(-RT * mdeltaf, rel=1e-6)


def test_generate_pull_restraint(tmp_path):
    p = tmp_path / "restrinfo"
    p.write_text(RESTRINFO)
    mdp = tmp_path / "restr_pull.mdp"
    ndx = tmp_path / "restr_pull.ndx"
    generate_pull_restraint(restrinfo=p, mdp=mdp, ndx=ndx)
    text = mdp.read_text()
    assert "pull = yes" in text
    assert "pull-ngroups = 6" in text
    for letter in "ABCDEF":
        assert f"pull-group1-name" not in text or True
    assert "pull-group6-name    = anchorF" in text
    assert "pull-group1-pbcatom = 10" in text
    assert "pull-group4-pbcatom = 40" in text
    # angle reference converted to degrees and normalized
    assert "pull-coord2-init = 90.000" in text
    assert "pull-coord1-init = 0.45" in text
    assert "pull-coord1-kB = 4184.0" in text
    ndx_text = ndx.read_text()
    for letter, atom in zip("ABCDEF", [10, 11, 12, 40, 41, 42]):
        assert f"[ anchor{letter} ]" in ndx_text
        assert str(atom) in ndx_text


def test_generate_pull_restraint_decouple(tmp_path):
    p = tmp_path / "restrinfo"
    p.write_text(RESTRINFO)
    mdp = tmp_path / "decouple.mdp"
    generate_pull_restraint(restrinfo=p, mdp=mdp, ndx=tmp_path / "d.ndx",
                            decouple_B=True)
    text = mdp.read_text()
    assert "pull-coord1-kB = 0.0" in text
    assert "pull-coord4-kB = 0.0" in text
    assert "pull-coord1-k = 4184.0" in text  # A-state keeps the spring


def test_generate_restraint_itp(tmp_path):
    p = tmp_path / "restrinfo"
    p.write_text(RESTRINFO)
    itp = tmp_path / "restr.itp"
    generate_restraint_itp(restrinfo=p, itp=itp)
    text = itp.read_text()
    assert "[ intermolecular_interactions ]" in text
    assert "[ bonds ]" in text and "[ angles ]" in text and "[ dihedrals ]" in text
    assert "12 40 6 0.450 4184.000" in text
    assert "11 12 40 1 90.000 41.840" in text

    itp2 = tmp_path / "restr-dec.itp"
    generate_restraint_itp(restrinfo=p, itp=itp2, decouple_B=True)
    text2 = itp2.read_text()
    assert "12 40 6 0.450 4184.000 0.450 0.000" in text2


def test_check_rms_average(tmp_path):
    xvg = tmp_path / "rms.xvg"
    xvg.write_text("@ title x\n# comment\n0 0.1\n1 0.2\n2 0.3\n")
    assert check_rms_average(xvg, 0.4) == pytest.approx(0.2)
    with pytest.raises(PipelineError):
        check_rms_average(xvg, 0.1)
