import math

import numpy
import pytest

from feplab import charge_correction as cc
from feplab.errors import PipelineError

from test_ligand import PDB_HEADER, _pdb_line
from test_topology import NDX, TOP


@pytest.fixture()
def system(tmp_path):
    top = tmp_path / "pp.top"
    top.write_text(TOP)
    ndx = tmp_path / "index.ndx"
    ndx.write_text(NDX)
    pdb = tmp_path / "conf.pdb"
    lines = [PDB_HEADER]
    coords = [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.0, 0.3, 0.0),
              (1.0, 0.0, 0.0), (1.0, 0.3, 0.0)]
    names = ["C1", "C2", "C3", "CA", "CB"]
    resnames = ["MOL", "MOL", "MOL", "PRO", "PRO"]
    serial = 1
    for name, res, (x, y, z) in zip(names, resnames, coords):
        lines.append(_pdb_line(serial, name, res, 1, x, y, z))
        serial += 1
    for w in range(2):
        lines.append(_pdb_line(serial, "OW", "SOL", 2 + w, 2.0, w * 0.5, 0.0)); serial += 1
        lines.append(_pdb_line(serial, "HW1", "SOL", 2 + w, 2.1, w * 0.5, 0.0)); serial += 1
        lines.append(_pdb_line(serial, "HW2", "SOL", 2 + w, 1.9, w * 0.5, 0.1)); serial += 1
    lines.append(_pdb_line(serial, "NA", "NA", 9, 3.0, 0.0, 0.0)); serial += 1
    lines.append(_pdb_line(serial, "CL", "CL", 9, -3.0, 0.0, 0.0))
    pdb.write_text("".join(lines))
    return top, ndx, pdb


def test_lattice_info(tmp_path):
    pdb = tmp_path / "box.pdb"
    pdb.write_text("CRYST1   50.000   40.000   30.000  90.00  90.00  90.00 P 1           1\n")
    li = cc.LatticeInfo.find_pdb_box(pdb)
    assert (li.a, li.b, li.c) == (5.0, 4.0, 3.0)  # angstrom -> nm
    assert li.get_volume() == pytest.approx(60.0)
    ortho = cc.LatticeInfo(1.0, 1.0, 1.0, 90.0, 90.0, 90.0)
    assert ortho.get_volume() == pytest.approx(1.0)
    nocryst = tmp_path / "nocryst.pdb"
    nocryst.write_text("ATOM      1  C   MOL A   1       0.000   0.000   0.000\n")
    with pytest.raises(PipelineError):
        cc.LatticeInfo.find_pdb_box(nocryst)


def test_build_topology_info(system):
    top, ndx, pdb = system
    info = cc.build_topology_info(str(top), str(ndx), "Ligand", "Receptor")
    assert len(info.sigma) == 13
    assert list(info.mtype[:3]) == [cc.TOP_LIGAND] * 3
    assert list(info.mtype[3:5]) == [cc.TOP_RECEPTOR] * 2
    assert list(info.mtype[5:11]) == [cc.TOP_WATER] * 6
    assert list(info.mtype[11:13]) == [cc.TOP_IONS] * 2
    assert info.guess_water_type() == "tip3p"


def test_quadrupole_and_netcharge(system, tmp_path, monkeypatch):
    # The correction writes fixed-name pqr intermediates into the cwd,
    # like v1; run_sample wraps each sample in its own directory.
    monkeypatch.chdir(tmp_path)
    top, ndx, pdb = system
    info = cc.build_topology_info(str(top), str(ndx), "Ligand", "Receptor")
    corr = cc.RokhlinChargeCorrection(info, str(pdb), 300.0)
    corr.load_guess_initial_params(str(pdb))
    assert corr.QL == pytest.approx(0.0)      # 0.1 - 0.2 + 0.1
    assert corr.QP == pytest.approx(0.0)
    assert corr.QS == pytest.approx(0.0)      # neutralized by NA/CL
    nsol, gamma = corr.calc_quadrupole(str(pdb))
    assert nsol == 2
    assert gamma != 0.0
    assert corr.calc_cubic_coulomb() == pytest.approx(math.pi / 2 - 3.0 * math.log(2.0 + math.sqrt(3.0)))
    assert corr.epsS > 20  # TIP3P dielectric table interpolated at 300 K


def test_save_pqr_and_inp(system, tmp_path, monkeypatch):
    top, ndx, pdb = system
    monkeypatch.chdir(tmp_path)
    info = cc.build_topology_info(str(top), str(ndx), "Ligand", "Receptor")
    caller = cc.APBSCaller(info)
    caller.setup(str(pdb))
    # pro1lig0 shows ligand+receptor (5 atoms), pro0lig1 shows ligand+receptor too,
    # onlylig1 shows only the ligand (3 atoms).
    assert len(open("pro1lig0.pqr").read().strip().splitlines()) == 5
    assert len(open("onlylig1.pqr").read().strip().splitlines()) == 3
    # charges zeroed outside enable_radii: receptor-only file has protein charges
    caller.save_inp(6.0, 0.15, 70.0, 300.0, 0.1369, 0.2513)
    inp = open("apbs.in").read()
    assert "dime" in inp and "sdie 70.0" in inp  # template rendered


def test_average_pbs_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    caller = cc.APBSCaller.__new__(cc.APBSCaller)
    with pytest.raises(PipelineError):
        caller.average_pbs("nonexistent")


def test_average_pbs_parses_dx(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dx = tmp_path / "x-PE0.dx"
    # object header followed by 4 double values
    dx.write_text("object 1 class gridpositions counts 2 2 1\n"
                  "object 3 class array type double rank 0 items 4\n"
                  "1.0 2.0\n3.0 4.0\n")
    caller = cc.APBSCaller.__new__(cc.APBSCaller)
    assert caller.average_pbs("x") == pytest.approx(2.5)
    assert open("x.avg").read().strip() == "2.500000e+00"


def test_charge_sample_serialization():
    sample = cc.ChargeSample(lattice=4.0, xi_LS=-1.4, volume=64.0, eps_s=70.0,
                             q_s=-1.0, q_l=1.0, q_p=0.0, nsol=1000,
                             gamma_s=1.5e-3, ip=10.0, il=-3.0, l_ref=6.0,
                             temp=300.0)
    d = sample.to_dict()
    assert d["xi_LS"] == -1.4 and d["eps_s"] == 70.0
