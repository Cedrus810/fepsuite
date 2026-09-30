import random

import pytest

from restlab.neutralize import WaterModel, neutralize_topology, rewrite_gro
from restlab.waterion import WATER_ION_MODELS_DIR

# dtot = -1: one water of the last SOL block becomes SOL2pos
TOPOLOGY = """[ moleculetype ]
MOL 2

[ atoms ]
1 CT 1 MOL C1 1 0.000000 12.010000  CT2 -1.000000 12.010000
2 CT 1 MOL C2 2 0.000000 12.010000

[ moleculetype ]
SOL 3

[ atoms ]
1 OW 1 SOL OW 1 0.000000 16.000000
2 HW 1 SOL HW 2 0.000000 1.008000
3 HW 1 SOL HW 3 0.000000 1.008000

[ system ]
whatever

[ molecules ]
MOL 1
SOL 5
"""

NA_MOLTYPE = ("[ moleculetype ]\nNA 1\n\n[ atoms ]\n"
              "1 NA 1 NA NA 1 1.000000 22.990000\n\n")


def water_model():
    return WaterModel(str(WATER_ION_MODELS_DIR / "amber.ion.itp"))


def gro_line(resid, resname, name, no, x, y, z):
    return "%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n" % (resid, resname, name, no, x, y, z)


def make_gro(title, lines, box=(2.0, 2.0, 2.0)):
    return (title + "\n" + "%5d\n" % len(lines)
            + "".join(lines)
            + "%8.5f%8.5f%8.5f\n" % box)


def test_water_model_parses_shipped_amber_ion_itp():
    wm = water_model()
    assert wm.natom == 3
    assert wm.posresname == "NA" and wm.negresname == "CL"
    assert len(wm.atomnames) == 3
    assert len(wm.coord) == 3 and len(wm.coord[0]) == 3
    assert wm.contents  # the itp body is kept for splicing into output


def test_neutralize_water_to_positive_ion(tmp_path):
    top = tmp_path / "pp.top"
    top.write_text(TOPOLOGY)
    out = tmp_path / "neut.top"
    wm = water_model()
    plan = neutralize_topology(str(top), str(out), wm, mode="auto")
    assert plan.dtot == pytest.approx(-1.0)
    assert plan.nchg == 1
    assert plan.exchange_mol == 1  # the last SOL block ([MOL 1, SOL 5])
    assert plan.fepmol == "SOL2pos" and plan.tomol == "NA"
    text = out.read_text()
    # the hybrid ion itp is spliced in before [ system ]
    assert "; SOLINFO 3  NA CL" in text
    # the last SOL block shrinks by one and gains the hybrid molecule
    assert "MOL     1\n" in text
    assert "SOL     4\n" in text
    assert "SOL2pos     1\n" in text


def test_neutralize_posonly_mode(tmp_path):
    topology = TOPOLOGY.replace(
        "CT2 -1.000000", "CT2 1.000000").replace(
        "MOL 1\nSOL 5", "MOL 1\nNA 3\nSOL 5").replace(
        "[ system ]", NA_MOLTYPE + "[ system ]")
    top = tmp_path / "pp.top"
    top.write_text(topology)
    out = tmp_path / "neut.top"
    wm = water_model()
    plan = neutralize_topology(str(top), str(out), wm, mode="posonly")
    assert plan.dtot == pytest.approx(1.0)
    assert plan.frommol == "NA" and plan.nfrom == 1
    assert plan.fepmol == "pos2SOL" and plan.tomol == "SOL"
    text = out.read_text()
    assert "NA     2\n" in text
    assert "pos2SOL     1\n" in text


def test_noninteger_charge_rejected(tmp_path):
    topology = TOPOLOGY.replace("CT2 -1.000000", "CT2 -0.500000")
    top = tmp_path / "pp.top"
    top.write_text(topology)
    with pytest.raises(RuntimeError, match="Noninteger"):
        neutralize_topology(str(top), str(tmp_path / "o.top"),
                            water_model(), mode="auto")


def test_missing_solvent_block_rejected(tmp_path):
    topology = TOPOLOGY.replace("MOL 1\nSOL 5", "MOL 1")
    top = tmp_path / "pp.top"
    top.write_text(topology)
    with pytest.raises(RuntimeError, match="Unable to find"):
        neutralize_topology(str(top), str(tmp_path / "o.top"),
                            water_model(), mode="auto")


def test_no_perturbation_leaves_molecules_alone(tmp_path):
    topology = TOPOLOGY.replace("CT2 -1.000000", "CT2 0.000000")
    top = tmp_path / "pp.top"
    top.write_text(topology)
    plan = neutralize_topology(str(top), str(tmp_path / "o.top"),
                               water_model(), mode="auto")
    assert plan.nchg == 0 and plan.exchange_mol is None


def test_rewrite_gro_water_to_ion_keeps_coordinates(tmp_path):
    # nfrom == nto (3): a water -> hybrid-ion rewrite is a topology-only
    # change; the chosen water's coordinates stay in place.  The SOL
    # block count must agree with the gro (v1 derives the atom ranges
    # of the exchange block from the [ molecules ] table).
    topology = TOPOLOGY.replace("MOL 1\nSOL 5", "MOL 1\nSOL 1")
    top = tmp_path / "pp.top"
    top.write_text(topology)
    out_top = tmp_path / "neut.top"
    wm = water_model()
    plan = neutralize_topology(str(top), str(out_top), wm, mode="auto")
    assert plan.exchange_mol == 1
    gro = make_gro("tiny system", [
        gro_line(1, "MOL", "C1", 1, 0.000, 0.000, 0.000),
        gro_line(1, "MOL", "C2", 2, 0.100, 0.000, 0.000),
        gro_line(2, "SOL", "OW", 3, 0.300, 0.000, 0.000),
        gro_line(2, "SOL", "HW1", 4, 0.400, 0.000, 0.000),
        gro_line(2, "SOL", "HW2", 5, 0.200, 0.100, 0.000),
    ])
    gro_path = tmp_path / "npt.gro"
    gro_path.write_text(gro)
    out_gro = tmp_path / "neut.gro"
    rewrite_gro(str(gro_path), str(out_gro), plan, wm, rng=random.Random(1))
    assert out_gro.read_text() == gro


def test_rewrite_gro_posonly_expands_ion_to_water(tmp_path):
    topology = TOPOLOGY.replace(
        "CT2 -1.000000", "CT2 1.000000").replace(
        "MOL 1\nSOL 5", "MOL 1\nNA 1\nSOL 5").replace(
        "[ system ]", NA_MOLTYPE + "[ system ]")
    top = tmp_path / "pp.top"
    top.write_text(topology)
    out_top = tmp_path / "neut.top"
    wm = water_model()
    plan = neutralize_topology(str(top), str(out_top), wm, mode="posonly")
    assert plan.nfrom == 1 and plan.nto == 3
    # the ion sits within exclude_distance of the solute, the bulk SOL
    # block (0.9 nm away) does not
    gro = make_gro("posonly", [
        gro_line(1, "MOL", "C1", 1, 0.000, 0.000, 0.000),
        gro_line(1, "MOL", "C2", 2, 0.100, 0.000, 0.000),
        gro_line(2, "NA", "NA", 3, 0.300, 0.000, 0.000),
        gro_line(3, "SOL", "OW", 4, 0.900, 0.000, 0.000),
        gro_line(3, "SOL", "HW1", 5, 1.000, 0.000, 0.000),
        gro_line(3, "SOL", "HW2", 6, 0.800, 0.100, 0.000),
    ])
    gro_path = tmp_path / "npt.gro"
    gro_path.write_text(gro)
    out_gro = tmp_path / "neut.gro"
    rewrite_gro(str(gro_path), str(out_gro), plan, wm, rng=random.Random(1))
    lines = out_gro.read_text().splitlines()
    assert lines[1].strip() == "8"  # 6 atoms + (3 - 1)
    # layout: title, natom, 2 MOL atoms, 3 expanded SOL, 3 bulk SOL, box
    expanded = lines[4:7]
    assert all(l[5:10] == "SOL  " for l in expanded)
    assert float(expanded[0][20:28]) == pytest.approx(0.300)
    assert float(expanded[0][28:36]) == pytest.approx(0.000)
    # the untouched SOL block follows, then the box
    assert "    3SOL" in lines[7] and "OW" in lines[7]
    assert lines[-1].split() == ["2.00000", "2.00000", "2.00000"]
