import pytest

from restlab.errors import PipelineError
from restlab.hotregion import (add_underline, molecule_atom_starts,
                               parse_perturbed_atoms, system_perturbed_atoms,
                               write_hot_index)

TOPOLOGY = """[ atomtypes ]
CT  6  12.010000  0.000000  A   0.3399669  0.4577296

[ moleculetype ]
MOL 4

[ atoms ]
1 CT 1 MOL C1 1 -0.100000 12.010000  CT2 0.000000 12.010000
2 CT 1 MOL C2 2 -0.100000 12.010000  CT2 0.000000 12.010000
3 CT 1 MOL C3 3 -0.100000 12.010000
4 CT 1 MOL C4 4 -0.100000 12.010000

[ moleculetype ]
SOL 2

[ atoms ]
1 OW 1 SOL OW 1 0.000000 16.000000
2 HW 1 SOL HW 2 0.000000 1.008000

[ molecules ]
MOL 1
SOL 2
"""

# atoms at 0 / 3 / 9 / 15 A; with a 0.4 nm cutoff only the first two
# residues (AAA, BBB) are "hot" around the two perturbed atoms
PDB = """ATOM      1  C   AAA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  C   BBB A   2       3.000   0.000   0.000  1.00  0.00           C
ATOM      3  C   CCC A   3       9.000   0.000   0.000  1.00  0.00           C
ATOM      4  C   DDD A   4      15.000   0.000   0.000  1.00  0.00           C
TER
END
"""


def test_parse_perturbed_atoms(tmp_path):
    top = tmp_path / "pp.top"
    top.write_text(TOPOLOGY)
    perturbed, composition, natoms = parse_perturbed_atoms(str(top), {"SOL"})
    assert perturbed == [("MOL", 0), ("MOL", 1)]
    assert composition == [("MOL", 1), ("SOL", 2)]
    assert natoms == {"MOL": 4, "SOL": 2}


def test_multiple_molecule_perturbation_rejected(tmp_path):
    top = tmp_path / "pp.top"
    top.write_text(TOPOLOGY)
    perturbed, composition, natoms = parse_perturbed_atoms(str(top), {"SOL"})
    _, starts = molecule_atom_starts(composition, natoms)
    assert starts == {"MOL": [0], "SOL": [4, 6]}
    # MOL listed twice -> perturbing it is refused (v1 behavior)
    composition_multi = [("MOL", 2), ("SOL", 1)]
    _, starts_multi = molecule_atom_starts(composition_multi, natoms)
    with pytest.raises(PipelineError, match="multiple"):
        system_perturbed_atoms(perturbed, starts_multi,
                               ignore_perturbing_multiple_molecules=False)
    # and can be explicitly allowed
    idx = system_perturbed_atoms(perturbed, starts_multi,
                                 ignore_perturbing_multiple_molecules=True)
    assert idx == [0, 1, 4, 5]


def test_add_underline(tmp_path):
    top = tmp_path / "pp.top"
    top.write_text(TOPOLOGY)
    conf = tmp_path / "conf.pdb"
    conf.write_text(PDB)
    out = tmp_path / "underlined.top"
    add_underline(structure=str(conf), topology=str(top), output=str(out),
                  distance=0.4, target_molecule="all",
                  non_perturbed_moleculetype="SOL")
    text = out.read_text()
    # v1's comment line (typo included) records the hot residues
    assert "; undelined resids" in text
    assert "[1, 2]" in text
    lines = text.splitlines()

    def atom_line(no):
        return next(l for l in lines if l.split()[:1] == [str(no)]
                    and len(l.split()) > 7)

    # MOL atoms 1/2 (system indices 0/1, hot residues) are underlined,
    # including their B-state type
    f1 = atom_line(1).split()
    assert f1[1] == "CT_" and f1[8] == "CT2"
    f2 = atom_line(2).split()
    assert f2[1] == "CT_"
    # MOL atoms 3/4 are more than 0.4 nm away: untouched
    assert atom_line(3).split()[1] == "CT"
    assert atom_line(4).split()[1] == "CT"
    # solvent is in the ignore list: untouched
    sol = next(l for l in lines if l.split()[:1] == ["1"]
               and "OW" in l.split())
    assert "OW_" not in sol


def test_write_hot_index(tmp_path):
    underlined = """[ moleculetype ]
MOL 4

[ atoms ]
1 CT_ 1 MOL C1 1 -0.100000 12.010000  CT2 0.000000 12.010000
2 CT_ 1 MOL C2 2 -0.100000 12.010000
3 CT 1 MOL C3 3 -0.100000 12.010000
4 CT 1 MOL C4 4 -0.100000 12.010000

[ moleculetype ]
SOL 2

[ atoms ]
1 OW 1 SOL OW 1 0.000000 16.000000
2 HW 1 SOL HW 2 0.000000 1.008000

[ molecules ]
MOL 1
SOL 2
"""
    top = tmp_path / "underlined.top"
    top.write_text(underlined)
    out = tmp_path / "for_rest.ndx"
    write_hot_index(str(top), str(out))
    text = out.read_text()
    assert "[ System ]" in text and "[ hot ]" in text
    hot = text.split("[ hot ]")[1].split()
    # the two underlined MOL atoms (1-origin), nothing else
    assert hot == ["1", "2"]
    system = text.split("[ System ]")[1].split("[ hot ]")[0].split()
    assert system == [str(i + 1) for i in range(8)]  # 4 + 2*2 atoms
