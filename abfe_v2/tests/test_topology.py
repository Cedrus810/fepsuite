import pytest

from feplab.errors import PipelineError
from feplab.topology import (find_ligand_range, parse_index, parse_top,
                             write_ndx_group)

TOP = """
* synthetic preprocessed topology for tests

[ defaults ]
1 2 yes 0.8333 0.5

[ atomtypes ]
CT 6 12.01 0.0 A 0.35 0.40
OW 8 16.00 -0.834 A 0.315061 0.60
HW 1 1.008 0.417 A 0.0 0.0
NA 11 22.99 1.0 A 0.25 0.1
CL 17 35.45 -1.0 A 0.35 0.1

[ moleculetype ]
MOL 3
[ atoms ]
1 CT 1 MOL C1 1 0.10
2 CT 1 MOL C2 2 -0.20
3 CT 1 MOL C3 3 0.10

[ moleculetype ]
PRO 3
[ atoms ]
1 CT 1 PRO CA 1 0.00
2 CT 1 PRO CB 2 0.00

[ moleculetype ]
SOL 3
[ atoms ]
1 OW 1 SOL OW 1 -0.834
2 HW 1 SOL HW1 2 0.417
3 HW 1 SOL HW2 3 0.417

[ moleculetype ]
NA 3
[ atoms ]
1 NA 1 NA NA 1 1.0

[ moleculetype ]
CL 3
[ atoms ]
1 CL 1 CL CL 1 -1.0

[ system ]
test

[ molecules ]
MOL 1
PRO 1
SOL 2
NA 1
CL 1
"""

NDX = """
[ Ligand ]
1 2 3
[ Receptor ]
4 5
"""


def test_parse_top(tmp_path):
    p = tmp_path / "pp.top"
    p.write_text(TOP)
    top = parse_top(p)
    assert top.defaults[1] == 2  # comb-rule
    assert top.atomtypes["CT"][4] == 0.0  # charge slot
    assert top.atomtypes["OW"][5] == pytest.approx(0.315061)  # sigma slot
    assert top.system == [("MOL", 1), ("PRO", 1), ("SOL", 2), ("NA", 1), ("CL", 1)]
    assert len(top.moleculetypes["MOL"]) == 3
    assert top.moleculetypes["SOL"][0].charge == pytest.approx(-0.834)


def test_parse_top_requires_preprocessed(tmp_path):
    p = tmp_path / "raw.top"
    p.write_text("#include \"ff.ff\"\n" + TOP)
    with pytest.raises(RuntimeError, match="not preprocessed"):
        parse_top(p)


def test_parse_index(tmp_path):
    p = tmp_path / "index.ndx"
    p.write_text(NDX)
    ndx = parse_index(p)
    assert ndx["Ligand"] == [0, 1, 2]
    assert ndx["Receptor"] == [3, 4]
    p2 = tmp_path / "dup.ndx"
    p2.write_text("[ A ]\n1\n[ A ]\n2\n")
    with pytest.raises(RuntimeError):
        parse_index(p2)


def test_find_ligand_range(tmp_path):
    p = tmp_path / "pp.top"
    p.write_text(TOP)
    top = parse_top(p)
    assert find_ligand_range(top, "MOL") == [0, 1, 2]
    with pytest.raises(RuntimeError):
        find_ligand_range(top, "NOPE")
    with pytest.raises(RuntimeError):
        find_ligand_range(top, "SOL")  # count 2 -> unsupported


def test_write_ndx_group(tmp_path):
    p = tmp_path / "out.ndx"
    with open(p, "w") as fh:
        write_ndx_group(fh, "G", [0, 1, 2])
    text = p.read_text()
    assert "[ G ]" in text
    assert "1 2 3" in text  # 1-origin
