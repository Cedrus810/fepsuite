from restlab.heavyhydrogen import turn_heavy
from restlab.waterion import recover_water

TOPOLOGY = """[ moleculetype ]
MOL 2

[ atoms ]
1 CT 1 MOL C1 1 0.100000 12.010000
2 CT 1 MOL H1 2 0.000000 1.008000  CT 0.000000 1.008000 ; a comment

[ moleculetype ]
SOL 2

[ atoms ]
1 OW 1 SOL OW 1 0.000000 16.000000
2 HW 1 SOL HW 2 0.000000 1.008000

[ molecules ]
MOL 1
SOL 1
"""


def test_turn_heavy_replaces_hydrogen_masses(tmp_path):
    top = tmp_path / "top.top"
    top.write_text(TOPOLOGY)
    out = tmp_path / "heavy.top"
    turn_heavy(str(top), str(out))
    text = out.read_text()
    lines = text.splitlines()

    def atom_line(no):
        return next(l for l in lines if l.split()[:1] == [str(no)]
                    and len(l.split()) > 7)

    # MOL hydrogen: both A and B masses buffed to 8.0 (comment dropped, as v1)
    h = atom_line(2)
    assert h.split()[7] == "8.0" and h.split()[10] == "8.0"
    assert "; a comment" not in h
    # carbon untouched
    assert atom_line(1).split()[7] == "12.010000"
    # SOL is ignored by default: water hydrogen keeps 1.008
    sol_h = next(l for l in lines if "HW" in l.split())
    assert "1.008000" in sol_h.split()


def test_recover_water_splices_flexible_water(tmp_path):
    preprocessed = """[ moleculetype ]
MOL 2

[ atoms ]
1 CT 1 MOL C1 1 0.100000 12.010000

[ moleculetype ]
SOL 2

[ atoms ]
1 OW 1 SOL OW 1 -0.834000 16.000000
2 HW 1 SOL HW 2 0.417000 1.008000

[ settles ]
1 1 0.0957 0.1514

[ molecules ]
MOL 1
SOL 1
"""
    top = tmp_path / "pp.top"
    top.write_text(preprocessed)
    out = tmp_path / "light.top"
    recover_water(str(top), str(out), ff="amber")
    text = out.read_text()
    # the SOL body is replaced by the force-field water model (its
    # #ifndef FLEXIBLE / #else branches included) ...
    assert "#else" in text
    assert "0.15139" in text
    # ... the rigid line of the preprocessed copy is gone ...
    assert "1 1 0.0957 0.1514\n" not in text
    # ... without duplicating the [ moleculetype ] header ...
    assert text.count("[ moleculetype ]") == 2
    # ... and the MOL section is untouched.
    assert "1 CT 1 MOL C1 1 0.100000 12.010000" in text


def test_recover_water_custom_model(tmp_path):
    top = tmp_path / "pp.top"
    top.write_text(TOPOLOGY)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "foo.water.itp").write_text(
        "[ moleculetype ]\nWAT 1\n\n[ atoms ]\n1 O 1 WAT O 1 0.0 16.0\n")
    out = tmp_path / "light.top"
    recover_water(str(top), str(out), ff="foo", water_moltype="SOL",
                  water_dir=model_dir)
    text = out.read_text()
    assert "1 O 1 WAT O 1 0.0 16.0" in text
    assert text.count("[ moleculetype ]") == 2
