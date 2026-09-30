import pytest

from restlab.rest2.scaling import Rest2Options, convert_topology

# scale = 300/1200 = 0.25, sqrt(scale) = 0.5
OPTS = Rest2Options(temp0=300.0, temp=1200.0)

UNDERLINED_TOPOLOGY = """[ defaults ]
1 1

[ atomtypes ]
CT  6  12.010000  0.000000  A   0.3399669  0.4577296
DUM 0  0.000000  0.000000  A   0.0000000  0.0000000

[ dihedraltypes ]
CT CT CT CT 9 0.500000 0.000000 3 1.000000 180.00 2

[ moleculetype ]
MOL 4

[ atoms ]
1 CT 1 MOL C1 1 0.000000 12.010000
2 CT_ 1 MOL C2 2 0.100000 12.010000  DUM 0.000000 12.010000
3 CT 1 MOL C3 3 -0.200000 12.010000  DUM 0.500000 12.010000
4 CT_ 1 MOL C4 4 0.300000 12.010000

[ dihedrals ]
1 2 3 4 9 0.500000 0.000000 3 1.000000 180.00 2

[ molecules ]
MOL 1
"""


def test_scaled_atomtype_emitted(tmp_path):
    src = tmp_path / "pp.top"
    src.write_text(UNDERLINED_TOPOLOGY)
    out = tmp_path / "out.top"
    convert_topology(str(src), str(out), OPTS)
    text = out.read_text()
    # the underlined atomtypes are duplicated with scaled LJ/charge,
    # followed by the original line ('%4s' right-aligns the name)
    lines = [l for l in text.splitlines() if l.split()[:1] == ["CT_"]]
    assert len(lines) == 1
    fields = lines[0].split()
    # atomtype bondtype atomic-number mass charge particle c6 c12
    assert fields[0] == "CT_" and fields[1] == "CT_"
    assert float(fields[4]) == pytest.approx(0.0)  # charge * sqrt(scale)
    assert float(fields[6]) == pytest.approx(0.3399669 * 0.25)
    assert float(fields[7]) == pytest.approx(0.4577296 * 0.25)
    assert "; scaled" in lines[0]
    # and the original atomtypes survive
    assert any(l.split()[:1] == ["CT"] and "scaled" not in l
               for l in text.splitlines())


def test_dihedraltypes_suppressed(tmp_path):
    src = tmp_path / "pp.top"
    src.write_text(UNDERLINED_TOPOLOGY)
    out = tmp_path / "out.top"
    convert_topology(str(src), str(out), OPTS)
    text = out.read_text()
    assert "[ dihedraltypes ]" in text  # header kept
    assert "CT CT CT CT 9" not in text  # entries suppressed (v1 behavior)


def test_scaled_atom_charge_and_suffix(tmp_path):
    src = tmp_path / "pp.top"
    src.write_text(UNDERLINED_TOPOLOGY)
    out = tmp_path / "out.top"
    convert_topology(str(src), str(out), OPTS)
    text = out.read_text()
    # atom 2: CT_ (scaled) with B-state DUM.
    # unify_charge (lambda 0? none given -> charge unchanged), then
    # charge scaled by sqrt(0.25)=0.5 -> 0.1 * 0.5 = 0.05
    atoms = [l for l in text.splitlines() if l.split()[:1] == ["2"]]
    assert len(atoms) == 1
    fields = atoms[0].split()
    assert fields[1] == "CT_"
    assert float(fields[6]) == pytest.approx(0.05)
    # B-state atomtype gains the suffix, its charge is scaled too
    assert fields[8] == "DUM_"
    assert float(fields[9]) == pytest.approx(0.0)
    # masses unified to the maximum (both 12.01 here)
    assert float(fields[7]) == pytest.approx(12.01)
    assert float(fields[10]) == pytest.approx(12.01)
    # non-scaled atom (has a B-state, no underline) keeps its atomtype
    atoms3 = [l for l in text.splitlines() if l.split()[:1] == ["3"]]
    assert atoms3[0].split()[1] == "CT"
    assert atoms3[0].split()[8] == "DUM"  # B-state not suffixed here
    # the other hot end atom (no B-state): scaled charge, no sqrt on mass
    atoms4 = [l for l in text.splitlines() if l.split()[:1] == ["4"]]
    assert atoms4[0].split()[1] == "CT_"
    assert float(atoms4[0].split()[6]) == pytest.approx(0.15)


def test_unify_charge_interpolates(tmp_path):
    opts = Rest2Options(temp0=300.0, temp=1200.0, unify_charge=True,
                        charge_lambda=0.3)
    src = tmp_path / "pp.top"
    src.write_text(UNDERLINED_TOPOLOGY)
    out = tmp_path / "out.top"
    convert_topology(str(src), str(out), opts)
    text = out.read_text()
    atoms = {l.split()[0]: l.split() for l in text.splitlines()
             if l.split()[:1] and l.split()[0].isdigit() and
             len(l.split()) >= 8 and l.split()[4].startswith("C")}
    # atom 2 (scaled): qA=0.1, qB=0.0 -> unified 0.07, then *0.5 -> 0.035
    assert float(atoms["2"][6]) == pytest.approx(0.035)
    assert float(atoms["2"][9]) == pytest.approx(0.035)
    # atom 3 (not scaled): qA=-0.2, qB=0.5 -> unified 0.01, no sqrt
    assert float(atoms["3"][6]) == pytest.approx(0.01)
    assert float(atoms["3"][9]) == pytest.approx(0.01)


def test_dihedral_scaling(tmp_path):
    src = tmp_path / "pp.top"
    src.write_text(UNDERLINED_TOPOLOGY)
    out = tmp_path / "out.top"
    convert_topology(str(src), str(out), OPTS)
    text = out.read_text()
    dihed = [l for l in text.splitlines()
             if l.startswith("    1     2     3     4")]
    # fn=9 with explicit params, one line with both phases:
    # (phi0, k, mult) x2; the force constant (index 1) is scaled by 0.5,
    # the periodicity stays an integer
    assert len(dihed) == 1
    fields = dihed[0].split()
    assert float(fields[5]) == pytest.approx(0.5)    # phi0 unscaled
    assert float(fields[6]) == pytest.approx(0.0)    # k (already zero)
    assert fields[7] == "3"                          # multiplicity integer
    assert float(fields[8]) == pytest.approx(1.0)    # phi0 of phase 2
    assert float(fields[9]) == pytest.approx(90.0)   # k 180 * 0.5
    assert fields[10] == "2"


def test_peptide_bond_exclusion(tmp_path, capsys):
    topology = """[ defaults ]
1 1

[ atomtypes ]
CT  6  12.010000  0.000000  A   0.3399669  0.4577296

[ moleculetype ]
PEP 4

[ atoms ]
1 CT_ 1 ALA O 1 0.000000 12.010000
2 CT 1 ALA C 2 0.000000 12.010000
3 CT 1 ALA N 3 0.000000 12.010000
4 CT 1 ALA H 4 0.000000 12.010000

[ dihedrals ]
1 2 3 4 1 0.000000 0.500000 3

[ molecules ]
PEP 1
"""
    src = tmp_path / "pp.top"
    src.write_text(topology)
    out = tmp_path / "out.top"
    opts = Rest2Options(temp0=300.0, temp=1200.0)
    convert_topology(str(src), str(out), opts)
    dihed = [l for l in out.read_text().splitlines()
             if l.startswith("    1     2     3     4")]
    fields = dihed[0].split()
    # O-C-N-H main chain dihedral of a non-proline residue: the force
    # constant (index 1) is NOT scaled
    assert float(fields[6]) == pytest.approx(0.5)
    assert "Prevented peptide bond scaling" in capsys.readouterr().err


def test_no_peptide_exclusion_when_disabled(tmp_path, capsys):
    topology = """[ defaults ]
1 1

[ atomtypes ]
CT  6  12.010000  0.000000  A   0.3399669  0.4577296

[ moleculetype ]
PEP 4

[ atoms ]
1 CT_ 1 ALA O 1 0.000000 12.010000
2 CT 1 ALA C 2 0.000000 12.010000
3 CT 1 ALA N 3 0.000000 12.010000
4 CT 1 ALA H 4 0.000000 12.010000

[ dihedrals ]
1 2 3 4 1 0.000000 0.500000 3

[ molecules ]
PEP 1
"""
    src = tmp_path / "pp.top"
    src.write_text(topology)
    out = tmp_path / "out.top"
    opts = Rest2Options(temp0=300.0, temp=1200.0, exclude_peptide=False)
    convert_topology(str(src), str(out), opts)
    dihed = [l for l in out.read_text().splitlines()
             if l.startswith("    1     2     3     4")]
    # force constant (index 1) gets the sqrt scale when not excluded
    assert float(dihed[0].split()[6]) == pytest.approx(0.25)


def test_unpreprocessed_toplogy_rejected(tmp_path):
    src = tmp_path / "raw.top"
    src.write_text("#include \"ff.itp\"\n")
    with pytest.raises(RuntimeError, match="not preprocessed"):
        convert_topology(str(src), str(tmp_path / "out.top"), OPTS)


def test_unify_charge_requires_lambda(tmp_path):
    opts = Rest2Options(unify_charge=True)
    with pytest.raises(RuntimeError, match="charge_lambda"):
        convert_topology(str(tmp_path / "missing.top"),
                         str(tmp_path / "out.top"), opts)
