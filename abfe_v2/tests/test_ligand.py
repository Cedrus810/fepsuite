import math

import numpy
import pytest

from feplab.errors import PipelineError
from feplab.ligand import (extract_ligand, ligand_diameter, make_ndx,
                           read_safe_diameter, resurrect_flexible,
                           write_diameter_txt)

from test_topology import NDX, TOP

PDB_HEADER = "CRYST1   40.000   40.000   40.000  90.00  90.00  90.00 P 1           1\n"


def _pdb_line(serial, name, resname, resid, x, y, z):
    return ("ATOM  %5d %-4s %3s  %4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n"
            % (serial, name, resname, resid, x, y, z, name[0]))


def write_system_pdb(path):
    # PDB coordinates are in angstrom (mdtraj converts them to nm).
    # 3 ligand atoms + 2 protein + 6 water + NA + CL = 13 atoms
    lines = [PDB_HEADER]
    coords = [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 3.0, 0.0),
              (10.0, 0.0, 0.0), (10.0, 3.0, 0.0)]
    names = ["C1", "C2", "C3", "CA", "CB"]
    resnames = ["MOL", "MOL", "MOL", "PRO", "PRO"]
    serial = 1
    for name, res, (x, y, z) in zip(names, resnames, coords):
        lines.append(_pdb_line(serial, name, res, 1, x, y, z))
        serial += 1
    for w in range(2):
        lines.append(_pdb_line(serial, "OW", "SOL", 2 + w, 20.0, w * 5.0, 0.0)); serial += 1
        lines.append(_pdb_line(serial, "HW1", "SOL", 2 + w, 21.0, w * 5.0, 0.0)); serial += 1
        lines.append(_pdb_line(serial, "HW2", "SOL", 2 + w, 19.0, w * 5.0, 1.0)); serial += 1
    lines.append(_pdb_line(serial, "NA", "NA", 9, 30.0, 0.0, 0.0)); serial += 1
    lines.append(_pdb_line(serial, "CL", "CL", 10, -30.0, 0.0, 0.0))
    path.write_text("".join(lines))


@pytest.fixture()
def system(tmp_path):
    top = tmp_path / "pp.top"
    top.write_text(TOP)
    ndx = tmp_path / "index.ndx"
    ndx.write_text(NDX)
    pdb = tmp_path / "conf.pdb"
    write_system_pdb(pdb)
    return top, ndx, pdb


def test_extract_ligand(system, tmp_path):
    top, ndx, pdb = system
    out_struct = tmp_path / "ligand.pdb"
    out_top = tmp_path / "ligand.top"
    charge = tmp_path / "totalcharge.txt"
    extract_ligand(topology=top, mol="MOL", structure=pdb, index=ndx,
                   output_ligand_structure=out_struct,
                   output_ligand_topology=out_top, total_charge=charge)
    text = out_top.read_text()
    assert "[ molecules ]" in text
    molecules = text.split("[ molecules ]")[1].split()
    assert "MOL" in molecules and "SOL" not in molecules
    assert charge.read_text().strip() == "0.00000"  # 0.1 - 0.2 + 0.1
    import mdtraj
    trj = mdtraj.load(out_struct)
    assert trj.n_atoms == 3


def test_extract_ligand_index_mismatch(system):
    top, ndx, pdb = system
    bad_ndx = ndx.with_name("bad.ndx")
    bad_ndx.write_text("[ Ligand ]\n1 2 4\n")
    with pytest.raises(PipelineError, match="mismatched"):
        extract_ligand(topology=top, mol="MOL", structure=pdb, index=bad_ndx,
                       output_ligand_structure="x", output_ligand_topology="y",
                       total_charge="z")


def test_resurrect_flexible(system, tmp_path):
    top, ndx, pdb = system
    flexible = tmp_path / "pp_flex.top"
    # "Flexible" water: same atoms but with a bonds section appended.
    sol_block = ("[ moleculetype ]\nSOL 3\n[ atoms ]\n1 OW 1 SOL OW 1 -0.834\n"
                 "2 HW 1 SOL HW1 2 0.417\n3 HW 1 SOL HW2 3 0.417\n")
    flex_block = sol_block + "[ bonds ]\n1 2 1\n1 3 1\n2 3 1\n"
    flexible.write_text(top.read_text().replace(sol_block, flex_block))
    # Remove the second (rigid) SOL from the flexible variant by truncation:
    # build a rigid ligand-only topology for the test instead.
    rigid = tmp_path / "ligand-ion.top"
    rigid.write_text(TOP)
    out = tmp_path / "ligand-ion-flex.top"
    resurrect_flexible(flexible=flexible, topology=rigid, output=out)
    text = out.read_text()
    assert "#ifndef FLEXIBLE" in text
    assert "#else" in text and "#endif" in text
    assert "[ bonds ]" in text  # flexible variant inserted


def test_diameter_roundtrip(tmp_path):
    write_diameter_txt(tmp_path / "diameter.txt", 1.0, 2.0, 2.5)
    assert read_safe_diameter(tmp_path / "diameter.txt") == 2.5
    assert read_safe_diameter(tmp_path / "missing.txt") is None


def test_ligand_diameter_single_frame(system):
    import mdtraj
    top, ndx, pdb = system
    avg, mx, safe = ligand_diameter(structure=pdb, index=ndx, ligand_mol="Ligand")
    # ligand atoms 1-3 at (0,0,0), (0.3,0,0), (0,0.3,0) -> max pairwise 0.3*sqrt(2)
    assert mx == pytest.approx(0.3 * math.sqrt(2), abs=1e-3)
    assert safe == pytest.approx(mx * 1.5)


def test_ligand_diameter_multiframe(system, tmp_path):
    import mdtraj
    top, ndx, pdb = system
    trj = mdtraj.load(pdb)
    far = trj.xyz[0].copy()   # nm
    far[1, 0] += 0.2          # stretch the second ligand atom
    both = mdtraj.Trajectory(numpy.stack([trj.xyz[0], far]), trj.topology)
    xtc = tmp_path / "lig.xtc"
    both.save_xtc(str(xtc))
    avg, mx, safe = ligand_diameter(structure=pdb, trajectory=xtc, index=ndx,
                                    ligand_mol="Ligand")
    assert mx > 0.3 * math.sqrt(2)  # the stretched frame widens the diameter
    assert safe >= mx * 0.99


def test_make_ndx(system, tmp_path):
    top, ndx, pdb = system
    out = tmp_path / "lig.ndx"
    make_ndx(structure=pdb, topology=top, output=out, ligand="MOL", receptor=None)
    text = out.read_text()
    assert "[ System ]" in text and "[ Ligand ]" in text and "[ Ligand_center ]" in text
    assert "[ Receptor ]" not in text
