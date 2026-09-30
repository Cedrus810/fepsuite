from pathlib import Path

import pytest

from feplab.errors import PipelineError
from feplab.mdp import (Mdp, apply_rlist, assemble_product_mdp, compute_rlist,
                        read_mdp_text, set_nsteps, strip_for_traj)

RUN = """
integrator = sd
nsteps = 2000000 ; 4 ns
dt = 0.002
rcoulomb = 1.0
rvdw = 1.0
nstxout-compressed = 10000
dispcorr = EnerPres
ref_t = 300
"""

ANNIH = """
;LRCONLY_BEGIN
nstxout_compressed = 1000
compressed_x_precision = 10000
nstenergy = 1000
;LRCONLY_END

couple_moltype = {group_mol}
"""


def test_read_merge_without():
    run = read_mdp_text(RUN)
    assert run["integrator"] == "sd"
    assert run["ref_t"] == "300"
    add = read_mdp_text("rcoulomb = 1.3\nfree_energy = yes\n")
    merged = run.merged(add)
    assert merged["rcoulomb"] == "1.3"
    assert merged["integrator"] == "sd"
    assert merged["free_energy"] == "yes"
    assert "dispcorr" not in merged.without({"dispcorr"})


def test_get_float_errors():
    mdp = read_mdp_text(RUN)
    assert mdp.get_float("dt") == 0.002
    with pytest.raises(PipelineError):
        mdp.get_float("nonexistent")


def test_strip_for_traj_strip_mode():
    text = RUN + ANNIH
    stripped = strip_for_traj(text, keep_traj=False)
    assert "nstxout_compressed" not in stripped
    assert "compressed_x_precision" not in stripped
    assert "couple_moltype" in stripped
    assert "LRCONLY_BEGIN" not in stripped  # markers removed in strip mode
    assert "nstxout-compressed" not in stripped  # run.mdp compressed line removed


def test_strip_for_traj_keep_mode():
    text = RUN + ANNIH
    kept = strip_for_traj(text, keep_traj=True)
    assert "nstxout_compressed = 1000" in kept
    assert "nstxout-compressed" not in kept  # run.mdp line still removed
    assert "couple_moltype" in kept


def test_strip_without_markers():
    text = RUN + "\ncouple_moltype = MOL\n"
    out = strip_for_traj(text, keep_traj=False)
    assert "nstxout-compressed" not in out  # sed range extends to EOF without markers
    assert "couple_moltype" in out
    assert assemble_product_mdp(RUN, ANNIH, None) == RUN + "\n" + ANNIH  # None = untouched


def test_rlist():
    mdp = read_mdp_text(RUN)
    rlist = compute_rlist(mdp, ligand_diameter=0.0, safe_rlist=None)
    assert rlist == pytest.approx(1.2)
    assert compute_rlist(mdp, 0.0, 2.5) == pytest.approx(2.5)  # safe value wins
    assert compute_rlist(mdp, 1.7, None) == pytest.approx(1.7)  # manual override wins
    apply_rlist(mdp, rlist, None)
    assert mdp["rlist"] == "1.2"
    assert mdp["verlet-buffer-tolerance"] == "-1"
    assert "nstlist" not in mdp
    apply_rlist(mdp, rlist, 25)
    assert mdp["nstlist"] == "25"
    mdp2 = read_mdp_text("integrator = steep\nrvdw = 1.0\n")
    apply_rlist(mdp2, rlist, 25)
    assert "nstlist" not in mdp2  # never set for minimizers


def test_set_nsteps_and_dt():
    mdp = read_mdp_text(RUN)
    set_nsteps(mdp, int(50.0 / mdp.get_float("dt")))
    assert mdp["nsteps"] == "25000"


def test_render_roundtrip(tmp_path):
    mdp = read_mdp_text(RUN)
    mdp["free_energy"] = "yes"
    p = tmp_path / "out.mdp"
    mdp.write(p, header="test")
    again = Mdp.read(p)
    assert again == mdp
    assert p.read_text().splitlines()[0] == "; test"


def test_assemble_with_real_templates():
    """The real template/rundir files must assemble into valid product mdps."""
    root = Path(__file__).resolve().parent.parent
    run_text = (root / "rundir_template/mdp/run.mdp").read_text()
    annih = (root / "template/annihilation.mdp").read_text()
    addenda = (annih + "pull = yes\n").replace("{lambdas_formatted}", "0.0 0.5 1.0")
    addenda = addenda.replace("{group_mol}", "MOL").replace("{lambda_state}", "3")
    text = assemble_product_mdp(run_text, addenda, traj_keep=False)
    mdp = read_mdp_text(text)
    assert mdp["init_lambda_state"] == "3"
    assert mdp["couple_moltype"] == "MOL"
    assert mdp["ref_t"] == "300"
    assert "nstxout-compressed" not in mdp
