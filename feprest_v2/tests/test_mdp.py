from conftest import MDP_DIR

from restlab import mdp


def test_strip_fep_on_real_template():
    text = (MDP_DIR / "cg.mdp").read_text()
    stripped = mdp.strip_fep(text)
    assert "%LAMBDA%" not in stripped
    assert "%VDWLAMBDA%" not in stripped
    assert "free-energy = no" in stripped
    assert "fep-lambdas" not in stripped
    # only the token-carrying lines are dropped; everything else survives
    assert "init-lambda-state = 0" in stripped
    assert "integrator = cg" in stripped
    assert "sc-alpha = 0.5" in stripped


def test_strip_fep_replaces_any_free_energy_line():
    # the v1 sed pattern was unanchored: any line mentioning free-energy
    # is replaced by "free-energy = no"
    assert mdp.strip_fep("free-energy = yes\n") == "free-energy = no\n"
    assert mdp.strip_fep("; free-energy remark\n") == "free-energy = no\n"
    assert mdp.strip_fep("a = b\n") == "a = b\n"


def test_strip_fep_on_cginit_and_nvtinit():
    # v1 stage 1: cginit -> minA (has FEP tokens -> stripped);
    # stage 2: nvtinit -> nvtA (has no FEP content at all -> no-op)
    stripped = mdp.strip_fep((MDP_DIR / "cginit.mdp").read_text())
    assert "%LAMBDA%" not in stripped
    assert "free-energy = no" in stripped
    nvt = mdp.strip_fep((MDP_DIR / "nvtinit.mdp").read_text())
    assert nvt == (MDP_DIR / "nvtinit.mdp").read_text()


def test_nptinit_has_no_fep_tokens():
    # v1 stage 3 copies nptinit.mdp verbatim, so it must not carry tokens
    text = (MDP_DIR / "nptinit.mdp").read_text()
    assert mdp.strip_fep(text) == text


def test_set_integrator():
    text = (MDP_DIR / "cginit.mdp").read_text()
    steep = mdp.set_integrator(mdp.strip_fep(text), "steep")
    assert "integrator = steep" in steep
    assert "integrator = cg" not in steep


def test_remove_posres_define():
    text = "define = -DFLEXIBLE -DPOSRES\nnsteps = 10\n"
    # first occurrence per line only (sed s///)
    assert mdp.remove_posres_define(text) == "define = -DFLEXIBLE  \nnsteps = 10\n"
    assert "-DPOSRES" not in mdp.remove_posres_define(text)


def test_topology_defines_posres():
    assert mdp.topology_defines_posres("#ifdef POSRES\n#endif\n")
    assert not mdp.topology_defines_posres("[ atoms ]\n")


def test_cg_to_steep_minimization():
    cg = (MDP_DIR / "cg.mdp").read_text()
    steep = mdp.cg_to_steep_minimization(cg)
    assert "integrator = steep" in steep
    assert "nsteps = 500" in steep
    assert "integrator = cg" not in steep


def test_cg_to_steep_replaces_first_occurrence_only():
    # sed s/cg/steep/ replaces once per line
    assert mdp.cg_to_steep_minimization("cg cg\n") == "steep cg\n"


def test_append_hot_group():
    out = mdp.append_hot_group("a = 1")
    assert out == "a = 1\nenergygrps = hot\nuserint1 = 1\n"


def test_render_lambda_template():
    out = mdp.render_lambda_template(
        "fep-lambdas = %LAMBDA%\nvdw-lambdas = %VDWLAMBDA%\n",
        0.123456789, 0.5)
    assert out == "fep-lambdas = 0.1234568\nvdw-lambdas = 0.5000000\n"


def test_production_temperature():
    text = (MDP_DIR / "run.mdp").read_text()
    assert mdp.production_temperature(text) == 300.0
    assert mdp.production_temperature("ref-t = 310 ; K\n") == 310.0
    assert mdp.production_temperature("ref_t=320\n") == 320.0
