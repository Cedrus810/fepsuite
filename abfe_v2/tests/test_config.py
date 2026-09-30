import pytest

from feplab.config import Config, ConfigError, parse_para_conf

SAMPLE = """
# comment
NRESTR=4
NCHARGE = 12
NANNIH=12
LIG_PARA=2
COMPLEX_PARA=12
TPP=1
LIG_GMX="MOL"
RECEPTOR_MDTRAJ='protein'
IONIC_STRENGTH=0.150
RUN_PROD=2000 # trailing comment
APBS=apbs
"""


def test_parse_basic(tmp_path):
    p = tmp_path / "para_conf.zsh"
    p.write_text(SAMPLE)
    values = parse_para_conf(p)
    assert values["NRESTR"] == "4"
    assert values["NCHARGE"] == "12"
    assert values["LIG_GMX"] == "MOL"
    assert values["RECEPTOR_MDTRAJ"] == "protein"
    assert values["IONIC_STRENGTH"] == "0.150"
    assert values["RUN_PROD"] == "2000"
    assert "APBS" in values


def test_parse_rejects(tmp_path):
    bad = tmp_path / "bad.zsh"
    bad.write_text("FOO=(1 2 3)\n")
    with pytest.raises(ConfigError):
        parse_para_conf(bad)
    bad.write_text("FOO=$(date)\n")
    with pytest.raises(ConfigError):
        parse_para_conf(bad)
    bad.write_text("FOO=${BAR}\n")
    with pytest.raises(ConfigError):
        parse_para_conf(bad)
    bad.write_text("this is not an assignment\n")
    with pytest.raises(ConfigError):
        parse_para_conf(bad)


def test_config_load_override(tmp_path):
    base = tmp_path / "para_conf.zsh"
    base.write_text(SAMPLE)
    (tmp_path / "mol1").mkdir()
    per_id = tmp_path / "mol1" / "para_conf.zsh"
    per_id.write_text('NCHARGE=8\nLIG_GMX="LIG"\n')

    cfg = Config.load(tmp_path, "mol1")
    assert cfg.get_int("NCHARGE") == 8  # per-ID wins
    assert cfg.get_int("NRESTR") == 4   # base kept
    assert cfg.lig_gmx == "LIG"
    assert cfg.complex_para == 12
    assert cfg.tpp == 1
    assert cfg.run_prod == 2000.0
    assert cfg.ionic_strength == 0.150
    assert cfg.nrestr == 4 and cfg.nannih == 12


def test_config_defaults_and_missing():
    cfg = Config({}, sources=[])
    assert cfg.lig_gmx == "MOL"
    assert cfg.receptor_sel == "protein"
    assert cfg.eq_rmsd_cutoff == 0.4
    assert cfg.ligand_diameter == 0.0
    assert cfg.annih_lambda_opt == 5
    assert cfg.annih_lambda_opt_length == 50.0
    assert cfg.water_structure == "spc216"
    assert cfg.water_thickness == 1.0
    assert cfg.ion_positive == "NA" and cfg.ion_negative == "CL"
    assert cfg.solvent_name == "SOL"
    assert cfg.nstlist is None
    with pytest.raises(ConfigError):
        cfg.get_int("LIG_PARA")


def test_validate():
    cfg = Config({"LIG_PARA": "1", "COMPLEX_PARA": "1", "NCHARGE": "1"}, sources=[])
    with pytest.raises(ConfigError):
        cfg.validate()
    cfg = Config({"LIG_PARA": "1", "COMPLEX_PARA": "1", "NCHARGE": "12",
                  "ANNIH_LAMBDA_OPT": "0"}, sources=[])
    with pytest.raises(ConfigError):
        cfg.validate()
    cfg = Config({"LIG_PARA": "1", "COMPLEX_PARA": "1"}, sources=[])
    cfg.validate()  # defaults pass
