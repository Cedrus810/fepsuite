import pytest

from restlab.config import Config, parse_para_conf
from restlab.errors import ConfigError


def write_conf(tmp_path, text):
    p = tmp_path / "para_conf.zsh"
    p.write_text(text)
    return Config.load(tmp_path)


def test_parse_basic(tmp_path):
    cfg = write_conf(tmp_path, """
# comment
BASECONF=conf_ionized.pdb
BASETOP=topol_ionized.top
NREP=32
PARA=8
TPP=1
CHARGE=auto
FF=amber
REST2_TEMP=1200
REFINIT=conf_ionized.pdb
REFCRD=
SIMLENGTH=4000 # inline comment
LIG_GMX="MOL"
""")
    assert cfg.baseconf == "conf_ionized.pdb"
    assert cfg.basetop == "topol_ionized.top"
    assert cfg.nrep == 32
    assert cfg.para == 8
    assert cfg.charge_mode == "auto"
    assert cfg.ff == "amber"
    assert cfg.rest2_temp == 1200.0
    assert cfg.refinit == "conf_ionized.pdb"
    assert cfg.refcrd is None  # empty value -> no restraints
    assert cfg.simlength == 4000.0
    assert cfg.get_str("LIG_GMX") == "MOL"
    cfg.validate()


def test_defaults():
    cfg = Config({}, sources=[])
    assert cfg.nrep == 32
    assert cfg.ntune == 5
    assert cfg.replica_interval == 1000
    assert cfg.sampling_interval == 100
    assert cfg.basewarn == 1
    assert cfg.domain_shrink == 0.6
    assert cfg.rest2_region_distance == 0.4
    assert cfg.run_tpr_parallel
    assert cfg.repopt_extend_run_length == 50.0
    assert cfg.repopt_replex_interval == 100
    assert cfg.water_moltype == "SOL"
    with pytest.raises(ConfigError):
        cfg.baseconf  # noqa: B018 — required key


def test_expansion_rejected(tmp_path):
    # v1 feprest template used REFINIT=$BASECONF; the Python parser must
    # reject it loudly instead of misparsing
    with pytest.raises(ConfigError, match="expansion"):
        write_conf(tmp_path, "REFINIT=$BASECONF\n")


def test_array_rejected(tmp_path):
    with pytest.raises(ConfigError, match="array"):
        write_conf(tmp_path, "FOO=(a b c)\n")


def test_per_id_override(tmp_path):
    (tmp_path / "para_conf.zsh").write_text("NREP=32\nBASETOP=t.top\n")
    (tmp_path / "mol1").mkdir()
    (tmp_path / "mol1" / "para_conf.zsh").write_text("NREP=48\n")
    cfg = Config.load(tmp_path, "mol1")
    assert cfg.nrep == 48
    assert cfg.basetop == "t.top"
    assert len(cfg.sources) == 2


def test_validate_errors():
    cases = [
        ({"NREP": "1"}, "NREP"),
        ({"PARA": "0"}, "PARA"),
        ({"CHARGE": "sometimes"}, "CHARGE"),
        ({"REST2_TEMP": "300"}, "REST2_TEMP"),
        ({"NTUNE": "0"}, "NTUNE"),
    ]
    for values, key in cases:
        cfg = Config(dict(values), sources=[])
        with pytest.raises(ConfigError, match=key):
            cfg.validate()


def test_parse_rejects_garbage(tmp_path):
    with pytest.raises(ConfigError):
        write_conf(tmp_path, "this is not an assignment\n")
