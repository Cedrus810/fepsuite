import json
import sys

import pytest

from feplab.config import Config, ConfigError
from feplab.phases import query_line
from feplab.resources import detect_gpus, ranks_for_replicas


def fake_run(outputs):
    def run(cmd):
        class R:
            returncode = 0
            stdout = outputs.get(cmd[0], "")
        return R()
    return run


def test_detect_pbs_gpufile(tmp_path):
    f = tmp_path / "gpufile"
    f.write_text("node01:0\nnode01:1\nnode01:2\n")
    det = detect_gpus(env={"PBS_GPUFILE": str(f)}, run_cmd=fake_run({}))
    assert det.gpus == 3 and det.source == "PBS_GPUFILE"


def test_detect_qstat_json(tmp_path):
    qstat_out = json.dumps({"Jobs": {"1234.admin": {
        "resources_allocated": {"ngpus": "4", "ncpus": "48"}}}})
    det = detect_gpus(env={"PBS_JOBID": "1234.admin"},
                      run_cmd=fake_run({"qstat": qstat_out}))
    assert det.gpus == 4 and det.source == "qstat"


def test_detect_cuda_visible():
    det = detect_gpus(env={"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, run_cmd=fake_run({}))
    assert det.gpus == 4 and det.source == "CUDA_VISIBLE_DEVICES"
    det = detect_gpus(env={"NVIDIA_VISIBLE_DEVICES": "GPU-abc, GPU-def"},
                      run_cmd=fake_run({}))
    assert det.gpus == 2
    # "all"/"void" carry no countable information
    assert detect_gpus(env={"CUDA_VISIBLE_DEVICES": "void"}, run_cmd=fake_run({})) is None


def test_detect_nvidia_smi_fallback():
    det = detect_gpus(env={}, run_cmd=fake_run({"nvidia-smi": "GPU 0: A\nGPU 1: B\n"}))
    assert det.gpus == 2 and det.source == "nvidia-smi"


def test_detect_none():
    assert detect_gpus(env={}, run_cmd=fake_run({})) is None


def test_ranks_for_replicas():
    # more replicas than GPUs: share, keep 1 rank per replica
    assert ranks_for_replicas(4, 12) == (1, 12)
    # more GPUs than replicas: extra ranks per replica
    assert ranks_for_replicas(24, 12) == (2, 24)
    assert ranks_for_replicas(8, 1) == (8, 8)
    assert ranks_for_replicas(0, 4) == (1, 4)


def test_config_bool_keys():
    cfg = Config({}, sources=[])
    assert cfg.auto_resource is True    # defaults on
    assert cfg.prep_nonmpi is True
    assert cfg.gmx_nompi == "gmx"
    cfg = Config({"AUTO_RESOURCE": "no", "PREP_NONMPI": "off"}, sources=[])
    assert cfg.auto_resource is False and cfg.prep_nonmpi is False
    with pytest.raises(ConfigError):
        Config({"PREP_NONMPI": "maybe"}, sources=[]).prep_nonmpi


def test_query_line_resolved(tmp_path):
    from feplab.config import parse_para_conf
    conf = tmp_path / "para_conf.zsh"
    conf.write_text("LIG_PARA=2\nCOMPLEX_PARA=12\nPREP_NONMPI=yes\n")
    cfg = Config.load(tmp_path, None, extra_path=str(conf))
    # prep-like stages request a single rank under PREP_NONMPI
    assert "PPM=1 " in query_line("setup", cfg)
    assert "PPM=1 " in query_line("restraints", cfg)
    assert "PPM=1 " in query_line("prep-charging-lig", cfg)
    # GPU-dependent stages keep the literal shell references
    assert "PPM=$LIG_PARA" in query_line("charging-lig", cfg)
    assert "PPM=$COMPLEX_PARA" in query_line("equilibrate", cfg)
    # and with PREP_NONMPI=no everything is static, like v1
    cfg2 = Config(dict(cfg._values, PREP_NONMPI="no"), sources=[])
    assert "PPM=$COMPLEX_PARA" in query_line("setup", cfg2)
