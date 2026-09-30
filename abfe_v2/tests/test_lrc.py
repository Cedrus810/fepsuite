import math

import pytest

from feplab import lrc
from feplab.errors import PipelineError
from feplab.units import KCAL_PER_KJ, kbt


def test_align_frames_same_grid():
    kbt_value = kbt(300)
    orig = [(0.0, 10.0), (1.0, 11.0), (2.0, 12.0)]
    reeval = [(0.0, 12.0), (1.0, 13.0), (2.0, 14.0)]
    times, deltas = lrc.align_frames(orig, reeval, kbt_value, time_begin=-1.0)
    assert times == [0.0, 1.0, 2.0]
    assert deltas[0] == pytest.approx(-(12.0 - 10.0) / kbt_value)


def test_align_frames_different_grids():
    """Original EDR at 1 ps, rerun EDR at 20 ps (trajectory frames)."""
    orig = [(float(t) * 1.0, 10.0 + t) for t in range(0, 100)]
    reeval = [(float(t) * 20.0, 12.0 + t / 5) for t in range(0, 5)]
    times, deltas = lrc.align_frames(orig, reeval, 2.5, time_begin=-1.0)
    assert times == [0.0, 20.0, 40.0, 60.0, 80.0]
    assert len(deltas) == 5


def test_align_frames_time_begin():
    orig = [(0.0, 1.0), (10.0, 2.0), (20.0, 3.0)]
    reeval = [(0.0, 5.0), (10.0, 6.0), (20.0, 7.0)]
    times, _ = lrc.align_frames(orig, reeval, 2.5, time_begin=5.0)
    assert times == [10.0, 20.0]


def test_align_frames_float_noise():
    """xtc times pass through float32; noise above the tolerance must
    not break the matching."""
    orig = [(2000.00001, 1.0), (2020.00002, 2.0)]
    reeval = [(2000.0, 3.0), (2020.0, 4.0)]
    times, deltas = lrc.align_frames(orig, reeval, 2.5, time_begin=0.0)
    assert len(times) == 2


def test_normalize_and_resolve_terms():
    # GROMACS version-dependent spellings must match after normalization.
    assert lrc.normalize_term_name("LJ (SR)") == lrc.normalize_term_name("LJ-(SR)")
    assert lrc.normalize_term_name("LJ recip.") == lrc.normalize_term_name("LJ-recip.")
    available = ["Bond", "LJ-(SR)", "Coulomb-(SR)", "LJ-recip.", "LJ-14",
                 "Potential", "Disper.-corr."]
    resolved = lrc.resolve_terms(available)
    assert set(resolved) == {"Disper.-corr.", "LJ-14", "LJ-(SR)", "LJ-recip."}
    custom = lrc.resolve_terms(available, wanted=["Potential"])
    assert custom == ["Potential"]


def _fake_sum_series(short, long_):
    def fake(path, terms=None):
        if "short" in str(path):
            return short
        return long_
    return fake


def test_compute_lrc_empty_intersection(tmp_path, monkeypatch):
    short = ([0.0, 1.0], {"LJ-14": [1.0, 2.0]}, ["LJ-14"])
    long_ = ([0.5, 1.5], {"LJ-14": [1.0, 2.0]}, ["LJ-14"])
    monkeypatch.setattr(lrc, "_sum_series", _fake_sum_series(short, long_))
    with pytest.raises(PipelineError, match="share no timestamp"):
        lrc.compute_long_range_correction(long="long.edr", short="short.edr", temp=300,
                                          output=tmp_path / "out.txt")


def test_compute_lrc_end_to_end(tmp_path, monkeypatch):
    """Synthetic series: exponential average over aligned frames."""
    kbt_value = kbt(300)
    n = 50
    short = ([float(t) for t in range(n)], {"LJ-14": [100.0 + t for t in range(n)]},
             ["LJ-14"])
    long_ = ([float(t) for t in range(n)],
             {"LJ-14": [102.0 + t for t in range(n)],
              "LJ-recip.": [0.0] * n},  # matched in the file, cancels here
             ["LJ-14", "LJ-recip."])
    monkeypatch.setattr(lrc, "_sum_series", _fake_sum_series(short, long_))
    out = tmp_path / "out.txt"
    mean, std = lrc.compute_long_range_correction(
        long="long.edr", short="short.edr", temp=300, time_begin=0.5, output=out)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 6  # 5 blocks + final line
    import numpy
    deltas = numpy.array([-(2.0) / kbt_value] * (n - 1))  # t=0 dropped by time_begin
    import scipy.special
    expect = -1.0 / kbt_value * (scipy.special.logsumexp(deltas) - math.log(n - 1))
    assert mean == pytest.approx(expect, abs=1e-3)


def test_compute_lrc_no_terms(tmp_path, monkeypatch):
    empty = ([0.0], {}, [])
    monkeypatch.setattr(lrc, "_sum_series", _fake_sum_series(empty, empty))
    with pytest.raises(PipelineError, match="none of the LRC energy terms"):
        lrc.compute_long_range_correction(long="long.edr", short="short.edr", temp=300,
                                          output=tmp_path / "out.txt")


def test_report_sanity_guard(tmp_path):
    """A poisoned LRC term must abort the report instead of writing it."""
    from feplab.analysis import build_report
    basedir = _make_basedir(tmp_path)
    poisoned = basedir / "lr-complex.lrc.txt"
    poisoned.write_text("0 100 1.0\n-170000.0\t100.0\n")  # kJ, ~ -4e4 kcal
    with pytest.raises(PipelineError, match="sanity limit"):
        build_report(basedir=basedir, restrinfo=basedir / "restrinfo", temp=300.0)
    # raising the limit lets it pass (at the user's own risk)
    report = build_report(basedir=basedir, restrinfo=basedir / "restrinfo",
                          temp=300.0, sanity_limit_kcal=1e9)
    assert "total" in report


def test_report_skipped_lrc(tmp_path):
    """Skipped terms contribute zero without reading their files."""
    from feplab.analysis import build_report
    basedir = _make_basedir(tmp_path)
    (basedir / "lr-annihilation-lig.lrc.txt").unlink()
    (basedir / "lr-annihilation-complex.lrc.txt").unlink()
    report = build_report(basedir=basedir, restrinfo=basedir / "restrinfo",
                          temp=300.0,
                          skipped_lrc={"lr-annihilation-lig",
                                       "lr-annihilation-complex"})
    lines = report.strip().splitlines()
    entries = {l.split()[0]: l.split()[1] for l in lines[:lines.index("----")]}
    assert entries["lr-annihilation-lig"] == "0.000"
    assert entries["lr-annihilation-complex"] == "0.000"


def _make_basedir(tmp_path):
    from test_analysis import BAR, BAR_LOG, LRC, RESTRINFO
    basedir = tmp_path / "basedir"
    basedir.mkdir()
    for key, (energy, error) in BAR.items():
        (basedir / f"{key}.bar.log").write_text(BAR_LOG.format(energy=energy, error=error))
    for key, (mean, std) in LRC.items():
        (basedir / f"{key}.lrc.txt").write_text("0 100 1.0\n%.4f\t%.4f\n" % (mean, std))
    (basedir / "restrinfo").write_text(RESTRINFO)
    ccdir = basedir / "charge-correction"
    ccdir.mkdir()
    (ccdir / "complex.json").write_text("[]")
    (ccdir / "ligand.json").write_text("[]")
    return basedir


def test_config_lrc_keys():
    from feplab.config import Config, ConfigError
    cfg = Config({}, sources=[])
    assert cfg.lrc_sanity_limit == 50.0
    assert cfg.skip_annihilation_lrc is False
    assert cfg.lrc_energy_terms is None
    cfg = Config({"SKIP_ANNIHILATION_LRC": "yes",
                  "LRC_ENERGY_TERMS": "LJ (SR), LJ-14"}, sources=[])
    assert cfg.skip_annihilation_lrc is True
    assert cfg.lrc_energy_terms == ["LJ (SR)", "LJ-14"]
