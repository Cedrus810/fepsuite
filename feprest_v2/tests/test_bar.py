import pickle

import pytest

from restlab import bar


def test_parse_deltae(tmp_path):
    f = tmp_path / "deltae.xvg"
    content = ("@ title \"dE\"\n"
               "# comment\n"
               "0 0 -10.0 1 -12.0\n"
               "0.1 0 -10.1 1 -12.1\n"
               "0.2 0 -10.2 1 -12.2\n"
               "0.2 0 -10.3 1 -12.3\n")  # duplicate time: skipped
    f.write_text(content)
    # broken last line (no newline): skipped, as v1
    with open(f, "a") as fh:
        fh.write("0.3 0 -10.4 1 -12.4")
    data = bar.parse_deltae([str(f)])
    assert len(data) == 6  # 3 frames x 2 evaluation states
    assert data[0] == (0.0, 0, -10.0)
    assert data[1] == (0.0, 1, -12.0)
    times = {t for (t, _, _) in data}
    assert times == {0.0, 0.1, 0.2}


def test_parse_deltae_subsample(tmp_path):
    f = tmp_path / "deltae.xvg"
    f.write_text("0 0 -10.0\n0.1 0 -10.1\n0.2 0 -10.2\n0.3 0 -10.3\n")
    data = bar.parse_deltae([str(f)], subsample=2)
    times = [t for (t, _, _) in data]
    assert times == [0.0, 0.2]


def test_constants():
    # unit conversion sanity: the report is in kcal/mol
    assert bar.gasconstant * 300 * bar.KCAL_OF_KJ == pytest.approx(
        bar.gasconstant_kcal * 300, rel=1e-5)


def test_run_analysis_orchestration(tmp_path, monkeypatch, capsys):
    # orchestration test with a stubbed BAR kernel (no pymbar needed)
    for isim in range(3):
        f = tmp_path / f"deltae_{isim}.xvg"
        rows = []
        for i in range(10):
            rows.append("%f %d %f %d %f" % (i * 0.1, isim, -10.0 - isim,
                                            isim + 1, -12.0 - isim))
        f.write_text("\n".join(rows) + "\n")

    calls = []

    def fake_bar(emat, time_all, nsim, btime, etime, show_intermediate):
        calls.append((nsim, btime, etime))
        return 0.001

    monkeypatch.setattr(bar, "bar", fake_bar)
    result = bar.run_analysis(str(tmp_path / "deltae_%sim.xvg"),
                              nsim=3, temp=300.0, save_dir=str(tmp_path),
                              split=4)
    # sliding windows + latter-half splits
    assert len(calls) == 8
    assert (3,) == calls[0][:1]
    assert result == pytest.approx(0.001 * bar.gasconstant * 300 * bar.KCAL_OF_KJ)
    out = capsys.readouterr().out
    assert "Performing sliding-window" in out
    assert "Final estimate [kcal/mol]" in out
    assert out.strip().endswith("BAR 0.00 0.00")
    assert (tmp_path / "results-sliding.pickle").exists()
    assert (tmp_path / "results-normalsplit.pickle").exists()
    loaded = pickle.load(open(tmp_path / "results-normalsplit.pickle", "rb"))
    assert len(loaded) == 4


def test_bar_kernel_needs_pymbar(monkeypatch):
    # the real kernel imports pymbar lazily; without it we get a clean
    # ImportError that the pipeline turns into the v1 non-fatal message
    monkeypatch.setitem(__import__("sys").modules, "pymbar", None)
    with pytest.raises(ImportError):
        bar.bar({}, {}, 2, 0.0, 1.0)
