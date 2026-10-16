"""Tests de bout en bout. DEM synthetique par defaut ; LANDLEVEL_DEM=... pour le tien."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "examples"))

from landlevel import read_dem  # noqa: E402
from run_dem import run_monoplan, run_multiplan  # noqa: E402


def test_monoplan(dem_path, tmp_path):
    grid = read_dem(dem_path)
    r = run_monoplan(grid, tmp_path)
    cut, fill = r["lp"]
    assert abs(abs(cut) - fill) < 0.01 * fill  # cut = fill


def test_multiplan(dem_path, tmp_path):
    grid = read_dem(dem_path)
    mono = run_monoplan(grid)["lp"]
    r = run_multiplan(grid, 3, tmp_path)
    assert len(r["sections"]) >= 2
    assert abs(r["cut"]) + r["fill"] <= (abs(mono[0]) + mono[1]) * 1.05
