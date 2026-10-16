"""Fixture `dem_path` : DEM synthetique, ou le tien via LANDLEVEL_DEM=/chemin/dem.tif."""
import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


@pytest.fixture(scope="session")
def dem_path(tmp_path_factory):
    env = os.environ.get("LANDLEVEL_DEM")
    if env:
        return env
    rng = np.random.default_rng(0)
    n = 120
    yy, xx = np.mgrid[0:n, 0:n]
    z = 100 + 0.01 * xx * 3 + 0.02 * yy * 3 + 0.4 * np.sin(xx / 9) * np.cos(yy / 11)
    z = (z + rng.normal(0, 0.02, z.shape)).astype("float32")
    z[:6, :6] = -9999  # nodata
    path = tmp_path_factory.mktemp("dem") / "synthetic.tif"
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
                       crs="EPSG:32618", transform=from_origin(500000, 5000000, 3, 3),
                       nodata=-9999) as dst:
        dst.write(z, 1)
    return str(path)
