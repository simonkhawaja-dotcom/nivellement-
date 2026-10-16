"""Lecture de DEM raster (GeoTIFF) pour le solveur de nivellement."""

from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine


@dataclass
class DEMGrid:
    x: np.ndarray          # 2D, coordonnees X (centre de cellule)
    y: np.ndarray          # 2D, coordonnees Y (centre de cellule)
    z: np.ndarray          # 2D, elevation naturelle, NaN si invalide
    mask: np.ndarray       # 2D bool, True = cellule valide
    transform: Affine
    crs: CRS
    cell_size: float
    shape: tuple

    @property
    def cell_area(self) -> float:
        return self.cell_size ** 2

    def valid_points(self):
        """Retourne (x, y, z) en tableaux 1D, cellules valides uniquement."""
        return self.x[self.mask], self.y[self.mask], self.z[self.mask]

    def n_valid(self) -> int:
        return int(self.mask.sum())

    def area_ha(self) -> float:
        return self.n_valid() * self.cell_area / 10_000.0


def read_dem(path: str) -> DEMGrid:
    """Lit un DEM GeoTIFF mono-bande directement avec rasterio.

    Gere le nodata correctement (masque les cellules invalides plutot que
    de les traiter comme des elevations reelles).
    """
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(f"DEM mono-bande attendu, {src.count} bandes trouvees")

        band = src.read(1).astype(np.float64)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        rows, cols = band.shape

        if abs(transform.b) > 1e-9 or abs(transform.d) > 1e-9:
            raise ValueError("transform avec rotation/cisaillement non supporte")

        cell_size_x = abs(transform.a)
        cell_size_y = abs(transform.e)
        if abs(cell_size_x - cell_size_y) > 1e-6:
            raise ValueError("pixels non carres non supportes")

        col_idx, row_idx = np.meshgrid(np.arange(cols), np.arange(rows))
        x = transform.c + (col_idx + 0.5) * transform.a
        y = transform.f + (row_idx + 0.5) * transform.e

        mask = np.isfinite(band)
        if nodata is not None:
            mask &= band != nodata

    z = np.where(mask, band, np.nan)

    return DEMGrid(
        x=x, y=y, z=z, mask=mask,
        transform=transform, crs=crs,
        cell_size=cell_size_x, shape=(rows, cols),
    )
