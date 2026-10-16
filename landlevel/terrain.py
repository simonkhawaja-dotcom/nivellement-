"""Slope/aspect et segmentation multiplan pour le nivellement agricole."""

from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_objects
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def fill_nodata_nearest(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remplit les cellules invalides avec la valeur de la cellule valide la
    plus proche (plus proche voisin), pour eviter la contamination NaN lors
    du calcul de gradient pres des bords du champ."""
    if mask.all():
        return z
    _, indices = distance_transform_edt(~mask, return_distances=True, return_indices=True)
    return z[tuple(indices)]


def compute_slope_aspect(grid):
    """Calcule slope (deg) et aspect (deg, 0=Nord, sens horaire) sur la
    grille complete a partir du gradient centre du DEM.

    Le nodata est rempli par plus-proche-voisin avant le calcul du gradient,
    puis le resultat hors du masque valide n'est pas utilise en aval.
    """
    z_filled = fill_nodata_nearest(grid.z, grid.mask)

    dzdrow, dzdcol = np.gradient(z_filled)
    dzdy = dzdrow / grid.transform.e  # transform.e < 0 (y decroit avec la ligne)
    dzdx = dzdcol / grid.transform.a  # transform.a > 0 (x croit avec la colonne)

    slope_deg = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    # direction de plus forte pente descendante, azimut compas (0=Nord, horaire)
    aspect_deg = np.degrees(np.arctan2(-dzdx, -dzdy)) % 360.0

    return slope_deg, aspect_deg


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    """Moyenne circulaire d'angles en degres (evite le probleme 0/360)."""
    rad = np.radians(angles_deg)
    s, c = np.sin(rad).mean(), np.cos(rad).mean()
    return float(np.degrees(np.arctan2(s, c)) % 360.0)


def segment_field(grid, aspect_deg: np.ndarray, n_clusters: int = 3,
                   min_size_cells: int = 400, smooth_size: int = 21,
                   random_state: int = 0) -> np.ndarray:
    """Segmente les cellules valides en `n_clusters` sections via KMeans sur
    [x_norm, y_norm, cos(aspect), sin(aspect)] (features spatiales incluses
    pour forcer des clusters geographiquement compacts), puis nettoie les
    petits ilots isoles et lisse les frontieres.

    Retourne un tableau 2D d'entiers (label 0..n_clusters-1, -1 = invalide).
    """
    mask = grid.mask
    aspect_rad = np.radians(aspect_deg)
    cos_a, sin_a = np.cos(aspect_rad), np.sin(aspect_rad)

    features = np.column_stack([
        grid.x[mask], grid.y[mask], cos_a[mask], sin_a[mask],
    ])
    features_scaled = StandardScaler().fit_transform(features)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels_flat = km.fit_predict(features_scaled)

    labels = np.full(grid.shape, -1, dtype=np.int32)
    labels[mask] = labels_flat

    return clean_segments(labels, n_clusters, min_size_cells, smooth_size)


def clean_segments(labels: np.ndarray, n_clusters: int, min_size_cells: int,
                    smooth_size: int = 5) -> np.ndarray:
    """Supprime les petits ilots isoles (scikit-image) puis lisse les
    frontieres par vote majoritaire dans une fenetre glissante (vectorise
    via convolution, une carte de comptage par cluster)."""
    valid = labels >= 0
    cleaned = labels.copy()

    for k in range(n_clusters):
        cluster_mask = cleaned == k
        kept = remove_small_objects(cluster_mask, max_size=min_size_cells)
        removed = cluster_mask & ~kept
        cleaned[removed] = -2  # ilot retire, sera reassigne par le lissage

    kernel = np.ones((smooth_size, smooth_size))
    counts = np.stack([
        ndi.convolve((cleaned == k).astype(float), kernel, mode="nearest")
        for k in range(n_clusters)
    ])
    majority = np.argmax(counts, axis=0).astype(np.int32)
    majority[~valid] = -1
    return majority


def export_sections_shapefile(labels: np.ndarray, transform, crs, out_path) -> None:
    """Vectorise les sections (rasterio.features.shapes) et les exporte en
    shapefile polygone (pyshp), un enregistrement par section avec son
    identifiant de cluster. Ecrit aussi le fichier .prj si le CRS est connu.
    """
    import shapefile  # pyshp
    from rasterio.features import shapes as rio_shapes
    from shapely.geometry import shape
    from shapely.geometry.polygon import orient
    from shapely.ops import unary_union

    out_path = Path(out_path)
    mask = labels >= 0

    geoms_by_cluster: dict[int, list] = {}
    for geom, value in rio_shapes(labels.astype(np.int32), mask=mask, transform=transform):
        geoms_by_cluster.setdefault(int(value), []).append(shape(geom))

    with shapefile.Writer(str(out_path), shapeType=shapefile.POLYGON) as w:
        w.field("cluster", "N")
        for value, geoms in sorted(geoms_by_cluster.items()):
            merged = unary_union(geoms)
            polys = [merged] if merged.geom_type == "Polygon" else list(merged.geoms)
            for poly in polys:
                poly = orient(poly, sign=-1.0)  # convention shapefile: exterieur horaire
                rings = [list(poly.exterior.coords)]
                rings += [list(interior.coords) for interior in poly.interiors]
                w.poly(rings)
                w.record(value)

    if crs is not None:
        with open(str(out_path) + ".prj", "w") as f:
            f.write(crs.to_wkt())
