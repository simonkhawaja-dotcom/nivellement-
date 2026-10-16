"""Fit d'un plan LP contraint (cut=fill) independamment par zone.

Reutilise pour deux scenarios distincts qui partagent la meme mecanique :
  - multiplan : zones = clusters KMeans par aspect
  - zones nommees : parcelles regroupees par fosse

Dans les deux cas, chaque zone est une unite d'equilibrage independante :
son propre plan, sa propre contrainte cut=fill.
"""

import numpy as np

from .grading import compute_volumes, fit_plane_balanced_l1
from .terrain import circular_mean_deg


def fit_planes_by_zone(grid, labels: np.ndarray, zone_names, aspect_deg=None,
                        min_points: int = 10, ratio: float = 1.0, min_slope: float = 0.0):
    """Fitte un plan contraint (cut=fill, ou cut=ratio*fill) par zone.

    labels : grille 2D d'entiers (index dans zone_names, -1 = invalide/hors zone)
    zone_names : liste des noms de zone (labels prend ses valeurs dans
        range(len(zone_names)))
    aspect_deg : grille 2D d'aspect (deg), optionnelle — si fournie, chaque
        zone rapporte aussi son aspect moyen (moyenne circulaire)
    min_points : zones plus petites ignorees (fit degenere)
    ratio : ratio cut/fill cible, transmis a fit_plane_balanced_l1 (1.0 =
        equilibre strict ; voir grading.py pour le ratio de foisonnement)
    min_slope : pente resultante minimale par zone (fraction, 0.0015 = 0,15 %),
        transmise a fit_plane_balanced_l1 (voir son docstring)

    Retourne (z_design_full, total_cut, total_fill, zones) ou `zones` est
    une liste de dicts avec les stats de chaque zone (plane, cut, fill,
    area_ha, plane_bearing_deg, aspect_bearing_deg le cas echeant).
    """
    z_design_full = np.full(grid.shape, np.nan)
    zones = []
    total_cut = 0.0
    total_fill = 0.0

    for idx, zname in enumerate(zone_names):
        sel = (labels == idx) & grid.mask
        n_pts = int(sel.sum())
        if n_pts < min_points:
            continue

        x_k, y_k, z_k = grid.x[sel], grid.y[sel], grid.z[sel]
        plane_k = fit_plane_balanced_l1(x_k, y_k, z_k, ratio=ratio, min_slope=min_slope)
        z_design_k = plane_k.elevation(x_k, y_k)
        cut_k, fill_k, net_k = compute_volumes(z_k, z_design_k, grid.cell_area)

        z_design_full[sel] = z_design_k
        total_cut += cut_k
        total_fill += fill_k

        entry = {
            "zone": zname,
            "cluster": idx,
            "n_points": n_pts,
            "area_ha": n_pts * grid.cell_area / 10_000.0,
            "plane": plane_k,
            "slope_pct": plane_k.slope_pct(),
            "slope_constrained": plane_k.slope_constrained,
            "plane_bearing_deg": plane_k.bearing_deg(),
            "cut": cut_k,
            "fill": fill_k,
            "net": net_k,
        }
        if aspect_deg is not None:
            entry["aspect_bearing_deg"] = circular_mean_deg(aspect_deg[sel])
        zones.append(entry)

    return z_design_full, total_cut, total_fill, zones
