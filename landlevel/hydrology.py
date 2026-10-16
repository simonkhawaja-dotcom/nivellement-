"""Ecoulement de surface (flow accumulation) pour validation de nivellement.

Implementation D8 simplifiee, sans remplissage de depressions (pas de
priority-flood) : chaque cellule ecoule vers son voisin le plus bas parmi
les 8 adjacents s'il existe une pente descendante, sinon elle est un puits
local (l'eau s'accumule sans aval — utile en soi pour reperer les creux
residuels d'un plan de nivellement). L'accumulation est calculee en
parcourant les cellules par elevation decroissante, chacune transferant
son accumulation courante vers son aval.

Bords du champ : un voisin hors domaine (fosse, terrain voisin) n'est pas
ignore. Son altitude est extrapolee (plan local du design via `grad`, ou DEM
brut via `ext_elev`) ; s'il est plus bas, la cellule est un EXUTOIRE (l'eau
quitte le champ, pas de transfert). Ignorer ces voisins deviait l'eau le
long du bord du champ et gonflait artificiellement les concentrations.

C'est une simplification deliberee (une vraie analyse hydrologique
remplirait les depressions avant le routage D8) : suffisante pour une
visualisation de validation, pas pour un dimensionnement de drainage.
"""

import numpy as np

_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

PIT = -1    # aucun voisin plus bas : puits local
EXIT = -2   # l'eau quitte le domaine (voisin hors champ plus bas)


def compute_flow(elev: np.ndarray, mask: np.ndarray, cell_size: float,
                 grad=None, ext_elev=None):
    """Retourne (accum, down).

    elev, mask : grilles 2D (domaine = cellules valides du masque).
    grad : (a, b) deux grilles 2D du gradient local du plan de design
        (dz/dx, dz/dy), pour extrapoler l'altitude des voisins hors domaine.
    ext_elev : grille 2D d'altitudes connues hors domaine (DEM brut), pour
        le meme usage quand le terrain naturel est analyse.
    accum : accumulation (nombre de cellules amont, soi inclus), 0 hors domaine.
    down : (rows, cols, 3) ; down[..., 0] == PIT (-1) pour un puits, == EXIT
        (-2) pour un exutoire, sinon indice de ligne aval (down[..., 1] =
        colonne aval). down[..., 2] = direction d'ecoulement, indice dans
        _NEIGHBORS ((dr, dc) : 0=NO, 1=N, 2=NE, 3=O, 4=E, 5=SO, 6=S, 7=SE),
        -1 pour un puits, -2 hors domaine — sert a dessiner les fleches.
    """
    rows, cols = elev.shape
    valid = mask & np.isfinite(elev)
    dist = [cell_size * 2 ** 0.5 if dr and dc else cell_size for dr, dc in _NEIGHBORS]

    idxs = np.argwhere(valid)
    down = np.full((rows, cols, 3), PIT, dtype=np.int32)
    down[..., 2] = -2  # direction : -2 = hors domaine

    for r, c in idxs:
        z0 = elev[r, c]
        best_slope = 0.0
        best = (PIT, PIT)
        best_k = -1
        for k, ((dr, dc), d) in enumerate(zip(_NEIGHBORS, dist)):
            rr, cc = r + dr, c + dc
            inside = 0 <= rr < rows and 0 <= cc < cols
            if inside and valid[rr, cc]:
                znb = elev[rr, cc]
                target = (rr, cc)
            else:
                if grad is not None:
                    znb = z0 + grad[0][r, c] * dc * cell_size + grad[1][r, c] * (-dr) * cell_size
                elif ext_elev is not None and inside and np.isfinite(ext_elev[rr, cc]):
                    znb = ext_elev[rr, cc]
                else:
                    continue
                target = (EXIT, EXIT)
            slope = (z0 - znb) / d
            if slope > best_slope:
                best_slope = slope
                best = target
                best_k = k
        down[r, c, 0], down[r, c, 1], down[r, c, 2] = best[0], best[1], best_k

    accum = np.zeros((rows, cols), dtype=np.float64)
    accum[valid] = 1.0
    order = idxs[np.argsort(-elev[idxs[:, 0], idxs[:, 1]])]
    for r, c in order:
        dr, dc = down[r, c, 0], down[r, c, 1]
        if dr >= 0:
            accum[dr, dc] += accum[r, c]

    return accum, down


def compute_flow_accumulation(elev: np.ndarray, mask: np.ndarray, cell_size: float,
                               grad=None, ext_elev=None) -> np.ndarray:
    """Accumulation de flux D8 seule (voir compute_flow)."""
    return compute_flow(elev, mask, cell_size, grad=grad, ext_elev=ext_elev)[0]


def flow_works(accum: np.ndarray, down: np.ndarray, zone_labels: np.ndarray,
               exits_per_zone: int = 2, pit_min_cells: int = 60, max_pits_per_zone: int = 3,
               waterway_min_cells: int = 150):
    """Repere les "ouvrages" indicatifs a partir du routage D8 du plan final.

    - exutoires : cellules ou l'eau quitte le champ (points a raccorder au
      fosse), les `exits_per_zone` plus fortes accumulations par zone ;
    - creux : puits internes avec au moins `pit_min_cells` cellules amont
      (risque de stagnation d'eau) ;
    - voies d'eau : cellules dont l'accumulation atteint `waterway_min_cells`
      (trajets d'ecoulement concentres, candidats a une voie d'eau engazonnee).

    Retourne (markers, waterway_cells) : markers = liste de dicts
    {row, col, accum, zone_index, kind} ; waterway_cells = indices plats (row*cols+col).
    """
    markers = []
    for zidx in sorted(int(z) for z in np.unique(zone_labels) if z >= 0):
        in_zone = zone_labels == zidx
        exits = np.argwhere(in_zone & (down[..., 0] == EXIT))
        pits = np.argwhere(in_zone & (down[..., 0] == PIT) & (accum >= pit_min_cells))
        for kind, cells, limit in (("exutoire", exits, exits_per_zone), ("creux", pits, max_pits_per_zone)):
            if not len(cells):
                continue
            order = np.argsort(-accum[cells[:, 0], cells[:, 1]])[:limit]
            for r, c in cells[order]:
                markers.append({
                    "row": int(r), "col": int(c), "accum": float(accum[r, c]),
                    "zone_index": zidx, "kind": kind,
                })

    waterway = np.flatnonzero((accum >= waterway_min_cells) & (zone_labels >= 0))
    return markers, waterway.tolist()
