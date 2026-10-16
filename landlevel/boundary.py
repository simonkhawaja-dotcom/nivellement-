"""Chargement d'un contour de champ (boundary) et clipping du DEM sur ce contour.

Le masque nodata seul (voir dem.read_dem) ne suffit pas a isoler le champ :
un DEM LIDAR brut couvre souvent aussi les fosses, chemins, bandes boisees
ou champs voisins qui n'ont pas de nodata. Ce module permet de restreindre
le DEM au polygone reel du champ avant tout calcul de plan/volume.
"""

import json
from pathlib import Path

import numpy as np
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape
from shapely.ops import unary_union


def load_boundary(path, default_crs: str = "EPSG:4326"):
    """Charge un contour de champ depuis un shapefile (.shp) ou un GeoJSON.

    Plusieurs entites sont fusionnees (unary_union) en une seule geometrie.
    Le CRS est lu depuis le .prj (shapefile) quand il existe, sinon
    `default_crs` est suppose (les exports GeoJSON sont conventionnellement
    en WGS84).

    Retourne (geom, crs) : geometrie shapely + CRS source (rasterio.crs.CRS).
    """
    path = Path(path)

    if path.suffix.lower() == ".shp":
        import shapefile  # pyshp

        with shapefile.Reader(str(path)) as reader:
            geoms = [shape(s.__geo_interface__) for s in reader.shapes()]
        prj_path = path.with_suffix(".prj")
        crs = CRS.from_wkt(prj_path.read_text(encoding="utf-8")) if prj_path.exists() \
            else CRS.from_string(default_crs)
    elif path.suffix.lower() in (".geojson", ".json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("type") == "FeatureCollection":
            features = data["features"]
        elif data.get("type") == "Feature":
            features = [data]
        else:
            features = [{"geometry": data}]
        geoms = [shape(f["geometry"]) for f in features]
        crs = CRS.from_string(default_crs)
    else:
        raise ValueError(f"Format de contour non supporte : {path.suffix} (attendu .shp/.geojson/.json)")

    if not geoms:
        raise ValueError(f"Aucune geometrie trouvee dans {path}")

    return unary_union(geoms), crs


def clip_grid_to_boundary(grid, boundary_geom, boundary_crs):
    """Restreint grid.mask (et grid.z) aux cellules a l'interieur du contour.

    Le contour est reprojete dans le CRS du DEM si necessaire, puis
    rasterise sur la grille exacte du DEM (meme transform/shape). Le
    resultat est combine (ET logique) avec le masque nodata existant :
    le clipping ne peut que retirer des cellules, jamais en ajouter.

    Modifie `grid` en place et le retourne (pour chainer les appels).
    """
    geom_dict = mapping(boundary_geom)
    if boundary_crs != grid.crs:
        geom_dict = transform_geom(boundary_crs, grid.crs, geom_dict)

    boundary_mask = rasterize(
        [(geom_dict, 1)],
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)

    n_before = int(grid.mask.sum())
    grid.mask = grid.mask & boundary_mask
    grid.z = np.where(grid.mask, grid.z, np.nan)
    n_after = int(grid.mask.sum())

    if n_after == 0:
        raise ValueError(
            "Le contour ne recoupe aucune cellule valide du DEM — "
            "verifiez le CRS et l'emprise du fichier de contour."
        )
    if n_after > n_before * 0.999:
        import warnings
        warnings.warn(
            "Le clipping par contour n'a retire quasiment aucune cellule "
            f"({n_before} -> {n_after}) : le contour couvre peut-etre plus "
            "que le DEM, verifiez qu'il correspond bien au bon champ."
        )

    return grid


def rasterize_named_zones(grid, features, groups, name_prop: str = "name",
                           src_crs: str = "EPSG:4326"):
    """Rasterise des groupes de parcelles nommees en zones de nivellement.

    Cas d'usage : un plan de ferme complet (features GeoJSON)
    contient plusieurs petites parcelles cadastrales, dont certaines
    sont contigues (pas de fosse, meme unite de nivellement) et d'autres
    separees par un fosse (unites independantes : le materiau ne peut pas
    etre deplace de l'une a l'autre). `groups` decrit ce regroupement.

    features : liste de dicts GeoJSON Feature (chaque avec .properties et .geometry)
    groups : dict ordonne {nom_de_zone: [nom_de_parcelle, ...]} ; les
        parcelles listees dans un meme groupe sont fusionnees (union) en une
        seule zone.

    Retourne (labels, zone_names) :
        labels : grille 2D d'entiers (meme shape que grid), -1 = hors de
            toute zone (ou nodata), sinon index dans zone_names.
        zone_names : liste des noms de zone, dans l'ordre de `groups`.
    """
    by_name: dict[str, list] = {}
    for feat in features:
        nm = feat.get("properties", {}).get(name_prop)
        if nm is not None:
            by_name.setdefault(nm, []).append(shape(feat["geometry"]))

    zone_names = list(groups.keys())
    labels = np.full(grid.shape, -1, dtype=np.int32)

    for idx, zname in enumerate(zone_names):
        geoms = []
        for field_name in groups[zname]:
            if field_name not in by_name:
                raise ValueError(f"Parcelle '{field_name}' introuvable (zone '{zname}')")
            geoms.extend(by_name[field_name])

        geom_dict = mapping(unary_union(geoms))
        geom_dict = transform_geom(src_crs, grid.crs, geom_dict)

        zone_mask = rasterize(
            [(geom_dict, 1)],
            out_shape=grid.shape,
            transform=grid.transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)

        labels[zone_mask & grid.mask] = idx

    return labels, zone_names
