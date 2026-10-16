# Nivellement — solveur (landlevel)

Calcule un plan de nivellement agricole à partir d'un DEM raster : plan unique
(moindres carres ou LP avec cut = fill), multiplan par sections, zones
d'équilibrage nommées (séparées par fossé), pente minimale, écoulement D8.

## Installation

```bash
python -m venv venv && source venv/bin/activate   # Windows : venv\Scripts\activate
pip install -e . pytest
```

## Essayer sur ton DEM

Le DEM doit être un GeoTIFF mono-bande, en CRS projeté (mètres), pixels carrés.

```bash
python examples/run_dem.py mon_dem.tif                       # monoplan + multiplan, cartes dans outputs/
python examples/run_dem.py mon_dem.tif --boundary contour.geojson --k 4 --min-slope 0.0015
python examples/run_dem.py mon_dem.tif --zones parcelles.geojson --zone-groups groupes.json
```

- `--boundary` : contour du champ (.shp/.geojson) pour exclure fossés, chemins, bandes boisées.
- `--zones` / `--zone-groups` : GeoJSON (EPSG:4326) de parcelles avec propriété `name`, et JSON
  `{"zone A": ["p1", "p2"], "zone B": ["p3"]}` ; chaque groupe est une unité d'équilibrage indépendante.
- `--min-slope` : pente finale minimale (fraction ; 0.0015 = 0,15 %).

Utilisation en Python : voir `examples/run_dem.py` (`run_monoplan`, `run_multiplan`, `run_zones`).

## Tests

```bash
pytest                                   # DEM synthétique généré automatiquement
LANDLEVEL_DEM=/chemin/mon_dem.tif pytest # sur ton propre DEM
```

## Limites connues

- Hydrologie : D8 sans remplissage de dépressions ; l'eau qui atteint le bord est comptée comme sortie ;
  seuils de voies d'eau/creux empiriques.
- Plans indépendants par section (marches d'altitude aux raccords, non adoucies) ; pente minimale à
  direction figée.
- Foisonnement : paramètre `ratio` du solveur, non exposé par le script d'exemple.
