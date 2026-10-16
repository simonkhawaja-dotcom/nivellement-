"""Fait tourner le solveur sur un DEM GeoTIFF et affiche les volumes cut/fill.

Utilisation :
    python examples/run_dem.py mon_dem.tif
    python examples/run_dem.py mon_dem.tif --boundary contour.geojson --k 3 --out sortie/
    python examples/run_dem.py mon_dem.tif --zones parcelles.geojson \
        --zone-groups groupes.json --min-slope 0.0015

Le DEM doit etre mono-bande, en CRS projete (metres), pixels carres.
`--zones` : GeoJSON (EPSG:4326) dont les features portent une propriete `name`.
`--zone-groups` : JSON {"zone A": ["parcelle1", "parcelle2"], "zone B": ["p3"]}
  (les parcelles d'un meme groupe forment une seule zone d'equilibrage).
"""

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from landlevel import (  # noqa: E402
    clip_grid_to_boundary, compute_volumes, fit_plane_balanced_l1, fit_plane_ols,
    fit_planes_by_zone, load_boundary, read_dem,
)
from landlevel.boundary import rasterize_named_zones  # noqa: E402
from landlevel.terrain import compute_slope_aspect, segment_field  # noqa: E402
from landlevel.viz import plot_cutfill_map, plot_sections_map  # noqa: E402


def run_monoplan(grid, out_dir=None, min_slope=0.0):
    """Plan unique : moindres carres (non contraint) vs LP (cut = fill)."""
    x, y, z = grid.valid_points()
    ols = fit_plane_ols(x, y, z)
    cut_o, fill_o, net_o = compute_volumes(z, ols.elevation(x, y), grid.cell_area)
    bal = fit_plane_balanced_l1(x, y, z, min_slope=min_slope)
    cut_b, fill_b, net_b = compute_volumes(z, bal.elevation(x, y), grid.cell_area)

    print("\n=== Monoplan ===")
    for nom, p, c, f, n in (("OLS (non contraint)", ols, cut_o, fill_o, net_o),
                            ("LP (cut = fill)", bal, cut_b, fill_b, net_b)):
        print(f"  {nom:20s} pente Y {p.grade_y_pct():+.3f}% / X {p.grade_x_pct():+.3f}% | "
              f"cut {c:,.0f} m3 | fill {f:,.0f} m3 | net {n:,.0f} m3")

    if out_dir:
        plot_cutfill_map(grid, ols.elevation(grid.x, grid.y), "Cut/Fill - OLS",
                         out_dir / "monoplan_ols.png")
        plot_cutfill_map(grid, bal.elevation(grid.x, grid.y), "Cut/Fill - LP cut=fill",
                         out_dir / "monoplan_lp.png")
    return {"ols": (cut_o, fill_o), "lp": (cut_b, fill_b), "plane_lp": bal}


def run_multiplan(grid, k=3, out_dir=None, min_slope=0.0):
    """k sections (KMeans sur position + aspect), un plan LP par section."""
    _, aspect = compute_slope_aspect(grid)
    labels = segment_field(grid, aspect, n_clusters=k)
    z_design, cut, fill, sections = fit_planes_by_zone(
        grid, labels, list(range(k)), aspect_deg=aspect, min_slope=min_slope)

    print(f"\n=== Multiplan (k={k}) ===")
    for s in sections:
        print(f"  Section {s['cluster']} : {s['area_ha']:.2f} ha | "
              f"cut {s['cut']:,.0f} m3 | fill {s['fill']:,.0f} m3")
    print(f"  Total deplace : {abs(cut) + fill:,.0f} m3")

    if out_dir:
        plot_sections_map(grid, labels, k, f"Sections (k={k})", out_dir / "multiplan_sections.png")
        plot_cutfill_map(grid, z_design, f"Cut/Fill - multiplan (k={k})",
                         out_dir / "multiplan_cutfill.png")
    return {"labels": labels, "cut": cut, "fill": fill, "sections": sections}


def run_zones(grid, features, groups, out_dir=None, min_slope=0.0):
    """Une zone d'equilibrage par groupe de parcelles nommees."""
    labels, names = rasterize_named_zones(grid, features, groups)
    _, aspect = compute_slope_aspect(grid)
    z_design, cut, fill, zones = fit_planes_by_zone(
        grid, labels, names, aspect_deg=aspect, min_slope=min_slope)

    print("\n=== Zones nommees ===")
    for z in zones:
        print(f"  {z['zone']} : {z['area_ha']:.2f} ha | cut {z['cut']:,.0f} m3 | fill {z['fill']:,.0f} m3")
    print(f"  Total deplace : {abs(cut) + fill:,.0f} m3")

    if out_dir:
        plot_cutfill_map(grid, z_design, "Cut/Fill - zones", out_dir / "zones_cutfill.png")
    return {"labels": labels, "cut": cut, "fill": fill, "zones": zones}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dem")
    ap.add_argument("--boundary", help="contour du champ (.shp ou .geojson) pour clipper le DEM")
    ap.add_argument("--k", type=int, default=3, help="nombre de sections du multiplan")
    ap.add_argument("--min-slope", type=float, default=0.0, help="pente minimale (fraction, 0.0015 = 0,15 %%)")
    ap.add_argument("--zones", help="GeoJSON de parcelles nommees (propriete `name`)")
    ap.add_argument("--zone-groups", help="JSON {zone: [parcelles]}")
    ap.add_argument("--out", default="outputs", help="dossier de sortie des cartes")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = read_dem(args.dem)
    if args.boundary:
        geom, crs = load_boundary(args.boundary)
        clip_grid_to_boundary(grid, geom, crs)
    print(f"{grid.n_valid()} cellules valides ({grid.area_ha():.3f} ha), "
          f"resolution {grid.cell_size} m, CRS {grid.crs}")

    run_monoplan(grid, out_dir, args.min_slope)
    run_multiplan(grid, args.k, out_dir, args.min_slope)
    if args.zones and args.zone_groups:
        feats = json.loads(pathlib.Path(args.zones).read_text(encoding="utf-8"))["features"]
        groups = json.loads(pathlib.Path(args.zone_groups).read_text(encoding="utf-8"))
        run_zones(grid, feats, groups, out_dir, args.min_slope)
    print(f"\nCartes dans {out_dir}/")


if __name__ == "__main__":
    main()
