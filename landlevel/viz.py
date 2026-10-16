"""Fonctions d'affichage (cartes cut/fill, sections) pour le nivellement."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm


def plot_cutfill_map(grid, z_design_grid: np.ndarray, title: str, out_path) -> None:
    """Carte cut/fill (design - naturel), palette diverging rouge/bleu centree sur 0."""
    diff = np.where(grid.mask, z_design_grid - grid.z, np.nan)
    vmax = np.nanmax(np.abs(diff))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(diff, cmap="RdBu", norm=norm)
    ax.set_title(title)
    ax.set_xlabel("colonne")
    ax.set_ylabel("ligne")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("design - naturel (m)  |  rouge = deblai, bleu = remblai")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sections_map(grid, labels: np.ndarray, n_clusters: int, title: str, out_path) -> None:
    """Carte des sections detectees, une couleur par cluster."""
    display = np.ma.masked_where(labels < 0, labels)
    cmap = ListedColormap(plt.get_cmap("tab10").colors[:max(n_clusters, 1)])

    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(display, cmap=cmap, vmin=-0.5, vmax=n_clusters - 0.5)
    ax.set_title(title)
    ax.set_xlabel("colonne")
    ax.set_ylabel("ligne")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, ticks=range(n_clusters))
    cbar.set_label("section")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
