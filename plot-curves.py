"""
plot-curves.py — Per-curve sign plots for inspection.

For each curve in the catalog, draw a standalone figure that shows:
  • the curve itself (F = 0, thick black line)
  • shaded regions where F > 0 (positive) and F < 0 (negative)
  • a descriptive title and legend

Usage:
    python plot-curves.py          # saves PNGs for every curve
    python plot-curves.py C1 C3    # saves only C1 and C3
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

from data import CURVES

# ── output directory ──────────────────────────────────────────────────────

FIGS_DIR = os.path.join(os.path.dirname(__file__) or '.', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

# ── plot domain (same as quick-plot) ──────────────────────────────────────

MU_MIN, MU_MAX = -2.5, 2.0
C_MIN, C_MAX = -0.75, 7.5
GRID = 1000

MU_grid, C_grid = np.meshgrid(
    np.linspace(MU_MIN, MU_MAX, GRID),
    np.linspace(C_MIN, C_MAX, GRID),
)


def plot_single_curve(curve):
    """Produce a standalone sign-plot for one curve."""
    name = curve.name or 'curve'
    label = curve.metadata.get('label', name)

    # Evaluate
    Z = curve(MU_grid, C_grid)

    neg_color = '#cc0000'   # strong red  for F < 0
    pos_color = '#0055cc'   # strong blue for F > 0
    zero_color = '#111111'

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlabel('μ')
    ax.set_ylabel('c')
    ax.set_title(f'{label}  —  sign regions', fontsize=13)
    ax.grid(True, alpha=0.25)

    # ── filled contour for sign regions ─────────────────────────────────
    # Two-level: everything < 0 gets neg_color, > 0 gets pos_color
    # We use a two-colour ListedColormap + contourf with levels=[-inf, 0, +inf]
    cmap = ListedColormap([neg_color, pos_color])
    # need to extend the grid slightly beyond min/max so no white holes
    levels = [-1e20, 0, 1e20]
    cf = ax.contourf(MU_grid, C_grid, Z, levels=levels, cmap=cmap,
                     extend='both')

    # ── the curve itself (F = 0) ───────────────────────────────────────
    ax.contour(MU_grid, C_grid, Z, levels=[0],
               colors=zero_color, linewidths=2.5, linestyles='-')

    # ── legend ─────────────────────────────────────────────────────────
    neg_patch = Patch(facecolor=neg_color, edgecolor='none',
                      label='F(μ, c) < 0')
    pos_patch = Patch(facecolor=pos_color, edgecolor='none',
                      label='F(μ, c) > 0')
    zero_line = plt.Line2D([0], [0], color=zero_color, linewidth=2.5,
                           label='F(μ, c) = 0')
    ax.legend(handles=[neg_patch, pos_patch, zero_line],
              loc='upper right', fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fname = f'sign-{name}.png'
    fpath = os.path.join(FIGS_DIR, fname)
    fig.savefig(fpath, dpi=150)
    print(f'  Saved  {fpath}')
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Which curves to plot?
    args = sys.argv[1:]
    if args:
        # user supplied curve names on the command line
        names_wanted = set(args)
        to_plot = [c for c in CURVES if c.name in names_wanted]
        if not to_plot:
            print(f'No matching curves for {args}.  Available: '
                  f'{[c.name for c in CURVES]}')
            sys.exit(1)
    else:
        to_plot = CURVES

    for curve in to_plot:
        print(f'Plotting {curve.name} …')
        plot_single_curve(curve)

    print('Done.')