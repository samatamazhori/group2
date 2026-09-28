"""gallery.py — Grid of local phase portraits.

Reuses the same ``PhasePortrait`` + layers unchanged.
Designed for papers and Jupyter notebooks.
"""

import matplotlib.pyplot as plt
import numpy as np

from .masks import R2Mask
from .layers import EquilibriaLayer, NullclineLayer, QuiverLayer, OrbitLayer
from .portrait import PhasePortrait


def gallery_local(system, branches=None, n_cols=3, figsize=None, **layer_kwargs):
    """Create a grid of local phase portraits, one per equilibrium.

    Parameters
    ----------
    system : System
        Dynamical system (must be at the desired μ, c).
    branches : list of str, optional
        Subset of equilibrium names to include (e.g. ``['E0', 'E_A⁺']``).
        ``None`` means all that exist.
    n_cols : int
        Number of columns in the grid (default 3).
    figsize : tuple, optional
        Override figure size.
    **layer_kwargs
        Passed to each local PhasePortrait's layers.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
    portraits : list of PhasePortrait
    """
    system.classify_all()
    all_eqs = system.real_equilibria()

    if branches is not None:
        all_eqs = [e for e in all_eqs if e.name in branches]

    if not all_eqs:
        print('⚠  No equilibria exist at this (μ, c).')
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, 'No equilibria', ha='center', va='center')
        return fig, [ax], []

    n = len(all_eqs)
    n_rows = int(np.ceil(n / n_cols))
    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    portraits = []

    for idx, eq in enumerate(all_eqs):
        ax = axes_flat[idx]

        # Local mask around this equilibrium
        zoom_r = _auto_zoom_radius(eq, all_eqs)
        mask = R2Mask(eq.x_num - zoom_r, eq.x_num + zoom_r,
                      eq.y_num - zoom_r, eq.y_num + zoom_r)

        portrait = PhasePortrait(system, ax=ax, mask=mask,
                                 title=f'{eq.label}  [{eq.type or "?"}]')

        # Default layers for local view
        portrait.add(QuiverLayer(grid_res=20, **layer_kwargs))
        portrait.add(NullclineLayer(**layer_kwargs))
        portrait.add(EquilibriaLayer(branches=[eq.name], show_labels=False,
                                     **layer_kwargs))

        # Circular IC net around the equilibrium
        ics = _circular_ics(eq, zoom_r, n_circles=4, n_angles=12)
        portrait.add(OrbitLayer(ics=ics, **layer_kwargs))

        portrait.draw()
        portraits.append(portrait)

    # Hide unused subplots
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    return fig, axes_flat[:n], portraits


# ── helpers ─────────────────────────────────────────────────────────────

def _auto_zoom_radius(eq, all_eqs, default=0.25, min_radius=0.15,
                      scale=0.35):
    """Pick a zoom radius that scales with distance to nearest other eq."""
    others = [e for e in all_eqs if e is not eq]
    if others:
        dists = [np.hypot(e.x_num - eq.x_num, e.y_num - eq.y_num)
                 for e in others]
        return max(min_radius, min(dists) * scale)
    return default


def _circular_ics(eq, radius, n_circles=4, n_angles=12):
    """Generate a circular net of ICs around an equilibrium."""
    rings = np.linspace(0.025, radius * 0.4, n_circles)
    ics = []
    for r in rings:
        for theta in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
            ics.append((
                eq.x_num + r * np.cos(theta),
                eq.y_num + r * np.sin(theta),
            ))
    return ics