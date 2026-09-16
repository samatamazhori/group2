"""
quick-plot.py — Draw all six curves and intersection points.

Shows the μ-c parameter plane with:
  • contour lines for each curve (solid, colour from metadata)
  • markers + labels for every intersection point
  • legend pulled from curve metadata

Usage:  python quick-plot.py
"""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from data import CURVES, ALL_POINTS


# ── plot domain ──────────────────────────────────────────────────────────────

# covers all intersection points with comfortable padding
MU_MIN, MU_MAX = -2.5, 2.0
C_MIN, C_MAX = -0.75, 7.5
GRID = 1000

MU_grid, C_grid = np.meshgrid(
    np.linspace(MU_MIN, MU_MAX, GRID),
    np.linspace(C_MIN, C_MAX, GRID),
)


# ── figure ───────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlabel('μ')
ax.set_ylabel('c')
ax.set_title('Bifurcation curves in the μ–c plane')
ax.grid(True, alpha=0.3)


# ── plot curves ──────────────────────────────────────────────────────────────

legend_handles = []

for curve in CURVES:
    label = curve.metadata.get('label', curve.name)
    color = curve.metadata.get('color', '#333333')
    lw = curve.metadata.get('linewidth', 2)
    ls = curve.metadata.get('linestyle', '-')

    try:
        Z = curve(MU_grid, C_grid)
        cs = ax.contour(MU_grid, C_grid, Z, levels=[0],
                        colors=color, linewidths=lw, linestyles=ls)
    except Exception as e:
        print(f"  ⚠  {curve.name} failed to plot: {e}")
        continue

    # dummy line for legend
    legend_handles.append(Line2D([0], [0], color=color, linewidth=lw,
                                 linestyle=ls, label=label))


# ── plot intersection points ────────────────────────────────────────────────

xs = [p.mu for p in ALL_POINTS]
ys = [p.c for p in ALL_POINTS]
labels = [p.name for p in ALL_POINTS]

ax.scatter(xs, ys, c='#333333', s=30, zorder=5, edgecolors='white',
           linewidths=0.6)

# Annotate with alternating offset to reduce overlap
for i, (x, y, lab) in enumerate(zip(xs, ys, labels)):
    dx = 8 if i % 2 == 0 else -8
    dy = 6 if i % 3 == 0 else -6
    ax.annotate(lab, (x, y), xytext=(dx, dy),
                textcoords='offset points', fontsize=7,
                color='#333333', fontfamily='monospace',
                arrowprops=dict(arrowstyle='->', color='#999999',
                                lw=0.5))


# ── legend ───────────────────────────────────────────────────────────────────

ax.legend(handles=legend_handles, loc='upper right', fontsize=11,
          framealpha=0.9, edgecolor='#cccccc')

fig.tight_layout()
plt.savefig('mu_c_plane.png', dpi=200)
print('Saved  mu_c_plane.png')