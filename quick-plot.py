"""
quick-plot.py — Draw all seven curves, intersection points, and box-edge points.

Shows the μ-c parameter plane with:
  • contour lines for each curve (solid, colour from metadata)
  • markers for real intersection points (circles) + box-edge points (squares)
  • labels for every named point
  • legend pulled from curve metadata

Prints edge-crossing points to terminal so you can copy-paste into explorer.py.

Usage:  python quick-plot.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from data import CURVES, ALL_POINTS, INFINITY_POINTS
from box_utils import BOX

# ── plot domain ──────────────────────────────────────────────────────────────

MU_MIN, MU_MAX = BOX[0], BOX[1]
C_MIN, C_MAX   = BOX[2], BOX[3]
GRID = 1000

MU_grid, C_grid = np.meshgrid(
    np.linspace(MU_MIN, MU_MAX, GRID),
    np.linspace(C_MIN, C_MAX, GRID),
)

# ── infinity points are registered in data.py as INFINITY_POINTS ──────
#    Each one carries (.mu, .c, .name) and plots as a black rectangle.

ALL_POINTS_PLOT = ALL_POINTS + INFINITY_POINTS

print(f'── {len(INFINITY_POINTS)} infinity points (box-edge crossings for C1\u2013C7) ──')
for p in INFINITY_POINTS:
    print(f'  {p.name}: ({p.mu:.6g}, {p.c:.6g})  [{p.edge}] on {p.curve.name}')

# ── figure ───────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlabel('μ')
ax.set_ylabel('c')
ax.set_title('Bifurcation curves in the μ–c plane')
ax.grid(True, alpha=0.3)

# draw box frame for clarity
ax.plot([MU_MIN, MU_MAX, MU_MAX, MU_MIN, MU_MIN],
        [C_MIN, C_MIN, C_MAX, C_MAX, C_MIN],
        color='#cccccc', linewidth=1, linestyle='--')

# ── plot curves ──────────────────────────────────────────────────────────────

legend_handles = []

for curve in CURVES:
    label = curve.metadata.get('label', curve.name)
    color = curve.metadata.get('color', '#333333')
    lw = curve.metadata.get('linewidth', 2)
    ls = curve.metadata.get('linestyle', '-')

    try:
        Z = curve(MU_grid, C_grid)
        ax.contour(MU_grid, C_grid, Z, levels=[0],
                   colors=color, linewidths=lw, linestyles=ls)
    except Exception as e:
        print(f"  ⚠  {curve.name} failed to plot: {e}")
        continue

    legend_handles.append(Line2D([0], [0], color=color, linewidth=lw,
                                 linestyle=ls, label=label))

# ── plot points ──────────────────────────────────────────────────────────────

# real intersection points — circles
xs = [p.mu for p in ALL_POINTS]
ys = [p.c for p in ALL_POINTS]
ax.scatter(xs, ys, c='#333333', s=30, zorder=5, edgecolors='white',
           linewidths=0.6, marker='o', label='intersection')

# infinity points — black rectangles
xi = [p.mu for p in INFINITY_POINTS]
yi = [p.c for p in INFINITY_POINTS]
ax.scatter(xi, yi, c='#000000', s=40, zorder=5, edgecolors='#000000',
           linewidths=1.2, marker='s', label='box edge')

# ── annotate all ────────────────────────────────────────────────────────────

for p in ALL_POINTS_PLOT:
    dx = 8 if hash(p.name) % 2 == 0 else -8
    dy = 6 if hash(p.name) % 3 == 0 else -6
    ax.annotate(p.name, (p.mu, p.c), xytext=(dx, dy),
                textcoords='offset points', fontsize=7,
                color='#333333', fontfamily='monospace',
                arrowprops=dict(arrowstyle='->', color='#999999', lw=0.5))

# ── legend ───────────────────────────────────────────────────────────────────

legend_handles.append(
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#333333',
               markersize=6, label='intersection (real)'))
legend_handles.append(
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#000000',
               markersize=7, label='box edge (infinity point)'))

ax.legend(handles=legend_handles, loc='upper right', fontsize=11,
          framealpha=0.9, edgecolor='#cccccc')

fig.tight_layout()
#plt.savefig('mu_c_plane.png', dpi=200)
#print('Saved  mu_c_plane.png')
plt.show()
