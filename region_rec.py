"""
region_rec.py — Interactive region reconnaissance tool.

Click anywhere on the μ-c plane to see the sign (+/−) of each curve at
that point.  Helps you visually identify the sign signature of each
region, which you can then use to define Region objects in data.py.

Usage:  python region_rec.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from data import CURVES, ALL_POINTS
from box_utils import edge_intersections, BOX

# ── plot domain ──────────────────────────────────────────────────────────────

MU_MIN, MU_MAX = BOX[0], BOX[1]
C_MIN, C_MAX   = BOX[2], BOX[3]
GRID = 1000

MU_grid, C_grid = np.meshgrid(
    np.linspace(MU_MIN, MU_MAX, GRID),
    np.linspace(C_MIN, C_MAX, GRID),
)

# ── edge points (for visual reference) ───────────────────────────────────────

EDGE_POINTS = []
for curve in CURVES:
    EDGE_POINTS.extend(edge_intersections(curve, BOX))

ALL_POINTS_PLOT = ALL_POINTS + EDGE_POINTS

# ── build figure ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlabel('μ')
ax.set_ylabel('c')
ax.set_title('Click anywhere to see curve signs', fontsize=13)
ax.grid(True, alpha=0.3)

# box frame
ax.plot([MU_MIN, MU_MAX, MU_MAX, MU_MIN, MU_MIN],
        [C_MIN, C_MIN, C_MAX, C_MAX, C_MIN],
        color='#cccccc', linewidth=1, linestyle='--')

# ── plot curves ──────────────────────────────────────────────────────────────

legend_handles = []
for curve in CURVES:
    label = curve.metadata.get('label', curve.name)
    color = curve.metadata.get('color', '#333333')
    lw = curve.metadata.get('linewidth', 2)
    try:
        Z = curve(MU_grid, C_grid)
        cs = ax.contour(MU_grid, C_grid, Z, levels=[0],
                        colors=color, linewidths=lw, linestyles='-')
    except Exception as e:
        print(f"  ⚠  {curve.name} failed to plot: {e}")
        continue
    legend_handles.append(Line2D([0], [0], color=color, linewidth=lw,
                                 linestyle='-', label=label))

# ── plot points ──────────────────────────────────────────────────────────────

# real intersection points
xs = [p.mu for p in ALL_POINTS]
ys = [p.c for p in ALL_POINTS]
ax.scatter(xs, ys, c='#333333', s=30, zorder=5, edgecolors='white',
           linewidths=0.6, marker='o')

# edge points
xe = [p.mu for p in EDGE_POINTS]
ye = [p.c for p in EDGE_POINTS]
ax.scatter(xe, ye, c='#cc0000', s=40, zorder=5, edgecolors='white',
           linewidths=0.8, marker='s')

# annotate point names
for p in ALL_POINTS_PLOT:
    dx = 8 if hash(p.name) % 2 == 0 else -8
    dy = 6 if hash(p.name) % 3 == 0 else -6
    ax.annotate(p.name, (p.mu, p.c), xytext=(dx, dy),
                textcoords='offset points', fontsize=7,
                color='#333333', fontfamily='monospace',
                arrowprops=dict(arrowstyle='->', color='#999999', lw=0.5))

#── legend ────────────────────────────────────────────────────────────────────

legend_handles.append(
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#333333',
               markersize=6, label='intersection'))
legend_handles.append(
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#cc0000',
               markersize=7, label='box edge'))
ax.legend(handles=legend_handles, loc='upper right', fontsize=10,
          framealpha=0.9, edgecolor='#cccccc')

# ── info text box (updated on click) ────────────────────────────────────────

info_box = ax.text(
    0.02, 0.02, '',
    transform=ax.transAxes,
    fontsize=10, fontfamily='monospace',
    va='bottom', ha='left',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
              edgecolor='#333333', alpha=0.92),
    zorder=20,
)

# marker for last clicked point
click_marker, = ax.plot([], [], 'o', color='black', markersize=8,
                        markeredgecolor='white', markeredgewidth=1.5,
                        zorder=15)


def format_sign(val, tol=1e-10):
    """Return '+' or '-' for a curve evaluation, with small tolerance."""
    if val > tol:
        return '+'
    elif val < -tol:
        return '-'
    else:
        return '0'


def on_click(event):
    """Handle mouse click: evaluate all curves and display signs."""
    if event.inaxes is None:
        return
    mu_click, c_click = event.xdata, event.ydata

    # update marker
    click_marker.set_data([mu_click], [c_click])

    # evaluate each curve
    lines = []
    lines.append(f'Click: μ={mu_click:.6g}  c={c_click:.6g}')
    lines.append(f'{"Curve":>6}  {"Sign":>5}  {"Value":>14}')
    lines.append('-' * 30)
    for curve in CURVES:
        val = curve(mu_click, c_click)
        sgn = format_sign(val)
        label = curve.metadata.get('label', curve.name)
        lines.append(f'{label:>6}  {sgn:>5}  {val:+.6e}')

    info_box.set_text('\n'.join(lines))
    fig.canvas.draw_idle()


# ── connect click handler ────────────────────────────────────────────────────

fig.canvas.mpl_connect('button_press_event', on_click)

# ── initial message ─────────────────────────────────────────────────────────

info_box.set_text('Click anywhere on the\nμ-c plane to see\ncurve signs.')
fig.tight_layout()
plt.show()