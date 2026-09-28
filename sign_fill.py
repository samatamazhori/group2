"""
sign_fill.py — Given a sign signature, colour the region(s) of the μ-c
plane where that pattern holds true.

Usage:
    python sign_fill.py  "+ + + - - - -"
    python sign_fill.py  "+ - - + + + +"

The signature must have exactly 7 tokens (one per curve C1…C7), each
being '+' or '-'.  An additional '0' means "on the curve" (≈0).

Examples:
    # region below C5 (c=0) and below C4 (c=μ+2) → negative for both
    python sign_fill.py  "? ? ? - - ? ?"

    # region between C2 branches, above C5, below C4
    python sign_fill.py  "? - ? - + ? ?"
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from data import CURVES, ALL_POINTS
from box_utils import edge_intersections, BOX

# ── plot domain ──────────────────────────────────────────────────────────────

MU_MIN, MU_MAX = BOX[0], BOX[1]
C_MIN, C_MAX   = BOX[2], BOX[3]
#MU_MIN, MU_MAX = -0.1, 0.6
#C_MIN, C_MAX   = -0.5, 3
GRID = 2000  # high-res for clean region boundary

MU_grid, C_grid = np.meshgrid(
    np.linspace(MU_MIN, MU_MAX, GRID),
    np.linspace(C_MIN, C_MAX, GRID),
)

# ── parse signature ─────────────────────────────────────────────────────────

USAGE = (
    "Usage: python sign_fill.py  <7 signs>\n"
    "  e.g.  python sign_fill.py  '+ + + - - - -'\n"
    "  each sign is +, -, ? (wildcard), or 0 (on-curve)"
)


def parse_signature(argv):
    """Return list of 7 codes: +1, -1, 0, or None (wildcard)."""
    # join all args into one string, then split
    raw = ' '.join(argv[1:])
    tokens = raw.replace(',', ' ').split()
    tokens = [t.strip() for t in tokens if t.strip()]

    if len(tokens) < 7:
        print(USAGE)
        print(f"Got {len(tokens)} token(s), need exactly 7.")
        sys.exit(1)

    codes = []
    for t in tokens[:7]:
        if t in ('+', '+1', '1'):
            codes.append(+1)
        elif t in ('-', '-1', '-1'):
            codes.append(-1)
        elif t in ('0',):
            codes.append(0)
        elif t in ('?', '*', 'any', '.'):
            codes.append(None)
        else:
            print(f"Unknown sign token: {t!r}")
            print(USAGE)
            sys.exit(1)
    return codes


signature = parse_signature(sys.argv)

# ── build mask ───────────────────────────────────────────────────────────────

print('Evaluating curves…')
tol = 1e-10
mask = np.ones_like(MU_grid, dtype=bool)

for i, (curve, target_sign) in enumerate(zip(CURVES, signature)):
    label = curve.metadata.get('label', curve.name)
    if target_sign is None:
        print(f'  {label}: wildcard (ignored)')
        continue

    Z = curve(MU_grid, C_grid)

    if target_sign == 0:
        # near-zero (on-curve) — use small band
        condition = np.abs(Z) <= tol
        print(f'  {label}: near zero  (band ±{tol})')
    elif target_sign == +1:
        condition = Z > tol
        print(f'  {label}: positive  (F > 0)')
    else:
        condition = Z < -tol
        print(f'  {label}: negative  (F < 0)')

    mask &= condition

n_hit = np.sum(mask)
print(f'\n  → {n_hit} grid points match ({100 * n_hit / mask.size:.1f}%)')

# ── figure ───────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlabel('μ')
ax.set_ylabel('c')

# build signature string for title
sig_str = ' '.join(
    '+' if s == 1 else ('−' if s == -1 else ('0' if s == 0 else '?'))
    for s in signature
)
curve_names = [c.metadata.get('label', c.name) for c in CURVES]
title_parts = [f'{cn}={s}' for cn, s in zip(curve_names, signature)]
ax.set_title(f'Region:  {"  ".join(title_parts)}', fontsize=12)
ax.grid(True, alpha=0.3)

# box frame
ax.plot([MU_MIN, MU_MAX, MU_MAX, MU_MIN, MU_MIN],
        [C_MIN, C_MIN, C_MAX, C_MAX, C_MIN],
        color='#cccccc', linewidth=1, linestyle='--')

# ── filled region ───────────────────────────────────────────────────────────

if n_hit > 0:
    # use contourf with one level to get a clean filled polygon
    ax.contourf(MU_grid, C_grid, mask.astype(float),
                levels=[0.5, 1.5], colors=['#66c2a5'], alpha=0.55, zorder=1)
else:
    ax.text(0.5, 0.5, 'No matching region found',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=15, color='#999999', fontstyle='italic',
            zorder=20)

# ── edge points ─────────────────────────────────────────────────────────────

EDGE_POINTS = []
for curve in CURVES:
    EDGE_POINTS.extend(edge_intersections(curve, BOX))

# ── curves (on top of fill) ─────────────────────────────────────────────

legend_handles = []
for curve in CURVES:
    label = curve.metadata.get('label', curve.name)
    color = curve.metadata.get('color', '#333333')
    lw = curve.metadata.get('linewidth', 2)
    Z = curve(MU_grid, C_grid)
    ax.contour(MU_grid, C_grid, Z, levels=[0],
               colors=color, linewidths=lw, linestyles='-')
    legend_handles.append(Line2D([0], [0], color=color, linewidth=lw,
                                 linestyle='-', label=label))

# ── points ──────────────────────────────────────────────────────────────────

# real intersection points
filter_points = [p for p in ALL_POINTS if (p.mu < MU_MAX and p.mu > MU_MIN) and (p.c < C_MAX and p.c > C_MIN) ]
xs = [p.mu for p in filter_points]
ys = [p.c for p in filter_points]

ax.scatter(xs, ys, c='#333333', s=30, zorder=8, edgecolors='white',
           linewidths=0.6, marker='o')

# edge points
#xe = [p.mu for p in EDGE_POINTS]
#ye = [p.c for p in EDGE_POINTS]
#ax.scatter(xe, ye, c='#cc0000', s=40, zorder=8, edgecolors='white',
#           linewidths=0.8, marker='s')

# annotate all points
#ALL_POINTS_PLOT = ALL_POINTS #+ EDGE_POINTS
for p in filter_points:
    dx = 8 if hash(p.name) % 2 == 0 else -8
    dy = 6 if hash(p.name) % 3 == 0 else -6
    ax.annotate(p.name, (p.mu, p.c), xytext=(dx, dy),
                textcoords='offset points', fontsize=7,
                color='#333333', fontfamily='monospace',
                arrowprops=dict(arrowstyle='->', color='#999999', lw=0.5),
                zorder=12)
#
# ── legend ───────────────────────────────────────────────────────────────────

legend_handles.append(
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#333333',
               markersize=6, label='intersection'))
legend_handles.append(
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#cc0000',
               markersize=7, label='box edge'))
legend_handles.append(
    plt.Line2D([0], [0], color='#66c2a5', linewidth=8, alpha=0.55,
               label='matched region'))
ax.legend(handles=legend_handles, loc='upper right', fontsize=10,
          framealpha=0.9, edgecolor='#cccccc')

fig.tight_layout()
plt.show()