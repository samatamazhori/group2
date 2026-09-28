"""
explorer.py — Interactive workspace for defining arcs and regions.

Workflow:
  1. Run quick-plot.py to see the full map + edge-point printouts
  2. Copy those edge-point definitions into EDGE_POINTS below
  3. Define arcs between consecutive points on each curve
  4. Define regions as closed boundary loops
  5. Run this script to visualise arcs and regions
  6. When satisfied, move definitions into data.py

Usage:  python explorer.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Patch
from matplotlib.colors import to_rgba

from data_model import Arc, Region, Boundary, Constraint, Point
from data import CURVES, INTERSECTIONS, ALL_POINTS, C1, C2, C3, C4, C5, C6
from box_utils import BOX

# ── curve lookup map ──────────────────────────────────────────────────────────
CURVE_MAP = {c.name: c for c in CURVES}

# ═════════════════════════════════════════════════════════════════════════════
# Edge points — copy-paste from quick-plot.py terminal output
# ═════════════════════════════════════════════════════════════════════════════

EDGE_POINTS = [
    # Paste after running quick-plot.py.  Example structure:
    # Point(-2.5, 0.286078, name='C2_edge_0',
    #       metadata={'edge': 'left', 'artificial': True}),
    Point(-2.5, 0, name='C5_edge_0',
        metadata={'edge': 'left', 'artificial': True}),
    Point(-2.5, -0.5, name='C4_edge_0',
        metadata={'edge': 'left', 'artificial': True}),
    Point(-2.5, 0.298870198981962, name='C2_edge_0',
        metadata={'edge': 'left', 'artificial': True}),
    Point(-0.360423281494103, 7.5, name='C2_edge_2',
        metadata={'edge': 'top', 'artificial': True}),
]

# ═════════════════════════════════════════════════════════════════════════════
# All named points on the plane
# ═════════════════════════════════════════════════════════════════════════════




POINTS = list(ALL_POINTS) + EDGE_POINTS
POINT_MAP = {p.name: p for p in POINTS}

# ═════════════════════════════════════════════════════════════════════════════
# Arcs — curve segments between consecutive named points
# ═════════════════════════════════════════════════════════════════════════════

ARCS = [
    # Example:
    # Arc(curve=CURVE_MAP['C5'], start=POINT_MAP['C5_edge_0'], end=POINT_MAP['I45_P1'],
    #     name='A1'),
    Arc(curve=C5, start=POINT_MAP['C5_edge_0'], end=POINT_MAP['I45_P1'],
         name='A1'),
    Arc(curve=C4, start=POINT_MAP['I45_P1'], end=POINT_MAP['C4_edge_0'],
         name='A2'),
    Arc(curve=C2, start=POINT_MAP['C2_edge_0'], end=POINT_MAP['I24_P2'],
         name='A3'),
    Arc(curve=C4, start=POINT_MAP['I45_P1'], end=POINT_MAP['I24_P2'],
         name='A4'),
    Arc(curve=C4, start=POINT_MAP['I24_P2'], end=POINT_MAP['I24_P1'],
         name='A5'),
    Arc(curve=C5, start=POINT_MAP['I45_P1'], end=POINT_MAP['I25_P1'],
         name='A6'),
    Arc(curve=C2, start=POINT_MAP['I24_P2'], end=POINT_MAP['I25_P1'],
         name='A7'),
    Arc(curve=C2, start=POINT_MAP['I25_P1'], end=POINT_MAP['I26_P1'],
         name='A8'),
    Arc(curve=C2, start=POINT_MAP['I24_P1'], end=POINT_MAP['C2_edge_2'],
         name='A9'),


]
ARC_MAP = {p.name: p for p in ARCS}

# ═════════════════════════════════════════════════════════════════════════════
# Regions — sign-defined zones bounded by arcs
# ═════════════════════════════════════════════════════════════════════════════

REGIONS = [
    # Example:
    # Region(
    #     name='R1',
    #     signs={C1: +1, C2: -1, C3: +1},
    #     vertices=[...],
    #     boundary_arcs=[...],
    #     test_point=(0.3, 0.5),
    # ),
    Region(
        name="R1",
        signs={C5:-1, C4: +1},
        vertices=[POINT_MAP['C5_edge_0'],POINT_MAP['I45_P1'],POINT_MAP['C4_edge_0']],
        boundary_arcs=[ARC_MAP['A1'],ARC_MAP['A2']],
        test_point=(-2.424,-1.058)

    ),
    Region(
        name="R2",
        signs={C2:-1, C4: +1},
        vertices=[POINT_MAP['C5_edge_0'],POINT_MAP['I45_P1'],POINT_MAP['I24_P2'],POINT_MAP['C2_edge_0']],
        boundary_arcs=[ARC_MAP['A6']],
        test_point=(-1.0,0.143)

    ),
    Region(
        name="R3",
        signs={C2:-1,C4: -1,C5: +1},
        vertices=[POINT_MAP['I25_P1'],POINT_MAP['I45_P1'],POINT_MAP['I24_P2'],POINT_MAP['C2_edge_0']],
        boundary_arcs=[ARC_MAP['A1'],ARC_MAP['A3'],ARC_MAP['A4']],
        test_point=(-2.25,0.200)

    ),
    Region(
        name="R4",
        signs={C1:+1,C2: -1,C3: -1, C4: +1, C5: +1, C6: -1},
        vertices=[POINT_MAP['C2_edge_0'],POINT_MAP['I24_P2'],POINT_MAP['I24_P1'],POINT_MAP['C2_edge_2']],
        boundary_arcs=[ARC_MAP['A3'],ARC_MAP['A5'],ARC_MAP['A9']],
        test_point=(-1.00,3.000)

    )
]

# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════

MU_MIN, MU_MAX, C_MIN, C_MAX = BOX
GRID = 1000
MU_grid, C_grid = np.meshgrid(
    np.linspace(MU_MIN, MU_MAX, GRID),
    np.linspace(C_MIN, C_MAX, GRID),
)


def plot(arcs=ARCS, regions=REGIONS):
    """Plot the current state: curves, points, arcs, and regions."""
    n_regions = len(regions)
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlabel('μ')
    ax.set_ylabel('c')
    ax.set_title(
        f'Explorer — {len(arcs)} arcs, {n_regions} regions'
        if n_regions else f'Explorer — {len(arcs)} arcs, no regions yet',
        fontsize=13)
    ax.grid(True, alpha=0.3)

    # box frame
    ax.plot([MU_MIN, MU_MAX, MU_MAX, MU_MIN, MU_MIN],
            [C_MIN, C_MIN, C_MAX, C_MAX, C_MIN],
            color='#cccccc', linewidth=1, linestyle='--')

    # ── regions ───────────────────────────────────────────────────────────
    region_colors = ['#cce5ff', '#ffcccb', '#ccffcc', '#ffffcc',
                     '#e5ccff', '#ffcccc', '#ccffff', '#ffe5cc']
    for ri, region in enumerate(regions):
        if region.boundary_arcs:
            # collect polygon vertices from boundary arcs
            verts = []
            for arc in region.boundary_arcs:
                verts.append((arc.start.mu, arc.start.c))
            verts.append((region.boundary_arcs[-1].end.mu,
                          region.boundary_arcs[-1].end.c))
            color = region_colors[ri % len(region_colors)]
            poly = Polygon(verts, closed=True, facecolor=color,
                           edgecolor='none', alpha=0.35, zorder=1)
            ax.add_patch(poly)

        # label region at test point
        if region.test_point:
            ax.text(region.test_point[0], region.test_point[1],
                    region.name, fontsize=11, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#999999', alpha=0.85),
                    zorder=10)

    # ── arcs ──────────────────────────────────────────────────────────────
    for arc in arcs:
        mu0, c0 = arc.start.mu, arc.start.c
        mu1, c1 = arc.end.mu, arc.end.c
        # small local grid covering just this arc's bounding box
        # pad by epsilon if range is zero (horizontal/vertical curve)
        eps_mu = 1e-6 if mu0 == mu1 else 0
        eps_c  = 1e-6 if c0 == c1 else 0
        local_mu = np.linspace(min(mu0, mu1) - eps_mu, max(mu0, mu1) + eps_mu, 200)
        local_c  = np.linspace(min(c0, c1) - eps_c,  max(c0, c1) + eps_c,  200)
        LM, LC = np.meshgrid(local_mu, local_c)
        try:
            Z = arc.curve(LM, LC)
            ax.contour(LM, LC, Z, levels=[0],
                       colors='black', linewidths=3, zorder=5)
        except Exception:
            pass
        # endpoints — thick red dots
        ax.scatter([mu0, mu1], [c0, c1], c='red', s=80, zorder=6,
                   edgecolors='black', linewidths=1.2)

        # label at midpoint
        ax.annotate(arc.name or '',
                    xy=((mu0 + mu1) / 2, (c0 + c1) / 2), fontsize=8,
                    color='black', fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.15',
                              facecolor='white', edgecolor='black', alpha=0.8),
                    zorder=9)

    # ── curves (thin background) ──────────────────────────────────────────
    for curve in CURVES:
        color = curve.metadata.get('color', '#333333')
        try:
            Z = curve(MU_grid, C_grid)
            ax.contour(MU_grid, C_grid, Z, levels=[0],
                       colors=color, linewidths=1, linestyles='-', alpha=0.4,
                       zorder=0)
        except Exception:
            pass

    # ── points ────────────────────────────────────────────────────────────
    xs_int = [p.mu for p in ALL_POINTS]
    ys_int = [p.c for p in ALL_POINTS]
    ax.scatter(xs_int, ys_int, c='#333333', s=30, zorder=8,
               edgecolors='white', linewidths=0.6, marker='o')

    if EDGE_POINTS:
        xe = [p.mu for p in EDGE_POINTS]
        ye = [p.c for p in EDGE_POINTS]
        ax.scatter(xe, ye, c='#cc0000', s=40, zorder=8,
                   edgecolors='white', linewidths=0.8, marker='s')

    # annotate
    for p in POINTS:
        dx = 8 if hash(p.name) % 2 == 0 else -8
        dy = 6 if hash(p.name) % 3 == 0 else -6
        ax.annotate(p.name, (p.mu, p.c), xytext=(dx, dy),
                    textcoords='offset points', fontsize=7,
                    color='#333333', fontfamily='monospace',
                    arrowprops=dict(arrowstyle='->', color='#999999',
                                    lw=0.5),
                    zorder=12)

    fig.tight_layout()
    plt.show()


def print_arcs(arcs):
    """Pretty-print arc definitions for review / copying into data.py."""
    for arc in arcs:
        print(f'Arc(curve=CURVE_MAP[{arc.curve.name!r}],')
        print(f'    start=POINT_MAP[{arc.start.name!r}],')
        print(f'    end=POINT_MAP[{arc.end.name!r}],')
        if arc.constraint:
            print(f'    constraint={arc.constraint!r},')
        print(f'    name={arc.name!r}),')
        print()


def print_regions(regions):
    """Pretty-print region definitions for review / copying into data.py."""
    for r in regions:
        signs = ', '.join(f"CURVE_MAP[{c.name!r}]: {s}" for c, s in r.signs.items())
        print(f"Region(name={r.name!r},")
        print(f"      signs={{{signs}}},")
        if r.boundary_arcs:
            print(f"      boundary_arcs=[{', '.join(a.name for a in r.boundary_arcs)}],")
        if r.test_point:
            print(f"      test_point={r.test_point},")
        print("),")
        print()


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    if '--arcs' in sys.argv:
        print_arcs(ARCS)
    elif '--regions' in sys.argv:
        print_regions(REGIONS)
    else:
        plot()
