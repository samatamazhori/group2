"""
data.py — Registered curves and intersections for the μ-c catalog.

Curves C1–C6 with plotting metadata (colors, legend labels).
Intersections I12..I56 with named points and symbolic metadata.
"""

import sympy as sp
import numpy as np

from data_model import Curve, Point, Intersection, InfinityPoint

# ── symbols used by all expressions ─────────────────────────────────────────

mu, c = sp.symbols('mu c')

# ══════════════════════════════════════════════════════════════════════════════
# Curves  (all solid lines, Tableau 10 colours)
# ══════════════════════════════════════════════════════════════════════════════

C1 = Curve(
    expr=(4 - 12*c + 9*c**2 - 44*mu + 50*c*mu - 24*c**2*mu
          + 121*mu**2 - 56*c*mu**2 + 16*c**2*mu**2),
    name='C1',
    symbols=(mu, c),
    metadata={
        'role': 'Delta of Q',
        'label': 'Δ_Q',
        'color': '#1f77b4',
        'linestyle': '-',
        'linewidth': 2,
    },
)

C2 = Curve(
    expr=-2*c + c**2 - 2*mu + 10*c*mu + 5*mu**2 - 16*c*mu**2,
    name='C2',
    symbols=(mu, c),
    metadata={
        'role': 'collision of branch A,B',
        'label': 'R',
        'color': '#d62728',
        'linestyle': '-',
        'linewidth': 2,
    },
)

C3 = Curve(
    expr=mu - sp.Rational(2, 5),
    name='C3',
    symbols=(mu, c),
    metadata={
        'role': 'collision of branch A',
        'label': 'μ = ²/₅',
        'color': '#2ca02c',
        'linestyle': '-',
        'linewidth': 2,
    },
)

C4 = Curve(
    expr=c - mu - 2,
    name='C4',
    symbols=(mu, c),
    metadata={
        'role': 'collision of B with 0,1',
        'label': 'c = μ + 2',
        'color': '#9467bd',
        'linestyle': '-',
        'linewidth': 2,
    },
)

C5 = Curve(
    expr=c,
    name='C5',
    symbols=(mu, c),
    metadata={
        'role': 'denominator singularity & collision branch B',
        'label': 'c = 0',
        'color': '#8c564b',
        'linestyle': '-',
        'linewidth': 2,
    },
)

C6 = Curve(
    expr=mu - sp.Rational(1, 4),
    name='C6',
    symbols=(mu, c),
    metadata={
        'role': "x² degeneracy",
        'label': 'μ = ¼',
        'color': '#ff7f0e',
        'linestyle': '-',
        'linewidth': 2,
    },
)


C7 = Curve(
    expr=mu - 1,
    name='C7',
    symbols=(mu, c),
    metadata={
        'role': "B sol degenarcy",
        'label': 'μ = 1',
        'color': '#e377c2',
        'linestyle': '-',
        'linewidth': 2,
    },
)

C8 = Curve(
    expr=(4 + c**3 - 52*mu + 165*mu**2 - 100*mu**3 + 64*mu**4
           - 3*c**2*(1 + 2*mu)
           + c*(-8 + 50*mu - 55*mu**2 + 16*mu**3)),
    name='C8',
    symbols=(mu, c),
    metadata={
        'role': "Hopf curve / TrB numerator",
        'label': 'S',
        'color': '#2ca02c',
        'linestyle': '--',
        'linewidth': 2,
    },
)

CURVES = [C1, C2, C3, C4, C5, C6, C7]


# ══════════════════════════════════════════════════════════════════════════════
# Intersections  —  I{lower#}{higher#}
# ══════════════════════════════════════════════════════════════════════════════

# ── C1 ∩ C2 (2 pts, numerical) ───────────────────────────────────────────
I12 = Intersection(
    curves=(C1, C2),
    points=[
        Point(0.191185, 0.895806, name='I12_P1',
              metadata={'label': 'P₁'}),
        Point(0.967111, 6.89609, name='I12_P2',
              metadata={'label': 'P₂'}),
    ],
    name='I12',
    metadata={'certificate': 'numerical', 'curves_label': 'C1 ∩ C2'},
)

# ── C1 ∩ C5 (1 pt, symbolic: μ = 2/11, c = 0) ─────────────────────────────
I15 = Intersection(
    curves=(C1, C5),
    points=[
        Point(sp.Rational(2, 11), 0.0, name='I15_P1',
              metadata={
                  'label': 'P₁',
                  'symbolic': 'μ = 2/11, c = 0',
              }),
    ],
    name='I15',
    metadata={'certificate': 'symbolic', 'curves_label': 'C1 ∩ C5'},
)

# ── C1 ∩ C6 (1 pt, symbolic: μ = 1/4, c = 3/8) ────────────────────────────
I16 = Intersection(
    curves=(C1, C6),
    points=[
        Point(0.25, 0.375, name='I16_P1',
              metadata={
                  'label': 'P₁',
                  'symbolic': 'μ = 1/4, c = 3/8',
              }),
    ],
    name='I16',
    metadata={'certificate': 'symbolic', 'curves_label': 'C1 ∩ C6'},
)

# ── C2 ∩ C3 (2 pts, symbolic) ──────────────────────────────────────────────
I23 = Intersection(
    curves=(C2, C3),
    points=[
        Point(0.4, 0.0, name='I23_P1',
              metadata={
                  'label': 'P₁',
                  'symbolic': 'μ = 2/5, c = 0',
              }),
        Point(0.4, sp.Rational(14, 25), name='I23_P2',
              metadata={
                  'label': 'P₂',
                  'symbolic': 'μ = 2/5, c = 14/25',
              }),
    ],
    name='I23',
    metadata={'certificate': 'symbolic', 'curves_label': 'C2 ∩ C3'},
)

# ── C2 ∩ C4 (3 pts, symbolic) ──────────────────────────────────────────────
I24 = Intersection(
    curves=(C2, C4),
    points=[
        Point(0.0, 2.0, name='I24_P1',
              metadata={'label': 'P₁', 'symbolic': 'μ = 0, c = 2'}),
        Point(-2 + (3 - np.sqrt(6)) / 2,
              (3 - np.sqrt(6)) / 2,
              name='I24_P2',
              metadata={
                  'label': 'P₂',
                  'symbolic': 'μ = -2 + (3-√6)/2, c = (3-√6)/2',
              }),
        Point(-2 + (3 + np.sqrt(6)) / 2,
              (3 + np.sqrt(6)) / 2,
              name='I24_P3',
              metadata={
                  'label': 'P₃',
                  'symbolic': 'μ = -2 + (3+√6)/2, c = (3+√6)/2',
              }),
    ],
    name='I24',
    metadata={'certificate': 'symbolic', 'curves_label': 'C2 ∩ C4'},
)

# ── C2 ∩ C5 (2 pts, symbolic) ──────────────────────────────────────────────
I25 = Intersection(
    curves=(C2, C5),
    points=[
        Point(0.0, 0.0, name='I25_P1',
              metadata={'label': 'P₁', 'symbolic': 'μ = 0, c = 0'}),
        Point(0.4, 0.0, name='I25_P2',
              metadata={'label': 'P₂', 'symbolic': 'μ = 2/5, c = 0'}),
    ],
    name='I25',
    metadata={'certificate': 'symbolic', 'curves_label': 'C2 ∩ C5'},
)

# ── C2 ∩ C6 (2 pts, symbolic) ──────────────────────────────────────────────
I26 = Intersection(
    curves=(C2, C6),
    points=[
        Point(0.25, -0.25, name='I26_P1',
              metadata={'label': 'P₁', 'symbolic': 'μ = 1/4, c = -1/4'}),
        Point(0.25, 0.75, name='I26_P2',
              metadata={'label': 'P₂', 'symbolic': 'μ = 1/4, c = 3/4'}),
    ],
    name='I26',
    metadata={'certificate': 'symbolic', 'curves_label': 'C2 ∩ C6'},
)

# ── C3 ∩ C4 (1 pt, symbolic) ───────────────────────────────────────────────
I34 = Intersection(
    curves=(C3, C4),
    points=[
        Point(0.4, sp.Rational(12, 5), name='I34_P1',
              metadata={'label': 'P₁', 'symbolic': 'μ = 2/5, c = 12/5'}),
    ],
    name='I34',
    metadata={'certificate': 'symbolic', 'curves_label': 'C3 ∩ C4'},
)

# ── C3 ∩ C5 (1 pt, symbolic) ───────────────────────────────────────────────
I35 = Intersection(
    curves=(C3, C5),
    points=[
        Point(0.4, 0.0, name='I35_P1',
              metadata={'label': 'P₁', 'symbolic': 'μ = 2/5, c = 0'}),
    ],
    name='I35',
    metadata={'certificate': 'symbolic', 'curves_label': 'C3 ∩ C5'},
)

# ── C4 ∩ C5 (1 pt, symbolic) ───────────────────────────────────────────────
I45 = Intersection(
    curves=(C4, C5),
    points=[
        Point(-2.0, 0.0, name='I45_P1',
              metadata={'label': 'P₁', 'symbolic': 'μ = -2, c = 0'}),
    ],
    name='I45',
    metadata={'certificate': 'symbolic', 'curves_label': 'C4 ∩ C5'},
)

# ── C4 ∩ C6 (1 pt, symbolic) ───────────────────────────────────────────────
I46 = Intersection(
    curves=(C4, C6),
    points=[
        Point(0.25, sp.Rational(9, 4), name='I46_P1',
              metadata={'label': 'P₁', 'symbolic': 'μ = 1/4, c = 9/4'}),
    ],
    name='I46',
    metadata={'certificate': 'symbolic', 'curves_label': 'C4 ∩ C6'},
)

# ── C5 ∩ C6 (1 pt, symbolic) ───────────────────────────────────────────────
I56 = Intersection(
    curves=(C5, C6),
    points=[
        Point(0.25, 0.0, name='I56_P1',
              metadata={'label': 'P₁', 'symbolic': 'μ = 1/4, c = 0'}),
    ],
    name='I56',
    metadata={'certificate': 'symbolic', 'curves_label': 'C5 ∩ C6'},
)

# ── C2 ∩ C7 (2 pts, symbolic) ──────────────────────────────────────────────
I27 = Intersection(
    curves=(C2, C7),
    points=[
        Point(1.0, 4 - np.sqrt(13), name='I27_P1',
              metadata={
                  'label': 'P₁',
                  'symbolic': 'μ = 1, c = 4 - √13',
              }),
        Point(1.0, 4 + np.sqrt(13), name='I27_P2',
              metadata={
                  'label': 'P₂',
                  'symbolic': 'μ = 1, c = 4 + √13',
              }),
    ],
    name='I27',
    metadata={'certificate': 'symbolic', 'curves_label': 'C2 ∩ C7'},
)

# ── C4 ∩ C7 (1 pt, symbolic) ───────────────────────────────────────────────
I47 = Intersection(
    curves=(C4, C7),
    points=[
        Point(1.0, 3.0, name='I47_P1',
              metadata={
                  'label': 'P₁',
                  'symbolic': 'μ = 1, c = 3',
              }),
    ],
    name='I47',
    metadata={'certificate': 'symbolic', 'curves_label': 'C4 ∩ C7'},
)

# ── C5 ∩ C7 (1 pt, symbolic) ───────────────────────────────────────────────
I57 = Intersection(
    curves=(C5, C7),
    points=[
        Point(1.0, 0.0, name='I57_P1',
              metadata={
                  'label': 'P₁',
                  'symbolic': 'μ = 1, c = 0',
              }),
    ],
    name='I57',
    metadata={'certificate': 'symbolic', 'curves_label': 'C5 ∩ C7'},
)

INTERSECTIONS = [I12, I15, I16, I23, I24, I25, I26, I27, I34, I35, I45, I46, I47, I56, I57]


# ── convenience: flat list of all points (for plotting) ─────────────────────

ALL_POINTS = []
for ix in INTERSECTIONS:
    ALL_POINTS.extend(ix.points)


# ── Infinity Points (box-edge crossings for C1–C7) ──────────────────────────
"""
Each InfinityPoint represents where a curve crosses the box boundary.
The curve + edge + index triplet uniquely identifies the crossing.
Change the box and the (\u03bc,c) recomputes automatically.
"""

from box_utils import edge_intersections, BOX as _BOX

INFINITY_POINTS = []
for _curve in [C1, C2, C3, C4, C5, C6, C7]:
    _pts = edge_intersections(_curve, _BOX)
    _edge_counts = {}
    for _p in _pts:
        _e = _p.metadata['edge']
        _idx = _edge_counts.get(_e, 0)
        _suffix = '' if _idx == 0 else f'_{_idx}'
        _ip = InfinityPoint(
            _curve, _e, index=_idx, box=_BOX,
            name=f'{_curve.name}_inf_{_e}{_suffix}',
        )
        INFINITY_POINTS.append(_ip)
        _edge_counts[_e] = _idx + 1
