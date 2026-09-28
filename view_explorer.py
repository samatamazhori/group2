"""
view_explorer.py — Plot the 4 new views and save figures.

Views:
  1. right_panel   : μ ∈ [-4,   0  ], c ∈ [-3,  7]   (left half of plane)
  2. bottom_center : μ ∈ [ 0,   0.5], c ∈ [-1,  3]   (zoom into busy bottom)
  3. left_panel    : μ ∈ [ 0.3, 2  ], c ∈ [-1,  4]   (centre-right)
  4. top_panel     : μ ∈ [ 0.4, 1  ], c ∈ [ 5, 15]   (high-c strip)
"""

import matplotlib
matplotlib.use('Agg')  # headless for saving

from view import View
from plane import PlaneBuilder
from data import C1, C2, C3, C4, C5, C6, C7, C8, ALL_POINTS, INFINITY_POINTS

# ── layers shared by every view ─────────────────────────────────────────

curves = [C1, C2, C3, C4, C5, C6, C7]
points = ALL_POINTS

# ── views to render ─────────────────────────────────────────────────────

views = [
    View.right_panel(),
    View.bottom_center(),
    View.left_panel(),
    View.top_panel(),
]

print('=' * 65)
print('VIEW EXPLORER — plotting 4 new views')
print('=' * 65)

for view in views:
    print(f'\n── {view.name} ──')
    print(f'    {view}')

    pb = PlaneBuilder(view=view, figsize=(12, 8))
    pb.add(curves)
    pb.add(points)

    path = pb.save(dpi=200)
    print(f'    → {path}')

print('\nDone — all 4 views saved.')