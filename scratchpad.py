"""
scratchpad.py — Workspace for defining arcs, boundaries, and regions.

All definitions here are experimental — once validated they get promoted
to ``data.py``.

Current region under construction:
  Signature:  C1=+  C2=+  C3=−  C4=−  C5=−  C6=−  C7=−
  Boundary:   C6_inf_bottom → I26_P1   (along C6, μ=¼)
              I26_P1        → I25_P1   (along C2)
              I25_P1        → I45_P1   (along C5, c=0)
              I45_P1        → C4_inf_left  (along C4, c=μ+2)
              box edges C4_inf_left → C6_inf_bottom  (left + bottom)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless for save; change to 'MacOSX' for interactive

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Imports
# ─────────────────────────────────────────────────────────────────────────────

from data import (
    C1, C2, C3, C4, C5, C6, C7,
    I16, I25, I26, I45, I46, I56,
    ALL_POINTS, INFINITY_POINTS, INTERSECTIONS,
)
from data_model import Arc, Region, Boundary, Point
from view import View, DEFAULT_VIEWS
from plane import PlaneBuilder

print('=' * 65)
print('SCRATCHPAD — Region definition')
print('=' * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Locate the specific points we need
# ─────────────────────────────────────────────────────────────────────────────

def find_point(name):
    """Scour ALL_POINTS + INFINITY_POINTS + intersection points for *name*."""
    for p in ALL_POINTS:
        if p.name == name:
            return p
    for ip in INFINITY_POINTS:
        if ip.name == name:
            return ip
    for ix in INTERSECTIONS:
        for p in ix.points:
            if p.name == name:
                return p
    raise KeyError(f'Point {name!r} not found')

# Grab the endpoints
I26_P1   = find_point('I26_P1')
I25_P1   = find_point('I25_P1')
I45_P1   = find_point('I45_P1')
C6_inf_bottom = find_point('C6_inf_bottom')
C4_inf_left   = find_point('C4_inf_left')

print('\n── Endpoints ──')
for p in [I26_P1, I25_P1, I45_P1, C6_inf_bottom, C4_inf_left]:
    print(f'  {p.name:20s}  μ={p.mu:.6g}  c={p.c:.6g}')

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Create Arcs (segments of a curve between two points)
# ─────────────────────────────────────────────────────────────────────────────

# Arc 1: C6 from bottom-edge infinity point up to I26_P1
arc_1 = Arc(
    curve=C6,
    start=C6_inf_bottom,
    end=I26_P1,
    name='C6_bottom→I26',
    metadata={'color': C6.metadata['color']},
)

# Arc 2: C2 from I26_P1 to I25_P1
arc_2 = Arc(
    curve=C2,
    start=I26_P1,
    end=I25_P1,
    name='C2_I26→I25',
    metadata={'color': C2.metadata['color']},
)

# Arc 3: C5 from I25_P1 to I45_P1  (c=0, leftward)
arc_3 = Arc(
    curve=C5,
    start=I25_P1,
    end=I45_P1,
    name='C5_I25→I45',
    metadata={'color': C5.metadata['color']},
)

# Arc 4: C4 from I45_P1 to C4_inf_left  (c=μ+2, leftward to box edge)
arc_4 = Arc(
    curve=C4,
    start=I45_P1,
    end=C4_inf_left,
    name='C4_I45→left',
    metadata={'color': C4.metadata['color']},
)

arcs = [arc_1, arc_2, arc_3, arc_4]

print('\n── Arcs ──')
for a in arcs:
    print(f'  {a.name:20s}  {a.start.name:20s} → {a.end.name:20s}  on {a.curve.name}')
    print(f'      endpoints: ({a.start.mu:.4g}, {a.start.c:.4g}) → ({a.end.mu:.4g}, {a.end.c:.4g})')

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Create Boundaries (arc + left/right region annotations)
# ─────────────────────────────────────────────────────────────────────────────
# 5.  Build the Region vertices — closed polygon from the 4 arcs + box edges
# ─────────────────────────────────────────────────────────────────────────────

R_scratch_name = 'R_scratch'

# The region is bounded by the 4 arcs above plus the box edges that connect
# C4_inf_left → C6_inf_bottom (left edge downward, then bottom edge rightward).
#
# Vertices in clockwise order:
#   C6_inf_bottom → (along C6 up) → I26_P1 → (along C2) → I25_P1
#   → (along C5 left) → I45_P1 → (along C4 up-left) → C4_inf_left
#   → (left edge down) → (-2.5, -0.75) → (bottom edge right) → C6_inf_bottom

c_lo = -0.75
mu_lo = -2.5

# Box-edge corner
corner_bl = Point(mu_lo, c_lo, name='corner_BL')

vertices = [
    C6_inf_bottom,
    I26_P1,
    I25_P1,
    I45_P1,
    C4_inf_left,
    corner_bl,
]

print('\n── Region vertices (clockwise) ──')
for v in vertices:
    print(f'  {v.name:20s}  μ={v.mu:.6g}  c={v.c:.6g}')

# Compute centroid for test-point
mu_cent = np.mean([v.mu for v in vertices])
c_cent  = np.mean([v.c for v in vertices])

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Create the Region
# ─────────────────────────────────────────────────────────────────────────────

# Sign signature: C1=+  C2=+  C3=−  C4=−  C5=−  C6=−  C7=−
R_scratch = Region(
    name=R_scratch_name,
    signs={
        C1: +1,
        C2: +1,
        C3: -1,
        C4: -1,
        C5: -1,
        C6: -1,
        C7: -1,
    },
    vertices=vertices,
    boundary_arcs=arcs,
    test_point=(mu_cent, c_cent),
    metadata={
        'color': '#fff4b0',    # light yellow
        'alpha': 0.35,
        'edge_color': '#888888',
    },
)

print('\n── Region ──')
print(f'  name:       {R_scratch.name}')
print(f'  test point: μ={mu_cent:.4g}, c={c_cent:.4g}')
print(f'  vertices:   {len(vertices)}')
print(f'  arcs:       {len(arcs)}')
print('\n  signs:')
for curve, sgn in R_scratch.signs.items():
    print(f'    {curve.name:4s}  →  {"+" if sgn > 0 else "-"}')

# Verify test point satisfies signs
ok = R_scratch.contains(mu_cent, c_cent, tol=1e-6)
print(f'\n  contains(test_point)?  {ok}')

# ─────────────────────────────────────────────────────────────────────────────
# 6b.  Create Boundaries (must come after the Region object)
# ─────────────────────────────────────────────────────────────────────────────

boundaries = [
    Boundary(region_left=None, region_right=R_scratch, arc=arc_1,
             name='B_C6_bottom'),
    Boundary(region_left=None, region_right=R_scratch, arc=arc_2,
             name='B_C2_I26_I25'),
    Boundary(region_left=None, region_right=R_scratch, arc=arc_3,
             name='B_C5_I25_I45'),
    Boundary(region_left=None, region_right=R_scratch, arc=arc_4,
             name='B_C4_I45_left'),
]

print('\n── Boundaries ──')
for b in boundaries:
    print(f'  {b.name:20s}  arc={b.arc.name}')

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Plot everything
# ─────────────────────────────────────────────────────────────────────────────

print('\n── Plotting ──')

# Custom view that shows the full region
view = View(name='scratch', mu_min=-2.5, mu_max=1.5, c_min=-0.75, c_max=1.5,
            description='Scratch region + context')

pb = PlaneBuilder(view=view, figsize=(14, 8), title='Scratch: region definition')

# Layers: regions at back, then curves, then points, then annotations
pb.add(R_scratch)                   # filled region
pb.add([C1, C2, C3, C4, C5, C6, C7])  # all curves for context
pb.add([I26_P1, I25_P1, I45_P1])   # key intersection points
pb.add([C6_inf_bottom, C4_inf_left])  # infinity endpoints
pb.add(boundaries)                  # boundary annotations

# Draw the arc labels directly
print('\n── Arc annotation text placements ──')
for a in arcs:
    mx = (a.start.mu + a.end.mu) / 2
    my = (a.start.c + a.end.c) / 2
    print(f'  {a.name:20s}  mid=({mx:.4g}, {my:.4g})  on {a.curve.name}')

# Save
path = pb.save('scratch_region', dpi=200)
print(f'\nSaved → {path}')

# Extra: verify containment for some nearby points
print('\n── Container check (nearby points) ──')
test_points = [
    ('inside',      mu_cent, c_cent),
    ('near I25_P1', 0.05, 0.05),
    ('near I26_P1', 0.24, -0.20),
    ('near I45_P1', -1.9, 0.05),
    ('outside top', 0.5, 0.5),
    ('outside right', 1.0, -0.3),
]
for label, mu_t, c_t in test_points:
    ok = R_scratch.contains(mu_t, c_t, tol=1e-6)
    print(f'  {label:15s}  μ={mu_t:.4g}  c={c_t:.4g}  →  {"IN" if ok else "OUT"}')

print('\nDone.')