"""
test-data-model.py — Smoke tests for the data model.

Run with:  python test-data-model.py

These are not unit tests — they are basic sanity checks that every object
can be constructed, evaluated, and displays something reasonable.
"""

import numpy as np
import sympy as sp

from data_model import (
    BaseObject, Curve, Constraint, Point, InfinityPoint,
    Intersection, Arc, Region, Boundary,
)

# Curves with simple shapes for InfinityPoint tests
from data import C2, C3, C4, C5

passed = 0
failed = 0


def check(label, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f'  ✓  {label}')
    else:
        failed += 1
        print(f'  ✗  {label} (FAILED)')


# ══════════════════════════════════════════════════════════════════════════════
# 1.  BaseObject
# ══════════════════════════════════════════════════════════════════════════════
print('\n── BaseObject ──')

obj = BaseObject(name='test', metadata={'colour': 'blue', 'style': 'dashed'})
check('repr contains class and name', 'BaseObject' in repr(obj) and 'test' in repr(obj))
check('name property', obj.name == 'test')
check('metadata property', obj.metadata['colour'] == 'blue')

# __getattr__ fallback to metadata
check('attr → metadata (flat)', obj.colour == 'blue')

# Nested metadata lookup
obj2 = BaseObject(metadata={'plotting': {'colour': 'red', 'width': 2}})
check('attr → nested metadata', obj2.colour == 'red')
check('attr → nested metadata (width)', obj2.width == 2)

# Setting name retroactively
obj2.name = 'retro'
check('name setter', obj2.name == 'retro')

# __str__ shows metadata
str_out = str(obj)
check('str contains metadata', 'colour' in str_out and 'style' in str_out)

# AttributeError for missing key
try:
    _ = obj.nonexistent
    check('missing attr raises AttributeError', False)
except AttributeError:
    check('missing attr raises AttributeError', True)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Curve
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Curve ──')

MU, C = sp.symbols('mu c')

# F3: μ - 2/5 = 0  (simple)
c3 = Curve(MU - 0.4, name='mu_2_5')
check('curve from sympy', c3.expr == MU - 0.4)
check('curve name', c3.name == 'mu_2_5')
check('curve eval scalar', abs(c3(0.4, 0.0)) < 1e-15)
check('curve eval scalar (non-zero)', abs(c3(1.0, 0.0) - 0.6) < 1e-15)

# F5: c = 0
c5 = Curve('c', name='c_0')
check('curve from str', c5.expr == sp.Symbol('c'))
check('curve numpy func exists', c5.numpy_func is not None)

# Vectorised evaluation
mu_grid = np.array([0.0, 0.5, 1.0])
c_grid = np.array([0.0, 0.0, 0.0])
vals = c3(mu_grid, c_grid)
check('curve vectorised', isinstance(vals, np.ndarray))
check('curve vectorised values', np.allclose(vals, [-0.4, 0.1, 0.6]))

# str display
str_out = str(c3)
check('curve str contains expr', 'μ' in str_out or 'mu' in str_out)

# Metadata attr fallback
c3.metadata['plotting'] = {'colour': 'green', 'linestyle': '--'}
check('curve attr → nested metadata', c3.colour == 'green')
check('curve attr → nested metadata (linestyle)', c3.linestyle == '--')

# Custom numpy function
def my_func(mu, c):
    return mu + c - 1.0
c_custom = Curve('mu + c - 1', name='custom', numpy_func=my_func)
check('custom numpy func', c_custom.numpy_func is not None)
check('custom numpy eval', abs(c_custom(0.3, 0.7)) < 1e-15)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Constraint
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Constraint ──')

# μ > 0.4
con1 = Constraint('mu', lower=0.4, lower_strict=True)
check('constraint μ > 0.4 (true)', con1(0.5, 0.0) is True)
check('constraint μ > 0.4 (false)', con1(0.3, 0.0) is False)
check('constraint μ > 0.4 (edge)', con1(0.4, 0.0) is False)  # strict

# c ≤ 2
con2 = Constraint('c', upper=2.0, upper_strict=False)
check('constraint c ≤ 2 (true)', con2(0.0, 1.9) is True)
check('constraint c ≤ 2 (edge)', con2(0.0, 2.0) is True)
check('constraint c ≤ 2 (false)', con2(0.0, 2.1) is False)

# 2/5 < μ < 1
con3 = Constraint('mu', lower=0.4, upper=1.0)
check('constraint 0.4 < μ < 1 (inside)', con3(0.6, 0.0) is True)
check('constraint 0.4 < μ < 1 (below)', con3(0.3, 0.0) is False)
check('constraint 0.4 < μ < 1 (above)', con3(1.5, 0.0) is False)

# mu > -inf  (equivalent to no lower bound)
con4 = Constraint('mu', upper=1.0)
check('constraint μ < 1 (true)', con4(0.0, 0.0) is True)
check('constraint μ < 1 (false)', con4(2.0, 0.0) is False)

# Callable constraint
con5 = Constraint(lambda mu, c: mu + c > 1.0)
check('constraint callable (true)', con5(0.7, 0.5) is True)
check('constraint callable (false)', con5(0.3, 0.3) is False)

# repr
r = repr(con3)
check('constraint repr', '0.4' in r and 'μ' in r and '1.0' in r)

# Vectorised
arr = np.array([0.1, 0.5, 0.8, 1.2])
result = con3(arr, np.zeros(4))
check('constraint vectorised', isinstance(result, np.ndarray) and result.dtype == bool)
check('constraint vectorised shape', result.shape == (4,))
check('constraint vectorised values', (result == [False, True, True, False]).all())


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Point
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Point ──')

p = Point(0.5, 1.0, name='P1')
check('point mu', p.mu == 0.5)
check('point c', p.c == 1.0)
check('point name', p.name == 'P1')
check('point repr', 'μ=0.5' in repr(p) and 'c=1.0' in repr(p))

# Coordinate mutation
p.mu = 0.6
check('point mu setter', p.mu == 0.6)

# Metadata entry for mu/c
check('point mu in metadata', p.metadata['mu'] == 0.6)
check('point c in metadata', p.metadata['c'] == 1.0)

# __getattr__ fallback
check('point attr → metadata mu', p.mu == 0.6)

p2 = Point(0.0, 0.0)
check('point (0,0) repr', 'μ=0.0' in repr(p2) and 'c=0.0' in repr(p2))


# ══════════════════════════════════════════════════════════════════════════════
# 4b.  InfinityPoint
# ══════════════════════════════════════════════════════════════════════════════
print('\n── InfinityPoint ──')

# C5: c = 0  (horizontal line).  Crosses left edge at (-2.5, 0)
#            and right edge at (2.0, 0).  Does not cross top/bottom.
inf_left = InfinityPoint(C5, 'left', name='C5_inf_left')
check('inf left mu = -2.5', abs(inf_left.mu - (-2.5)) < 1e-10)
check('inf left c = 0', abs(inf_left.c) < 1e-10)
check('inf left is_at_infinity', inf_left.is_at_infinity is True)
check('inf left curve is C5', inf_left.curve is C5)
check('inf left edge', inf_left.edge == 'left')

# Right edge of C5
inf_right = InfinityPoint(C5, 'right', name='C5_inf_right')
check('inf right mu = 2.0', abs(inf_right.mu - 2.0) < 1e-10)
check('inf right c = 0', abs(inf_right.c) < 1e-10)

# C3: μ = 2/5 = 0.4  (vertical line).  Crosses bottom at (0.4, -0.75)
#            and top at (0.4, 7.5).
inf_bottom = InfinityPoint(C3, 'bottom', name='C3_inf_bottom')
check('inf bottom mu = 0.4', abs(inf_bottom.mu - 0.4) < 1e-10)
check('inf bottom c = -0.75', abs(inf_bottom.c - (-0.75)) < 1e-10)

inf_top = InfinityPoint(C3, 'top', name='C3_inf_top')
check('inf top mu = 0.4', abs(inf_top.mu - 0.4) < 1e-10)
check('inf top c = 7.5', abs(inf_top.c - 7.5) < 1e-10)

# Edge with no crossing → nan
inf_none = InfinityPoint(C5, 'top', name='C5_inf_top')
check('inf no-crossing mu is nan', np.isnan(inf_none.mu))
check('inf no-crossing c is nan', np.isnan(inf_none.c))

# C4: c = μ + 2  (diagonal).  Crosses left edge at (-2.5, -0.5).
inf_c4 = InfinityPoint(C4, 'left', name='C4_inf_left')
check('C4 inf left mu = -2.5', abs(inf_c4.mu - (-2.5)) < 1e-10)
check('C4 inf left c = -0.5', abs(inf_c4.c - (-0.5)) < 1e-10)

# ── custom box ──────────────────────────────────────────────────────────────
# Shrink box to (-1, 1, -1, 3). C5 left edge should now be at (-1, 0).
zoom_box = (-1.0, 1.0, -1.0, 3.0)
inf_zoom = InfinityPoint(C5, 'left', box=zoom_box, name='C5_zoom')
check('zoom box mu = -1.0', abs(inf_zoom.mu - (-1.0)) < 1e-10)
check('zoom box c = 0', abs(inf_zoom.c) < 1e-10)

# ── change box after construction ───────────────────────────────────────────
inf_left.box = zoom_box
check('after box setter mu = -1.0', abs(inf_left.mu - (-1.0)) < 1e-10)
check('after box setter c = 0', abs(inf_left.c) < 1e-10)

# ── index parameter (multiple crossings on same edge) ───────────────────────
# C2 crosses the left edge once (index 0).  Index > 0 should be nan.
inf_c2 = InfinityPoint(C2, 'left', index=0, name='C2_inf_left')
check('C2 left index=0 mu finite', np.isfinite(inf_c2.mu))
check('C2 left index=0 c finite', np.isfinite(inf_c2.c))

inf_c2_bad = InfinityPoint(C2, 'left', index=99, name='C2_inf_left_bad')
check('C2 left index=99 mu is nan', np.isnan(inf_c2_bad.mu))
check('C2 left index=99 c is nan', np.isnan(inf_c2_bad.c))

# ── repr / str ──────────────────────────────────────────────────────────────
rep = repr(inf_left)
check('repr contains InfinityPoint', 'InfinityPoint' in rep)
check('repr contains curve name', 'C5' in rep)
check('repr contains edge', 'left' in rep)

s = str(inf_left)
check('str contains InfinityPoint', 'InfinityPoint' in s)
check('str contains mu,c', 'μ=' in s and 'c=' in s)

# ── default box (auto-picked from box_utils) ────────────────────────────────
def_inf = InfinityPoint(C5, 'left')
check('default box gives mu=-2.5', abs(def_inf.mu - (-2.5)) < 1e-10)

# ── works as Arc endpoint (duck-type compatible) ────────────────────────────
p = Point(0.5, 1.0, name='P1')
arc_with_inf = Arc(curve=C5, start=inf_left, end=p, name='test_arc')
check('Arc with InfinityPoint start', 'test_arc' in repr(arc_with_inf))
check('Arc start preserved', arc_with_inf.start is inf_left)
check('Arc end preserved', arc_with_inf.end is p)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Intersection
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Intersection ──')

p_int = Point(1.0, 2.0, name='X1')
inter = Intersection(curves=(c3, c5), points=[p_int], name='X')
check('intersection curves', c3 in inter.curves and c5 in inter.curves)
check('intersection points', inter.points[0].mu == 1.0)
check('intersection name', inter.name == 'X')

# Multiple points
p2_int = Point(0.0, 0.0, name='X2')
inter2 = Intersection(curves=(c3,), points=[p_int, p2_int])
check('intersection multiple points', len(inter2.points) == 2)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Arc
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Arc ──')

p_start = Point(0.0, 0.0)
p_end = Point(1.0, 1.0)
arc = Arc(curve=c3, start=p_start, end=p_end, name='A1')
check('arc curve', arc.curve is c3)
check('arc start', arc.start.mu == 0.0)
check('arc end', arc.end.c == 1.0)
check('arc name', arc.name == 'A1')
check('arc repr', 'A1' in repr(arc) and '→' in repr(arc))

# With constraint
arc_con = Arc(curve=c3, start=p_start, end=p_end, constraint=con3)
check('arc with constraint', arc_con.constraint is con3)

# Arc without constraint
check('arc None constraint', arc.constraint is None)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Region
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Region ──')

# Build a toy scenario: two curves, F=c (c=0) and F=μ-0.4 (μ=0.4)
# In reality these would partition the plane, but here we just test the API.
r = Region(
    name='R_1',
    signs={c3: -1, c5: +1},       # c3 < 0 and c5 > 0  (c3 = μ-0.4, c5 = c)
    vertices=[Point(0.0, 0.0), Point(1.0, 1.0)],
    boundary_arcs=[arc],
    test_point=(0.5, 0.5),
)
check('region name', r.name == 'R_1')
check('region signs count', len(r.signs) == 2)
check('region vertices count', len(r.vertices) == 2)
check('region test_point', r.test_point == (0.5, 0.5))
check('region repr', '2 signs' in repr(r) and '2 vertices' in repr(r))

# Region str
str_r = str(r)
check('region str has name', 'R_1' in str_r)
check('region str has signs', '+' in str_r or '-' in str_r)

# Contains
r2 = Region(
    name='upper_right',
    signs={c3: +1, c5: +1},  # μ > 0.4, c > 0
)
check('region contains (true)', r2.contains(0.6, 1.0))
check('region contains (false: mu too low)', not r2.contains(0.3, 1.0))
check('region contains (false: c too low)', not r2.contains(0.6, -0.1))
check('region contains (on boundary, within tol)', r2.contains(0.4, 1.0, tol=0.01))


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Boundary
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Boundary ──')

r_left = Region(name='R_left')
r_right = Region(name='R_right')
b = Boundary(region_left=r_left, region_right=r_right, arc=arc, name='B1')
check('boundary region_left', b.region_left.name == 'R_left')
check('boundary region_right', b.region_right.name == 'R_right')
check('boundary arc', b.arc is arc)
check('boundary name', b.name == 'B1')
check('boundary repr', 'R_left' in repr(b) and 'R_right' in repr(b))
check('boundary str', 'R_left' in str(b) and 'R_right' in str(b))


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n── Result: {passed} passed / {failed} failed / {passed + failed} total ──\n')

if failed > 0:
    print('Some checks failed — review above.')
    exit(1)
else:
    print('All good.  Data model is ready for registration.')
    exit(0)