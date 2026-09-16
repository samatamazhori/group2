"""
data-model.py — Core data objects for the μ-c bifurcation catalog.

Hierarchy:
  BaseObject  (metadata, dynamic attr access, repr/str)
  ├── Curve         — implicit curve F(μ, c) = 0
  ├── Constraint    — inequality bound on μ or c
  ├── Point         — (μ, c) coordinate
  ├── Intersection  — curves × points
  ├── Arc           — curve segment between two points
  ├── Region        — sign-defined region of the μ-c plane
  └── Boundary      — arc separating two regions
"""

import sympy as sp
import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_bound(val, default):
    """Normalise a bound value to float or np.inf."""
    if val is None:
        return default
    if isinstance(val, str) and val.strip().lower() == 'inf':
        return np.inf
    return float(val)


# ── BaseObject ───────────────────────────────────────────────────────────────

class BaseObject:
    """Lightweight base providing metadata, dynamic attr access, and pretty
    representations.  Every domain object inherits from this.

    Metadata entries stored in ``obj.metadata['foo']`` are also readable as
    ``obj.foo`` via ``__getattr__`` fallback (including one level of nested
    dict lookup).  This means you can set ``curve.metadata['colour'] = 'red'``
    and then access ``curve.colour`` directly.
    """

    def __init__(self, name=None, metadata=None):
        self._metadata = dict(metadata or {})
        self._name = name

    # ── metadata ─────────────────────────────────────────────────────────────

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata.clear()
        self._metadata.update(dict(value or {}))

    # ── name ─────────────────────────────────────────────────────────────────

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    # ── dynamic attribute fallback ───────────────────────────────────────────

    def __getattr__(self, key):
        """Fallback: look up *key* in ``_metadata`` (flat or one level of
        nesting).  This enables e.g. ``curve.colour`` reaching
        ``curve.metadata['plotting']['colour']``."""
        if key.startswith('_'):
            raise AttributeError(key)
        md = self._metadata
        if key in md:
            return md[key]
        for v in md.values():
            if isinstance(v, dict) and key in v:
                return v[key]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{key}'"
        )

    # ── display ──────────────────────────────────────────────────────────────

    def __repr__(self):
        cls = type(self).__name__
        tag = f' "{self._name}"' if self._name else ''
        return f'<{cls}{tag} at 0x{id(self):x}>'

    def __str__(self):
        lines = [f'{type(self).__name__}: {self._name or "(unnamed)"}']
        if self._metadata:
            lines.append('  metadata:')
            for k, v in self._metadata.items():
                lines.append(f'    {k}: {v!r}')
        return '\n'.join(lines)


# ── Curve ────────────────────────────────────────────────────────────────────

class Curve(BaseObject):
    """An implicit curve *F*(μ, c) = 0 in the parameter plane.

    The canonical source is a **sympy expression**.  A **numpy-compatible
    callable** is derived automatically via ``lambdify`` on construction, so
    the curve can be evaluated on meshgrids for contour plotting.

    Parameters
    ----------
    expr : sympy.Expr or str
        Symbolic expression defining *F*(μ, c) = 0.
    name : str, optional
    symbols : tuple of str or sp.Symbol, optional
        Variable names or symbols (default ``('mu', 'c')``).
    numpy_func : callable, optional
        Pre-compiled numpy function.  If omitted, ``lambdify`` is tried.
    metadata : dict, optional
    """

    def __init__(self, expr, name=None, symbols=('mu', 'c'),
                 numpy_func=None, metadata=None):
        # Canonical source: sympy expression
        if isinstance(expr, str):
            expr = sp.sympify(expr)
        self._sympy_expr = sp.sympify(expr)

        # Symbols
        if symbols is not None and isinstance(symbols[0], str):
            self._symbols = sp.symbols(symbols)
        else:
            self._symbols = symbols

        # Numpy callable — auto-derive if not provided
        self._numpy_func = numpy_func
        if self._numpy_func is None and self._symbols is not None:
            try:
                self._numpy_func = sp.lambdify(
                    self._symbols, self._sympy_expr, modules='numpy'
                )
            except Exception:
                pass  # caller can set numpy_func later

        super().__init__(name=name, metadata=metadata)

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def expr(self):
        """The canonical sympy expression."""
        return self._sympy_expr

    @property
    def symbols(self):
        return self._symbols

    @property
    def numpy_func(self):
        return self._numpy_func

    @numpy_func.setter
    def numpy_func(self, func):
        self._numpy_func = func

    # ── evaluation ───────────────────────────────────────────────────────────

    def __call__(self, mu, c):
        """Evaluate *F*(μ, c).

        If a numpy function is available, it is used (fast, vectorised).
        Otherwise falls back to sympy substitution (slow, scalar).
        """
        if self._numpy_func is not None:
            return self._numpy_func(mu, c)
        sub = dict(zip(self._symbols, (mu, c)))
        return float(self._sympy_expr.subs(sub))

    # ── display ──────────────────────────────────────────────────────────────

    def __repr__(self):
        name = f' "{self._name}"' if self._name else ''
        return f'<Curve{name} at 0x{id(self):x}>'

    def __str__(self):
        expr_str = sp.pretty(self._sympy_expr, use_unicode=False)
        lines = [f'Curve: {self._name or "(unnamed)"}']
        lines.append(f'  F(μ, c) = {expr_str.strip()}')
        if self._numpy_func is not None:
            lines.append('  (numpy-compiled)')
        if self._metadata:
            lines.append('  metadata:')
            for k, v in self._metadata.items():
                lines.append(f'    {k}: {v!r}')
        return '\n'.join(lines)


# ── Constraint ───────────────────────────────────────────────────────────────

class Constraint(BaseObject):
    """An inequality bound on a variable, e.g. ``2/5 < μ < 1``.

    Parameters
    ----------
    var : str or callable
        ``'mu'``, ``'c'``, or a callable ``f(mu, c) → bool``.
    lower, upper : float or str, optional
        Bounds.  Pass ``'inf'`` or leave ``None`` for unbounded.
    lower_strict, upper_strict : bool
        Whether the inequality is strict (default ``True`` = ``<`` / ``>``).
    name : str, optional
    metadata : dict, optional
    """

    def __init__(self, var, lower=None, upper=None,
                 lower_strict=True, upper_strict=True,
                 name=None, metadata=None):
        self._var = var
        self._lower = _to_bound(lower, -np.inf)
        self._upper = _to_bound(upper, np.inf)
        self._lower_strict = lower_strict
        self._upper_strict = upper_strict
        super().__init__(name=name, metadata=metadata)

    # ── evaluation ───────────────────────────────────────────────────────────

    def __call__(self, mu, c):
        """Evaluate the constraint.

        Returns a bool (scalar) or boolean numpy array (vectorised).
        """
        if callable(self._var):
            return self._var(mu, c)

        val = mu if self._var == 'mu' else c
        is_arr = isinstance(val, np.ndarray)

        if is_arr:
            ok = np.ones_like(val, dtype=bool)
            if not np.isinf(self._lower):
                op = np.greater if self._lower_strict else np.greater_equal
                ok &= op(val, self._lower)
            if not np.isinf(self._upper):
                op = np.less if self._upper_strict else np.less_equal
                ok &= op(val, self._upper)
            return ok
        else:
            if not np.isinf(self._lower):
                if self._lower_strict and not (val > self._lower):
                    return False
                if not self._lower_strict and not (val >= self._lower):
                    return False
            if not np.isinf(self._upper):
                if self._upper_strict and not (val < self._upper):
                    return False
                if not self._upper_strict and not (val <= self._upper):
                    return False
            return True

    # ── display ──────────────────────────────────────────────────────────────

    def _var_str(self):
        if callable(self._var):
            return 'f(μ,c)'
        return 'μ' if self._var == 'mu' else self._var

    def __repr__(self):
        parts = []
        var_s = self._var_str()
        if not np.isinf(self._lower):
            op = '<' if self._lower_strict else '≤'
            parts.append(f'{self._lower} {op}')
        parts.append(var_s)
        if not np.isinf(self._upper):
            op = '<' if self._upper_strict else '≤'
            parts.append(f'{op} {self._upper}')
        tag = f' "{self._name}"' if self._name else ''
        return f'<Constraint{tag}: {" ".join(parts)}>'

    def __str__(self):
        return f'Constraint: {" ".join(self.__repr__().split(": ")[1:])}'


# ── Point ────────────────────────────────────────────────────────────────────

class Point(BaseObject):
    """A single point (μ, c) in the parameter plane.

    Coordinates are stored in ``_metadata`` and exposed via properties,
    so they are also reachable through ``__getattr__`` fallback.
    """

    def __init__(self, mu, c, name=None, metadata=None):
        super().__init__(name=name, metadata=metadata)
        self._mu = float(mu)
        self._c = float(c)
        # Also expose in metadata for serialisation / attr fallback
        self._metadata['mu'] = self._mu
        self._metadata['c'] = self._c

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = float(value)
        self._metadata['mu'] = self._mu

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, value):
        self._c = float(value)
        self._metadata['c'] = self._c

    # ── display ──────────────────────────────────────────────────────────────

    def __repr__(self):
        tag = f' "{self._name}"' if self._name else ''
        return f'<Point{tag}: μ={self._mu}, c={self._c}>'

    def __str__(self):
        return f'Point: (μ={self._mu}, c={self._c})'


# ── Intersection ─────────────────────────────────────────────────────────────

class Intersection(BaseObject):
    """The intersection of two (or more) curves, yielding one or more Points.

    Curves and points are stored in ``_metadata`` so they are reachable
    as attributes.
    """

    def __init__(self, curves, points, name=None, metadata=None):
        super().__init__(name=name, metadata=metadata)
        self._curves = tuple(curves)
        self._points = list(points)
        self._metadata['curves'] = self._curves
        self._metadata['points'] = self._points

    @property
    def curves(self):
        return self._curves

    @property
    def points(self):
        return self._points


# ── Arc ──────────────────────────────────────────────────────────────────────

class Arc(BaseObject):
    """A segment of a Curve between two Points, optionally bounded by a
    Constraint."""

    def __init__(self, curve, start, end, constraint=None,
                 name=None, metadata=None):
        super().__init__(name=name, metadata=metadata)
        self._curve = curve
        self._start = start
        self._end = end
        self._constraint = constraint
        self._metadata['curve'] = curve
        self._metadata['start'] = start
        self._metadata['end'] = end
        self._metadata['constraint'] = constraint

    @property
    def curve(self):
        return self._curve

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def constraint(self):
        return self._constraint

    def __repr__(self):
        tag = f' "{self._name}"' if self._name else ''
        return f'<Arc{tag}: {self._start.mu:.3g}→{self._end.mu:.3g} on {self._curve.name or "?"}>'

    def __str__(self):
        return (f'Arc: {self._name or "(unnamed)"} — '
                f'({self._start.mu:.3g}, {self._start.c:.3g}) → '
                f'({self._end.mu:.3g}, {self._end.c:.3g}) on '
                f'{self._curve.name or "unnamed curve"}')


# ── Boundary ─────────────────────────────────────────────────────────────────

class Boundary(BaseObject):
    """An Arc that separates two Regions."""

    def __init__(self, region_left, region_right, arc,
                 name=None, metadata=None):
        super().__init__(name=name, metadata=metadata)
        self._region_left = region_left
        self._region_right = region_right
        self._arc = arc
        self._metadata['region_left'] = region_left
        self._metadata['region_right'] = region_right
        self._metadata['arc'] = arc

    @property
    def region_left(self):
        return self._region_left

    @property
    def region_right(self):
        return self._region_right

    @property
    def arc(self):
        return self._arc

    def __repr__(self):
        tag = f' "{self._name}"' if self._name else ''
        l = self._region_left.name or '?'
        r = self._region_right.name or '?'
        return f'<Boundary{tag}: {l} │ {r}>'

    def __str__(self):
        l = self._region_left.name or '(unnamed)'
        r = self._region_right.name or '(unnamed)'
        return f'Boundary: {l} ║ {r}'


# ── Region ───────────────────────────────────────────────────────────────────

class Region(BaseObject):
    """A region of the μ-c plane defined by the sign of one or more curves.

    Parameters
    ----------
    name : str
    signs : dict
        ``{Curve: +1 or -1}`` — the sign that *F*(μ, c) takes inside the region.
    vertices : list of Point, optional
        Corner points from intersecting boundary curves.
    boundary_arcs : list of Arc, optional
        The arcs that enclose this region.
    test_point : tuple of (mu, c), optional
        A point known to lie inside the region.
    metadata : dict, optional
    """

    def __init__(self, name, signs=None, vertices=None,
                 boundary_arcs=None, test_point=None, metadata=None):
        super().__init__(name=name, metadata=metadata)
        self._signs = dict(signs or {})
        self._vertices = list(vertices or [])
        self._boundary_arcs = list(boundary_arcs or [])
        self._test_point = test_point
        self._metadata['signs'] = self._signs
        self._metadata['vertices'] = self._vertices
        self._metadata['boundary_arcs'] = self._boundary_arcs
        self._metadata['test_point'] = test_point

    @property
    def signs(self):
        return self._signs

    @property
    def vertices(self):
        return self._vertices

    @property
    def boundary_arcs(self):
        return self._boundary_arcs

    @property
    def test_point(self):
        return self._test_point

    @test_point.setter
    def test_point(self, value):
        self._test_point = value
        self._metadata['test_point'] = value

    # ── containment ─────────────────────────────────────────────────────────

    def contains(self, mu, c, tol=1e-10):
        """Check whether (μ, c) satisfies all sign conditions of this region.

        Returns ``True`` if *F*(μ, c) has the correct sign for every curve
        in ``self.signs`` (within tolerance).
        """
        for curve, sign in self._signs.items():
            val = curve(mu, c)
            if sign > 0 and val <= -tol:
                return False
            if sign < 0 and val >= tol:
                return False
        return True

    # ── display ──────────────────────────────────────────────────────────────

    def __repr__(self):
        tag = f' "{self._name}"' if self._name else ''
        n_curves = len(self._signs)
        n_verts = len(self._vertices)
        return f'<Region{tag}: {n_curves} signs, {n_verts} vertices>'

    def __str__(self):
        lines = [f'Region: {self._name or "(unnamed)"}']
        if self._signs:
            lines.append('  signs:')
            for curve, sgn in self._signs.items():
                lines.append(f'    {curve.name or "?"} → {"+" if sgn > 0 else "-"}')
        if self._vertices:
            lines.append(f'  vertices ({len(self._vertices)}):')
            for p in self._vertices:
                lines.append(f'    ({p.mu:.4g}, {p.c:.4g})')
        if self._test_point:
            lines.append(f'  test point: (μ={self._test_point[0]}, c={self._test_point[1]})')
        return '\n'.join(lines)