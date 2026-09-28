"""
box_utils.py — Find where a curve exits a given bounding box.

For each curve, sample each of the four box edges at high resolution
and locate sign changes of F(μ,c).  Returns artificial Point objects
on the box boundary, tagged with metadata for arc registration.
"""

import numpy as np

from data_model import Point

# ── default box (matches quick-plot / plot-curves) ───────────────────────

MU_MIN, MU_MAX = -2.5, 2.0
C_MIN, C_MAX   = -0.75, 7.5
BOX = (MU_MIN, MU_MAX, C_MIN, C_MAX)
GRID = 1000


# ── helpers ───────────────────────────────────────────────────────────────

def _crossings_1d(f, ts):
    """Find all roots of *f(t)* on the sorted array *ts*.

    Detects sign changes between adjacent samples and refines by linear
    interpolation.  Also picks up exact zeros.
    """
    vals = f(ts)
    # scalar result → curve doesn't vary along this edge (e.g. μ=constant
    # and the expression ignores c).  No crossings unless the constant
    # happens to be exactly zero, which is degenerate.
    if np.ndim(vals) == 0:
        return [float(ts[0])] if abs(float(vals)) < 1e-14 else []

    roots = []
    for i in range(len(ts) - 1):
        v0, v1 = vals[i], vals[i + 1]
        # exact zero
        if v0 == 0.0:
            roots.append(ts[i])
        # sign change
        if v0 * v1 < 0:
            t0, t1 = ts[i], ts[i + 1]
            t_root = t0 - v0 * (t1 - t0) / (v1 - v0)
            roots.append(t_root)
    # check last point
    if vals[-1] == 0.0:
        roots.append(ts[-1])
    return roots


# ── main entry point ─────────────────────────────────────────────────────

def edge_intersections(curve, box=BOX, n_samples=GRID):
    """Return a list of ``Point`` for every crossing of *curve* with *box*.

    Parameters
    ----------
    curve : Curve
    box : (mu_min, mu_max, c_min, c_max)
    n_samples : int
        Number of samples per edge (default 1000).

    Returns
    -------
    list[Point]
        Each point is named ``{curve.name}_edge_{n}`` and carries metadata
        ``{'edge': 'left'|'right'|'bottom'|'top', 'artificial': True}``.
    """
    mu_lo, mu_hi, c_lo, c_hi = box
    mu, c = curve.symbols
    count = 0
    points = []

    # ── left edge  (μ = mu_lo, c varies) ────────────────────────────────
    ts = np.linspace(c_lo, c_hi, n_samples)
    roots = _crossings_1d(lambda t: curve(mu_lo, t), ts)
    for r in roots:
        points.append(Point(
            mu_lo, r,
            name=f'{curve.name}_edge_{count}',
            metadata={'edge': 'left', 'artificial': True},
        ))
        count += 1

    # ── right edge (μ = mu_hi, c varies) ───────────────────────────────
    roots = _crossings_1d(lambda t: curve(mu_hi, t), ts)
    for r in roots:
        points.append(Point(
            mu_hi, r,
            name=f'{curve.name}_edge_{count}',
            metadata={'edge': 'right', 'artificial': True},
        ))
        count += 1

    # ── bottom edge (c = c_lo, μ varies) ───────────────────────────────
    ts = np.linspace(mu_lo, mu_hi, n_samples)
    roots = _crossings_1d(lambda t: curve(t, c_lo), ts)
    for r in roots:
        points.append(Point(
            r, c_lo,
            name=f'{curve.name}_edge_{count}',
            metadata={'edge': 'bottom', 'artificial': True},
        ))
        count += 1

    # ── top edge  (c = c_hi, μ varies) ─────────────────────────────────
    roots = _crossings_1d(lambda t: curve(t, c_hi), ts)
    for r in roots:
        points.append(Point(
            r, c_hi,
            name=f'{curve.name}_edge_{count}',
            metadata={'edge': 'top', 'artificial': True},
        ))
        count += 1

    return points


# ── batch: compute edge points for all curves ────────────────────────────

def all_edge_points(curves, box=BOX, n_samples=GRID):
    """Convenience wrapper: edge points for every curve in *curves*.

    Returns a flat list (usable directly in ``data.py``).
    """
    result = []
    for curve in curves:
        result.extend(edge_intersections(curve, box, n_samples))
    return result
