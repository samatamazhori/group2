"""masks.py — Domain masks for phase portraits.

Masks are applied once, up-front, before any layer sees data.
They clip points, fields, and provide axis limits.
"""

from abc import ABC, abstractmethod
import numpy as np


class Mask(ABC):
    """Abstract base for a spatial domain mask on the (x, y) plane.

    Subclasses must implement all abstract methods.
    """

    @abstractmethod
    def clip_points(self, points):
        """Filter a list of ``(x, y)`` tuples to those inside the mask.

        Parameters
        ----------
        points : list of (float, float)

        Returns
        -------
        list of (float, float)
        """

    @abstractmethod
    def clip_field(self, X, Y, U, V):
        """Mask the vector field arrays, returning masked arrays.

        Parameters
        ----------
        X, Y, U, V : np.ndarray
            Meshgrid arrays.

        Returns
        -------
        U_masked, V_masked : np.ma.MaskedArray
        """

    @abstractmethod
    def contains(self, x, y):
        """Check whether a point (or array of points) lies inside the mask.

        Parameters
        ----------
        x, y : float or np.ndarray

        Returns
        -------
        bool or np.ndarray[bool]
        """

    @abstractmethod
    def get_limits(self):
        """Return ``(xmin, xmax, ymin, ymax)`` for axis limits.

        Returns
        -------
        tuple of 4 floats
        """


class R2Mask(Mask):
    """Rectangular box mask.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        Axis-aligned bounding box.
    """

    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)

    # ── Mask API ───────────────────────────────────────────────────────

    def clip_points(self, points):
        return [(x, y) for x, y in points
                if self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax]

    def clip_field(self, X, Y, U, V):
        mag = np.sqrt(U ** 2 + V ** 2)
        # Mask: outside box OR extreme values (blow-up)
        outside = (X < self.xmin) | (X > self.xmax) \
                | (Y < self.ymin) | (Y > self.ymax)
        extreme = mag > 1e3
        mask = outside | extreme
        return np.ma.masked_where(mask, U), np.ma.masked_where(mask, V)

    def contains(self, x, y):
        return (self.xmin <= x <= self.xmax) & (self.ymin <= y <= self.ymax)

    def get_limits(self):
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    # ── helpers ────────────────────────────────────────────────────────

    @classmethod
    def from_equilibria(cls, eqs, pad_factor=5.5, min_pad=0.5):
        """Auto-compute bounds from a list of equilibrium objects.

        Parameters
        ----------
        eqs : list of Equilibrium
        pad_factor : float
        min_pad : float

        Returns
        -------
        R2Mask
        """
        xs = np.array([e.x_num for e in eqs if e.exists])
        ys = np.array([e.y_num for e in eqs if e.exists])
        if len(xs) == 0:
            return cls(-1, 1, -1, 1)

        xc = (xs.min() + xs.max()) / 2
        yc = (ys.min() + ys.max()) / 2
        dx = max(xs.max() - xs.min(), min_pad) * pad_factor
        dy = max(ys.max() - ys.min(), min_pad) * pad_factor
        return cls(xc - dx / 2, xc + dx / 2, yc - dy / 2, yc + dy / 2)

    def __repr__(self):
        return (f'<R2Mask x∈[{self.xmin:.4g}, {self.xmax:.4g}]'
                f' y∈[{self.ymin:.4g}, {self.ymax:.4g}]>')


# ── stubs for future masks ─────────────────────────────────────────────

class HalfPlaneMask(Mask):
    """Stub: half-plane mask (e.g. x ≥ 0)."""
    def __init__(self):
        raise NotImplementedError('HalfPlaneMask is a stub')

    def clip_points(self, points): ...
    def clip_field(self, X, Y, U, V): ...
    def contains(self, x, y): ...
    def get_limits(self): ...


class SimplexMask(Mask):
    """Stub: triangular / simplex mask."""
    def __init__(self):
        raise NotImplementedError('SimplexMask is a stub')

    def clip_points(self, points): ...
    def clip_field(self, X, Y, U, V): ...
    def contains(self, x, y): ...
    def get_limits(self): ...