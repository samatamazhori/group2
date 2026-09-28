"""layers.py — Composable phase-portrait layers.

Each layer wraps one visual element (equilibria, nullclines, quiver,
orbits).  Layers are mixed freely in a ``PhasePortrait``.
"""

from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    """Abstract base for a single visual layer.

    Subclasses implement ``prepare()``, which creates matplotlib artists
    and appends them to ``self._artists``.
    """

    def __init__(self, name=None):
        self.name = name or type(self).__name__
        self.visible = True
        self._artists = []

    @abstractmethod
    def prepare(self, system, mask, ax):
        """Create artists for this layer.

        Parameters
        ----------
        system : System
            The dynamical system at a fixed (μ, c).
        mask : Mask
            Domain mask for clipping.
        ax : matplotlib.axes.Axes
            Target axes.

        Returns
        -------
        list[Artist]
            The created artists.
        """
        ...

    def draw(self, ax):
        """Draw (or re-draw) the layer's artists.

        Default implementation returns artists created by ``prepare()``.
        Override for dynamic layers (e.g. orbits added after init).
        """
        return self._artists

    def set_visible(self, visible):
        """Toggle visibility of all artists."""
        self.visible = visible
        for a in self._artists:
            a.set_visible(visible)

    def update(self, system, mask):
        """Optional: update internal state when the system changes.
        """
        pass

    def clear(self):
        """Remove all artists from the axes and reset."""
        for a in self._artists:
            try:
                a.remove()
            except Exception:
                pass
        self._artists.clear()

    def __repr__(self):
        return f'<{type(self).__name__} "{self.name}" visible={self.visible}>'


# ═══════════════════════════════════════════════════════════════════════
#  EquilibriaLayer
# ═══════════════════════════════════════════════════════════════════════

# Default marker styles keyed by stability type
_DEFAULT_EQ_STYLES = {
    'stable node':     {'marker': 'o', 'c': '#2ca02c', 's': 90, 'edgecolors': 'white', 'linewidths': 1.5},
    'stable spiral':   {'marker': 'o', 'c': '#2ca02c', 's': 90, 'edgecolors': 'white', 'linewidths': 1.5},
    'unstable node':   {'marker': 'o', 'c': '#d62728', 's': 90, 'edgecolors': 'white', 'linewidths': 1.5},
    'unstable spiral': {'marker': 'o', 'c': '#d62728', 's': 90, 'edgecolors': 'white', 'linewidths': 1.5},
    'saddle':          {'marker': 'D', 'c': '#ff7f0e', 's': 90, 'edgecolors': 'white', 'linewidths': 1.5},
    'centre':          {'marker': 'o', 'c': '#1f77b4', 's': 90, 'edgecolors': 'white', 'linewidths': 1.5},
    'degenerate':      {'marker': 's', 'c': '#9467bd', 's': 90, 'edgecolors': 'white', 'linewidths': 1.5},
}


class EquilibriaLayer(Layer):
    """Layer showing equilibrium points with stability-based styling.

    Parameters
    ----------
    branches : list of str, optional
        Subset of branch names to show (e.g. ``['E0', 'E_A⁺']``).
        ``None`` means all existing equilibria.
    styles : dict, optional
        Override styles keyed by type name or equilibrium name.
        Merged on top of ``_DEFAULT_EQ_STYLES``.
    show_labels : bool
        Annotate each equilibrium with its label (default True).
    name : str
    """

    def __init__(self, branches=None, styles=None, show_labels=True, name='equilibria'):
        super().__init__(name)
        self.branches = branches
        self.show_labels = show_labels
        self._styles = dict(_DEFAULT_EQ_STYLES)
        if styles:
            for k, v in styles.items():
                if k in self._styles:
                    self._styles[k].update(v)
                else:
                    self._styles[k] = v

    def prepare(self, system, mask, ax):
        system.classify_all()
        eqs = system.real_equilibria()

        # Filter by requested branches
        if self.branches is not None:
            eqs = [e for e in eqs if e.name in self.branches]

        # Filter by mask
        eqs = [e for e in eqs if mask.contains(e.x_num, e.y_num)]

        for eq in eqs:
            style = dict(self._styles.get(eq.type, self._styles['saddle']))
            # Per-equilibrium-name override
            eq_override = self._styles.get(eq.name)
            if eq_override:
                style.update(eq_override)

            # Point drawing
            pt = ax.scatter(eq.x_num, eq.y_num, zorder=10, **style)
            self._artists.append(pt)

            if self.show_labels:
                label = ax.annotate(
                    eq.label, (eq.x_num, eq.y_num),
                    xytext=(10, 8), textcoords='offset points',
                    fontsize=8, fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.75),
                    zorder=12,
                )
                self._artists.append(label)

        return self._artists


# ═══════════════════════════════════════════════════════════════════════
#  NullclineLayer
# ═══════════════════════════════════════════════════════════════════════

class NullclineLayer(Layer):
    """Layer showing nullclines (dx/dt=0 and dy/dt=0).

    Parameters
    ----------
    grid_res : int
        Grid resolution per axis (default 400).
    x_style, y_style : dict
        Contour style overrides for dx/dt=0 and dy/dt=0 nullclines.
    name : str
    """

    _X_DEFAULT = {'colors': '#1f77b4', 'linewidths': 1.0, 'linestyles': '--', 'alpha': 0.7}
    _Y_DEFAULT = {'colors': '#d62728', 'linewidths': 1.0, 'linestyles': '--', 'alpha': 0.7}

    def __init__(self, grid_res=400, x_style=None, y_style=None, name='nullclines'):
        super().__init__(name)
        self.grid_res = grid_res
        self._x_style = {**self._X_DEFAULT, **(x_style or {})}
        self._y_style = {**self._Y_DEFAULT, **(y_style or {})}

    def prepare(self, system, mask, ax):
        # Use current view limits for the grid (adapts to zoom/pan)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xs = np.linspace(xmin, xmax, self.grid_res)
        ys = np.linspace(ymin, ymax, self.grid_res)
        X, Y_grid = np.meshgrid(xs, ys)

        F = system.f(X, Y_grid)
        G = system.g(X, Y_grid)

        c1 = ax.contour(X, Y_grid, F, levels=[0], zorder=2, **self._x_style)
        c2 = ax.contour(X, Y_grid, G, levels=[0], zorder=2, **self._y_style)

        self._artists.extend(c1.collections + c2.collections)
        return self._artists


    # Should implment redraw and draw based on curent x limits and y limits 


# ═══════════════════════════════════════════════════════════════════════
#  QuiverLayer
# ═══════════════════════════════════════════════════════════════════════

class QuiverLayer(Layer):
    """Layer showing the vector field as a quiver plot.

    Parameters
    ----------
    grid_res : int
        Number of arrows per axis (default 25).
    scale : float, optional
        Arrow scale (passed to ``ax.quiver``).  Auto if ``None``.
    style : dict
        Style overrides for ``ax.quiver``.
    name : str
    """

    _DEFAULT_STYLE = {
        'color': '#bbbbbb',
        'width': 0.003,
        'headwidth': 4,
        'headlength': 5,
        'alpha': 0.6,
        'zorder': 1,
    }

    def __init__(self, grid_res=25, scale=None, style=None, name='quiver'):
        super().__init__(name)
        self.grid_res = grid_res
        self._scale = scale
        self._style = {**self._DEFAULT_STYLE, **(style or {})}

    def prepare(self, system, mask, ax):
        # Use current view limits for the grid (adapts to zoom/pan)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xs = np.linspace(xmin, xmax, self.grid_res)
        ys = np.linspace(ymin, ymax, self.grid_res)
        X, Y = np.meshgrid(xs, ys)

        U = system.f(X, Y)
        V = system.g(X, Y)
        U_m, V_m = mask.clip_field(X, Y, U, V)

        q = ax.quiver(X, Y, U_m, V_m,
                      scale=self._scale, **self._style)
        self._artists.append(q)
        return self._artists
    
    


# ═══════════════════════════════════════════════════════════════════════
#  OrbitLayer
# ═══════════════════════════════════════════════════════════════════════

class OrbitLayer(Layer):
    """Layer showing integrated trajectories from initial conditions.

    Supports both pre-set ICs and dynamic addition via ``add_ic()``.
    Call ``redraw()`` on the parent ``PhasePortrait`` after adding ICs
    to re-render all layers.

    Parameters
    ----------
    ics : list of (float, float), optional
        Pre-defined initial conditions.
    t_span : tuple
        Integration time span (default ``(0, 20)``).
    dt : float
        Max integration step (default 0.01).
    style : dict
        Line style for trajectories.
    name : str
    """

    _DEFAULT_STYLE = {
        'color': '#333333',
        'alpha': 0.5,
        'zorder': 2,
    }

    def __init__(self, ics=None, t_span=(0, 20), dt=0.01,
                 style=None, name='orbits'):
        super().__init__(name)
        self._ics = list(ics) if ics else []
        self.t_span = t_span
        self.dt = dt
        self._style = {**self._DEFAULT_STYLE, **(style or {})}

    # ── IC management ─────────────────────────────────────────────────

    @property
    def ics(self):
        """List of ``(x0, y0)`` initial conditions (read-only copy)."""
        return list(self._ics)

    def add_ic(self, x0, y0):
        """Append one initial condition.

        Call ``prepare()`` or the portrait's ``redraw()`` to render.

        Returns ``self`` for chaining.
        """
        self._ics.append((float(x0), float(y0)))
        return self

    def clear_ics(self):
        """Remove all ICs.

        Returns ``self`` for chaining.
        """
        self._ics.clear()
        return self

    # ── Layer API ─────────────────────────────────────────────────────

    def prepare(self, system, mask, ax):
        for x0, y0 in self._ics:
            try:
                t, xs, ys = system.trajectory(x0, y0,
                                              t_span=self.t_span,
                                              dt=self.dt)
                if np.all(np.isfinite(xs)) and np.all(np.isfinite(ys)):
                    line = ax.plot(xs, ys, **self._style)[0]
                    self._artists.append(line)
            except Exception:
                continue
        return self._artists