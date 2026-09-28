"""
system.py — The 2D dynamical system and its equilibrium analysis.

    dx/dt = x[(c-4)y - 5μ + 4μ(x+y) + x(x+3y-3) + 2]
    dy/dt = y[c(y-1) + x(x+3y-1)]

At most 6 equilibrium points::

    Branch A (y = 0)     :  E0 ≡ (0,0),  E_A⁺, E_A⁻
    Branch B (y ≠ 0)     :  E1 ≡ (0,1),  E_B⁺, E_B⁻

Usage::

    sys = System(mu=0.3, c=1.5)
    sys.equilibria()          → list of 6 Equilibrium objects (all info filled)
    sys.num_equilibria()       → count of real, finite equilibria at (μ,c)
    sys.classify_all()         → fill classification for each valid eq
    sys.plot_global()          → global phase portrait
    sys.plot_local(eq)         → local portrait with circular IC net
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.integrate import solve_ivp
from pathlib import Path


# ────────────────────────────────────────────────────────────────────────
#  symbols used throughout  (class-level refs so they never get shadowed
#  by constructor parameters)
# ────────────────────────────────────────────────────────────────────────
X, Y, MU, C = sp.symbols('x y mu c')
x, y, mu, c = X, Y, MU, C  # alias for compact expression-writing below

# ────────────────────────────────────────────────────────────────────────
#  Equilibrium — lightweight data container
# ────────────────────────────────────────────────────────────────────────

class Equilibrium:
    """All info for one equilibrium point at a given (μ,c).

    Attributes are filled by ``System._evaluate_equilibria()`` and
    ``System.classify()``.
    """
    __slots__ = (
        'name',       # str:  "E0", "E1", "E_A⁺", "E_A⁻", "E_B⁺", "E_B⁻"
        'branch',     # str:  "A" or "B"
        'label',      # str:  nice Unicode label for plots
        'x_expr',     # sp.Expr  (symbolic)
        'y_expr',     # sp.Expr  (symbolic)
        'x_num',      # float or NaN
        'y_num',      # float or NaN
        'exists',     # bool
        'type',       # str or None  (filled by classify)
        'trace',      # float or None
        'det',        # float or None
        'eigenvals',  # np.ndarray or None
        'jacobian',   # np.ndarray or None
    )

    def __init__(self, name, branch, label, x_expr, y_expr):
        self.name = name
        self.branch = branch
        self.label = label
        self.x_expr = x_expr
        self.y_expr = y_expr
        self.x_num = np.nan
        self.y_num = np.nan
        self.exists = False
        self.type = None
        self.trace = None
        self.det = None
        self.eigenvals = None
        self.jacobian = None

    def __repr__(self):
        if not self.exists:
            return f'<{self.name}: does not exist>'
        return (f'<{self.name} @ ({self.x_num:.5g}, {self.y_num:.5g})'
                f'  {self.type or "?"}>')


# ────────────────────────────────────────────────────────────────────────
#  System
# ────────────────────────────────────────────────────────────────────────

class System:
    """2D polynomial system parameterised by (μ, c) ∈ ℝ².

    Parameters
    ----------
    mu, c : float
        Parameter values.
    """

    # ── class-level symbol references (never shadowed by constructor params) ──
    X, Y, MU, C = sp.symbols('x y mu c')

    # ── symbolic definitions (class-level, built once) ────────────────
    _f_expr = X * ((C - 4) * Y - 5 * MU + 4 * MU * (X + Y)
                   + X * (X + 3 * Y - 3) + 2)
    _g_expr = Y * (C * (Y - 1) + X * (X + 3 * Y - 1))

    # Jacobian matrix (symbolic) — 2×2 matrix of partial derivatives
    _J_expr = sp.Matrix([
        [sp.diff(_f_expr, X), sp.diff(_f_expr, Y)],
        [sp.diff(_g_expr, X), sp.diff(_g_expr, Y)],
    ])

    # Trace and determinant (symbolic)
    _trace_expr = sp.simplify(_J_expr[0, 0] + _J_expr[1, 1])
    _det_expr   = sp.simplify(_J_expr[0, 0] * _J_expr[1, 1]
                              - _J_expr[0, 1] * _J_expr[1, 0])

    # Discriminants
    _Delta_A = 1 - 4 * MU + 16 * MU ** 2
    _Delta_B = (-4 * (-2 + 8 * MU) * (-2 * C + C ** 2 - C * MU)
                + (2 + C - 11 * MU + 4 * C * MU) ** 2)

    # Branch-A x coordinates (the two roots of the y=0 sub-system)
    _x_A_plus  = (3 - 4 * MU + sp.sqrt(_Delta_A)) / 2
    _x_A_minus = (3 - 4 * MU - sp.sqrt(_Delta_A)) / 2

    # Branch-B x coordinates
    _denom_B = 2 * (-2 + 8 * MU)
    _common_B = -2 - C + 11 * MU - 4 * C * MU

    _x_B_plus  = (_common_B - sp.sqrt(_Delta_B)) / _denom_B   # user: x^p
    _x_B_minus = (_common_B + sp.sqrt(_Delta_B)) / _denom_B   # user: x^n

    # Branch-B y: from dy/dt = 0 with y ≠ 0 →
    #   C(y-1) + X(X+3y-1) = 0  ⇒  y = (C + X - X²) / (C + 3X)
    # We use the *symbolic* x expressions to obtain a closed form.
    _y_B_expr = (C + X - X ** 2) / (C + 3 * X)

    # Build the equilibrium list *template* (expressions only, no numerics)
    _EQUILIBRIUM_TEMPLATES = [
        Equilibrium("E0",    "A", r"(0, 0)",          sp.Integer(0), sp.Integer(0)),
        Equilibrium("E1",    "B", r"(0, 1)",          sp.Integer(0), sp.Integer(1)),
        Equilibrium("E_A⁺",  "A", r"(x⁺_A, 0)",       _x_A_plus, sp.Integer(0)),
        Equilibrium("E_A⁻",  "A", r"(x⁻_A, 0)",       _x_A_minus, sp.Integer(0)),
        Equilibrium("E_B⁺",  "B", r"(x⁺_B, y⁻_B)",   _x_B_plus,
                     _y_B_expr.subs({X: _x_B_plus})),
        Equilibrium("E_B⁻",  "B", r"(x⁻_B, y⁺_B)",   _x_B_minus,
                     _y_B_expr.subs({X: _x_B_minus})),
    ]

    # ── instance ─────────────────────────────────────────────────────

    def __init__(self, mu, c, xlim=None, ylim=None):
        self.mu = float(mu)
        self.c = float(c)
        # Optional user-specified domain bounds (else auto-calculated)
        self._xlim = xlim
        self._ylim = ylim

        # Lambdified numpy functions for the vector field
        self._f_np = sp.lambdify((self.X, self.Y, self.MU, self.C),
                                 self._f_expr, modules='numpy')
        self._g_np = sp.lambdify((self.X, self.Y, self.MU, self.C),
                                 self._g_expr, modules='numpy')

        # Lambdified Jacobian components
        self._J_np = sp.lambdify((self.X, self.Y, self.MU, self.C),
                                 self._J_expr, modules='numpy')

        # Build and evaluate the 6 equilibria for this (μ, c)
        self._equilibria = []
        self._evaluate_equilibria()

    # ── public helpers ────────────────────────────────────────────────

    def f(self, x, y):
        """dx/dt evaluated as a numpy array."""
        return self._f_np(x, y, self.mu, self.c)

    def g(self, x, y):
        """dy/dt evaluated as a numpy array."""
        return self._g_np(x, y, self.mu, self.c)

    def vector_field(self, x, y):
        """(dx/dt, dy/dt) as a tuple."""
        return self.f(x, y), self.g(x, y)

    def jacobian(self, x, y):
        """2×2 Jacobian matrix at (x, y), as numpy array."""
        return np.array(self._J_np(x, y, self.mu, self.c)).reshape(2, 2)

    # ── equilibria ────────────────────────────────────────────────────

    def _evaluate_equilibria(self):
        """Fill numerical values and existence flags for all 6 equilibria."""
        self._equilibria = []
        for tmpl in self._EQUILIBRIUM_TEMPLATES:
            eq = self._build_eq(tmpl)
            self._equilibria.append(eq)

    def _build_eq(self, template):
        """Deep-copy a template equilibrium and evaluate it at (μ,c)."""
        eq = Equilibrium(template.name, template.branch,
                         template.label,
                         template.x_expr, template.y_expr)
        try:
            # Substitute μ, c into the symbolic expressions
            val = template.x_expr.subs({mu: self.mu, c: self.c})
            # Check for complex results (e.g. sqrt of negative with denom=0)
            if val.is_complex and not val.is_real:
                eq.exists = False
                return eq
            x_val = complex(val).real  # in case sympy returns e.g. 5 + 0*I
            val = template.y_expr.subs({mu: self.mu, c: self.c})
            if val.is_complex and not val.is_real:
                eq.exists = False
                return eq
            y_val = complex(val).real
        except Exception:
            eq.exists = False
            return eq

        # Existence checks
        eq.x_num = x_val
        eq.y_num = y_val

        if not np.isfinite(x_val) or not np.isfinite(y_val):
            eq.exists = False
            return eq

        if np.isnan(x_val) or np.isnan(y_val):
            eq.exists = False
            return eq

        # Check discriminant conditions
        if 'A' in eq.name and eq.name != 'E0':
            dA = float(self._Delta_A.subs({mu: self.mu}))
            if dA < -1e-12:
                eq.exists = False
                return eq

        if 'B' in eq.name and eq.name != 'E1':
            denom_B_val = float(self._denom_B.subs({mu: self.mu, c: self.c}))
            if abs(denom_B_val) < 1e-12:
                eq.exists = False
                return eq
            dB = float(self._Delta_B.subs({mu: self.mu, c: self.c}))
            if dB < -1e-12:
                eq.exists = False
                return eq

        eq.exists = True
        return eq

    def equilibria(self):
        """Return the list of all 6 ``Equilibrium`` objects (info filled)."""
        return self._equilibria

    def real_equilibria(self):
        """Return only the subset that exist at the current (μ,c)."""
        return [eq for eq in self._equilibria if eq.exists]

    def num_equilibria(self):
        """Return the number of equilibria that exist (ignoring coincidences)."""
        return sum(1 for eq in self._equilibria if eq.exists)

    # ── classification ────────────────────────────────────────────────

    @staticmethod
    def _classify_type(T, D):
        """Given trace *T* and determinant *D*, return a type string.

        This is the standard linear classification for 2D autonomous ODEs.
        """
        atol = 1e-10
        if D < -atol:
            return 'saddle'
        if D > atol:
            disc = T * T - 4 * D
            if T < -atol:
                return 'stable node' if disc > atol else 'stable spiral'
            elif T > atol:
                return 'unstable node' if disc > atol else 'unstable spiral'
            else:
                return 'centre'
        # D ≈ 0
        if abs(T) < atol:
            return 'degenerate (double zero)'
        return 'degenerate (zero det)'

    def classify(self, eq):
        """Fill classification info (trace, det, eigenvalues, type) for one
        equilibrium.

        Modifies *eq* in-place and returns it.
        """
        if not eq.exists:
            eq.type = 'does not exist'
            return eq

        J = self.jacobian(eq.x_num, eq.y_num)
        eq.jacobian = J
        eq.trace = float(np.trace(J))
        eq.det = float(np.linalg.det(J))
        eq.eigenvals = np.linalg.eigvals(J)
        eq.type = self._classify_type(eq.trace, eq.det)
        return eq

    def classify_all(self):
        """Classify all existing equilibria.  Returns list of classified eqs."""
        for eq in self._equilibria:
            self.classify(eq)
        return self.real_equilibria()

    # ── domain calculation ────────────────────────────────────────────

    def _auto_limits(self, pad_factor=1.5, min_pad=0.5):
        """Auto-calculate (xmin, xmax, ymin, ymax) from existing equilibria.

        If the user provided explicit limits via the constructor, those
        are returned instead.
        """
        if self._xlim is not None and self._ylim is not None:
            return (*self._xlim, *self._ylim)

        eqs = self.real_equilibria()
        if not eqs:
            return (-1.0, 1.0, -1.0, 1.0)   # fallback

        xs = np.array([e.x_num for e in eqs])
        ys = np.array([e.y_num for e in eqs])
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        dx = max(xmax - xmin, min_pad) * pad_factor
        dy = max(ymax - ymin, min_pad) * pad_factor
        xc = (xmin + xmax) / 2
        yc = (ymin + ymax) / 2

        return (xc - dx / 2, xc + dx / 2, yc - dy / 2, yc + dy / 2)

    # ── trajectory integration ────────────────────────────────────────

    def trajectory(self, x0, y0, t_span=(0, 20), dt=0.01, **kwargs):
        """Integrate from ``(x0, y0)`` using ``scipy.integrate.solve_ivp``.

        Returns ``(t, x, y)`` arrays.
        """
        def rhs(t, xy):
            xv, yv = xy
            return [self.f(xv, yv), self.g(xv, yv)]

        sol = solve_ivp(rhs, t_span, [x0, y0],
                        max_step=dt, dense_output=True,
                        method='RK45', **kwargs)
        t = sol.t
        xy = sol.y
        return t, xy[0], xy[1]

    # ── plotting ──────────────────────────────────────────────────────

    def _plot_nullclines(self, ax, x_grid, y_grid):
        """Draw nullclines: dx/dt=0 (blue dashed), dy/dt=0 (red dashed)."""
        F = self.f(x_grid, y_grid)
        G = self.g(x_grid, y_grid)

        ax.contour(x_grid, y_grid, F, levels=[0],
                   colors='#1f77b4', linewidths=1.0, linestyles='--', alpha=0.7)
        ax.contour(x_grid, y_grid, G, levels=[0],
                   colors='#d62728', linewidths=1.0, linestyles='--', alpha=0.7)

    def _plot_equilibria(self, ax, eqs=None):
        """Mark equilibrium points with filled circles and labels."""
        if eqs is None:
            eqs = self.real_equilibria()

        for eq in eqs:
            marker = 'o'
            color = '#2ca02c' if eq.type and 'stable' in eq.type else (
                '#d62728' if eq.type and 'unstable' in eq.type else (
                    '#ff7f0e' if eq.type == 'saddle' else '#1f77b4'))
            if eq.type == 'does not exist':
                continue
            ax.scatter(eq.x_num, eq.y_num, marker=marker,
                       color=color, s=60, zorder=10,
                       edgecolors='white', linewidths=1.0)
            ax.annotate(eq.label, (eq.x_num, eq.y_num),
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=8, fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white', alpha=0.75),
                        zorder=12)

    def _plot_trajectories(self, ax, initial_conditions, t_span=(0, 15),
                           color='#333333', alpha=0.5, lw=0.6):
        """Plot trajectories from a list of ``(x0, y0)`` initial conditions."""
        for x0, y0 in initial_conditions:
            try:
                t, xs, ys = self.trajectory(x0, y0, t_span=t_span)
                # Only plot if the trajectory stays finite
                if np.all(np.isfinite(xs)) and np.all(np.isfinite(ys)):
                    ax.plot(xs, ys, color=color, alpha=alpha,
                            linewidth=lw, zorder=2)
            except Exception:
                pass  # skip ICs that blow up immediately

    # ── global portrait ───────────────────────────────────────────────

    def plot_global(self, grid_res=400, t_span=(0, 25), pad_factor=1.5,
                    min_pad=0.5, ic_nx=8, ic_ny=8,
                    save=True, show=True, filename='global_portrait',
                    title=None):
        """Global phase portrait: vector field, nullclines, equilibria.

        Parameters
        ----------
        grid_res : int
            Resolution for nullcline / streamline grid.
        t_span : tuple
            Integration time span for trajectories.
        pad_factor, min_pad : float
            Domain auto-computation parameters.
        ic_nx, ic_ny : int
            Number of initial conditions per axis (rectangular grid).
        save, show : bool
        filename : str
            Saved under ``figs/Rtest(c_{c}-mu_{mu})/``.
        title : str or None
        """
        self.classify_all()

        xmin, xmax, ymin, ymax = self._auto_limits(pad_factor, min_pad)
        xs = np.linspace(xmin, xmax, grid_res)
        ys = np.linspace(ymin, ymax, grid_res)
        X, Y = np.meshgrid(xs, ys)

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # stream plot (vector field)
        U = self.f(X, Y)
        V = self.g(X, Y)
        # Mask out extreme values for clean streamlines
        mag = np.sqrt(U ** 2 + V ** 2)
        mask = mag > 1e3
        U_masked = np.ma.masked_where(mask, U)
        V_masked = np.ma.masked_where(mask, V)

        try:
            ax.streamplot(X, Y, U_masked, V_masked,
                          color='#bbbbbb', linewidth=0.5,
                          density=1.0, arrowsize=0.6, zorder=1)
        except Exception:
            pass  # fallback: skip streamplot if it fails

        # nullclines
        self._plot_nullclines(ax, X, Y)

        # trajectories from rectangular grid of ICs
        ic_x = np.linspace(xmin + 0.1 * (xmax - xmin),
                           xmax - 0.1 * (xmax - xmin), ic_nx)
        ic_y = np.linspace(ymin + 0.1 * (ymax - ymin),
                           ymax - 0.1 * (ymax - ymin), ic_ny)
        ics = [(xi, yi) for xi in ic_x for yi in ic_y]
        self._plot_trajectories(ax, ics, t_span=t_span)

        # equilibria
        self._plot_equilibria(ax)

        # annotations
        ax.axhline(0, color='#cccccc', linewidth=0.5)
        ax.axvline(0, color='#cccccc', linewidth=0.5)
        ax.grid(True, alpha=0.2)

        if title is None:
            title = (f'Global phase portrait  (μ={self.mu:.3g}, c={self.c:.3g})'
                     f'  —  {self.num_equilibria()} eq.')
        ax.set_title(title, fontsize=11)

        fig.tight_layout()
        self._save_or_show(fig, save, show, filename)
        return fig, ax

    # ── local portrait ────────────────────────────────────────────────

    def plot_local(self, eq, zoom_radius=None, grid_res=300,
                   t_span=(0, 20), n_circles=4, n_angles=12,
                   save=True, show=True, filename=None):
        """Local phase portrait centred on one equilibrium.

        Uses a **circular net** of initial conditions — concentric rings
        around the equilibrium — which naturally reveals spiral and nodal
        dynamics.

        Parameters
        ----------
        eq : Equilibrium
            The equilibrium to zoom into (must exist).
        zoom_radius : float or None
            Half-width of the plot window.  Auto-computed if ``None``.
        grid_res : int
        t_span : tuple
        n_circles : int
            Number of concentric rings for ICs.
        n_angles : int
            Number of evenly-spaced angles per ring.
        save, show : bool
        filename : str or None
        """
        self.classify(eq)
        if not eq.exists:
            print(f'⚠  {eq.name} does not exist at μ={self.mu}, c={self.c}')
            return None, None

        # — domain —
        if zoom_radius is None:
            # Pick a radius that scales with distance to nearest other eq
            others = [e for e in self.real_equilibria() if e is not eq]
            if others:
                dists = [np.hypot(e.x_num - eq.x_num, e.y_num - eq.y_num)
                         for e in others]
                zoom_radius = max(0.15, min(dists) * 0.25)
            else:
                zoom_radius = 0.25

        xmin = eq.x_num - zoom_radius
        xmax = eq.x_num + zoom_radius
        ymin = eq.y_num - zoom_radius
        ymax = eq.y_num + zoom_radius

        # — circular IC net —
        rings = np.linspace(0.025, zoom_radius * 0.4, n_circles)
        ics_circle = []
        for r in rings:
            for theta in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
                ics_circle.append((
                    eq.x_num + r * np.cos(theta),
                    eq.y_num + r * np.sin(theta),
                ))

        xs = np.linspace(xmin, xmax, grid_res)
        ys = np.linspace(ymin, ymax, grid_res)
        X, Y = np.meshgrid(xs, ys)

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # stream plot
        U = self.f(X, Y)
        V = self.g(X, Y)
        mag = np.sqrt(U ** 2 + V ** 2)
        U_masked = np.ma.masked_where(mag > 1e3, U)
        V_masked = np.ma.masked_where(mag > 1e3, V)
        try:
            ax.streamplot(X, Y, U_masked, V_masked,
                          color='#bbbbbb', linewidth=0.4,
                          density=0.8, arrowsize=0.5, zorder=1)
        except Exception:
            pass

        # nullclines
        self._plot_nullclines(ax, X, Y)

        # trajectories from circular IC net
        self._plot_trajectories(ax, ics_circle, t_span=t_span,
                                color='#333333', alpha=0.55, lw=0.7)

        # also shoot a few trajectories *outward* from very near the eq
        # to reveal the unstable manifold direction
        if eq.type and 'unstable' in eq.type:
            # small perturbation along eigenvector directions
            if eq.eigenvals is not None and eq.jacobian is not None:
                evals, evecs = np.linalg.eig(eq.jacobian)
                for i, val in enumerate(evals):
                    if val.real > 1e-10:   # unstable direction
                        vec = evecs[:, i].real
                        vec = vec / np.linalg.norm(vec) * 0.02 * zoom_radius
                        for sign in (+1, -1):
                            x0 = eq.x_num + sign * vec[0]
                            y0 = eq.y_num + sign * vec[1]
                            self._plot_trajectories(
                                ax, [(x0, y0)], t_span=(0, t_span[1]),
                                color='#e6550d', alpha=0.7, lw=1.0)

        # equilibrium
        self._plot_equilibria(ax, [eq])

        # draw the rings visually (faded) to show the IC layout
        for r in rings:
            circ = Circle((eq.x_num, eq.y_num), r,
                          fill=False, linestyle=':', linewidth=0.6,
                          color='#999999', alpha=0.4, zorder=0)
            ax.add_patch(circ)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

        t_str = eq.type or '?'
        title = (f'Local portrait: {eq.label}  [{t_str}]  '
                 f'(μ={self.mu:.3g}, c={self.c:.3g})')
        ax.set_title(title, fontsize=10)

        fig.tight_layout()

        fname = filename or f'local_{eq.name}'
        self._save_or_show(fig, save, show, fname)
        return fig, ax

    def plot_all_local(self, **kwargs):
        """Plot local portraits for every existing equilibrium."""
        figs = []
        for eq in self.real_equilibria():
            f, ax = self.plot_local(eq, **kwargs)
            figs.append(f)
        return figs

    # ── save / show ───────────────────────────────────────────────────

    def _save_or_show(self, fig, save, show, name):
        """Save figure to ``figs/Rtest(c_{c}-mu_{mu})/`` and/or show."""
        if save:
            mu_str = f'{self.mu:.10g}'.replace('.', '_')
            c_str = f'{self.c:.10g}'.replace('.', '_')
            out_dir = Path('figs') / f'Rtest(c_{c_str}-mu_{mu_str})'
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f'{name}.pdf'
            fig.savefig(str(path), bbox_inches='tight', dpi=150)
            print(f'  → saved {path}')
        if show:
            plt.show()
        else:
            plt.close(fig)

    # ── summaries ─────────────────────────────────────────────────────

    def symbolic_summary(self):
        """Print symbolic expressions for all 6 equilibria."""
        lines = ['Symbolic equilibrium expressions:',
                 f'  μ = {self.mu:.6g}, c = {self.c:.6g}', '']
        for tmpl in self._EQUILIBRIUM_TEMPLATES:
            x_str = sp.pretty(tmpl.x_expr, use_unicode=True)
            y_str = sp.pretty(tmpl.y_expr, use_unicode=True)
            lines.append(f'  {tmpl.name}  [{tmpl.branch}]  {tmpl.label}:')
            lines.append(f'    x = {x_str}')
            lines.append(f'    y = {y_str}')
            lines.append('')
        print('\n'.join(lines))

    def numerical_summary(self):
        """Print numerical values, existence, and classification."""
        self.classify_all()
        lines = [f'Numerical summary  —  μ = {self.mu:.6g}, c = {self.c:.6g}',
                 f'  {self.num_equilibria()} / 6 equilibria exist',
                 '']
        lines.append(f'  {"Name":>8}  {"Branch":>6}  {"x":>12}  {"y":>12}  '
                     f'{"Type":>20}  {"Trace":>10}  {"Det":>10}')
        lines.append('  ' + '-' * 84)
        for eq in self._equilibria:
            if eq.exists:
                lines.append(
                    f'  {eq.name:>8}  {eq.branch:>6}  {eq.x_num:>12.6g}  '
                    f'{eq.y_num:>12.6g}  {eq.type or "?":>20}  '
                    f'{eq.trace if eq.trace is None else f"{eq.trace:>10.4g}":>10}  '
                    f'{eq.det if eq.det is None else f"{eq.det:>10.4g}":>10}')
            else:
                lines.append(
                    f'  {eq.name:>8}  {eq.branch:>6}  {"—":>12}  {"—":>12}  '
                    f'{"does not exist":>20}  {"—":>10}  {"—":>10}')
        print('\n'.join(lines))