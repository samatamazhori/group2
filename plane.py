"""
plane.py — Layered μ–c parameter-plane plotter.

Usage
-----
    pb = PlaneBuilder()                 # default: global view
    pb.add(C1, C2, C3)                  # curves
    pb.add(ALL_POINTS)                  # points from intersections
    pb.add(INFINITY_POINTS)             # box-edge markers
    pb.show()                           # render interactively

    # Zoom in
    pb.view = View.r1_detail()
    pb.show()

    # Save
    pb.save()                           # auto-named
    pb.save('my_custom_name')           # custom name

    # Switch layers on the fly
    pb.remove(C3)
    pb.add(R1, R2)                      # regions are filled
    pb.show()
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from data_model import Curve, Point, InfinityPoint, Region, Boundary, Intersection
from view import View


# ── style maps ───────────────────────────────────────────────────────────

# Z-order: what draws first (back) → last (front)
Z_ORDER = {
    Region: 0,
    Curve: 1,
    Boundary: 2,
    Intersection: 3,
    Point: 4,
    InfinityPoint: 4,
}

# Point marker styles by type
POINT_STYLES = {
    Point: {
        'marker': 'o',
        'c': '#333333',
        's': 30,
        'edgecolors': 'white',
        'linewidths': 0.6,
        'label': 'intersection (real)',
    },
    InfinityPoint: {
        'marker': 's',
        'c': '#000000',
        's': 40,
        'edgecolors': '#000000',
        'linewidths': 1.2,
        'label': 'box edge (infinity point)',
    },
}


def _view_str(view):
    """Short readable tag for a view, e.g. ``"mu-0.5-1.5_c-0.5-3.0"``."""
    return f'mu{view.mu_min:.4g}-{view.mu_max:.4g}_c{view.c_min:.4g}-{view.c_max:.4g}'


# ═══════════════════════════════════════════════════════════════════════════
# PlaneBuilder
# ═══════════════════════════════════════════════════════════════════════════

class PlaneBuilder:
    """Layered μ–c parameter-plane plotter.

    Accepts any mix of ``Curve``, ``Point``, ``InfinityPoint``, ``Region``,
    ``Boundary``, or ``Intersection`` objects via the unified ``.add()``
    method.  Objects are drawn in z-order (regions → curves → points).

    Change ``.view`` between calls to ``.show()`` or ``.save()`` to zoom/pan.
    """

    def __init__(self, view=None, figsize=(12, 8),
                 title=None, grid=True):
        self.view = view or View.global_view()
        self.figsize = figsize
        self.title = title
        self.grid = grid
        self._objects = []          # (priority, obj)
        self._fig = None
        self._ax = None

    # ── object management ───────────────────────────────────────────────

    def add(self, *items):
        """Add drawable objects — curves, points, regions, boundaries, etc.

        Accepts any number of positional arguments, each being either
        a single object or a list/tuple of objects.  Chainable.

        Examples
        --------
            pb.add(C1, C2, C3)
            pb.add(ALL_POINTS)          # list of Point
            pb.add(INFINITY_POINTS)     # list of InfinityPoint
            pb.add(R1, R2).add(C4)      # chaining
        """
        for item in items:
            if isinstance(item, (list, tuple)):
                for sub in item:
                    self._add_one(sub)
            else:
                self._add_one(item)
        return self

    def _add_one(self, obj):
        priority = Z_ORDER.get(type(obj), 5)
        self._objects.append((priority, obj))

    def remove(self, obj):
        """Remove a specific object from the layer stack.

        Chainable.
        """
        self._objects = [(p, o) for p, o in self._objects if o is not obj]
        return self

    def clear(self):
        """Remove all objects.

        Chainable.
        """
        self._objects = []
        return self

    # ── by-type accessors (queries, not filters) ────────────────────────

    @property
    def curves(self):
        """List of all added Curve objects."""
        return [o for _, o in self._objects if isinstance(o, Curve)]

    @property
    def points(self):
        """List of all added Point and InfinityPoint objects."""
        return [o for _, o in self._objects
                if isinstance(o, (Point, InfinityPoint))]

    @property
    def regions(self):
        """List of all added Region objects."""
        return [o for _, o in self._objects if isinstance(o, Region)]

    @property
    def boundaries(self):
        """List of all added Boundary objects."""
        return [o for _, o in self._objects if isinstance(o, Boundary)]

    # ── rendering (internal) ────────────────────────────────────────────

    def _build(self):
        """Create figure, draw all layers, build legend.  Called by
        ``show()`` and ``save()``."""
        mu_lo, mu_hi = self.view.mu_min, self.view.mu_max
        c_lo, c_hi   = self.view.c_min, self.view.c_max

        self._fig, self._ax = plt.subplots(figsize=self.figsize)
        ax = self._ax

        ax.set_xlim(mu_lo, mu_hi)
        ax.set_ylim(c_lo, c_hi)
        ax.set_xlabel('μ')
        ax.set_ylabel('c')

        # title: prefer explicit, then view description, then view name
        ax.set_title(self.title or self.view.description or self.view.name)

        if self.grid:
            ax.grid(True, alpha=0.3)

        # box frame
        ax.plot([mu_lo, mu_hi, mu_hi, mu_lo, mu_lo],
                [c_lo, c_lo, c_hi, c_hi, c_lo],
                color='#cccccc', linewidth=1, linestyle='--', zorder=0)

        # draw layers in z-order
        sorted_objs = sorted(self._objects, key=lambda x: x[0])
        for _priority, obj in sorted_objs:
            self._draw_one(obj, ax)

        # legend
        self._build_legend(ax)

        self._fig.tight_layout()

    def _draw_one(self, obj, ax):
        """Dispatch drawing by object type."""
        if isinstance(obj, Curve):
            self._draw_curve(obj, ax)
        elif isinstance(obj, (Point, InfinityPoint)):
            self._draw_point(obj, ax)
        elif isinstance(obj, Region):
            self._draw_region(obj, ax)
        elif isinstance(obj, Intersection):
            for p in obj.points:
                self._draw_point(p, ax)
        elif isinstance(obj, Boundary):
            self._draw_boundary(obj, ax)

    # ── per-type draw methods ───────────────────────────────────────────

    def _draw_curve(self, curve, ax):
        """Contour line for one curve, clipped to view."""
        style = curve.metadata
        mu_lo, mu_hi = self.view.mu_min, self.view.mu_max
        c_lo, c_hi   = self.view.c_min, self.view.c_max

        # adaptive grid — finer for small views, coarser for large
        span = max(mu_hi - mu_lo, c_hi - c_lo)
        GRID = max(200, min(800, int(800 / span))) if span > 0 else 400

        MU_g, C_g = np.meshgrid(
            np.linspace(mu_lo, mu_hi, GRID),
            np.linspace(c_lo, c_hi, GRID),
        )

        try:
            Z = curve(MU_g, C_g)
            ax.contour(MU_g, C_g, Z, levels=[0],
                       colors=style.get('color', '#333333'),
                       linewidths=style.get('linewidth', 2),
                       linestyles=style.get('linestyle', '-'),
                       zorder=1)
        except Exception as e:
            print(f'  ⚠  {curve.name} failed to plot: {e}')

    def _draw_point(self, pt, ax):
        """Single point marker + label (if inside view)."""
        if not (self.view.mu_min <= pt.mu <= self.view.mu_max and
                self.view.c_min <= pt.c <= self.view.c_max):
            return  # skip — outside visible area

        style = POINT_STYLES.get(type(pt), POINT_STYLES[Point])

        ax.scatter([pt.mu], [pt.c],
                   marker=style['marker'],
                   c=style['c'], s=style['s'],
                   edgecolors=style['edgecolors'],
                   linewidths=style['linewidths'],
                   zorder=5)

        # annotation offset based on name hash for variety
        h = hash(pt.name) if pt.name else 0
        dx = 8 if h % 2 == 0 else -8
        dy = 6 if h % 3 == 0 else 7 - h % 13

        ax.annotate(pt.name, (pt.mu, pt.c), xytext=(dx, dy),
                    textcoords='offset points', fontsize=7,
                    color='#333333', fontfamily='monospace',
                    arrowprops=dict(arrowstyle='->', color='#999999', lw=0.5),
                    zorder=6)

    def _draw_region(self, region, ax):
        """Filled polygon for a region + boundary outline + centroid label."""
        verts = region.vertices
        if len(verts) < 3:
            return

        xs = [v.mu for v in verts]
        ys = [v.c for v in verts]

        # style from metadata, with sensible defaults
        colour = region.metadata.get('color', 'lightblue')
        alpha  = region.metadata.get('alpha', 0.3)
        edge   = region.metadata.get('edge_color', colour)

        ax.fill(xs, ys, color=colour, alpha=alpha, zorder=0)

        # closed boundary
        ax.plot(xs + [xs[0]], ys + [ys[0]],
                color=edge, linewidth=1.5, zorder=1)

        # label near centroid
        cx = np.mean(xs)
        cy = np.mean(ys)
        ax.text(cx, cy, region.name, fontsize=10, fontweight='bold',
                ha='center', va='center', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15',
                          fc='white', ec='none', alpha=0.7))

    def _draw_boundary(self, boundary, ax):
        """Draw a boundary arc between two regions."""
        arc = boundary.arc
        if arc is None:
            return

        # Sample along the curve between start and end
        start = arc.start
        end   = arc.end
        if start is None or end is None:
            return

        colour = boundary.metadata.get('color', '#666666')
        lw     = boundary.metadata.get('linewidth', 1.5)
        ls     = boundary.metadata.get('linestyle', '-')

        # linear interpolation (sufficient for visual)
        ts = np.linspace(0, 1, 120)
        xs = start.mu + ts * (end.mu - start.mu)
        ys = start.c + ts * (end.c - start.c)

        ax.plot(xs, ys, color=colour, linewidth=lw, linestyle=ls, zorder=2)

    # ── legend ──────────────────────────────────────────────────────────

    def _build_legend(self, ax):
        """Construct a deduplicated legend from the current layers."""
        handles = []

        # one handle per curve
        seen_curves = set()
        for curve in self.curves:
            if curve.name in seen_curves:
                continue
            seen_curves.add(curve.name)
            style = curve.metadata
            handles.append(Line2D(
                [0], [0],
                color=style.get('color', '#333333'),
                linewidth=style.get('linewidth', 2),
                linestyle=style.get('linestyle', '-'),
                label=style.get('label', curve.name),
            ))

        # one handle for intersection points
        has_intersections = any(
            isinstance(o, (Point, Intersection)) and not isinstance(o, InfinityPoint)
            for _, o in self._objects
        )
        if has_intersections:
            handles.append(Line2D(
                [0], [0],
                marker='o', color='w', markerfacecolor='#333333',
                markersize=6, label='intersection (real)',
            ))

        # one handle for infinity points
        has_infinity = any(
            isinstance(o, InfinityPoint) for _, o in self._objects
        )
        if has_infinity:
            handles.append(Line2D(
                [0], [0],
                marker='s', color='w', markerfacecolor='#000000',
                markersize=7, label='box edge (infinity point)',
            ))

        # one handle per region
        seen_regions = set()
        for region in self.regions:
            if region.name in seen_regions:
                continue
            seen_regions.add(region.name)
            colour = region.metadata.get('color', 'lightblue')
            handles.append(Line2D(
                [0], [0],
                color=colour, linewidth=2, label=region.name,
            ))

        if handles:
            ax.legend(handles=handles, loc='upper right', fontsize=11,
                      framealpha=0.9, edgecolor='#cccccc')

    # ── public API ──────────────────────────────────────────────────────

    def show(self):
        """Render and display the figure in an interactive window."""
        self._build()
        self._fig.show()
        plt.show()
        return self

    def save(self, name=None, dpi=200, close=True):
        """Save the figure under ``figs/{view.name}/{name}_{view.suffix}.png``.

        Parameters
        ----------
        name : str, optional
            Descriptive filename stem.  If ``None``, auto-generates from
            the layer types present (e.g. ``"curves+points"``).
        dpi : int
            Figure resolution (default 200).
        close : bool
            Whether to ``plt.close()`` the figure after saving (default ``True``).

        Returns
        -------
        str
            The full path to the saved file.
        """
        self._build()

        # ensure directory exists
        view_dir = f'figs/{self.view.name}'
        os.makedirs(view_dir, exist_ok=True)

        if name is None:
            name = self._auto_name()

        # sanitise name for filesystem
        safe = name.replace(' ', '_').replace('/', '_')

        path = f'{view_dir}/{safe}_{self.view.suffix}.png'

        self._fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f'Saved  {path}')

        if close:
            plt.close(self._fig)

        return path

    def _auto_name(self):
        """Auto-generate a filename stem from the layer types present."""
        kinds = []
        for _, o in self._objects:
            t = type(o)
            if t is Curve and 'curves' not in kinds:
                kinds.append('curves')
            elif t is InfinityPoint and 'infinity' not in kinds:
                kinds.append('infinity')
            elif t is Point and 'points' not in kinds:
                kinds.append('points')
            elif t is Region and 'regions' not in kinds:
                kinds.append('regions')
            elif t is Boundary and 'boundaries' not in kinds:
                kinds.append('boundaries')
            elif t is Intersection and 'intersections' not in kinds:
                kinds.append('intersections')
        return '+'.join(kinds) if kinds else 'plot'

    # ── snapshot / metadata ─────────────────────────────────────────────

    def snapshot(self):
        """Return a dict describing the current state (view + object count).

        Useful for saving alongside figures or for logging.
        """
        return {
            'view': {
                'name': self.view.name,
                'box': list(self.view.box),
                'description': self.view.description,
            },
            'n_curves': len(self.curves),
            'n_points': len(self.points),
            'n_regions': len(self.regions),
            'n_boundaries': len(self.boundaries),
            'n_total': len(self._objects),
        }

    def __repr__(self):
        return (f'<PlaneBuilder view="{self.view.name}" '
                f'{len(self._objects)} objects>')


# ── convenience: quick-plot replacement ─────────────────────────────────

def quick_plot(view=None, curves=None, points=None, regions=None,
               figsize=(12, 8), save=None, **kwargs):
    """One-shot plotting: build a PlaneBuilder, add objects, show/save.

    Parameters
    ----------
    view : View or str, optional
        Which view to use.  Strings are looked up in ``DEFAULT_VIEWS``.
    curves, points, regions : list, optional
        Objects to add.  ``curves`` can also be a single ``Curve``, etc.
    figsize : tuple, optional
    save : str, optional
        If given, save instead of showing.  The string is the filename stem.
    **kwargs
        Passed to ``PlaneBuilder``.

    Returns
    -------
    PlaneBuilder
    """
    from view import DEFAULT_VIEWS

    if isinstance(view, str):
        view = DEFAULT_VIEWS.get(view, View.global_view())

    pb = PlaneBuilder(view=view, figsize=figsize, **kwargs)

    # curves
    if curves is not None:
        if isinstance(curves, (Curve, list, tuple)):
            pb.add(curves)
        else:
            pb.add(curves)

    # points (including InfinityPoint)
    if points is not None:
        pb.add(points)

    # regions
    if regions is not None:
        pb.add(regions)

    if save:
        pb.save(save)
    else:
        pb.show()

    return pb