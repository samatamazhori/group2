"""
interactive_plane.py — Interactive μ–c plane viewer with sidebar controls.

Provides:
- Checkbox toggles for each curve (C1–C8)
- Checkbox toggles for intersection points and infinity points
- Scroll-to-zoom at cursor position
- Click-drag to pan
- Left-click to inspect — μ, c and curve signs shown in sidebar
- Keyboard shortcut: ``r`` to reset view / recompute contours
- Extensible click handler — override ``_format_inspect()`` to add output

Usage
-----
    from data import C1, C2, C3, C4, C5, C6, C7, C8, ALL_POINTS, INFINITY_POINTS
    from interactive_plane import InteractivePlane

    ip = InteractivePlane(
        curves=[C1, C2, C3, C4, C5, C6, C7, C8],
        intersection_points=ALL_POINTS,
        infinity_points=INFINITY_POINTS,
    )
    ip.show()
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

from data_model import Curve, Point, InfinityPoint, Region, Boundary
from view import View

from system import System

# ═══════════════════════════════════════════════════════════════════════════════
# InteractivePlane
# ═══════════════════════════════════════════════════════════════════════════════

class InteractivePlane:
    """Interactive μ–c parameter-plane viewer with a sidebar control panel.

    Parameters
    ----------
    curves : list of Curve, optional
        Curves to display on the plane (C1–C8, etc.).
    intersection_points : list of Point, optional
        Finite intersection points (e.g. ``ALL_POINTS``).
    infinity_points : list of InfinityPoint, optional
        Box-edge / infinity points (e.g. ``INFINITY_POINTS``).
    regions : list of Region, optional
        Filled region polygons.
    boundaries : list of Boundary, optional
        Boundary arcs between regions.
    view : View, optional
        Initial viewport.  Defaults to ``View.global_view()``.
    figsize : tuple of float, optional
        Figure size in inches (default ``(16, 9)``).
    grid_resolution : int, optional
        Number of grid points per axis for contour computation
        (default 600; higher = smoother but slower).
    """

    def __init__(self, curves=None, intersection_points=None,
                 infinity_points=None, regions=None, boundaries=None,
                 view=None, figsize=(16, 9), grid_resolution=600):
        # data references
        self.curves = list(curves) if curves else []
        self.intersection_points = list(intersection_points) if intersection_points else []
        self.infinity_points = list(infinity_points) if infinity_points else []
        self.regions = list(regions) if regions else []
        self.boundaries = list(boundaries) if boundaries else []
        self.view = view or View.global_view()
        self.grid_resolution = grid_resolution

        # internal state
        self._press_xy = None
        self._press_data = None
        self._xlim_start = None
        self._ylim_start = None
        self._is_dragging = False

        # artist registries for visibility toggles
        self._curve_artists = {}     # curve_name -> [LineCollection, ...]
        self._point_artists = {}     # 'intersections' | 'infinity' -> PathCollection

        # last-click state for change detection
        self._last_mu = None
        self._last_c = None
        self._last_signature = None
        self._last_eq_types = {}   # eq_name -> type_string

        # build the figure
        self.fig = plt.figure(figsize=figsize)
        self._setup_layout()
        self._build_regions()
        self._build_boundaries()
        self._build_curves()
        self._build_points()
        self._connect_events()
        self._apply_view()

    # ════════════════════════════════════════════════════════════════════
    # Layout
    # ════════════════════════════════════════════════════════════════════

    _MARGIN_L = 0.018
    _MARGIN_R = 0.155
    _PLOT_L   = 0.175
    _PLOT_R   = 0.975
    _BOTTOM   = 0.04
    _TOP      = 0.97

    def _setup_layout(self):
        """Create the main plot axes and all sidebar UI elements."""
        # main plot
        self.ax = self.fig.add_axes([
            self._PLOT_L, self._BOTTOM,
            self._PLOT_R - self._PLOT_L, self._TOP - self._BOTTOM,
        ])
        self.ax.set_xlabel('\u03bc')
        self.ax.set_ylabel('c')
        self.ax.grid(True, alpha=0.3)

        # sidebar title
        self.fig.text(self._MARGIN_L, self._TOP - 0.005,
                      '\u03bc\u2013c Controls',
                      fontweight='bold', fontsize=13, va='top', ha='left')

        self.fig.text(self._MARGIN_L, self._TOP - 0.035,
                      '\u2500\u2500 Curves \u2500\u2500',
                      fontweight='bold', fontsize=9, va='top', ha='left',
                      color='#555555')

        # curve checkboxes
        n = len(self.curves)
        ch = min(0.52, max(0.10, n * 0.055)) if n else 0.01
        y_bot_ch = self._TOP - 0.050 - ch
        self.curve_labels = [
            f"{c.name}  {c.metadata.get('label', '')}" for c in self.curves
        ]
        cax = self.fig.add_axes([self._MARGIN_L, y_bot_ch,
                                 self._MARGIN_R - self._MARGIN_L, ch])
        self.curve_check = CheckButtons(
            cax, self.curve_labels, [True] * n
        )
        self.curve_check.label_fontsize = 8
        self.curve_check.on_clicked(self._toggle_curve)

        # point section header
        y_pt_top = y_bot_ch - 0.015
        self.fig.text(self._MARGIN_L, y_pt_top,
                      '\u2500\u2500 Points \u2500\u2500',
                      fontweight='bold', fontsize=9, va='bottom', ha='left',
                      color='#555555')

        # point checkboxes
        pt_labels = []
        pt_init = []
        if self.intersection_points:
            pt_labels.append('Intersections')
            pt_init.append(True)
        if self.infinity_points:
            pt_labels.append('Infinity points')
            pt_init.append(True)

        if pt_labels:
            pch = min(0.10, max(0.05, len(pt_labels) * 0.045))
            y_bot_pt = y_pt_top - 0.005 - pch
            pax = self.fig.add_axes([self._MARGIN_L, y_bot_pt,
                                     self._MARGIN_R - self._MARGIN_L, pch])
            self.point_check = CheckButtons(pax, pt_labels, pt_init)
            self.point_check.label_fontsize = 8
            self.point_check.on_clicked(self._toggle_point_group)
            self.point_labels = pt_labels
        else:
            self.point_check = None
            y_bot_pt = y_pt_top

        # info section header
        y_info_top = y_bot_pt - 0.015
        self.fig.text(self._MARGIN_L, y_info_top,
                      '\u2500\u2500 Info \u2500\u2500',
                      fontweight='bold', fontsize=9, va='bottom', ha='left',
                      color='#555555')

        # info text panel
        iy = y_info_top - 0.005
        self.info_ax = self.fig.add_axes([
            self._MARGIN_L, self._BOTTOM + 0.005,
            self._MARGIN_R - self._MARGIN_L, iy - self._BOTTOM - 0.005,
        ])
        self.info_ax.axis('off')
        self.info_text = self.info_ax.text(
            0.05, 0.98, 'Click on the plane\nto inspect a point',
            transform=self.info_ax.transAxes,
            va='top', fontsize=9, fontfamily='monospace',
            color='#333333',
        )

        # control hint at bottom
        self.fig.text(self._MARGIN_L, 0.002,
                      'Scroll: zoom  |  Drag: pan  |  R: reset',
                      fontsize=7, color='#888888', va='bottom', ha='left')

    # ════════════════════════════════════════════════════════════════════
    # Rendering
    # ════════════════════════════════════════════════════════════════════

    def _build_curves(self):
        """Compute contour lines for all curves at the current view extent.
        Results are stored in ``self._curve_artists`` for toggle control.
        """
        mu_lo, mu_hi = self.ax.get_xlim()
        c_lo, c_hi = self.ax.get_ylim()
        span = max(mu_hi - mu_lo, c_hi - c_lo)
        GRID = max(200, min(self.grid_resolution,
                            int(self.grid_resolution / span))) if span > 0 else 400

        MU_g, C_g = np.meshgrid(
            np.linspace(mu_lo, mu_hi, GRID),
            np.linspace(c_lo, c_hi, GRID),
        )

        for curve in self.curves:
            style = curve.metadata
            try:
                Z = curve(MU_g, C_g)
                cs = self.ax.contour(
                    MU_g, C_g, Z, levels=[0],
                    colors=style.get('color', '#333333'),
                    linewidths=style.get('linewidth', 2),
                    linestyles=style.get('linestyle', '-'),
                    zorder=1,
                )
                self._curve_artists[curve.name] = cs.collections
            except Exception as e:
                print(f'  \u26a0  {curve.name} failed to plot: {e}')
                self._curve_artists[curve.name] = []

    def _build_points(self):
        """Scatter-plot intersection and infinity points."""
        if self.intersection_points:
            mus = [p.mu for p in self.intersection_points]
            cs = [p.c for p in self.intersection_points]
            self._point_artists['intersections'] = self.ax.scatter(
                mus, cs, marker='o', c='#333333', s=30,
                edgecolors='white', linewidths=0.6,
                zorder=5, label='intersections',
            )

        if self.infinity_points:
            mus = [p.mu for p in self.infinity_points]
            cs = [p.c for p in self.infinity_points]
            self._point_artists['infinity'] = self.ax.scatter(
                mus, cs, marker='s', c='#000000', s=40,
                edgecolors='#000000', linewidths=1.2,
                zorder=5, label='infinity',
            )

    def _build_regions(self):
        """Fill region polygons."""
        for region in self.regions:
            verts = region.vertices
            if len(verts) < 3:
                continue
            xs = [v.mu for v in verts]
            ys = [v.c for v in verts]
            colour = region.metadata.get('color', 'lightblue')
            alpha = region.metadata.get('alpha', 0.3)
            edge = region.metadata.get('edge_color', colour)
            self.ax.fill(xs, ys, color=colour, alpha=alpha, zorder=0)
            self.ax.plot(xs + [xs[0]], ys + [ys[0]],
                         color=edge, linewidth=1.5, zorder=1)
            cx, cy = np.mean(xs), np.mean(ys)
            self.ax.text(cx, cy, region.name, fontsize=10, fontweight='bold',
                         ha='center', va='center', zorder=6,
                         bbox=dict(boxstyle='round,pad=0.15',
                                   fc='white', ec='none', alpha=0.7))

    def _build_boundaries(self):
        """Draw boundary arcs between regions."""
        for boundary in self.boundaries:
            arc = boundary.arc
            if arc is None or arc.start is None or arc.end is None:
                continue
            colour = boundary.metadata.get('color', '#666666')
            lw = boundary.metadata.get('linewidth', 1.5)
            ls = boundary.metadata.get('linestyle', '-')
            ts = np.linspace(0, 1, 120)
            xs = arc.start.mu + ts * (arc.end.mu - arc.start.mu)
            ys = arc.start.c + ts * (arc.end.c - arc.start.c)
            self.ax.plot(xs, ys, color=colour, linewidth=lw,
                         linestyle=ls, zorder=2)

    def _apply_view(self):
        """Set axes limits from the current view object."""
        self.ax.set_xlim(self.view.mu_min, self.view.mu_max)
        self.ax.set_ylim(self.view.c_min, self.view.c_max)

    # �────────────────────────────────────────────────────────────────────────────────
    # Sidebar callbacks  (visibility toggles)
    # �─────────────────────────────────────────────────────────────────

    def _toggle_curve(self, label):
        """CheckButtons callback — show / hide a single curve."""
        curve_name = label.split()[0]
        idx = self.curve_labels.index(label)
        visible = self.curve_check.get_status()[idx]
        for coll in self._curve_artists.get(curve_name, []):
            coll.set_visible(visible)
        self.fig.canvas.draw_idle()

    def _toggle_point_group(self, label):
        """CheckButtons callback — show / hide a point group."""
        mapping = {
            'Intersections': 'intersections',
            'Infinity points': 'infinity',
        }
        key = mapping.get(label, label.lower())
        if key in self._point_artists:
            idx = self.point_labels.index(label)
            visible = self.point_check.get_status()[idx]
            self._point_artists[key].set_visible(visible)
            self.fig.canvas.draw_idle()

    # ╕───────────────────────────────────────────────────────────────────
    # Event handlers  (zoom, pan, inspec, keyboard)
    # ╕───────────────────────────────────────────────────────────────────

    def _connect_events(self):
        """Register all interactive event handlers with the figure canvas."""
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _on_scroll(self, event):
        """Scroll-wheel zoom, keeping the cursor position fixed."""
        if event.inaxes != self.ax:
            return
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return

        scale = 1.25 if event.button == 'up' else 1 / 1.25

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        x_width = cur_xlim[1] - cur_xlim[0]
        y_height = cur_ylim[1] - cur_ylim[0]

        new_width = x_width / scale
        new_height = y_height / scale

        rel_x = (xdata - cur_xlim[0]) / x_width
        rel_y = (ydata - cur_ylim[0]) / y_height

        self.ax.set_xlim([xdata - new_width * rel_x,
                          xdata + new_width * (1 - rel_x)])
        self.ax.set_ylim([ydata - new_height * rel_y,
                          ydata + new_height * (1 - rel_y)])
        self.fig.canvas.draw_idle()

    def _on_press(self, event):
        """Record the starting position for drag-to-pan / click detection."""
        if event.inaxes != self.ax or event.button != 1:
            return
        self._press_xy = (event.x, event.y)
        self._press_data = (event.xdata, event.ydata)
        self._xlim_start = self.ax.get_xlim()
        self._ylim_start = self.ax.get_ylim()
        self._is_dragging = False

    def _on_motion(self, event):
        """Drag-to-pan: shift the axes limits by the mouse delta."""
        if self._press_xy is None:
            return
        if event.inaxes != self.ax:
            return

        dx_px = event.x - self._press_xy[0]
        dy_px = event.y - self._press_xy[1]

        if abs(dx_px) > 3 or abs(dy_px) > 3:
            self._is_dragging = True

        if not self._is_dragging:
            return

        inv = self.ax.transData.inverted()
        x_now, y_now = inv.transform((event.x, event.y))
        x_start, y_start = self._press_data

        self.ax.set_xlim(self._xlim_start[0] + (x_start - x_now),
                         self._xlim_start[1] + (x_start - x_now))
        self.ax.set_ylim(self._ylim_start[0] + (y_start - y_now),
                         self._ylim_start[1] + (y_start - y_now))
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        """End pan.  If no drag occurred, treat as an inspect click."""
        if self._press_xy is None or event.button != 1:
            return

        if not self._is_dragging:
            self._on_inspect(self._press_data)

        self._press_xy = None
        self._is_dragging = False
        self._xlim_start = None
        self._ylim_start = None

    # ── inspection (extensible) ──────────────────────────────────────────

    def _on_inspect(self, pos):
        """Show (μ, c) and curve signs in the sidebar info panel.

        Override ``_format_inspect()`` to add custom output."""
        mu, c = pos
        if mu is None or c is None:
            return
        lines = self._format_inspect(mu, c)
        self.info_text.set_text('\n'.join(lines))
        self.fig.canvas.draw_idle()

    def _format_signs(self, mu, c):
        """Return (lines, signature) for curves signs at (µ, c).

        ``lines`` is a list of formatted strings, ``signature`` is a
        tuple of ``'+' | '-' | '0' | '?'`` (one per curve).

        When a previous click exists, signs that **changed** are
        marked with ``◀``.
        """
        sig = []
        for curve in self.curves:
            try:
                val = curve(mu, c)
                s = '+' if val > 1e-12 else ('-' if val < -1e-12 else '0')
            except Exception:
                s = '?'
            sig.append(s)
        signature = tuple(sig)

        change_map = {}
        if self._last_signature is not None and len(self._last_signature) == len(sig):
            for i, (a, b) in enumerate(zip(self._last_signature, sig)):
                if a != b:
                    change_map[i] = True

        lines = ['\u2500\u2500 signs \u2500\u2500']
        for i, curve in enumerate(self.curves):
            marker = ' \u25c6' if i in change_map else ''
            lines.append(f'  {curve.name}  {sig[i]}{marker}')

        return lines, signature

    def _format_equilibria(self, mu, c):
        """Return lines for equilibrium summary at (µ, c).

        When ``_last_eq_types`` holds previous types, any change is
        annotated with ``← was <oldtype>``.
        """
        try:
            sys = System(mu, c)
            sys.classify_all()
            eqs = sys.real_equilibria()
        except Exception as e:
            return [f'System error: {e}'], None

        if not eqs:
            return ['0 eq.'], None

        # compute column widths for alignment
        name_w = max(len(eq.name) for eq in eqs)
        type_w = max(len(eq.type or '?') for eq in eqs)
        type_w = max(type_w, 4)

        lines = [f'{len(eqs)} eq.']
        for eq in eqs:
            t = eq.type or '?'
            coord = f'({eq.x_num:.3g}, {eq.y_num:.3g})'

            old = self._last_eq_types.get(eq.name)
            if old is not None and old != t:
                suffix = f'  \u2190 was {old}'
            else:
                suffix = ''

            lines.append(
                f'  {eq.name:<{name_w}}  {t:<{type_w}}  {coord}{suffix}'
            )

        new_types = {eq.name: eq.type for eq in eqs}
        return lines, new_types

    def _format_inspect(self, mu, c):
        """Build the info-panel text for a given (µ, c)."""
        lines = [f'\u03bc = {mu:.5g}    c = {c:.5g}']

        # signs (with change markers)
        # i commented out these 
        sign_lines, signature = self._format_signs(mu, c)
        lines.extend(sign_lines)

        # equilibria (with type-change markers)
        lines.append('')
        eq_lines, new_types = self._format_equilibria(mu, c)
        lines.extend(eq_lines)

        # store state for next click
        self._last_mu = mu
        self._last_c = c
        self._last_signature = signature
        if new_types is not None:
            self._last_eq_types = new_types

        return lines

    # ── keyboard shortcuts ───────────────────────────────────────

    def _on_key(self, event):
        """Keyboard shortcut handler."""
        if event.key == 'r':
            self.recompute()

    # ════════════════════════════════════════════════════════════════════
    # Public API
    # ════════════════════════════════════════════════════════════════════

    def show(self):
        """Display the interactive figure."""
        self.fig.show()
        plt.show()
        return self

    def recompute(self):
        """Recompute contour lines at the current axes limits.

        Useful after extreme zoom-in where the original contour grid
        has become too coarse."""
        # remove old contour artists
        for name, colls in list(self._curve_artists.items()):
            for c in colls:
                try:
                    c.remove()
                except Exception:
                    pass
        self._curve_artists = {}

        # store current visibility states for each curve
        vis_map = {}
        for label in self.curve_labels:
            idx = self.curve_labels.index(label)
            curve_name = label.split()[0]
            vis_map[curve_name] = self.curve_check.get_status()[idx]

        # rebuild contours
        self._build_curves()

        # re-apply visibility
        for curve_name, visible in vis_map.items():
            for coll in self._curve_artists.get(curve_name, []):
                coll.set_visible(visible)

        self.fig.canvas.draw_idle()