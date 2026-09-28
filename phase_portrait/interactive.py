"""interactive.py — UI glue for PhasePortrait.

Backend-agnostic: works with Qt, TkAgg, MacOSX, and inline backends.
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button, TextBox
import numpy as np


def _robust_refresh(fig):
    """Belt-and-suspenders draw: draw_idle + draw + flush + pause."""
    fig.canvas.draw_idle()
    fig.canvas.draw()
    try:
        fig.canvas.flush_events()
    except Exception:
        pass
    #plt.pause(0.05)


class Interactor:
    """Attach interactive controls to a ``PhasePortrait``.

    Parameters
    ----------
    portrait : PhasePortrait
        The portrait to control.
    """

    def __init__(self, portrait):
        self.portrait = portrait
        self._toggle_ax = None
        self._toggle_check = None
        self._cids = []

    # ── helpers ────────────────────────────────────────────────────────

    def _orbit_layer(self):
        """Return the first ``OrbitLayer``, creating one if none exists."""
        from .layers import OrbitLayer
        for l in self.portrait._layers:
            if isinstance(l, OrbitLayer):
                return l
        ol = OrbitLayer(name='orbits')
        self.portrait.add(ol)
        return ol

    def _full_redraw_and_refresh(self):
        """Full redraw + robust refresh — used by clear, redraw, and scroll zoom."""
        self.portrait.redraw()
        _robust_refresh(self.portrait._fig)

    # ── layer toggles ──────────────────────────────────────────────────

    def add_toggles(self, ax=None):
        """Add a ``CheckButtons`` widget to toggle layer visibility."""
        layers = self.portrait.layers
        labels = [l.name for l in layers]
        visibilities = [l.visible for l in layers]

        if ax is None:
            fig = self.portrait._ax.figure
            n = len(labels)
            h = min(0.45, n * 0.07 + 0.04)
            ax = fig.add_axes([0.82, 0.55, 0.16, h])

        self._toggle_ax = ax
        self._toggle_check = CheckButtons(ax, labels, visibilities)

        for rect in ax.get_children():
            if hasattr(rect, 'set_linewidth'):
                rect.set_linewidth(0.5)
        ax.tick_params(labelsize=8)

        def _on_click(label):
            layer = self.portrait.get_layer(label)
            if layer is None:
                return
            idx = labels.index(label)
            visible = self._toggle_check.get_status()[idx]
            layer.set_visible(visible)
            self._full_redraw_and_refresh()

        self._toggle_check.on_clicked(_on_click)
        return self

    # ── click-to-orbit (right-click) ───────────────────────────────────

    def add_click_orbit(self, callback=None):
        """Right-click on the phase portrait to add an orbit.

        Left-click is reserved for pan/zoom (matplotlib default).
        """
        cid = self.portrait._fig.canvas.mpl_connect(
            'button_press_event', self._make_click_handler(callback)
        )
        self._cids.append(cid)
        return self

    def _make_click_handler(self, callback):
        ol = self._orbit_layer()
        portrait = self.portrait

        def _on_click(event):
            if event.inaxes != portrait._ax:
                return
            # Only right-click (button=3) adds orbits; left-click is for pan/zoom
            if event.button != 3:
                return

            x0, y0 = event.xdata, event.ydata
            print(f'  ➕ orbit at ({x0:.3f}, {y0:.3f})')

            ol.add_ic(x0, y0)
            self._full_redraw_and_refresh()

            if callback:
                callback(x0, y0)

        return _on_click

    # ── text-box orbit ─────────────────────────────────────────────────

    def add_text_orbit(self, ax=None, label='(x₀, y₀):', initial='0, 0'):
        """Text box to enter ICs manually."""
        if ax is None:
            fig = self.portrait._ax.figure
            ax = fig.add_axes([0.82, 0.44, 0.16, 0.06])

        text_box = TextBox(ax, label, initial=initial)

        def _on_submit(text):
            try:
                parts = text.replace('(', '').replace(')', '').split(',')
                x0, y0 = float(parts[0]), float(parts[1])
                print(f'  ➕ orbit at ({x0:.3f}, {y0:.3f})')
                ol = self._orbit_layer()
                ol.add_ic(x0, y0)
                self._full_redraw_and_refresh()
            except (ValueError, IndexError):
                print(f'  ⚠  Invalid input: {text!r}.  Use format: x, y')

        text_box.on_submit(_on_submit)
        return self

    # ── clear button ───────────────────────────────────────────────────

    def add_clear_button(self, ax=None, label='Clear orbits'):
        """Clear all orbits."""
        if ax is None:
            fig = self.portrait._ax.figure
            ax = fig.add_axes([0.82, 0.37, 0.16, 0.05])

        button = Button(ax, label)

        def _on_click(event):
            print('  🗑  clearing orbits')
            ol = self._orbit_layer()
            ol.clear_ics()
            self._full_redraw_and_refresh()

        button.on_clicked(_on_click)
        return self

    # ── reset view ─────────────────────────────────────────────────────

    def add_reset_button(self, ax=None, label='Reset view'):
        """Reset the view to mask limits."""
        if ax is None:
            fig = self.portrait._ax.figure
            ax = fig.add_axes([0.82, 0.30, 0.16, 0.05])

        button = Button(ax, label, color='#eeeeee', hovercolor='#cccccc')

        def _on_click(event):
            print('  🔄  reset view')
            xmin, xmax, ymin, ymax = self.portrait.mask.get_limits()
            self.portrait._ax.set_xlim(xmin, xmax)
            self.portrait._ax.set_ylim(ymin, ymax)
            self._full_redraw_and_refresh()

        button.on_clicked(_on_click)
        return self

    # ── manual full redraw ─────────────────────────────────────────────

    def add_manual_refresh(self, ax=None, label='🔄 Full redraw'):
        """Full rebuild — re-computes nullclines, quiver, orbits at current view."""
        if ax is None:
            fig = self.portrait._ax.figure
            ax = fig.add_axes([0.82, 0.23, 0.16, 0.05])

        button = Button(ax, label, color='#e8f5e9', hovercolor='#c8e6c9')

        def _on_click(event):
            print('  🔄  full redraw')
            self._full_redraw_and_refresh()

        button.on_clicked(_on_click)
        return self

    # ── scroll-to-zoom ─────────────────────────────────────────────────

    def add_scroll_zoom(self, zoom_factor=1.15):
        """Scroll wheel zooms, re-rendering all layers at the new view.

        Parameters
        ----------
        zoom_factor : float
            Scale factor per scroll step (>1 zooms in, <1 zooms out).
        """
        def _on_scroll(event):
            if event.inaxes != self.portrait._ax:
                return

            ax = self.portrait._ax
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            xdata, ydata = event.xdata, event.ydata

            # Direction: scroll up = zoom in, scroll down = zoom out
            factor = zoom_factor if event.button == 'up' else 1.0 / zoom_factor

            # Zoom toward cursor position
            new_xlim = (
                xdata - (xdata - xlim[0]) * factor,
                xdata + (xlim[1] - xdata) * factor,
            )
            new_ylim = (
                ydata - (ydata - ylim[0]) * factor,
                ydata + (ylim[1] - ydata) * factor,
            )

            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)

            # Re-render grid-based layers at new zoom level
            self._full_redraw_and_refresh()

        cid = self.portrait._fig.canvas.mpl_connect(
            'scroll_event', _on_scroll
        )
        self._cids.append(cid)
        return self

    # ── keyboard shortcuts ─────────────────────────────────────────────

    def add_keyboard_shortcuts(self):
        """Keyboard shortcuts for common actions.

        - ``c`` — clear orbits
        - ``r`` — full redraw
        - ``v`` — reset view
        - ``h`` — print help
        """
        def _on_key(event):
            if event.inaxes != self.portrait._ax:
                return
            key = event.key.lower()
            if key == 'c':
                print('  🗑  [c]lear orbits')
                ol = self._orbit_layer()
                ol.clear_ics()
                self._full_redraw_and_refresh()
            elif key == 'r':
                print('  🔄  [r]edraw')
                self._full_redraw_and_refresh()
            elif key == 'v':
                print('  🔄  reset [v]iew')
                xmin, xmax, ymin, ymax = self.portrait.mask.get_limits()
                self.portrait._ax.set_xlim(xmin, xmax)
                self.portrait._ax.set_ylim(ymin, ymax)
                self._full_redraw_and_refresh()
            elif key == 'h':
                print('  ── Keyboard shortcuts ──')
                print('  c   clear orbits')
                print('  r   full redraw')
                print('  v   reset view')
                print('  h   this help')
                print('  ─────────────────────────')

        cid = self.portrait._fig.canvas.mpl_connect(
            'key_press_event', _on_key
        )
        self._cids.append(cid)
        return self

    # ── help label ─────────────────────────────────────────────────────

    def add_help_label(self, ax=None):
        """Small instruction label."""
        if ax is None:
            fig = self.portrait._ax.figure
            ax = fig.add_axes([0.82, 0.17, 0.16, 0.05])
        ax.axis('off')
        ax.text(0.5, 0.5,
                'Right-click → orbit\nScroll → zoom  |  c/r/v/h keys',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=7, color='#666666', style='italic')
        return self

    def __repr__(self):
        return f'<Interactor>'