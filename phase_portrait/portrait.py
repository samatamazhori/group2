"""portrait.py — PhasePortrait fluent builder.

Composes layers into a single figure.  No widgets or event handlers.
"""

import matplotlib.pyplot as plt
import numpy as np

from .masks import R2Mask
from .layers import Layer


class PhasePortrait:
    """Composable phase portrait for a fixed (μ, c).

    Parameters
    ----------
    system : System
        The dynamical system (already at the desired μ, c).
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into.  Creates a new figure if ``None``.
    mask : Mask, optional
        Domain mask.  Defaults to ``R2Mask`` auto-sized from equilibria.
    figsize : tuple of float
        Figure size in inches (default ``(9, 7)``).
    title : str, optional
        Plot title.  Auto-generated if ``None``.
    """

    def __init__(self, system, ax=None, mask=None, figsize=(9, 7), title=None):
        self.system = system
        self._fig = None
        self._ax = ax
        self._own_ax = ax is None          # did we create the axes?
        self._figsize = figsize
        self._title = title
        self._layers = []                  # list of Layer
        self._drawn = False

        # Auto-mask from equilibria if not provided
        if mask is None:
            eqs = system.real_equilibria()
            if eqs:
                self.mask = R2Mask.from_equilibria(eqs)
            else:
                self.mask = R2Mask(-1, 1, -1, 1)
        else:
            self.mask = mask

    # ── layer management ───────────────────────────────────────────────

    def add(self, layer):
        """Add a layer.

        Parameters
        ----------
        layer : Layer

        Returns
        -------
        self
        """
        if not isinstance(layer, Layer):
            raise TypeError(f'Expected a Layer, got {type(layer).__name__}')
        self._layers.append(layer)
        return self

    def remove(self, name):
        """Remove a layer by name.

        Parameters
        ----------
        name : str

        Returns
        -------
        self
        """
        self._layers = [l for l in self._layers if l.name != name]
        return self

    def get_layer(self, name):
        """Retrieve a layer by name, or ``None``.

        Parameters
        ----------
        name : str

        Returns
        -------
        Layer or None
        """
        for l in self._layers:
            if l.name == name:
                return l
        return None

    def toggle(self, name, visible=None):
        """Toggle visibility of a layer by name.

        Parameters
        ----------
        name : str
        visible : bool, optional
            Explicit visibility.  Flips current state if ``None``.
        """
        layer = self.get_layer(name)
        if layer is None:
            raise KeyError(f'Layer {name!r} not found')
        if visible is None:
            visible = not layer.visible
        layer.set_visible(visible)
        return self

    @property
    def layers(self):
        """List of all registered layers."""
        return list(self._layers)

    # ── orbits (convenience) ───────────────────────────────────────────

    def add_orbit(self, x0, y0, **kwargs):
        """Convenience: add an initial condition to the first ``OrbitLayer``.

        Parameters
        ----------
        x0, y0 : float
        **kwargs
            Passed to ``OrbitLayer.add_ic()``.

        Returns
        -------
        self
        """
        from .layers import OrbitLayer
        for l in self._layers:
            if isinstance(l, OrbitLayer):
                l.add_ic(x0, y0)
                return self
        # No OrbitLayer exists — create one with defaults
        ol = OrbitLayer(ics=[(x0, y0)], **kwargs)
        self.add(ol)
        return self

    def clear_orbits(self):
        """Clear all ICs from all ``OrbitLayer`` instances."""
        from .layers import OrbitLayer
        for l in self._layers:
            if isinstance(l, OrbitLayer):
                l.clear_ics()
        return self

    # ── drawing ────────────────────────────────────────────────────────

    def _setup_axes(self):
        """Create or configure axes.

        Idempotent: only creates the figure/axes on the first call.
        Subsequent calls just re-apply limits, labels, and grid.
        """
        if self._ax is None:
            # First-time setup — create the figure
            if self._own_ax:
                self._fig, self._ax = plt.subplots(figsize=self._figsize)
            else:
                self._fig = self._ax.figure

        ax = self._ax

        # Always re-configure these (they get wiped by ax.clear())
        # NOTE: axhline/axvline here are the ONLY copies — they are NOT
        # duplicated in the first-time block above.
        xmin, xmax, ymin, ymax = self.mask.get_limits()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.axhline(0, color='#cccccc', linewidth=0.5, zorder=0)
        ax.axvline(0, color='#cccccc', linewidth=0.5, zorder=0)
        ax.grid(True, alpha=0.2)

        title = self._title or (f'Phase portrait  '
                                f'(\u03bc={self.system.mu:.3g}, '
                                f'c={self.system.c:.3g})')
        ax.set_title(title, fontsize=11)

    def draw(self):
        """Prepare all layers and render.

        Returns
        -------
        self
        """
        if self._ax is None:
            self._setup_axes()

        for layer in self._layers:
            if layer.visible:
                layer.prepare(self.system, self.mask, self._ax)

        if self._own_ax:
            self._fig.tight_layout()

        self._drawn = True
        return self

    def redraw(self):
        """Clear and re-draw everything, preserving the current view.

        Useful after adding ICs or changing the system.
        """
        if self._ax is None:
            self.draw()
            return self

        # Preserve current view limits through the redraw
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()

        # Remove all layer artists
        for layer in self._layers:
            layer.clear()

        # Clear axes — this wipes everything (labels, grid, ticks, artists)
        self._ax.clear()
        self._drawn = False

        # Reconfigure labels & grid WITHOUT touching limits
        self._ax.set_xlabel('x')
        self._ax.set_ylabel('y')
        self._ax.axhline(0, color='#cccccc', linewidth=0.5, zorder=0)
        self._ax.axvline(0, color='#cccccc', linewidth=0.5, zorder=0)
        self._ax.grid(True, alpha=0.2)
        # Re-apply the title (also wiped by clear)
        title = self._title or (f'Phase portrait  '
                                f'(\u03bc={self.system.mu:.3g}, '
                                f'c={self.system.c:.3g})')
        self._ax.set_title(title, fontsize=11)

        # Restore user's view — this is the ONLY limit-setting in redraw
        self._ax.set_xlim(xlim)
        self._ax.set_ylim(ylim)

        for layer in self._layers:
            if layer.visible:
                layer.prepare(self.system, self.mask, self._ax)

        self._drawn = True
        return self

    def refresh_layer(self, name):
        """Re-prepare a single layer by name, leaving everything else intact.

        Much cheaper than a full redraw — the other layers keep their
        artists and the axis limits are untouched.

        Parameters
        ----------
        name : str
            Layer name to refresh.

        Returns
        -------
        self
        """
        if self._ax is None:
            return self
        layer = self.get_layer(name)
        if layer is None:
            raise KeyError(f'Layer {name!r} not found')
        layer.clear()
        if layer.visible:
            layer.prepare(self.system, self.mask, self._ax)
        return self

    # ── display / export ───────────────────────────────────────────────

    def show(self):
        """Display the figure."""
        if not self._drawn:
            self.draw()
        import matplotlib
        if matplotlib.get_backend().lower() != 'agg':
            if self._fig is not None:
                self._fig.show()
            plt.show()
        return self

    def save(self, filename='phase_portrait', dpi=150, close=True):
        """Save the figure.

        Parameters
        ----------
        filename : str
            Path (without extension).  Saved as PDF.
        dpi : int
        close : bool
            Close the figure after saving.

        Returns
        -------
        str
            Full saved path.
        """
        if not self._drawn:
            self.draw()
        path = f'{filename}.pdf'
        self._fig.savefig(path, bbox_inches='tight', dpi=dpi)
        print(f'Saved  {path}')
        if close:
            plt.close(self._fig)
        return path

    def __repr__(self):
        return (f'<PhasePortrait μ={self.system.mu:.3g} c={self.system.c:.3g} '
                f'{len(self._layers)} layers drawn={self._drawn}>')