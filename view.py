"""
view.py — Named viewports (bounding boxes) on the μ–c parameter plane.

A ``View`` is just a rectangle with a name and description.  You can
switch views on the fly to zoom into interesting regions or pan around.

Usage
-----
    from view import View

    v = View('my_zoom', mu_min=-1, mu_max=1, c_min=0, c_max=3,
             description='Zoom into Hopf region')
    v.box          # → (-1, 1, 0, 3)
    v.suffix       # → "mu-1-1_c0-3"   (for filenames)

    # Presets
    View.global_view()
    View.r1_detail()
"""

from dataclasses import dataclass


@dataclass
class View:
    """A named rectangular viewport on the μ–c plane.

    Attributes
    ----------
    name : str
        Short identifier (used as directory name in ``figs/``).
    mu_min, mu_max : float
    c_min, c_max : float
    description : str
        Human-readable description (used as plot title).
    """

    name: str
    mu_min: float
    mu_max: float
    c_min: float
    c_max: float
    description: str = ''

    # ── box tuple (compatible with box_utils / edge_intersections) ──────

    @property
    def box(self):
        """Tuple ``(mu_min, mu_max, c_min, c_max)`` for passing to
        :func:`box_utils.edge_intersections` or other API that expects
        the raw 4-tuple."""
        return (self.mu_min, self.mu_max, self.c_min, self.c_max)

    # ── compact filename suffix ─────────────────────────────────────────

    @property
    def suffix(self):
        """Compact string like ``mu-2.5-2.0_c-0.75-7.5`` suitable for
        appending to figure filenames."""
        return f'mu{self.mu_min}-{self.mu_max}_c{self.c_min}-{self.c_max}'

    # ── common view presets (class methods for discovery) ───────────────

    @classmethod
    def global_view(cls):
        """The full domain: μ ∈ [-2.5, 2.0], c ∈ [-0.75, 7.5]."""
        return cls(
            'global', -2.5, 2.0, -0.75, 7.5,
            description='Full μ–c plane',
        )

    @classmethod
    def r1_detail(cls):
        """Zoom into the region around R1: μ ∈ [-0.5, 1.5], c ∈ [-0.5, 3.0]."""
        return cls(
            'r1_detail', -0.5, 1.5, -0.5, 3.0,
            description='R1 region close-up',
        )

    @classmethod
    def hopf_area(cls):
        """Region containing the Hopf curve C8: μ ∈ [0.0, 1.0], c ∈ [2.0, 5.0]."""
        return cls(
            'hopf_area', 0.0, 1.0, 2.0, 5.0,
            description='Hopf / C8 region',
        )

    @classmethod
    def c2_bottom(cls):
        """Bottom-left corner where C2 enters: μ ∈ [-0.1, 0.5], c ∈ [-0.75, 0.5]."""
        return cls(
            'c2_bottom', -0.1, 0.5, -0.75, 0.5,
            description='C2 near bottom edge',
        )

    @classmethod
    def r3_detail(cls):
        """R3a / R3b area: μ ∈ [0.0, 2.0], c ∈ [-0.75, 2.0]."""
        return cls(
            'r3_detail', 0.0, 2.0, -0.75, 2.0,
            description='R3a / R3b region',
        )

    @classmethod
    def top_right(cls):
        """Top-right corner: μ ∈ [0.0, 2.0], c ∈ [3.0, 7.5]."""
        return cls(
            'top_right', 0.0, 2.0, 3.0, 7.5,
            description='Top-right quadrant',
        )

    # ── new views ───────────────────────────────────────────────────────

    @classmethod
    def right_panel(cls):
        """Left half of the plane: μ ∈ [-4, 0], c ∈ [-3, 7]."""
        return cls(
            'right_panel', -4.0, 0.0, -3.0, 7.0,
            description='Left half (μ ≤ 0)',
        )

    @classmethod
    def bottom_center(cls):
        """Bottom-centre zoom: μ ∈ [0, 0.5], c ∈ [-1, 3]."""
        return cls(
            'bottom_center', 0.0, 0.5, -1.0, 3.0,
            description='Bottom centre (μ ∈ [0, 0.5])',
        )

    @classmethod
    def left_panel(cls):
        """Right-centre panel: μ ∈ [0.3, 2], c ∈ [-1, 4]."""
        return cls(
            'left_panel', 0.3, 2.0, -1.0, 4.0,
            description='Centre-right (μ ≥ 0.3)',
        )

    @classmethod
    def top_panel(cls):
        """Top strip: μ ∈ [0.5, 1.3], c ∈ [5, 15]."""
        return cls(
            'top_panel', 0.5, 1.3, 5.0, 15.0,
            description='Top strip (c ∈ [5, 15])',
        )

    # ── display ─────────────────────────────────────────────────────────

    def __repr__(self):
        return (f'<View "{self.name}": '
                f'μ∈[{self.mu_min}, {self.mu_max}], '
                f'c∈[{self.c_min}, {self.c_max}]>')

    def __str__(self):
        return (f'View: {self.description or self.name}  '
                f'(μ {self.mu_min} to {self.mu_max}, '
                f'c {self.c_min} to {self.c_max})')


# ── convenience: default views lookup ─────────────────────────────────────

DEFAULT_VIEWS = {
    'global':       View.global_view(),
    'r1_detail':    View.r1_detail(),
    'hopf_area':    View.hopf_area(),
    'c2_bottom':    View.c2_bottom(),
    'r3_detail':    View.r3_detail(),
    'top_right':    View.top_right(),
    'right_panel':  View.right_panel(),
    'bottom_center': View.bottom_center(),
    'left_panel':   View.left_panel(),
    'top_panel':    View.top_panel(),
}