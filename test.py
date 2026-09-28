"""test.py — Playground for the phase_portrait subproject.

Run sections individually to explore what each component does.

    $ python -c "from test import test_phase; test_phase()"
    $ python -c "from test import test_gallery; test_gallery()"
    $ python -c "from test import test_interactive; test_interactive()"

Or run this file directly (runs all three in sequence with Agg backend):

    $ python test.py
"""

import sys
import os

# ── Ensure the repo root is on the import path ──
# This lets us `from system import System` even when running from anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ────────────────────────────────────────────────────────────────────────────
#  test_phase — basic PhasePortrait with all layer types
# ────────────────────────────────────────────────────────────────────────────

def test_phase():
    """Demonstrate a basic PhasePortrait with every layer type.

    Shows:
        - Creating a System at a given (μ, c)
        - Building layers individually
        - Composing them into a PhasePortrait
        - Toggling layers on/off
        - Adding orbits dynamically
        - Redrawing and saving
    """
    # We'll create the System first.  The two parameters are:
    #   μ (mu) — controls growth rate
    #   c       — controls competition / predation strength
    from system import System
    from phase_portrait import (
        PhasePortrait,
        EquilibriaLayer,
        NullclineLayer,
        QuiverLayer,
        OrbitLayer,
        R2Mask,
    )

    # ── Step 1: create the dynamical system ──
    # Try changing μ and c to see different behaviours!
    s = System(mu=0.3, c=1.5)

    # Print a summary of what equilibria exist at this (μ, c)
    s.classify_all()
    eqs = s.real_equilibria()
    print(f'System at μ={s.mu:.3g}, c={s.c:.3g}')
    for eq in eqs:
        print(f'  {eq.name:>6}  ({eq.x_num:+.4f}, {eq.y_num:+.4f})  →  {eq.type}')
    print()

    # ── Step 2: build layers ──
    # Each layer is independent.  Add or remove them freely.

    # EquilibriaLayer — shows equilibrium points coloured by stability type
    #   green  = stable (node / spiral)
    #   red    = unstable (node / spiral)
    #   orange = saddle
    #   blue   = centre
    #   purple = degenerate
    # Pass branches=['E0', 'E_A⁺'] to show a subset.  None = show all.
    eq_layer = EquilibriaLayer(
        branches=None,           # show all existing equilibria
        show_labels=True,        # annotate with name
    )

    # NullclineLayer — contour lines where dx/dt=0 (blue) and dy/dt=0 (red)
    #   Equilibrium points live at intersections of these curves.
    nc_layer = NullclineLayer(
        grid_res=400,            # higher = smoother contours, slower
    )

    # QuiverLayer — tiny arrows showing direction of the flow
    qv_layer = QuiverLayer(
        grid_res=25,             # arrows per axis (25×25 = 625 arrows)
        scale=None,              # auto-scale, or pass a float to fix arrow length
    )

    # OrbitLayer — integrated trajectories from initial conditions
    # Pass a list of (x, y) starting points.  You can add more later.
    orb_layer = OrbitLayer(
        ics=[
            (0.30, 0.20),        # near E0
            (0.50, 0.50),        # somewhere in the middle
            (-0.10, 1.10),       # near E1
            (1.20, 0.40),        # far right
        ],
        t_span=(0, 20),          # integrate forward 20 time units
        dt=0.01,                 # max step size
        style={'color': '#444444', 'alpha': 0.6, 'linewidth': 0.6},
    )

    # ── Step 3: compose into a PhasePortrait ──
    # The mask controls which rectangle of the plane is visible.
    # R2Mask.from_equilibria() auto-sizes nicely, or pass explicit limits:
    #   mask = R2Mask(-3, 3, -3, 3)
    mask = R2Mask.from_equilibria(eqs, pad_factor=1.5, min_pad=0.5)

    p = (PhasePortrait(s, mask=mask, figsize=(9, 7))
         .add(eq_layer)
         .add(nc_layer)
         .add(qv_layer)
         .add(orb_layer)
         .draw())

    # ── Step 4: toggle layers on/off ──
    # Uncomment to hide the quiver or nullclines:
    # p.toggle('nullclines', visible=False)
    # p.toggle('quiver', visible=False)

    # ── Step 5: add more orbits interactively ──
    # You can add ICs after drawing, then call redraw().
    p.add_orbit(0.00, 0.80)
    p.add_orbit(0.80, 0.80)
    p.add_orbit(0.05, -0.30)
    p.redraw()

    # ── Step 6: inspect what we have ──
    print(f'Layers: {[l.name for l in p.layers]}')
    print(f'Mask:   {mask}')
    print(f'Total layer artists: {sum(len(l._artists) for l in p.layers)}')

    # ── Step 7: show or save ──
    p.show()
    p.save('/Users/sam/uni/group2/temp.pdf', dpi=150)

    plt.close('all')
    print('✅  test_phase done\n')


# ────────────────────────────────────────────────────────────────────────────
#  test_gallery — grid of local portraits
# ────────────────────────────────────────────────────────────────────────────

def test_gallery():
    """Demonstrate the local-portrait gallery.

    ``gallery_local()`` creates one zoomed-in PhasePortrait per equilibrium,
    each with a circular net of orbits around it.  Great for understanding
    local stability structure at a glance.
    """
    from system import System
    from phase_portrait import gallery_local

    # ── Try different (μ, c) pairs ──────────────────────────────────────
    #   (0.3, 1.5)  — standard, 6 equilibria
    #   (0.1, 0.5)  — fewer equilibria, simpler
    #   (0.5, 2.0)  — different balance
    s = System(mu=0.3, c=1.5)

    # gallery_local returns (figure, list_of_axes, list_of_portraits)
    # Pass branches=['E0', 'E1'] to show only those two.
    fig, axes, portraits = gallery_local(
        s,
        branches=None,       # all existing equilibria
        n_cols=3,            # grid columns
    )

    print(f'Gallery: {len(portraits)} portraits')
    for i, p in enumerate(portraits):
        print(f'  [{i}] {p}')

    fig.savefig('/tmp/test_gallery.pdf', bbox_inches='tight')
    plt.close('all')
    print('✅  test_gallery done\n')


# ────────────────────────────────────────────────────────────────────────────
#  test_interactive — Interactor demo (non-blocking)
# ────────────────────────────────────────────────────────────────────────────

def test_interactive():
    """Demonstrate the Interactor widgets.

    Opens a window with:
        - Check-boxes to toggle layers on/off
        - Click anywhere to add an orbit from that point
        - A text box for manual (x, y) entry
        - Clear orbits / Reset view buttons

    Close the window when done.
    """
    from system import System
    from phase_portrait import (
        PhasePortrait,
        EquilibriaLayer,
        NullclineLayer,
        QuiverLayer,
        OrbitLayer,
        Interactor,
    )

    s = System(mu=0.3, c=1.5)

    p = (PhasePortrait(s, figsize=(10, 8))
         .add(EquilibriaLayer())
         .add(NullclineLayer(grid_res=300))
         .add(QuiverLayer(grid_res=10))
         .add(OrbitLayer(ics=[(0.5, 0.5), (-0.3, 0.8)],
                         t_span=(0, 25), dt=0.01,
                         style={'color': '#333', 'alpha': 0.6, 'lw': 0.7}))
         .draw())

    # ── Wire up the interactor ──
    interactor = Interactor(p)

    # Add all interactive controls
    interactor \
        .add_toggles() \
        .add_click_orbit(callback=lambda x, y: print(f'  ➕ orbit from ({x:.3f}, {y:.3f})')) \
        .add_text_orbit(initial='0, 0.5') \
        .add_clear_button() \
        .add_reset_button() \
        .add_manual_refresh() \
        .add_scroll_zoom() \
        .add_keyboard_shortcuts() \
        .add_help_label()

    print('Interactive window open — close it when done.')
    print('Tip: RIGHT-click to add orbits, scroll to zoom, toggles on the right.')
    print('Keys: c=clear, r=redraw, v=reset view, h=help')
    p.show()
    plt.close('all')
    print('✅  test_interactive done\n')


# ────────────────────────────────────────────────────────────────────────────
#  Run all three when executed directly
# ────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Use Agg (non-interactive) backend for batch runs
    matplotlib.use('Agg')
    print('═' * 50)
    print('  Phase Portrait Subproject — Test Suite')
    print('═' * 50)
    #test_phase()
    #test_gallery()
    test_interactive()
    print('═' * 50)
    print('  All tests complete.  Check /tmp/*.pdf for output.')
    print('═' * 50)