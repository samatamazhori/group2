**Phase Portrait Subproject — `phase_portrait/` (built)**

### Goal
Composable, native-Matplotlib phase portraits for the 2-D system in `system.py`.
Layers are first-class and easy to extend. Interactivity is isolated. Masks live at the top.

### Binding to `system.py`
`System(mu, c)` — construct with parameters  
`.classify_all()` — solve symbolic branches, classify stability  
`.real_equilibria()` → list of `Equilibrium` objects (`.name`, `.x_num`, `.y_num`, `.type`)  
`.trajectory(x0, y0, t_span=(0,20), dt=0.01)` → (t, x, y) arrays  
Types: stable/unstable node, stable/unstable spiral, saddle, centre, degenerate  
Branches: E0, E_A⁺, E_A⁻ (Branch A, y=0) and E1, E_B⁺, E_B⁻ (Branch B, y≠0)

### Files (5 files, 1062 lines)

| File | Status | Role |
|------|--------|------|
| `masks.py` | ✅ | `Mask` ABC + `R2Mask` (rect, auto-sized from eqs) + `HalfPlaneMask`/`SimplexMask` stubs |
| `layers.py` | ✅ | `Layer` ABC + `EquilibriaLayer` (per-type styled, branch filter), `NullclineLayer`, `QuiverLayer`, `OrbitLayer` (pre-set ICs + `.add_ic()`) |
| `portrait.py` | ✅ | `PhasePortrait(system, ax, mask)` fluent builder: `.add()`, `.draw()`, `.redraw()`, `.toggle()`, `.save()` |
| `interactive.py` | ✅ | `Interactor(portrait)` — `add_toggles()`, `add_click_orbit()`, `add_text_orbit()`, `add_clear_button()`, `add_reset_button()` |
| `gallery.py` | ✅ | `gallery_local(system, branches)` — grid of local portraits, one per eq, with circular IC net + auto zoom |

### Design rules (enforced)
- Native Matplotlib artists only (`scatter`, `quiver`, `contour`, `plot`, `annotate`).
- Mix any subset of the 6 equilibria via `branches=[...]`.
- Static path works with zero interactive imports.
- No parameter animation (out of scope).
- Each file is focused; no god objects.

### Usage
```python
from system import System
from phase_portrait import PhasePortrait, EquilibriaLayer, NullclineLayer, QuiverLayer, OrbitLayer

s = System(mu=0.3, c=1.5)
p = (PhasePortrait(s)
     .add(EquilibriaLayer())
     .add(NullclineLayer())
     .add(QuiverLayer())
     .add(OrbitLayer(ics=[(0.5, 0.5)]))
     .draw()
     .show())
```