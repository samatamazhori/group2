# Project: Bifurcation Catalog of a 2D Dynamical System

## Mission

Build a **Python framework** (numpy, matplotlib, sympy, scipy) to **model and stratify the μ–c parameter plane** of a 2D dynamical system. The eight curves C1–C8 (implicit equations) partition the (μ, c) plane into sign-based regions with distinct bifurcation behaviour. The ultimate goal is to **catalogue the full bifurcation behaviour per region** — which equilibrium branches exist, their stability, and how they collide.

---

## System

**State variables:** x, y  
**Parameters:** μ, c

```
dx/dt = x[(c-4)y - 5μ + 4μ(x+y) + x(x+3y-3) + 2]
dy/dt = y[c(y-1) + x(x+3y-1)]
```

At most 6 equilibrium points, split across branch A (y=0, 3 equilibria) and branch B (y≠0, up to 3 equilibria).

---

## The Eight Curves

All curves registered in `data.py` as `Curve` objects with sympy expressions and plotting metadata.

| ID | Expression (F=0) | Role | Label | Colour | Style |
|----|-|-|-|-|-|
| C1 | `4 - 12c + 9c² - 44μ + 50cμ - 24c²μ + 121μ² - 56cμ² + 16c²μ²` | Δ of Q (discriminant) | **Δ₁** | `#1f77b4` blue | solid |
| C2 | `-2c + c² - 2μ + 10cμ + 5μ² - 16cμ²` | collision of branch A, B | **R** | `#d62728` red | solid |
| C3 | `μ - ²/₅` | collision of branch A | **μ = ²/₅** | `#2ca02c` green | solid |
| C4 | `c - μ - 2` | collision of B with (0,1) | **c = μ + 2** | `#9467bd` purple | solid |
| C5 | `c` | denominator singularity, collision branch B | **c = 0** | `#8c564b` brown | solid |
| C6 | `μ - ¼` | x² degeneracy | **μ = ¼** | `#ff7f0e` orange | solid |
| C7 | `μ - 1` | B-solution degeneracy | **μ = 1** | `#e377c2` pink | solid |
| C8 | S (Hopf curve) | **S** | `#17becf` teal | dashed |

C1–C7 determine region signatures (7 tokens). C8 is the Hopf curve, dashed, plotted for reference but excluded from sign-based region definitions.

---

## Data Model (`data_model.py`)

Flat file, no subpackages. All classes inherit `BaseObject` which provides `__getattr__` fallback to `self.metadata` (including one-level nested lookup).

```
BaseObject  (metadata dict + __getattr__ fallback)
 ├── Curve          — F(μ,c)=0 with sympy expr + auto-lambdified numpy callable
 ├── Constraint     — inequality bound (μ or c, lower/upper, strictness)
 ├── Point          — (μ, c) coordinate
 ├── InfinityPoint  — curve × box-edge crossing (lazy NaN-safe coord resolution)
 ├── Intersection   — set of curves × list of points they share
 ├── Arc            — curve segment between two points (optional constraint)
 ├── Region         — sign-defined region with contains(μ, c) method
 └── Boundary       — arc separating two regions
```

### InfinityPoint

Separate class (does NOT extend Point). Stores `curve`, `edge` (left/right/top/bottom), `index`, `box`. Lazily computes `mu, c` via `box_utils.edge_intersections`. `is_at_infinity` always True. Box setter triggers recompute. Returns NaN for non-crossing edges. 32 dedicated tests in `test-data-model.py`.

---

## Box Domain

```python
BOX = (MU_MIN, MU_MAX, C_MIN, C_MAX) = (-2.5, 2.0, -0.75, 7.5)
```

---

## Intersections

15 intersections between curve pairs. Naming: `I{C1#}{C2#}` — e.g. C1∩C2 → I12.

| Intersection | Curves | # Points | Notes |
|-------------|--------|:--------:|-------|
| I12 | C1, C2 | 2 | (0.191, 0.896), (0.967, 6.896) |
| I15 | C1, C5 | 1 | μ = 2/11, c = 0 |
| I16 | C1, C6 | 1 | μ = ¼, c = ⅜ |
| I23 | C2, C3 | 2 | c = 0 and c = 14/25 |
| I24 | C2, C4 | 3 | μ=0,c=2; (3±√6)/2 in μ |
| I25 | C2, C5 | 2 | (0,0), (²/₅,0) |
| I26 | C2, C6 | 2 | μ=¼, c=-¼ and c=¾ |
| I27 | C2, C7 | 1 | μ=1, c=? |
| I34 | C3, C4 | 1 | μ=²/₅, c=12/5 |
| I35 | C3, C5 | 1 | μ=²/₅, c=0 |
| I45 | C4, C5 | 1 | μ=-2, c=0 |
| I46 | C4, C6 | 1 | μ=¼, c=9/4 |
| I47 | C4, C7 | 1 | μ=1, c=3 |
| I56 | C5, C6 | 1 | μ=¼, c=0 |
| I57 | C5, C7 | 1 | μ=1, c=0 |

**ALL_POINTS** list in `data.py` contains all intersection points.

**INFINITY_POINTS** list in `data.py` contains 16 InfinityPoints — each curve's box-edge crossings:
- C1: top ×2
- C2: left, right, top ×2
- C3: bottom, top
- C4: left, right
- C5: left, right
- C6: bottom, top
- C7: bottom, top

---

## Files

| File | Purpose |
|------|---------|
| `context.md` | This file — mission, curves, decisions, progress |
| `notes.md` | Handover notes for next developer / LLM session |
| `data_model.py` | Core framework (~500 lines) — all classes, 108+ tests pass |
| `data.py` | Registered curves, intersections, infinity points |
| `view.py` | `View` dataclass — 10 named viewport presets (`DEFAULT_VIEWS`) |
| `plane.py` | `PlaneBuilder` — layered plotter with unified `add()`, `show()`, `save()` |
| `quick-plot.py` | Standalone CLI plot (superseded by `PlaneBuilder`) |
| `region_rec.py` | CLI click tool showing signs at mouse cursor |
| `sign_fill.py` | CLI: `python sign_fill.py "+ + + - - - -"` fills matching region |
| `end-to-end.py` | Full (μ,c) → system → classify → plot → dump |
| `explorer.py` | Arc/region workspace skeleton |
| `system.py` | `System(mu,c)` — equilibrium solving (6 branches), classification, trajectories |
| `box_utils.py` | Edge-crossing detection via linear interpolation |
| `interactive_plane.py` | Interactive μ–c viewer with sidebar, zoom, pan, inspect |
| `scratchpad.py` | Region `R_scratch` defined and verified (test in/out) |
| `view_explorer.py` | Batch-plot all 10 views via PlaneBuilder |
| `test-data-model.py` | 108+ smoke tests for data model classes |

---

## Plotting System

### PlaneBuilder (`plane.py`)

- Unified `.add()` accepts any `Curve | Point | InfinityPoint | Region | Boundary | Intersection` (or lists thereof)
- Auto z-order: regions back → curves middle → points front
- Adaptive grid resolution (finer for zooms, coarser for large views)
- Legend deduplication (one per unique curve, one per point group, one per region)
- `.show()`, `.save(name=None, dpi=200)`, `.clear()`, `.remove()`
- `.snapshot()` returns metadata dict of everything plotted
- Auto-stem filename from layer types
- View switching between `show()`/`save()` calls

### File naming

```
figs/{view.name}/{stem}_{view.suffix}.png
```

Example: `figs/r1_detail/curves+points_mu-0.5-1.5_c-0.5-3.0.png`

### View presets (`view.py`)

10 presets in `DEFAULT_VIEWS`:
- `global_view` — full domain
- `r1_detail` — R1 close-up
- `hopf_area` — Hopf/C8 region
- `c2_bottom` — C2 near bottom edge
- `r3_detail` — R3a/R3b area
- `top_right` — top-right quadrant
- `right_panel` — μ∈[-4,0], c∈[-3,7]
- `bottom_center` — μ∈[0,0.5], c∈[-1,3]
- `left_panel` — μ∈[0.3,2], c∈[-1,4]
- `top_panel` — μ∈[0.5,1.3], c∈[5,15]

---

## InteractivePlane (`interactive_plane.py`)

Self-contained interactive μ–c viewer (NOT subclassing PlaneBuilder).

### Layout
- Main plot axes + embedded sidebar (inside figure)
- Sidebar sections: curve checkboxes → point checkboxes → info text panel
- Control hints at figure bottom: "Scroll: zoom | Drag: pan | R: reset"

### Interactions
| Action | Behaviour |
|--------|-----------|
| Scroll up/down | Zoom in/out centered on cursor position |
| Click + drag (≥3px) | Pan the view |
| Left-click (<3px drag) | Inspect point → show μ,c + signs + equilibria |
| `r` key | Recompute contours at current view (reset) |
| Curve checkboxes | Toggle C1–C8 visibility |
| Point checkboxes | Toggle intersection / infinity point groups |

### Click inspection output

Three sections, auto-updated on each click:

1. **Coordinates:** `μ = 0.5    c = 2`
2. **Signs** (one per curve C1–C8): `C1  +`, `C2  -` — sign flips from previous click marked with `◆`
3. **Equilibria** (via `System` class): `E0    stable node      (0, 0)` — type changes from previous click shown as `← was <oldtype>`

### State tracking for change detection
- `_last_signature` — tuple of 8 signs `('+', '-', ...)`
- `_last_eq_types` — dict `{eq_name: type_string}`

Sign flips get `◆` marker; equilibrium type changes get `← was saddle` annotation.

### Key technical details
- Contour plotting via `ax.contour` with adaptive grid (200–600 pts)
- Visibility toggles via `set_visible()` on existing collections (no redraw)
- Scroll zoom preserves cursor-relative position
- 3-pixel drag threshold distinguishes click from pan
- `_apply_view()` called BEFORE `_build_curves()` (bug fix: view limits must be set before contour computation)
- `System` integration via `from system import System`

---

## Progress

### ✅ Done
- [x] Data model (`BaseObject`, `Curve`, `Constraint`, `Point`, `InfinityPoint`, `Intersection`, `Arc`, `Region`, `Boundary`) in `data_model.py`
- [x] 108+ smoke tests passing (`test-data-model.py`)
- [x] All 8 curves registered in `data.py` with sympy, metadata, colours
- [x] All 15 intersections registered with named points
- [x] 16 InfinityPoints registered (box-edge crossings)
- [x] `System(mu,c)` class with equilibrium solving, stability classification, trajectories
- [x] `PlaneBuilder` with unified `.add()`, z-ordering, adaptive grid, legend dedup, view switching
- [x] 10 view presets in `view.py` (`DEFAULT_VIEWS`)
- [x] `InteractivePlane` with zoom/pan/inspect/toggle/System integration
- [x] Sign change detection (`◆` marker)
- [x] Equilibrium type change detection (`← was <oldtype>` marker)
- [x] View order bug fixed (`_apply_view()` before `_build_*()`)
- [x] All typos fixed in `interactive_plane.py`
- [x] Scratchpad region `R_scratch` defined, tested (in/out verified)
- [x] `view_explorer.py` batch-plots all views
- [x] `sign_fill.py` CLI tool
- [x] `region_rec.py` interactive sign reconnaissance
- [x] `end-to-end.py` per-point workflow

### 🔲 Next
1. **Arc tracing** — For each curve C1–C7, collect intersection + infinity points, sort along curve, create `Arc` objects with bounding `Constraint`s
2. **Region definitions** — From `data.md` signatures (R1, R2, R3a, R3b), build closed boundary loops, assign sign signatures, test points, verify `contains()`
3. **Migrate `R_scratch`** from `scratchpad.py` to `data.py` when region definitions are finalised
4. **Bifurcation catalog** — Document equilibrium behaviour per region (branches, stability, collisions)
5. **Performance** — Cache contour computations, grid reuse across zooms

---

## Key Decisions

| Area | Decision |
|------|----------|
| **Metadata fallback** | `__getattr__` only (no `__setattr__` override). One-level nested dict lookup supported. |
| **Curve evaluation** | Auto-lambdified from sympy; manual override via `numpy_func` setter |
| **Curves** | 8 total (C1–C7 for sign regions, C8 as dashed Hopf reference) |
| **Signatures** | 7 tokens (C1–C7). C8 excluded from region definitions |
| **InfinityPoint** | Separate class (not Point). Lazy eval, NaN fallback, `is_at_infinity` always True |
| **Interactive viewer** | Separate class (not subclassing PlaneBuilder). Uses matplotlib widgets (`CheckButtons`). |
| **View bug fix** | `_apply_view()` must run before `_build_curves()` — contour grid uses axis limits |
| **Change detection** | `_last_signature` (sign flips → `◆`) + `_last_eq_types` (type changes → `← was <oldtype>`) |
| **Python** | Target: 3.12.3 (homebrew). Working: 3.11.4 (anaconda, has all packages) |