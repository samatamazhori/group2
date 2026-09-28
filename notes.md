# Notes for Next Developer / LLM Session

## What Has Been Built

A Python framework to catalogue the μ–c parameter plane of a 2D dynamical system (dx/dt, dy/dt). The plane is partitioned by 8 bifurcation curves. The framework can plot, explore interactively, analyse equilibrium behaviour, and will eventually define sign-based regions with full bifurcation catalogues.

---

## Data Model (`data_model.py`)

Every class inherits `BaseObject` which gives `__getattr__` fallback to `self.metadata` (including one level of nested dict lookup). The hierarchy:

```
BaseObject
 ├── Curve          — implicit F(μ,c)=0, sympy + auto-lambdified numpy
 ├── Constraint     — inequality bound
 ├── Point          — (μ, c) coordinate
 ├── InfinityPoint  — curve×box-edge crossing (lazy, NaN-safe)
 ├── Intersection   — curve set × point list
 ├── Arc            — curve segment between two points
 ├── Region         — sign-defined, contains(μ,c)
 └── Boundary       — arc between two regions
```

**InfinityPoint** is NOT a subclass of Point. It stores `curve`, `edge` (left/right/top/bottom), `index`, `box`. Coordinates resolved lazily via `box_utils.edge_intersections`. `is_at_infinity` always True. Box setter triggers recompute. NaN if edge doesn't cross.

---

## Registered Data (`data.py`)

- **8 curves** C1–C8 (see `context.md` for equations, colours, labels)
- **15 intersections** I12–I57, all with named Point objects
- **16 InfinityPoints** — each curve's box-edge crossings for the global BOX
- **ALL_POINTS** and **INFINITY_POINTS** lists exported

---

## Plotting System

### View (`view.py`)
Dataclass with `.box` property (4-tuple) and `.suffix` (filename-safe string). 10 presets in `DEFAULT_VIEWS`.

### PlaneBuilder (`plane.py`)
Unified `.add()` accepts any drawable. Auto z-order, adaptive grid, legend dedup, `.show()`, `.save()`, `.clear()`, `.remove()`. View switching between renders. Filenames: `figs/{view.name}/{stem}_{view.suffix}.png`.

### InteractivePlane (`interactive_plane.py`)
Self-contained interactive viewer (separate class). Features:
- Embedded sidebar with curve/point toggles (`matplotlib.widgets.CheckButtons`)
- Scroll-zoom at cursor, drag-pan (3px threshold), left-click inspect
- Click inspection: μ,c coordinates → curve signs with `◆` on flips → System equilibrium summary with `← was <oldtype>` on type changes
- `r` key recomputes contours
- `_format_inspect()` composes `_format_signs()` + `_format_equilibria()` — both private, decoupled
- State tracking: `_last_signature` (tuple) + `_last_eq_types` (dict)

**Critical:** `_apply_view()` must be called BEFORE `_build_curves()` — contour grid depends on axis limits. This was a bug that has been fixed.

---

## System Analysis (`system.py`)

`System(mu, c)` — construct with (μ,c), then:
- `classify_all()` — solve symbolic branches, evaluate numerically, classify type
- `real_equilibria()` — list of `Equilibrium` objects with `.name`, `.x_num`, `.y_num`, `.type`
- Types: `stable/unstable node`, `stable/unstable focus`, `saddle`, `stable/unstable spiral`, `center`, `degenerate (zero det)`, or `?`
- Plotting methods for phase portraits

---

## Scratchpad & Exploration

- `scratchpad.py` — `R_scratch` region defined (4 arcs), with `contains()` verification
- `view_explorer.py` — batch-plot all 10 views
- `sign_fill.py` — CLI fill by sign signature: `python sign_fill.py "+ + + - - - -"`
- `region_rec.py` — interactive sign reconnaissance
- `end-to-end.py` — full per-point workflow

---

## What Awaits (High Priority)

### 1. Arc Tracing (critical path for region definitions)

For each curve C1–C7, collect all relevant points along it (intersection + infinity points on that curve). Sort them along the curve (by arclength or by parameter). Create `Arc(curve, start_point, end_point, constraint=...)` objects for each segment. Each arc needs a `Constraint` bounding its valid μ/c range. This is the foundation for building region boundaries.

The `explorer.py` file is a skeleton workspace for this.

### 2. Region Definitions

From `data.md` — 4 region signatures are documented:

| Region | Sign Signature (C1 C2 C3 C4 C5 C6 C7) | Test Point |
|--------|---------------------------------------|------------|
| R1 | `+ + + + - + -` | |
| R2 | `+ + + - - - -` | |
| R3a | `- + + - - - +` | |
| R3b | `+ - + - - - -` | |

For each region, build a closed loop of `Arc` objects (traced in step 1), create a `Region` with sign signature and test point, verify `contains()` returns correct in/out for known points.

### 3. Migrate `R_scratch`

The scratchpad region `R_scratch` (signature `+ + – – – – –`) should be moved into `data.py` as a proper Region once the arc-tracing framework is finalised. Currently it lives in `scratchpad.py` with hardcoded vertex interpolation.

### 4. Bifurcation Catalog

Once regions are defined, run `System(mu, c).classify_all()` at a test point in each region. Document:
- Which equilibrium branches exist
- Their stability type
- Whether they are real or complex
- Any collisions/degeneracies at the boundaries

Save as structured data (perhaps a dict in `data.py` or a separate `catalog.py`).

---

## Key Constraints & Gotchas

- **Python interpreter:** Target is 3.12.3 (homebrew), but the only interpreter with all packages is `/opt/homebrew/anaconda3/bin/python3` (3.11.4). Use `python3` for actual runs, `python3.12 -c "import py_compile..."` for syntax checks.
- **Signature tokens:** 7 tokens (C1–C7). C8 (Hopf, dashed) is excluded from region definitions.
- **Box domain:** μ∈[-2.5, 2.0], c∈[-0.75, 7.5]. Some features (C8) extend beyond.
- **InfinityPoint evaluation** requires `box_utils.edge_intersections()` which is pure linear interpolation (1000 samples + refinement).
- **InteractivePlane** is a standalone class — do NOT modify PlaneBuilder to add interactivity.
- **Matplotlib widgets** only (CheckButtons). No tkinter, no custom GUI.
- **No right-click** context menus. Left-click only for inspection.
- **Contours via ax.contour** with adaptive grid (200–600 pts based on view span).
- **Change detection** uses `_last_signature` (tuple of 8 chars) and `_last_eq_types` (dict). Both reset when `recompute()` is called? (Currently no reset — verify behaviour).

---

## File Inventory (Oct 2024)

| File | Lines | Purpose |
|------|-------|---------|
| `data_model.py` | ~500 | Core data model classes + tests |
| `data.py` | ~310 | Registered curves, intersections, infinity points |
| `view.py` | ~160 | View dataclass + 10 presets |
| `plane.py` | ~420 | PlaneBuilder unified plotter |
| `system.py` | ~620 | System class for equilibrium analysis |
| `interactive_plane.py` | ~560 | Interactive viewer with sidebar |
| `scratchpad.py` | ~260 | Scratch region experiments |
| `box_utils.py` | ~100 | Edge-crossing detection |
| `quick-plot.py` | ~120 | CLI quick plot |
| `region_rec.py` | ~160 | Click sign tool |
| `sign_fill.py` | ~200 | CLI sign fill |
| `end-to-end.py` | ~200 | Full workflow per point |
| `explorer.py` | ~340 | Arc/region workspace (skeleton) |
| `view_explorer.py` | ~40 | Batch-view plotter |
| `test-data-model.py` | ~420 | Data model tests |
| `context.md` | ~240 | Project documentation |
| `notes.md` | This file | Handover notes |
| `data.md` | ~70 | Mathematica notes (equations, intersections) |
| `plan-0.md` | ~40 | Original architecture plan |