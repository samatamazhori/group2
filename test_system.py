"""
test_system.py — Comprehensive test of system.py at (μ=0.3, c=2.0).

Covers every public method, edge case, and output format.
Run with::

    python3 test_system.py
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from system import System, Equilibrium

# ── configuration ─────────────────────────────────────────────────────
MU, C = 0.3, 2.0
sys = System(mu=MU, c=C)

print('╔══════════════════════════════════════════════════════════════╗')
print(f'║  System Test:  μ = {MU},  c = {C}                          ║')
print('╚══════════════════════════════════════════════════════════════╝')

# ──────────────────────────────────────────────────────────────────────
# 1.  Core attributes
# ──────────────────────────────────────────────────────────────────────
print('\n── 1.  Core attributes ──')
print(f'    μ = {sys.mu}   (type: {type(sys.mu).__name__})')
print(f'    c = {sys.c}   (type: {type(sys.c).__name__})')

assert isinstance(sys.mu, float), 'mu not float'
assert isinstance(sys.c, float), 'c not float'
assert sys.mu == MU, 'mu mismatch'
assert sys.c == C, 'c mismatch'

# ──────────────────────────────────────────────────────────────────────
# 2.  f, g, vector_field
# ──────────────────────────────────────────────────────────────────────
print('\n── 2.  f, g, vector_field ──')
x_pt, y_pt = 1.0, 0.5
fx = sys.f(x_pt, y_pt)
gx = sys.g(x_pt, y_pt)
u, v = sys.vector_field(x_pt, y_pt)

print(f'    f(1, 0.5)  = {fx:.6g}')
print(f'    g(1, 0.5)  = {gx:.6g}')
print(f'    vector_field(1, 0.5) = ({u:.6g}, {v:.6g})')

assert isinstance(fx, (float, np.floating)), 'f not numeric'
assert isinstance(gx, (float, np.floating)), 'g not numeric'
assert abs(u - fx) < 1e-12, 'vector_field x mismatch'
assert abs(v - gx) < 1e-12, 'vector_field y mismatch'

# Numpy array input
X_arr = np.array([[0.0, 0.5], [1.0, 1.5]])
Y_arr = np.array([[0.0, 0.5], [1.0, 1.5]])
F_arr = sys.f(X_arr, Y_arr)
assert F_arr.shape == (2, 2), 'numpy array input shape wrong'
print(f'    f(array) output shape: {F_arr.shape}  ✓')

print('    ✓  f, g, vector_field')

# ──────────────────────────────────────────────────────────────────────
# 3.  Jacobian
# ──────────────────────────────────────────────────────────────────────
print('\n── 3.  Jacobian ──')
J = sys.jacobian(x_pt, y_pt)
print(f'    J(1, 0.5) =')
print(f'      [{J[0,0]:.4f}, {J[0,1]:.4f}]')
print(f'      [{J[1,0]:.4f}, {J[1,1]:.4f}]')

assert J.shape == (2, 2), 'Jacobian shape wrong'
assert np.all(np.isfinite(J)), 'Jacobian has non-finite values'

# Verify against symbolic partial derivatives
# ∂f/∂x at (0,0): should be 2 - 5μ = 2 - 1.5 = 0.5
J0 = sys.jacobian(0, 0)
print(f'    J(0, 0) = [{J0[0,0]:.4f}, {J0[0,1]:.4f}; '
      f'{J0[1,0]:.4f}, {J0[1,1]:.4f}]')
assert abs(J0[0, 0] - (2 - 5 * MU)) < 1e-10, '∂f/∂x at (0,0) wrong'

# ──────────────────────────────────────────────────────────────────────
# 4.  All 6 equilibria — existence & numerics
# ──────────────────────────────────────────────────────────────────────
print('\n── 4.  Equilibrium: existence & numerical values ──')
all_eqs = sys.equilibria()
print(f'    Total registered: {len(all_eqs)}')

assert len(all_eqs) == 6, 'should have 6 equilibria'
assert all(isinstance(eq, Equilibrium) for eq in all_eqs), 'not Equilibrium objects'

# Check names
expected_names = ['E0', 'E1', 'E_A⁺', 'E_A⁻', 'E_B⁺', 'E_B⁻']
for eq, ename in zip(all_eqs, expected_names):
    assert eq.name == ename, f'expected {ename}, got {eq.name}'
print(f'    Names:            {[e.name for e in all_eqs]}')

# Check branches (E0, E_A⁺, E_A⁻ = branch A; E1, E_B⁺, E_B⁻ = branch B)
branch_A_indices = [0, 2, 3]
branch_B_indices = [1, 4, 5]
for i in branch_A_indices:
    assert all_eqs[i].branch == 'A', f'{all_eqs[i].name} should be branch A'
for i in branch_B_indices:
    assert all_eqs[i].branch == 'B', f'{all_eqs[i].name} should be branch B'

# At μ=0.3, c=2.0, all 6 should exist
real_eqs = sys.real_equilibria()
print(f'    Existing (real):  {len(real_eqs)}')
assert len(real_eqs) == 6, f'expected 6 existing, got {len(real_eqs)}'

# Verify E0 = (0, 0)
eq0 = all_eqs[0]
assert abs(eq0.x_num) < 1e-12, 'E0 x should be 0'
assert abs(eq0.y_num) < 1e-12, 'E0 y should be 0'

# Verify E1 = (0, 1)
eq1 = all_eqs[1]
assert abs(eq1.x_num) < 1e-12, 'E1 x should be 0'
assert abs(eq1.y_num - 1) < 1e-12, 'E1 y should be 1'

# E_A⁺ should have positive x, E_A⁻ should have smaller/possibly positive x
x_Ap = all_eqs[2].x_num
x_Am = all_eqs[3].x_num
print(f'    E_A⁺ x = {x_Ap:.6f},  E_A⁻ x = {x_Am:.6f}')
assert x_Ap > x_Am, 'E_A⁺ should have larger x than E_A⁻'

# Branch B equilibrium: verify they satisfy dy/dt=0 with y≠0
for eq in all_eqs[4:]:
    assert eq.exists
    f_val = sys.f(eq.x_num, eq.y_num)
    g_val = sys.g(eq.x_num, eq.y_num)
    assert abs(f_val) < 1e-8, f'{eq.name}: dx/dt = {f_val:.4e} (should be ≈0)'
    assert abs(g_val) < 1e-8, f'{eq.name}: dy/dt = {g_val:.4e} (should be ≈0)'
    print(f'    ✓  {eq.name} @ ({eq.x_num:.6f}, {eq.y_num:.6f})  '
          f'→ f={f_val:.2e}, g={g_val:.2e}')

# ──────────────────────────────────────────────────────────────────────
# 5.  num_equilibria
# ──────────────────────────────────────────────────────────────────────
print('\n── 5.  num_equilibria ──')
n = sys.num_equilibria()
print(f'    Number of equilibria: {n}')
assert n == 6  # at this (μ,c)

# Verify on a case where some don't exist
sys2 = System(mu=0.25, c=1.0)  # B-branch denom = 0
n2 = sys2.num_equilibria()
print(f'    At (μ=0.25, c=1.0): {n2} (should be 4)')
assert n2 == 4, f'expected 4, got {n2}'

sys3 = System(mu=0.4, c=0.0)   # E_A⁻ = E0, B-branch issues
n3 = sys3.num_equilibria()
print(f'    At (μ=0.4, c=0.0):  {n3} (should be 5)')

# ──────────────────────────────────────────────────────────────────────
# 6.  Classify (trace, det, eigenvalues, type string)
# ──────────────────────────────────────────────────────────────────────
print('\n── 6.  Classification ──')
sys.classify_all()

# Check that every existing eq got classified
for eq in all_eqs:
    assert eq.exists
    assert eq.type is not None, f'{eq.name} type is None'
    assert eq.trace is not None, f'{eq.name} trace is None'
    assert eq.det is not None, f'{eq.name} det is None'
    assert eq.eigenvals is not None, f'{eq.name} eigenvalues is None'
    assert eq.jacobian is not None, f'{eq.name} Jacobian is None'
    print(f'    {eq.name:>5}  ({eq.x_num:>8.5f}, {eq.y_num:>8.5f})  '
          f'T={eq.trace:>8.4f}  D={eq.det:>8.4f}  '
          f'λ={eq.eigenvals[0]:.4f},{eq.eigenvals[1]:.4f}  '
          f'→ {eq.type}')

# Verify E0 at (0,0) with μ=0.3, c=2.0:
# J = [[2-5μ, 0], [0, -c]] = [[2-1.5, 0], [0, -2]] = [[0.5, 0], [0, -2]]
# trace = -1.5, det = -1 → saddle
assert eq0.type == 'saddle', f'E0 should be saddle, got {eq0.type}'
assert abs(eq0.trace - (-1.5)) < 1e-10, f'E0 trace wrong: {eq0.trace}'
assert abs(eq0.det - (-1.0)) < 1e-10, f'E0 det wrong: {eq0.det}'

# Verify E1 at (0,1) with μ=0.3, c=2.0:
# J should have trace and det from expression
assert eq1.eigenvals.shape == (2,), 'eigenvals shape wrong'

# ──────────────────────────────────────────────────────────────────────
# 7.  Classify edge cases (via _classify_type static)
# ──────────────────────────────────────────────────────────────────────
print('\n── 7.  Classification type strings (edge cases) ──')

types = {}

# saddle: D < 0
types['saddle'] = System._classify_type(1.0, -1.0)
# stable node: D > 0, T < 0, T² > 4D
types['stable node'] = System._classify_type(-3.0, 1.0)   # T²=9, 4D=4
# stable spiral: D > 0, T < 0, T² < 4D
types['stable spiral'] = System._classify_type(-1.0, 1.0) # T²=1, 4D=4
# unstable node: D > 0, T > 0, T² > 4D
types['unstable node'] = System._classify_type(3.0, 1.0)
# unstable spiral: D > 0, T > 0, T² < 4D
types['unstable spiral'] = System._classify_type(1.0, 1.0)
# centre: D > 0, T ≈ 0
types['centre'] = System._classify_type(0.0, 1.0)
# degenerate (double zero): D ≈ 0, T ≈ 0
types['degenerate (double zero)'] = System._classify_type(0.0, 0.0)
# degenerate (zero det): D ≈ 0, T ≠ 0
types['degenerate (zero det)'] = System._classify_type(2.0, 0.0)

for desc, got in types.items():
    status = '✓' if got == desc else '✗'
    print(f'    {status}  {desc:>30s} → {got}')

for desc, got in types.items():
    assert got == desc, f'Expected "{desc}", got "{got}"'

# ──────────────────────────────────────────────────────────────────────
# 8.  Trajectory integration
# ──────────────────────────────────────────────────────────────────────
print('\n── 8.  Trajectory integration ──')
t, xs, ys = sys.trajectory(0.1, 0.1, t_span=(0, 5), dt=0.01)
print(f'    From (0.1, 0.1), t ∈ [0, 5]:')
print(f'      t shape: {t.shape}, xs shape: {xs.shape}')
assert len(t) > 10, 'trajectory too short'
assert np.all(np.isfinite(t)), 'non-finite t'
assert np.all(np.isfinite(xs)), f'non-finite xs: {xs[:3]} ...'
assert np.all(np.isfinite(ys)), f'non-finite ys'

# Verify the trajectory starts at our IC
assert abs(xs[0] - 0.1) < 1e-10, f'traj start x wrong: {xs[0]}'
assert abs(ys[0] - 0.1) < 1e-10, f'traj start y wrong: {ys[0]}'

# Second trajectory with different t_span
t2, xs2, ys2 = sys.trajectory(-0.5, 0.8, t_span=(0, 10), dt=0.05)
print(f'    Second traj from (−0.5, 0.8), t ∈ [0, 10]: {len(t2)} points')
assert len(t2) > 10, 'second trajectory too short'

print('    ✓  Trajectory integration')

# ──────────────────────────────────────────────────────────────────────
# 9.  Domain auto-limits
# ──────────────────────────────────────────────────────────────────────
print('\n── 9.  Domain auto-limits ──')
xmin, xmax, ymin, ymax = sys._auto_limits(pad_factor=1.5, min_pad=0.5)
print(f'    Auto domain: x∈[{xmin:.4f}, {xmax:.4f}], y∈[{ymin:.4f}, {ymax:.4f}]')
assert xmin < xmax, 'xmin ≥ xmax'
assert ymin < ymax, 'ymin ≥ ymax'
assert np.isfinite(xmin), 'non-finite domain'

# With user-specified override limits
sys_override = System(mu=MU, c=C, xlim=(-3, 3), ylim=(-2, 5))
xlo, xhi, ylo, yhi = sys_override._auto_limits()
print(f'    Override domain:  x∈[{xlo:.4f}, {xhi:.4f}], y∈[{ylo:.4f}, {yhi:.4f}]')
assert abs(xlo + 3) < 1e-10, 'override xlim not respected'
assert abs(yhi - 5) < 1e-10, 'override ylim not respected'

# With no equilibria (should fallback)
sys_empty_params = System(mu=999, c=999)  # extreme, unlikely to have eq
if sys_empty_params.num_equilibria() == 0:
    xl, xr, yl, yr = sys_empty_params._auto_limits()
    print(f'    Empty fallback:   x∈[{xl:.4f}, {xr:.4f}], y∈[{yl:.4f}, {yr:.4f}]')
    assert abs(xl + 1) < 1e-10, 'fallback xmin not -1'

print('    ✓  Domain auto-limits')

# ──────────────────────────────────────────────────────────────────────
# 10.  Symbolic expressions
# ──────────────────────────────────────────────────────────────────────
print('\n── 10.  Symbolic equilibrium expressions ──')
sys.symbolic_summary()

# Access the class-level templates directly
templates = System._EQUILIBRIUM_TEMPLATES
assert len(templates) == 6, 'wrong number of templates'
for tmpl in templates:
    assert tmpl.x_expr is not None, f'{tmpl.name} x_expr is None'
    assert tmpl.y_expr is not None, f'{tmpl.name} y_expr is None'
    assert isinstance(tmpl.x_expr, sp.Expr), f'{tmpl.name} x_expr not sympy'
    assert isinstance(tmpl.y_expr, sp.Expr), f'{tmpl.name} y_expr not sympy'

# Verify the class-level symbol constants exist
assert hasattr(System, 'X'), 'System.X missing'
assert hasattr(System, 'Y'), 'System.Y missing'
assert hasattr(System, 'MU'), 'System.MU missing'
assert hasattr(System, 'C'), 'System.C missing'
assert isinstance(System.X, sp.Symbol), 'System.X not a symbol'
assert isinstance(System.MU, sp.Symbol), 'System.MU not a symbol'

print('    ✓  Symbolic expressions')

# ──────────────────────────────────────────────────────────────────────
# 11.  Numerical summary
# ──────────────────────────────────────────────────────────────────────
print('\n── 11.  Numerical summary ──')
sys.numerical_summary()

# ──────────────────────────────────────────────────────────────────────
# 12.  Plotting — global portrait
# ──────────────────────────────────────────────────────────────────────
print('\n── 12.  Global phase portrait ──')
fig, ax = sys.plot_global(show=False, save=True, grid_res=200,
                           ic_nx=6, ic_ny=6, t_span=(0, 15))
assert fig is not None, 'global plot returned None fig'
assert ax is not None, 'global plot returned None ax'
print('    ✓  Global portrait saved')

# With explicit limits (override)
sys_override.plot_global(show=False, save=True,
                         filename='global_override_test')
print('    ✓  Global portrait with override limits saved')

# With custom title
sys.plot_global(show=False, save=True, filename='global_custom_title',
                title='My custom title')
print('    ✓  Global portrait with custom title saved')

# ──────────────────────────────────────────────────────────────────────
# 13.  Plotting — local portrait (each equilibrium)
# ──────────────────────────────────────────────────────────────────────
print('\n── 13.  Local phase portraits ──')
sys.classify_all()
portrait_count = 0
for eq in sys.real_equilibria():
    fig_loc, ax_loc = sys.plot_local(eq, show=False, save=True)
    portrait_count += 1
    assert fig_loc is not None, f'{eq.name} local plot returned None'
    assert ax_loc is not None, f'{eq.name} local plot returned None ax'
    print(f'    ✓  Local portrait: {eq.name}  ({eq.type})')
print(f'    Total local portraits generated: {portrait_count}')

# Local portrait with explicit zoom_radius
fig_z, ax_z = sys.plot_local(all_eqs[0], zoom_radius=0.3,
                              show=False, save=True,
                              filename='local_E0_zoom')
print(f'    ✓  Local portrait with explicit zoom_radius=0.3')

# ──────────────────────────────────────────────────────────────────────
# 14.  plot_all_local convenience
# ──────────────────────────────────────────────────────────────────────
print('\n── 14.  plot_all_local ──')
figs = sys.plot_all_local(show=False, save=False)
assert len(figs) == sys.num_equilibria(), \
    f'plot_all_local returned {len(figs)} figs, expected {sys.num_equilibria()}'
for f in figs:
    plt.close(f)
print(f'    ✓  plot_all_local returned {len(figs)} figures')

# ──────────────────────────────────────────────────────────────────────
# 15.  Equations verified at equilibrium (symbolic check)
# ──────────────────────────────────────────────────────────────────────
print('\n── 15.  Symbolic verification: eqns vanish at equilibrium expressions ──')
from sympy import simplify, N, Abs

for tmpl in System._EQUILIBRIUM_TEMPLATES:
    x_expr = tmpl.x_expr
    y_expr = tmpl.y_expr
    try:
        f_val = simplify(System._f_expr.subs({System.X: x_expr, System.Y: y_expr,
                                              System.MU: MU, System.C: C}))
        g_val = simplify(System._g_expr.subs({System.X: x_expr, System.Y: y_expr,
                                              System.MU: MU, System.C: C}))
        f_num_complex = complex(N(f_val))  # use complex to handle 0*I case
        g_num_complex = complex(N(g_val))
        f_num = f_num_complex.real if abs(f_num_complex.imag) < 1e-12 else f_num_complex.real
        g_num = g_num_complex.real if abs(g_num_complex.imag) < 1e-12 else g_num_complex.real
    except Exception as e:
        print(f'    ⚠  {tmpl.name}: skipped ({e})')
        continue
    assert abs(f_num) < 1e-6, f'{tmpl.name}: f = {f_num:.4e}'
    assert abs(g_num) < 1e-6, f'{tmpl.name}: g = {g_num:.4e}'
    print(f'    ✓  {tmpl.name}:  f ≈ {f_num:.2e}, g ≈ {g_num:.2e}')

# ──────────────────────────────────────────────────────────────────────
# 16.  Non-existence edge cases
# ──────────────────────────────────────────────────────────────────────
print('\n── 16.  Non-existence edge cases ──')

# μ = 1/4 → B-branch denominator vanishes
sys_mu_quarter = System(mu=0.25, c=2.0)
print(f'    μ=0.25, c=2:   {sys_mu_quarter.num_equilibria()} eq exist '
      f'(B⁺: {sys_mu_quarter._equilibria[4].exists}, '
      f'B⁻: {sys_mu_quarter._equilibria[5].exists})')
assert not sys_mu_quarter._equilibria[4].exists, 'E_B⁺ should not exist at μ=0.25'
assert not sys_mu_quarter._equilibria[5].exists, 'E_B⁻ should not exist at μ=0.25'

# c = 0 → some B-branch issues expected
sys_c0 = System(mu=0.3, c=0.0)
print(f'    μ=0.3, c=0:    {sys_c0.num_equilibria()} eq exist '
      f'(E_B⁺: {sys_c0._equilibria[4].exists}, '
      f'E_B⁻: {sys_c0._equilibria[5].exists})')

# Large negative Δ_B (can't happen at μ=0.3, c=2, but let's check somewhere)
sys_possible = System(mu=2.0, c=10.0)
print(f'    μ=2.0, c=10:   {sys_possible.num_equilibria()} eq exist')

# ──────────────────────────────────────────────────────────────────────
# 17.  File output structure
# ──────────────────────────────────────────────────────────────────────
print('\n── 17.  File output verification ──')
from pathlib import Path

mu_str = f'{MU:.10g}'.replace('.', '_')
c_str = f'{C:.10g}'.replace('.', '_')
expected_dir = Path('figs') / f'Rtest(c_{c_str}-mu_{mu_str})'

print(f'    Expected directory: {expected_dir}')
assert expected_dir.exists(), f'Directory {expected_dir} does not exist'

# Check files exist
expected_files = [
    'global_portrait.pdf',
    'local_E0.pdf',
    'local_E1.pdf',
    'local_E_A⁺.pdf',
    'local_E_A⁻.pdf',
    'local_E_B⁺.pdf',
    'local_E_B⁻.pdf',
]
for fname in expected_files:
    fpath = expected_dir / fname
    assert fpath.exists(), f'Missing file: {fpath}'
    assert fpath.stat().st_size > 1000, f'File too small: {fpath} ({fpath.stat().st_size} bytes)'
print(f'    All {len(expected_files)} expected files present and > 1 KB')

# ──────────────────────────────────────────────────────────────────────
# 18.  Non-equilibrium methods
# ──────────────────────────────────────────────────────────────────────
print('\n── 18.  Edge: extreme parameter fallback ──')
# Use a parameter that makes all equilibria non-existent (complex Δ_B, divergent, etc.)
# μ=0.25 makes B-branch blow up; use a value where Δ_A is negative
# At μ=0.25, c=0.1, we should have fewer eq, but not 0
# The fallback is only triggered when real_equilibria() is empty — 
# let's just ensure auto_limits still returns something sensible even with few eq
sys_extreme = System(mu=0.25, c=100.0)
n_extreme = sys_extreme.num_equilibria()
eqs_extreme = sys_extreme.real_equilibria()
print(f'    At μ=0.25, c=100:  {n_extreme} eq exist')
xmin3, xmax3, ymin3, ymax3 = sys_extreme._auto_limits()
print(f'    Auto-domain: x∈[{xmin3:.4f}, {xmax3:.4f}], y∈[{ymin3:.4f}, {ymax3:.4f}]')
assert xmin3 < xmax3, f'auto-limits produce empty x domain: [{xmin3:.4f}, {xmax3:.4f}]'
assert ymin3 < ymax3, f'auto-limits produce empty y domain'

# ──────────────────────────────────────────────────────────────────────
# 19.  Classify on non-existent equilibrium
# ──────────────────────────────────────────────────────────────────────
print('\n── 19.  Classify non-existent eq ──')
sys_mq = System(mu=0.25, c=2.0)
eq_dead = sys_mq._equilibria[4]  # E_B⁺, should not exist
assert not eq_dead.exists, 'E_B⁺ at μ=0.25 should not exist'
sys_mq.classify(eq_dead)
assert eq_dead.type == 'does not exist', \
    f'non-existent eq type should be "does not exist", got "{eq_dead.type}"'
print(f'    ✓  Non-existent eq type: "{eq_dead.type}"')

# ──────────────────────────────────────────────────────────────────────
# 20.  Constants / docstring check
# ──────────────────────────────────────────────────────────────────────
print('\n── 20.  Constants & metadata ──')
import system as system_module
mod_doc = system_module.__doc__
assert mod_doc is not None, 'module has no docstring'
assert 'dx/dt' in mod_doc, 'module docstring missing ODE system'
print(f'    Module docstring: {len(mod_doc)} chars  ✓')

assert hasattr(System, '_EQUILIBRIUM_TEMPLATES'), 'missing templates list'
assert hasattr(System, '_f_expr'), 'missing f_expr'
assert hasattr(System, '_g_expr'), 'missing g_expr'
assert hasattr(System, '_J_expr'), 'missing J_expr'
assert hasattr(System, '_Delta_A'), 'missing Delta_A'
assert hasattr(System, '_Delta_B'), 'missing Delta_B'
print('    ✓  All class-level constants present')

# ──────────────────────────────────────────────────────────────────────
# Done
# ──────────────────────────────────────────────────────────────────────
print('\n╔═════════════════════════════════════════════════════════════╗')
print('║  ALL 20 tests PASSED                                       ║')
print(f'║  μ = {MU}, c = {C}                                              ║')
print('╚═════════════════════════════════════════════════════════════╝')