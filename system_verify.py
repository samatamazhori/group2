"""
system_verify.py — Visual verification of system.py at (μ=0.3, c=2.0).

Prints every detail to stdout so you can read and verify.
No assertions — pure display.
"""

from system import System, Equilibrium
import numpy as np

#MU, C = 0.724745, 2.724745
MU, C = -0.5,2
sys = System(mu=MU, c=C)

# ═══════════════════════════════════════════════════════════════
sep = '═' * 78
sub = '─' * 78

print(sep)
print(f'  SYSTEM VERIFICATION    μ = {MU}    c = {C}')
print(sep)

# ── 1. System equations ───────────────────────────────────────
print(f'\n  ╔══ 1. SYSTEM EQUATIONS ═══════════════════════════════╗')
print(f'  ║  dx/dt = x[(c-4)y - 5μ + 4μ(x+y) + x(x+3y-3) + 2]')
print(f'  ║  dy/dt = y[c(y-1) + x(x+3y-1)]')
print(f'  ╚══════════════════════════════════════════════════════╝')

# ── 2. Jacobian ───────────────────────────────────────────────
print(f'\n  ╔══ 2. JACOBIAN ═══════════════════════════════════════╗')
J0 = sys.jacobian(0, 0)
J1 = sys.jacobian(0, 1)
print(f'  ║  J(0,0) = [{J0[0,0]:.6f}, {J0[0,1]:.6f}]')
print(f'  ║           [{J1[0,0]:.6f}, {J1[0,1]:.6f}]')
print(f'  ╚══════════════════════════════════════════════════════╝')

# ── 3. All Equilibria ─────────────────────────────────────────
print(f'\n  ╔══ 3. ALL EQUILIBRIA ═══════════════════════════════════╗')
sys.classify_all()
all_eqs = sys.equilibria()
real_eqs = sys.real_equilibria()
print(f'  ║  Total registered: {len(all_eqs)}')
print(f'  ║  Existing (real):   {len(real_eqs)}')
print(f'  ╚══════════════════════════════════════════════════════╝')

for eq in all_eqs:
    print(f'\n{sub}')
    print(f'  [{eq.name}]')
    print(f'  Branch:        {eq.branch}')
    print(f'  Exists:        {eq.exists}')

    # Coordinates
    if eq.exists:
        print(f'  (x, y):        ({eq.x_num:.10f}, {eq.y_num:.10f})')
    else:
        print(f'  x_expr:        {eq.x_expr if eq.x_expr is not None else "—"}')
        print(f'  y_expr:        {eq.y_expr if eq.y_expr is not None else "—"}')

    # Verify that the numerical equilibrium satisfies the ODE
    if eq.exists:
        f_val = sys.f(eq.x_num, eq.y_num)
        g_val = sys.g(eq.x_num, eq.y_num)
        print(f'  f(x,y):        {f_val:.4e}  (should be ≈ 0)')
        print(f'  g(x,y):        {g_val:.4e}  (should be ≈ 0)')

    # Classification details
    print(f'  Trace:         {eq.trace:.10f}' if eq.trace is not None else '  Trace:         —')
    print(f'  Det:           {eq.det:.10f}' if eq.det is not None else '  Det:           —')
    if eq.eigenvals is not None:
        l1, l2 = eq.eigenvals
        l1s = f'{l1.real:.6f} + {l1.imag:.6f}i' if isinstance(l1, complex) else f'{l1:.6f}'
        l2s = f'{l2.real:.6f} + {l2.imag:.6f}i' if isinstance(l2, complex) else f'{l2:.6f}'
        print(f'  Eigenvalues:   [{l1s},  {l2s}]')
    else:
        print(f'  Eigenvalues:   —')
    print(f'  Type:          {eq.type}')

    if eq.jacobian is not None:
        J = eq.jacobian
        print(f'  Jacobian:')
        print(f'               [{J[0,0]:.10f},  {J[0,1]:.10f}]')
        print(f'               [{J[1,0]:.10f},  {J[1,1]:.10f}]')

# ── 4. Summary table ──────────────────────────────────────────
print(f'\n{sub}')
print(f'  ┌── SUMMARY TABLE ─{"─"*70}┐')
header = f'  │ {"Name":>8s} {"Br":>2s} {"Exists":>6s}  {"x":>12s}  {"y":>12s}  '
header += f'{"Trace":>10s}  {"Det":>10s}  {"Type":>22s} │'
print(header)
print(f'  ├{"─"*78}┤')
for eq in all_eqs:
    x_str = f'{eq.x_num:.8f}' if eq.exists else '—'
    y_str = f'{eq.y_num:.8f}' if eq.exists else '—'
    t_str = f'{eq.trace:.6f}' if eq.trace is not None else '—'
    d_str = f'{eq.det:.6f}' if eq.det is not None else '—'
    typ_str = eq.type if eq.type is not None else '—'
    e_str = '✓' if eq.exists else '✗'
    row = f'  │ {eq.name:>8s}  {eq.branch:1s}  {e_str:>5s}  '
    row += f'{x_str:>12s}  {y_str:>12s}  {t_str:>10s}  {d_str:>10s}  {typ_str:>22s} │'
    print(row)
print(f'  └{"─"*78}┘')

# ── 5. Eigenvalue detail ─────────────────────────────────────
print(f'\n  ╔══ 4. EIGENVALUE BREAKDOWN ═════════════════════════════╗')
for eq in all_eqs:
    if not eq.exists or eq.eigenvals is None:
        continue
    l1, l2 = eq.eigenvals
    l1c = complex(l1) if not isinstance(l1, complex) else l1
    l2c = complex(l2) if not isinstance(l2, complex) else l2
    print(f'  ║  {eq.name:>6s}:  λ₁ = {l1c.real:+.6f} {"+" if l1c.imag >= 0 else "-"} {abs(l1c.imag):.6f}i'
          f'  |  λ₂ = {l2c.real:+.6f} {"+" if l2c.imag >= 0 else "-"} {abs(l2c.imag):.6f}i')
    # check stability
    real_parts = [l1c.real, l2c.real]
    if all(r < 0 for r in real_parts):
        stab = 'stable'
    elif all(r > 0 for r in real_parts):
        stab = 'unstable'
    elif any(r == 0 for r in real_parts):
        stab = 'centre / marginal'
    else:
        stab = 'saddle'
    print(f'  ║         {"real" if abs(l1c.imag) < 1e-12 else "complex"}  |  stability: {stab}')
print(f'  ╚══════════════════════════════════════════════════════╝')

# ── 6. Trajectory sanity check ───────────────────────────────
print(f'\n  ╔══ 5. TRAJECTORY CHECK ═══════════════════════════════╗')
t, xs, ys = sys.trajectory(0.1, 0.1, t_span=(0, 5), dt=0.01)
print(f'  ║  IC = (0.1, 0.1), t ∈ [0, 5]')
print(f'  ║  Points:        {len(t)}')
print(f'  ║  t[-1]:         {t[-1]:.4f}')
print(f'  ║  x[-1]:         {xs[-1]:.6f}')
print(f'  ║  y[-1]:         {ys[-1]:.6f}')
t2, xs2, ys2 = sys.trajectory(-0.5, 0.8, t_span=(0, 10), dt=0.05)
print(f'  ║  IC = (-0.5, 0.8), t ∈ [0, 10]')
print(f'  ║  Points:        {len(t2)}')
print(f'  ║  t[-1]:         {t2[-1]:.4f}')
print(f'  ║  x[-1]:         {xs2[-1]:.6f}')
print(f'  ║  y[-1]:         {ys2[-1]:.6f}')
print(f'  ╚══════════════════════════════════════════════════════╝')

# ── 7. Domain limits ─────────────────────────────────────────
print(f'\n  ╔══ 6. AUTO-DOMAIN LIMITS ══════════════════════════════╗')
xmin, xmax, ymin, ymax = sys._auto_limits(pad_factor=1.5, min_pad=0.5)
print(f'  ║  x ∈ [{xmin:.6f}, {xmax:.6f}]   (span = {xmax-xmin:.4f})')
print(f'  ║  y ∈ [{ymin:.6f}, {ymax:.6f}]   (span = {ymax-ymin:.4f})')
print(f'  ╚══════════════════════════════════════════════════════╝')

# ── 8. Nullclines (brief) ────────────────────────────────────
print(f'\n  ╔══ 7. NULLCLINE SIGNATURES ════════════════════════════╗')
print(f'  ║  x-nullcline:  x = 0  OR  (c-4)y - 5μ + 4μ(x+y) + x(x+3y-3) + 2 = 0')
print(f'  ║  y-nullcline:  y = 0  OR  c(y-1) + x(x+3y-1) = 0')
print(f'  ║')
print(f'  ║  Branch A (y=0):   x = (3 - 4μ ± √Δ_A) / 2')
print(f'  ║    Δ_A = 16μ² - 4μ + 1')
print(f'  ║    Δ_A = {16*MU*MU - 4*MU + 1:.6f}')
print(f'  ║    Δ_A > 0? {"yes" if 16*MU*MU - 4*MU + 1 > 0 else "no"}')
print(f'  ║')
print(f'  ║  Branch B (x≠0, y≠0):  y = (c + x - x²) / (c + 3x)   (from dy/dt=0)')
print(f'  ╚══════════════════════════════════════════════════════╝')

# ── 9. Edge-case info ───────────────────────────────────────
print(f'\n  ╔══ 8. EDGE-CASE QUICK CHECK ═══════════════════════════╗')
sys_q = System(mu=0.25, c=2.0)
sys_q.classify_all()
n = sys_q.num_equilibria()
print(f'  ║  μ = 0.25,  c = 2.0:  {n} equilibria exist')
for eq in sys_q.equilibria():
    x_str = f'{eq.x_num:.4f}' if eq.exists else '—'
    y_str = f'{eq.y_num:.4f}' if eq.exists else '—'
    print(f'  ║    {eq.name:>6s}  exists={str(eq.exists):>5s}  '
          f'({x_str}, {y_str})  type={eq.type}')
print(f'  ╚══════════════════════════════════════════════════════╝')

# ── 10. File output ──────────────────────────────────────────
from pathlib import Path
mu_s = f'{MU:.10g}'.replace('.', '_')
c_s = f'{C:.10g}'.replace('.', '_')
d = Path('figs') / f'Rtest(c_{c_s}-mu_{mu_s})'
print(f'\n  ╔══ 9. OUTPUT FILES ════════════════════════════════════╗')
if d.exists():
    for f in sorted(d.glob('*.pdf')):
        sz = f.stat().st_size
        print(f'  ║  ✓  {f.name:45s}  {sz:>8,d} bytes')
else:
    print(f'  ║  Directory not yet generated (run plot commands)')
print(f'  ╚══════════════════════════════════════════════════════╝')

print(f'\n{sep}')
print(f'  END OF VERIFICATION')
print(sep)