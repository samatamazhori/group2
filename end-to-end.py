"""
end-to-end.py — Complete workflow: build the system, classify, plot, and
dump all info into ``name.txt`` inside ``figs/Rtest(c_{c}-mu_{μ})/``.

Change ``MU`` and ``C`` at the top, then run::

    python3 end-to-end.py
"""

import numpy as np
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
#  Parameters  —  edit these
# ═══════════════════════════════════════════════════════════════════════
MU = -0.812
C  = 3.0
# ═══════════════════════════════════════════════════════════════════════

from system import System

# ─── output directory ──────────────────────────────────────────────────

mu_str = f'{MU:.10g}'.replace('.', '_')
c_str = f'{C:.10g}'.replace('.', '_')
out_dir = Path('figs') / f'Rtest(c_{c_str}-mu_{mu_str})'
out_dir.mkdir(parents=True, exist_ok=True)

# ─── build system ──────────────────────────────────────────────────────

sys = System(mu=MU, c=C)
sys.classify_all()
eqs = sys.equilibria()
real_eqs = sys.real_equilibria()

# ═══════════════════════════════════════════════════════════════════════
#  name.txt  —  full numerical + classification dump
# ═══════════════════════════════════════════════════════════════════════

def fmt_val(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    if isinstance(v, (complex, np.complexfloating)):
        if abs(v.imag) < 1e-14:
            return f"{v.real:.10f}"
        return f"{v.real:.6f} {v.imag:+.6f}i"
    if isinstance(v, float):
        return f"{v:.10f}"
    return str(v)

def fmt_evec(v):
    return f"({v[0]:.6f}, {v[1]:.6f})"

lines = []
lines.append("=" * 90)
lines.append(f"  SYSTEM REPORT  —  μ = {MU:.10g},  c = {C:.10g}")
lines.append("=" * 90)
lines.append("")

# ── equations ──────────────────────────────────────────────────────────
lines.append("System equations:")
lines.append(f"  dx/dt = x[(c−4)y − 5μ + 4μ(x+y) + x(x+3y−3) + 2]")
lines.append(f"  dy/dt = y[c(y−1) + x(x+3y−1)]")
lines.append("")

# ── Jacobian (numeric at each eq) ──────────────────────────────────────
lines.append(f"Jacobian matrix (symbolic):")
J_sym = sys._J_expr
lines.append(f"  J11 = {J_sym[0,0]}")
lines.append(f"  J12 = {J_sym[0,1]}")
lines.append(f"  J21 = {J_sym[1,0]}")
lines.append(f"  J22 = {J_sym[1,1]}")
lines.append(f"  Tr(J) = {sys._trace_expr}")
lines.append(f"  det(J) = {sys._det_expr}")
lines.append("")

# ── equilibrium summary table ─────────────────────────────────────────
lines.append(f"Equilibria:  {len(real_eqs)} / {len(eqs)} exist")
lines.append("")

header = (f"  {'Name':>8}  {'Branch':>6}  {'Exists':>6}"
          f"  {'x':>14}  {'y':>14}"
          f"  {'Trace':>12}  {'Det':>12}"
          f"  {'Type':>20}")
lines.append(header)
lines.append("  " + "─" * (len(header) - 2))

for eq in eqs:
    if eq.exists:
        lines.append(
            f"  {eq.name:>8}  {eq.branch:>6}  {'✓':>6}"
            f"  {fmt_val(eq.x_num):>14}  {fmt_val(eq.y_num):>14}"
            f"  {fmt_val(eq.trace):>12}  {fmt_val(eq.det):>12}"
            f"  {eq.type:>20}")
    else:
        lines.append(
            f"  {eq.name:>8}  {eq.branch:>6}  {'✗':>6}"
            f"  {'—':>14}  {'—':>14}"
            f"  {'—':>12}  {'—':>12}"
            f"  {'does not exist':>20}")

lines.append("")

# ── detailed per-equilibrium ──────────────────────────────────────────
for eq in real_eqs:
    lines.append("─" * 80)
    lines.append(f"  {eq.name}  {eq.label}  [{eq.type}]")
    lines.append(f"    coordinates       = ({fmt_val(eq.x_num)}, {fmt_val(eq.y_num)})")
    lines.append(f"    f(x,y) residual   = {fmt_val(sys.f(eq.x_num, eq.y_num))}")
    lines.append(f"    g(x,y) residual   = {fmt_val(sys.g(eq.x_num, eq.y_num))}")
    lines.append(f"    trace             = {fmt_val(eq.trace)}")
    lines.append(f"    determinant       = {fmt_val(eq.det)}")
    delta_str = ""
    if eq.trace is not None and eq.det is not None and eq.det > 1e-12:
        delta = eq.trace**2 - 4*eq.det
        delta_str = f"  (Δ = Tr² − 4Det = {fmt_val(delta)})"
    lines.append(f"    eigenvalues       = {fmt_val(eq.eigenvals[0])},  {fmt_val(eq.eigenvals[1])}{delta_str}")
    if eq.jacobian is not None:
        J = eq.jacobian
        lines.append(f"    Jacobian          = [[{fmt_val(J[0,0])}, {fmt_val(J[0,1])}],")
        lines.append(f"                        [{fmt_val(J[1,0])}, {fmt_val(J[1,1])}]]")
        # eigenvectors
        evals, evecs = np.linalg.eig(J)
        for i in range(2):
            lines.append(f"    eigenvector λ_{i+1} = {fmt_evec(evecs[:, i])}  (λ = {fmt_val(evals[i])})")
    lines.append("")

# ── trajectory test ────────────────────────────────────────────────────
lines.append("─" * 80)
lines.append("  Trajectory integration (sample ICs):")
for name, ic in [("near origin", (0.05, 0.05)),
                 ("near E1", (0.05, 0.95))]:
    try:
        t, xs, ys = sys.trajectory(ic[0], ic[1], t_span=(0, 10))
        xf, yf = xs[-1], ys[-1]
        lines.append(f"    IC {name} ({ic[0]:.2f},{ic[1]:.2f}):  "
                      f"({xs[0]:.4f},{ys[0]:.4f}) → ({xf:.4f},{yf:.4f})  "
                      f"({len(t)} steps)")
    except Exception as e:
        lines.append(f"    IC {name}: failed — {e}")
lines.append("")

# ── nullcline grid statistics ─────────────────────────────────────────
xmin, xmax, ymin, ymax = sys._auto_limits()
lines.append("─" * 80)
lines.append("  Domain (auto):")
lines.append(f"    x ∈ [{fmt_val(xmin)}, {fmt_val(xmax)}]")
lines.append(f"    y ∈ [{fmt_val(ymin)}, {fmt_val(ymax)}]")
lines.append("")

# ── discriminant values ───────────────────────────────────────────────
lines.append("─" * 80)
lines.append("  Discriminants:")
DA = float(sys._Delta_A.subs({sys.MU: MU}))
lines.append(f"    Δ_A = {fmt_val(DA)}  (≥ 0 → Branch A has 2 real + E0 zeros)")
DB = float(sys._Delta_B.subs({sys.MU: MU, sys.C: C}))
lines.append(f"    Δ_B = {fmt_val(DB)}  (≥ 0 → Branch B has 2 real + E1 zeros)")
lines.append("")

lines.append("=" * 90)
lines.append("  END OF REPORT")
lines.append("=" * 90)

report = "\n".join(lines)
print(report)

# ── write name.txt ────────────────────────────────────────────────────
txt_path = out_dir / "name.txt"
txt_path.write_text(report)
print(f"\n  → saved {txt_path}")

# ═══════════════════════════════════════════════════════════════════════
#  Plots
# ═══════════════════════════════════════════════════════════════════════

print(f"\n  ── Generating plots ──")

# Global portrait
sys.plot_global(show=False)
sys.plot_global(title=f"Global portrait  μ={MU}, c={C}",
                filename="global_portrait", show=False)

# Local for each existing equilibrium
for eq in real_eqs:
    sys.plot_local(eq, show=False)

print(f"\n  All done.  Files in: {out_dir}/")