"""
quick-solve.py — Solve C1(μ, c) = 0 for μ at N values of c ∈ [0, 1].
"""

import numpy as np
from data import C1
import sympy as sp

mu, c = sp.symbols('mu c')
solutions = sp.solve(C1.expr, mu)
mu0_f = sp.lambdify(c, solutions[0], modules='numpy')
mu1_f = sp.lambdify(c, solutions[1], modules='numpy')

print(f'C1: {C1.name}\n')
print(f'{"c":>8}  {"μ₁":>14}  {"μ₂":>14}  {"real":>6}')
print('-' * 46)

for c_i in np.linspace(0.0, 1.0, 11):
    m0 = mu0_f(c_i)
    m1 = mu1_f(c_i)
    real = np.isfinite(m0) and np.isfinite(m1)
    print(f'{c_i:8.4f}  {m0:14.8f}  {m1:14.8f}  {str(real):>6}')
