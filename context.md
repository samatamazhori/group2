# Project 
This Project is for cataloging the entire dynamical behhaviour
of the following system:


\begin{align}
    \dot{x} &= x [(c-4) y-5 \mu +4 \mu  (x+y)+x (x+3 y-3)+2] \\
    \dot{y} &= y[c (y-1) +x  (x+3 y-1)]
\end{align}

As you can see the state vars are x and y and we have two params called mu and c. we are tyring to find the bifuirication of this system.

# Curves 
by solving the system symoblicaly we have found at most 6 equlibrium points in terms of mu and c and reached the followng 
curves which will give us a skelton and stratify the mu and c plane into regoins the follwoing curves are here 


# ---------------------------------------------------------------
# s1(d2): 4 - 12c + 9c^2 - 44μ + 50cμ - 24c^2μ + 121μ^2 - 56cμ^2 + 16c^2μ^2 = 0
# ---------------------------------------------------------------
F1 = (4 - 12*C + 9*C**2 - 44*MU + 50*C*MU - 24*C**2*MU
      + 121*MU**2 - 56*C*MU**2 + 16*C**2*MU**2)
ax.contour(MU, C, F1, levels=[0], colors='b', linewidths=2)

# ---------------------------------------------------------------
# s2(R): -2c + c^2 - 2μ + 10cμ + 5μ^2 - 16cμ^2 = 0
# ---------------------------------------------------------------
F2 = -2*C + C**2 - 2*MU + 10*C*MU + 5*MU**2 - 16*C*MU**2
ax.contour(MU, C, F2, levels=[0], colors='r', linewidths=2)

# ---------------------------------------------------------------
# s3: μ = 2/5
# ---------------------------------------------------------------
F3 = MU - 2/5
ax.contour(MU, C, F3, levels=[0], colors='g', linewidths=2, linestyles='--')

# ---------------------------------------------------------------
# s4: c = μ + 2
# ---------------------------------------------------------------
F4 = C - MU - 2
ax.contour(MU, C, F4, levels=[0], colors='m', linewidths=2)

# ---------------------------------------------------------------
# s5: c = 0
# ---------------------------------------------------------------
F5 = C
ax.contour(MU, C, F5, levels=[0], colors='k', linewidths=2, linestyles='-.')

# ---------------------------------------------------------------
# s6: μ = 1/4
# ---------------------------------------------------------------
F6 = MU - 1/4
ax.contour(MU, C, F6, levels=[0], colors='orange', linewidths=2, linestyles=':')

