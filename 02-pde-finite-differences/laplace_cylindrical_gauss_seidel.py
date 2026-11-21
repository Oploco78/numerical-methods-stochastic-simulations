# -*- coding: utf-8 -*-
"""
1D & 2D Cylindrical Laplace Equation Solver via Gauss-Seidel Relaxation.
Solves the boundary-value problem for electrostatic potential inside a finite
coaxial cylinder using finite differences in cylindrical coordinates, benchmarks
the 1D solution against the exact analytical expression, and computes the electric
field vector field via numerical differentiation.
"""

import numpy as np
import matplotlib.pyplot as plt

R_i = 1.0   # Inner radius (cm)
R_e = 4.0   # Outer radius (cm)
V_i = 2.5   # Inner potential (V)
V_e = 0.0   # Outer potential (V)
L = 20.0    # Cylinder length (cm)


# 1D MODEL

print("Solving 1D model...")
N = 100
r = np.linspace(R_i, R_e, N)
h = r[1] - r[0]

V = np.linspace(V_i, V_e, N)
V[0], V[-1] = V_i, V_e

tol = 1e-6
error_max = 1.0
iterations_1d = 0

while error_max > tol:
    error_max = 0.0
    for j in range(1, N - 1):
        V_old = V[j]
        V[j] = 0.5 * (V[j-1] * (1 - h / (2 * r[j])) + V[j+1] * (1 + h / (2 * r[j])))
        
        change = abs(V[j] - V_old)
        if change > error_max:
            error_max = change
    iterations_1d += 1

# Exact analytical solution and error
V_exact = V_i * (np.log(R_e / r) / np.log(R_e / R_i))
max_error = np.max(np.abs(V - V_exact))

# 1D electric field calculation
E_num = np.zeros(N)
E_num[1:-1] = -(V[2:] - V[:-2]) / (2 * h)
E_num[0] = -(V[1] - V[0]) / h
E_num[-1] = -(V[-1] - V[-2]) / h
E_exact = V_i / (r * np.log(R_e / R_i))


# 2D MODEL

print(f"1D model converged in {iterations_1d} iterations.")
print("Solving 2D model (this may take a few seconds)...")

N_r = 40   # Radial grid points
N_z = 100  # Longitudinal grid points
r_2d = np.linspace(R_i, R_e, N_r)
z_2d = np.linspace(0, L, N_z)
hr = r_2d[1] - r_2d[0]
hz = z_2d[1] - z_2d[0]

# 2D potential grid with Dirichlet boundary conditions
V2D = np.zeros((N_r, N_z))

V2D[0, :] = V_i
V2D[-1, :] = V_e
V2D[:, 0] = 0.0
V2D[:, -1] = 0.0

alpha = 1 / hr**2
beta = 1 / hz**2
denom = 2 * (alpha + beta)

tol_2d = 1e-4
error_max_2d = 1.0
iterations_2d = 0

while error_max_2d > tol_2d:
    error_max_2d = 0.0
    for j in range(1, N_r - 1):
        gamma = 1 / (2 * r_2d[j] * hr)
        for k in range(1, N_z - 1):
            V_old = V2D[j, k]
            
            V2D[j, k] = (1/denom) * (
                (alpha + gamma) * V2D[j+1, k] +
                (alpha - gamma) * V2D[j-1, k] +
                beta * (V2D[j, k+1] + V2D[j, k-1])
            )
            
            change = abs(V2D[j, k] - V_old)
            if change > error_max_2d:
                error_max_2d = change
    iterations_2d += 1

print(f"2D model converged in {iterations_2d} iterations.")

# 2D electric field components
E_r, E_z = np.gradient(-V2D, hr, hz)


# PLOTTING

plt.style.use('default')

# FIGURE 1: 1D Simulation
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(r, V_exact, 'k-', linewidth=3, label='Exact (Analytical)')
ax1.plot(r, V, 'r--', linewidth=2, label='Numerical (Gauss-Seidel)')
ax1.set_title(f'1D Electrostatic Potential $V(r)$ (Max error: {max_error:.2e} V)')
ax1.set_xlabel('Radius $r$ (cm)')
ax1.set_ylabel('Potential $V$ (V)')
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend()

ax2.plot(r, E_exact, 'k-', linewidth=3, label='Exact E')
ax2.plot(r, E_num, 'b--', linewidth=2, label='Numerical E (-dV/dr)')
ax2.set_title('1D Electric Field $E(r)$')
ax2.set_xlabel('Radius $r$ (cm)')
ax2.set_ylabel('Electric Field $E$ (V/cm)')
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend()

fig1.tight_layout()

# FIGURE 2: 2D Simulation
fig2, ax3 = plt.subplots(figsize=(12, 6))

Z, R = np.meshgrid(z_2d, r_2d)

contour_map = ax3.contourf(Z, R, V2D, levels=50, cmap='inferno')
cbar = plt.colorbar(contour_map, ax=ax3)
cbar.set_label('Potential (V)')

strm = ax3.streamplot(z_2d, r_2d, E_z, E_r, color='cyan', linewidth=1,
                      density=1.5, arrowsize=1.5)

ax3.set_title('2D Potential and Electric Field Streamlines $\\vec{E}$')
ax3.set_xlabel('Z-Axis - Length (cm)')
ax3.set_ylabel('R-Axis - Radius (cm)')

fig2.tight_layout()
plt.show()