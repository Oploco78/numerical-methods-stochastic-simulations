# -*- coding: utf-8 -*-
"""
2D Stochastic Random Walk, Explicit Finite-Difference Diffusion PDE & Entropy Evolution.
Simulates an ensemble of 2D isotropic random walkers, verifies diffusive scaling
(<r^2> = 4Dt), solves the macroscopic 2D diffusion equation via explicit FTCS finite
differences, compares both density distributions, and tracks coarse-grained Shannon entropy.
"""

import numpy as np
import random
import matplotlib.pyplot as plt

n = 100   # Number of steps
m = 1000  # Number of walkers

# a) Two-dimensional random walk

# Use n+1 to include the initial state (t = 0)
sum_r2 = np.zeros(n + 1, float)
positions = np.zeros((n + 1, m, 2))

random.seed(10)

for j in range(m):
    x = 0.0
    y = 0.0
    
    for i in range(1, n + 1):
        theta = random.random() * 2 * np.pi
        x += np.cos(theta)
        y += np.sin(theta)
        
        r2 = x**2 + y**2
        sum_r2[i] += r2
        positions[i, j, 0] = x
        positions[i, j, 1] = y

mean_r2 = sum_r2 / m

# Linear fit
time_rw = np.arange(n + 1)
coeffs = np.polyfit(time_rw[1:], mean_r2[1:], 1)
proportionality_const = coeffs[0]

print(f"Proportionality constant (Slope): {proportionality_const:.4f}")
print("For a 2D random walk with unit step length, the theoretical expected value is 1.0")

# b) Numerical integration of the 2D diffusion equation

D = 0.25             
L = 50               # Spatial domain [-50, 50]
dx = 1               
dy = 1               
dt = 0.2             
t_max = n            

x_space = np.arange(-L, L + 1, dx)
y_space = np.arange(-L, L + 1, dy)
NX = len(x_space)
NY = len(y_space)

rho = np.zeros((NY, NX), float)
center_x = NX // 2
center_y = NY // 2
rho[center_y, center_x] = m  # Initial condition: all walkers concentrated at the origin

save_times = [0, 50, 100]
results = {}
results[0] = rho.copy()

total_steps = int(t_max / dt)
rho_new = np.zeros_like(rho)
C = D * dt / (dx**2)

for t_step in range(1, total_steps + 1):
    rho_new[1:-1, 1:-1] = rho[1:-1, 1:-1] + C * (
        rho[1:-1, 2:] + rho[1:-1, :-2] +   
        rho[2:, 1:-1] + rho[:-2, 1:-1] -   
        4 * rho[1:-1, 1:-1]
    )
    rho = rho_new.copy()
    
    # Store snapshots at target time steps
    current_t = round(t_step * dt, 2)
    if current_t in save_times:
        results[current_t] = rho.copy()

# Compute 2D histograms of the stochastic walkers
hist_results = {}
# Define bin edges
space_bins = np.linspace(-L - dx/2, L + dx/2, 2*L + 2)

for t in save_times:
    H, _, _ = np.histogram2d(positions[t, :, 0], positions[t, :, 1], bins=(space_bins, space_bins))
    hist_results[t] = H.T  # Transpose so coordinates match imshow and rho (row=y, col=x)

# c) Entropy evolution

entropy = np.zeros(n + 1)
cell_edges = np.linspace(-L, L, 11)  # Creates a 10x10 coarse-grained grid

for t in range(n + 1):
    H_ent, _, _ = np.histogram2d(positions[t, :, 0], positions[t, :, 1], bins=(cell_edges, cell_edges))
    P_i = H_ent.flatten() / m
    P_i = P_i[P_i > 0]  # Filter out zero probabilities to avoid log(0)
    
    if len(P_i) > 0:
        entropy[t] = -np.sum(P_i * np.log(P_i))

# PLOTTING

# Figure 1: Trajectories, Mean Squared Displacement <r^2>, and Entropy
fig1, axs = plt.subplots(1, 3, figsize=(16, 5))

# Plot a.1) Sample trajectories
for j in range(5):  # Plot only 5 walkers for visual clarity
    axs[0].plot(positions[:, j, 0], positions[:, j, 1], '-', alpha=0.8, label=f'Walker {j+1}')
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].set_title('a.1) Sample Trajectories (5 Walkers)')
axs[0].grid(True)

# Plot a.2) Random walk <r^2> scaling
axs[1].plot(time_rw, mean_r2, "g-", linewidth=2, label='Simulation')
axs[1].plot(time_rw, proportionality_const * time_rw + coeffs[1], 'k--', label=f'Linear Fit (K={proportionality_const:.2f})')
axs[1].plot(time_rw, 4 * D * time_rw, 'r:', linewidth=2, label='Theoretical Diffusion (4Dt)')
axs[1].set_xlabel("Steps / Time (t)")
axs[1].set_ylabel(r"$\langle r^2 \rangle$")
axs[1].set_title('a.2) Diffusive Scaling')
axs[1].legend()
axs[1].grid(True)

# Plot c) Entropy evolution
axs[2].plot(time_rw, entropy, "b-", linewidth=2)
axs[2].set_xlabel("Steps / Time (t)")
axs[2].set_ylabel("Entropy S(t)")
axs[2].set_title('c) Shannon Entropy Evolution')
axs[2].grid(True)

fig1.tight_layout()

# Figure 2: 2D Comparison of Diffusion PDE vs. Stochastic Random Walk
fig2, axes = plt.subplots(2, 3, figsize=(15, 8))
fig2.suptitle(r'b) 2D Comparison: Diffusion PDE vs. Stochastic Random Walk')

for i, t in enumerate(save_times):
    
    # Row 1: Diffusion equation
    ax_dif = axes[0, i]
    im_dif = ax_dif.imshow(results[t], extent=[-L, L, -L, L], origin='lower', cmap='hot', interpolation='nearest')
    ax_dif.set_title(f"Numerical Diffusion t = {t}")
    if i == 0: ax_dif.set_ylabel("y")
    
    # Row 2: Random walk histogram
    ax_rw = axes[1, i]
    im_rw = ax_rw.imshow(hist_results[t], extent=[-L, L, -L, L], origin='lower', cmap='hot', interpolation='nearest')
    ax_rw.set_title(f"Random Walk t = {t}")
    ax_rw.set_xlabel("x")
    if i == 0: ax_rw.set_ylabel("y")

# Shared colorbar for Figure 2
fig2.subplots_adjust(right=0.9)
cbar_ax = fig2.add_axes([0.92, 0.15, 0.02, 0.7])
fig2.colorbar(im_dif, cax=cbar_ax)

# Figure 3: Central Cross-Section Comparison
fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.set_title(r'Band-Averaged Cross Section ($y \approx 0, t=100$)')

# Average across a 5-cell wide central band (center_y - 2 to center_y + 2)
half_width = 2
sim_band = np.mean(hist_results[100][center_y-half_width : center_y+half_width+1, :], axis=0)
diff_band = np.mean(results[100][center_y-half_width : center_y+half_width+1, :], axis=0)

ax3.plot(x_space, diff_band, 'r-', linewidth=2, label=r'Diffusion Equation (Band Average)')
ax3.plot(x_space, sim_band, 'bo', markersize=5, alpha=0.7, label='Monte Carlo Simulation (Band Average)')

ax3.set_xlabel("Position x")
ax3.set_ylabel("Mean Walker Density")
ax3.legend()
ax3.grid(True)