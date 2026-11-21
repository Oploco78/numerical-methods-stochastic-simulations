# -*- coding: utf-8 -*-
"""
2D Ising Model Monte Carlo Simulation & Magnetic Hysteresis Analysis.
Implements the Metropolis algorithm with periodic boundary conditions to simulate
magnetic hysteresis loops across temperature regimes and Monte Carlo relaxation times,
quantifying dissipated energy via trapezoidal numerical integration.
"""

import numpy as np
import matplotlib.pyplot as plt
from random import random
import math

# Model parameters
N = 8
J = 1
mu = 1

H_max = 5
delta_H = 0.5

# Magnetic field sweep vector (descending and ascending branches)
H_down = np.arange(H_max, -H_max - delta_H/2, -delta_H)
H_up = np.arange(-H_max + delta_H, H_max + delta_H/2, delta_H)
H_cycle = np.concatenate((H_down, H_up))

temperature_list = [0.25, 1.5, 3.0]
mc_steps_list = [5, 10, 20, 50, 100, 200, 500]

def montecarlo(spin, T, H):
    for i in range(N):
        for j in range(N):
            # Nearest-neighbor interaction with periodic boundary conditions
            neighbors = spin[i, (j-1)%N] + spin[i, (j+1)%N] + spin[(i-1)%N, j] + spin[(i+1)%N, j]
            delta_E = 2.0 * spin[i, j] * (J * neighbors + mu * H)
            
            # Metropolis acceptance criterion
            if delta_E <= 0:
                spin[i, j] *= -1
            elif random() < math.exp(-delta_E / T):
                spin[i, j] *= -1
    return spin

def compute_magnetization(spin):
    return np.sum(spin)


lattice_snapshots = {}
areas_by_T = {T: [] for T in temperature_list}

for T in temperature_list:
    plt.figure(figsize=(8, 6))
    captured_lattice = np.zeros((N, N))
    
    for mc_steps in mc_steps_list:
        spin = np.ones([N, N], int)
        mean_magnetizations = []
        
        for H in H_cycle:
            M_accumulated = 0
            measurement_steps = max(1, mc_steps // 2)
            
            for step in range(mc_steps):
                spin = montecarlo(spin, T, H)
                
                # Accumulate magnetization only during the second half to ensure thermal equilibrium
                if step >= (mc_steps - measurement_steps):
                    M_accumulated += compute_magnetization(spin) / (N*N)
            
            M = M_accumulated / measurement_steps
            mean_magnetizations.append(M)
            
            # Capture spin lattice configuration when magnetic field crosses zero
            if mc_steps == 100 and abs(H) < 1e-5:
                captured_lattice = np.copy(spin)
                
        # Numerical integration to compute hysteresis loop area
        area = np.abs(np.trapezoid(mean_magnetizations, H_cycle))
        areas_by_T[T].append(area)
        
        # Filter legend entries to avoid cluttering the plot
        if mc_steps in [5, 20, 100, 500]:
            plt.plot(H_cycle, mean_magnetizations, marker='o', linestyle='-', alpha=0.6, label=f'MC Steps={mc_steps}')
        else:
            plt.plot(H_cycle, mean_magnetizations, marker='o', linestyle='-', alpha=0.2)
            
    lattice_snapshots[T] = captured_lattice
    
    plt.title(f'Hysteresis Loop ({N}x{N} Lattice, T={T})')
    plt.xlabel('Magnetic Field (H)')
    plt.ylabel('Magnetization per Spin (M)')
    plt.grid(True)
    plt.legend()
    plt.show()


# Plot loop area versus Monte Carlo relaxation time
plt.figure(figsize=(8, 6))
colors = ['b', 'g', 'r']
for i, T in enumerate(temperature_list):
    plt.plot(mc_steps_list, areas_by_T[T], marker='s', color=colors[i], linestyle='--', linewidth=2, label=f'T={T}')

plt.title('Dissipated Energy (Hysteresis Loop Area) vs. Relaxation Time')
plt.xlabel('Monte Carlo Steps (per H value)')
plt.ylabel('Hysteresis Loop Area')
plt.xscale('log')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.show()


# Plot spin lattice snapshots at H = 0
fig, axes = plt.subplots(1, len(temperature_list), figsize=(12, 4))
for i, T in enumerate(temperature_list):
    ax = axes[i]
    im = ax.imshow(lattice_snapshots[T], cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title(f'Lattice State at T={T}\n(H=0)')
    ax.axis('off')
plt.tight_layout()
plt.show()