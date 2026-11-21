# -*- coding: utf-8 -*-
"""
Chaotic Dynamics of a Driven Damped Pendulum: Lyapunov Exponents & Poincare Sections.
Integrates the non-linear equations of motion via the semi-implicit Euler-Cromer method,
estimates the maximal Lyapunov exponent via linear regression of trajectory divergence,
and reconstructs the strange attractor in phase space alongside its Poincare section.
"""

import numpy as np
import matplotlib.pyplot as plt

l = 9.8
g = 9.8
q = 0.5
F_drive = 1.2
Omega_drive = 2/3

t_max = 150.0
dt = 0.01        
steps = int(t_max / dt)
t = np.linspace(0, t_max, steps)

def simulate_pendulums(q1, q2, delta_theta_0):
    
    theta1 = np.zeros(steps)
    omega1 = np.zeros(steps)

    theta2 = np.zeros(steps)
    omega2 = np.zeros(steps)
    
    theta1[0] = 0.20
    omega1[0] = 0.0
    
    theta2[0] = theta1[0] + delta_theta_0
    omega2[0] = 0.0
    
    for i in range(steps - 1):
        
        acc1 = -(g/l) * np.sin(theta1[i]) - q1 * omega1[i] + F_drive * np.sin(Omega_drive * t[i])
        omega1[i+1] = omega1[i] + acc1 * dt
        theta1[i+1] = theta1[i] + omega1[i+1] * dt
        
        acc2 = -(g/l) * np.sin(theta2[i]) - q2 * omega2[i] + F_drive * np.sin(Omega_drive * t[i])
        omega2[i+1] = omega2[i] + acc2 * dt
        theta2[i+1] = theta2[i] + omega2[i+1] * dt
        
    delta_theta = np.abs(theta2 - theta1)
    delta_theta = np.maximum(delta_theta, 1e-20)
    log_delta_theta = np.log(delta_theta)
    
    return log_delta_theta

# Part a) Lyapunov Exponent Estimation

q = 0.5
delta_theta_0 = 1e-5
log_delta_theta_a = simulate_pendulums(q, q, delta_theta_0)

# Fit a straight line only over the linear growth regime to estimate the Lyapunov exponent
lin_growth_a = (t > 10) & (t < 80)
fit_coeffs_a = np.polyfit(t[lin_growth_a], log_delta_theta_a[lin_growth_a], 1)
lyap_exp_a = fit_coeffs_a[0]


# Part b) Sensitivity to Damping Perturbation

q1 = 0.5
q2 = 0.501
log_delta_theta_b = simulate_pendulums(q1, q2, delta_theta_0)

lin_growth_b = (t > 5) & (t < 30)
fit_coeffs_b = np.polyfit(t[lin_growth_b], log_delta_theta_b[lin_growth_b], 1)
lyap_exp_b = fit_coeffs_b[0]


# Plots

plt.figure(figsize=(15, 5))

# Plot a)

plt.subplot(1, 2, 1)
plt.plot(t, log_delta_theta_a, color='green', linewidth=1)
plt.plot(t[lin_growth_a], fit_coeffs_a[0]*t[lin_growth_a] + fit_coeffs_a[1], 'r--', linewidth=2, label=f'Linear fit ($\\lambda \\approx$ {lyap_exp_a:.3f})')
plt.title("a) Identical Damping ($q_1 = q_2 = 0.5$)")
plt.xlabel("Time (t)")
plt.ylabel(r"$\log(\Delta\theta)$")
plt.legend()
plt.grid(True)

# Plot b)

plt.subplot(1, 2, 2)
plt.plot(t, log_delta_theta_b, color='orange', linewidth=1)
plt.plot(t[lin_growth_b], fit_coeffs_b[0]*t[lin_growth_b] + fit_coeffs_b[1], 'r--', linewidth=2, label=f'Linear fit ($\\lambda \\approx$ {lyap_exp_b:.3f})')
plt.title("b) Perturbed Damping ($q_1 = 0.5, q_2 = 0.501$)")
plt.xlabel("Time (t)")
plt.ylabel(r"$\log(\Delta\theta)$")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Phase Space Trajectory and Poincare Section

# Run a long simulation for a single pendulum to collect sufficient points
t_max_poincare = 5000.0  
steps_poincare = int(t_max_poincare / dt)
t_poincare = np.linspace(0, t_max_poincare, steps_poincare)

theta_p = np.zeros(steps_poincare)
omega_p = np.zeros(steps_poincare)
theta_p[0] = 0.20
omega_p[0] = 0.0

# Simulate
for i in range(steps_poincare - 1):
    acc = -(g/l) * np.sin(theta_p[i]) - q * omega_p[i] + F_drive * np.sin(Omega_drive * t_poincare[i])
    omega_p[i+1] = omega_p[i] + acc * dt
    theta_p[i+1] = theta_p[i] + omega_p[i+1] * dt

# Wrap angle onto the interval [-pi, pi]
theta_p_mod = ((theta_p + np.pi) % (2 * np.pi)) - np.pi

# Discard initial transient regime (first 100 seconds)
transient = int(100 / dt)

# Compute indices for the Poincare Section (sampled at multiples of driving period T)
T = 2 * np.pi / Omega_drive
poincare_indices = []
total_periods = int(t_max_poincare / T)

for n in range(total_periods):
    current_time = n * T
    if current_time > 100:  # Skip initial transient
        idx = int(current_time / dt)
        if idx < steps_poincare:
            poincare_indices.append(idx)
            
plt.figure(figsize=(15, 6))

# Phase Space
plt.subplot(1, 2, 1)
# Use small marker size and transparency to highlight the strange attractor structure
plt.scatter(theta_p_mod[transient:], omega_p[transient:], color='darkblue', s=0.5, alpha=0.3)
plt.title("Strange Attractor (Phase Space)")
plt.xlabel(r"Angle $\theta$ (rad)")
plt.ylabel(r"Angular Velocity $\omega$ (rad/s)")
plt.grid(True)

# Poincare Section
plt.subplot(1, 2, 2)
plt.scatter(theta_p_mod[poincare_indices], omega_p[poincare_indices], color='red', s=4)
plt.title("Poincaré Section")
plt.xlabel(r"Angle $\theta$ (rad)")
plt.ylabel(r"Angular Velocity $\omega$ (rad/s)")
plt.grid(True)

plt.tight_layout()
plt.show()