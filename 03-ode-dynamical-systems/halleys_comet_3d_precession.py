# -*- coding: utf-8 -*-
"""
3D Orbital Dynamics of Halley's Comet & Jovian Gravitational Precession.
Integrates the 3D two-body and restricted three-body (Sun-Jupiter-Comet) equations
of motion using the symplectic Euler-Cromer scheme, verifies energy and angular
momentum conservation, measures aphelion precession, and renders a 3D VPython animation.
"""

import numpy as np
from vpython import sphere, cone, rate, color, canvas, vector, mag
import matplotlib.pyplot as plt

# Simulation parameter definitions
GM_sun = 4 * np.pi**2       # G * Solar Mass in Astronomical Units (AU^3 / yr^2)
dt = 0.002                  # Time step (~0.73 days)
T = 76                      # Orbital period in years
total_time = 10 * T         # Total simulation duration (years)
NN = total_time / dt        # Number of steps
N = int(NN)

# Position and velocity arrays
x = np.zeros(N, float)
y = np.zeros(N, float)
z = np.zeros(N, float)
r = np.zeros(N, float)      # Radial distance from the Sun
vx = np.zeros(N, float)
vy = np.zeros(N, float)
vz = np.zeros(N, float)
t = np.zeros(N, float)

# Initial conditions in Astronomical Units (AU) and years
t[0] = 0
x[0] = 0.59
y[0] = 0
z[0] = 0
r[0] = np.sqrt(x[0]**2 + y[0]**2 + z[0]**2)

# Halley's orbital inclination (~162 degrees, retrograde orbit)
inclination = np.radians(162)
v_total = 11.47

vx[0] = 0.0
vy[0] = v_total * np.cos(inclination)
vz[0] = v_total * np.sin(inclination)

# Main integration loop using 3D Euler-Cromer method
for i in range(N - 1):
    vx[i+1] = vx[i] - GM_sun * (x[i] / r[i]**3) * dt
    vy[i+1] = vy[i] - GM_sun * (y[i] / r[i]**3) * dt
    vz[i+1] = vz[i] - GM_sun * (z[i] / r[i]**3) * dt
    
    x[i+1] = x[i] + vx[i+1] * dt
    y[i+1] = y[i] + vy[i+1] * dt
    z[i+1] = z[i] + vz[i+1] * dt
    
    r[i+1] = np.sqrt(x[i+1]**2 + y[i+1]**2 + z[i+1]**2)
    t[i+1] = t[i] + dt

# Part A: 3D Orbital Parameters
v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
v_max = np.max(v_mag)
aphelion = np.max(r)

print("PART A: Halley's 3D Orbital Parameters")
print(f"Initial velocity: {vy[0]:.2f} AU/yr")
print(f"Maximum orbital speed: {v_max:.2f} AU/yr")
print(f"Maximum distance from the Sun (Aphelion): {aphelion:.2f} AU\n")

print("PART B: Comparison with Pluto's Orbit")
print(f"Halley reaches {aphelion:.2f} AU, crossing Pluto's orbit (perihelion 29.7 AU, aphelion 49.3 AU).\n")

# Part C: Specific Energy and Angular Momentum Conservation
Ek = 0.5 * v_mag**2
Ep = -GM_sun / r
Et = Ek + Ep
L = np.sqrt((y*vz - z*vy)**2 + (z*vx - x*vz)**2 + (x*vy - y*vx)**2)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t[:int(2*T/dt)], Ek[:int(2*T/dt)], label='Kinetic Energy', color='blue')
plt.plot(t[:int(2*T/dt)], Ep[:int(2*T/dt)], label='Potential Energy', color='red')
plt.plot(t[:int(2*T/dt)], Et[:int(2*T/dt)], label='Total Energy', color='black', linestyle='--')
plt.title('Specific Orbital Energy (2 Orbits)')
plt.xlabel('Time (Years)')
plt.ylabel('Specific Energy (AU^2 / yr^2)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t[:int(2*T/dt)], L[:int(2*T/dt)], label='Angular Momentum', color='green')
plt.title('Specific Angular Momentum')
plt.xlabel('Time (Years)')
plt.ylabel('Specific Angular Momentum (AU^2 / yr)')
plt.ylim(0, np.max(L)*1.2)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Part D: 3D Orbital Precession Induced by Jupiter
mass_J_solar = 1.9e27 / 1.989e30
GM_jupiter = GM_sun * mass_J_solar
a_J = 5.2
omega_J = np.sqrt(GM_sun / a_J**3)

# Reset state arrays for perturbed simulation
x_p = np.zeros(N, float); y_p = np.zeros(N, float); z_p = np.zeros(N, float)
vx_p = np.zeros(N, float); vy_p = np.zeros(N, float); vz_p = np.zeros(N, float)
r_p = np.zeros(N, float)

x_p[0] = 0.59; y_p[0] = 0.0; z_p[0] = 0.0
r_p[0] = np.sqrt(x_p[0]**2 + y_p[0]**2 + z_p[0]**2)
vx_p[0] = 0.0; vy_p[0] = v_total * np.cos(inclination); vz_p[0] = v_total * np.sin(inclination)

for i in range(N - 1):
    # Jupiter's circular orbit in the XY plane (z = 0)
    xJ = a_J * np.cos(omega_J * t[i])
    yJ = a_J * np.sin(omega_J * t[i])
    zJ = 0.0
    dist_HJ = np.sqrt((x_p[i] - xJ)**2 + (y_p[i] - yJ)**2 + (z_p[i] - zJ)**2)
    
    ax = -GM_sun * (x_p[i] / r_p[i]**3) - GM_jupiter * ((x_p[i] - xJ) / dist_HJ**3)
    ay = -GM_sun * (y_p[i] / r_p[i]**3) - GM_jupiter * ((y_p[i] - yJ) / dist_HJ**3)
    az = -GM_sun * (z_p[i] / r_p[i]**3) - GM_jupiter * ((z_p[i] - zJ) / dist_HJ**3)
    
    vx_p[i+1] = vx_p[i] + ax * dt
    vy_p[i+1] = vy_p[i] + ay * dt
    vz_p[i+1] = vz_p[i] + az * dt
    
    x_p[i+1] = x_p[i] + vx_p[i+1] * dt
    y_p[i+1] = y_p[i] + vy_p[i+1] * dt
    z_p[i+1] = z_p[i] + vz_p[i+1] * dt
    r_p[i+1] = np.sqrt(x_p[i+1]**2 + y_p[i+1]**2 + z_p[i+1]**2)

# Numerical calculation of aphelion precession (Part D)
aphelion_angles = []
for i in range(1, N - 1):
    # Detect local radial maxima (Aphelia)
    if r_p[i] > r_p[i-1] and r_p[i] > r_p[i+1]:
        angle = np.degrees(np.arctan2(y_p[i], x_p[i]))
        aphelion_angles.append(angle)

print("PART D: Orbital Precession Due to Jupiter")
if len(aphelion_angles) > 1:
    total_precession = aphelion_angles[-1] - aphelion_angles[0]
    precession_per_orbit = total_precession / (len(aphelion_angles) - 1)
    print(f"Detected {len(aphelion_angles)} aphelia over {total_time} years.")
    print(f"Total orbital precession: {total_precession:.4f} degrees.")
    print(f"Average precession per orbit: {precession_per_orbit:.4f} degrees/orbit.\n")
else:
    print("Insufficient orbits detected to measure precession.\n")

# 3D Visualization (VPython + Dynamic Comet Tail)
canvas(title="3D Orbit of Halley's Comet with Dynamic Tail", x=0, y=0, width=800, height=800,
       center=vector(0, 0, 0), background=vector(0, 0, 0))

sun = sphere(pos=vector(0, 0, 0), color=color.yellow, radius=0.5, emissive=True)
halley = sphere(pos=vector(x_p[0], y_p[0], z_p[0]), color=color.cyan, radius=0.15, make_trail=True)
jupiter = sphere(pos=vector(a_J, 0, 0), color=color.orange, radius=0.3, make_trail=True)

# Initialize tail as invisible
tail = cone(pos=halley.pos, axis=vector(0, 0, 0), radius=0.1, color=color.white, opacity=0.6)

anim_steps = int(T / dt)
for i in range(0, anim_steps, 15):
    rate(100)
    current_pos = vector(x_p[i], y_p[i], z_p[i])
    halley.pos = current_pos
    
    # Dynamic comet tail logic (visible within 4.0 AU of the Sun)
    sun_dist = mag(current_pos)
    if sun_dist < 4.0:
        tail.visible = True
        tail.pos = current_pos
        # Tail points radially away from the Sun and grows closer to perihelion
        anti_solar_dir = current_pos / sun_dist
        tail_length = (4.0 - sun_dist) * 0.4
        tail.axis = anti_solar_dir * tail_length
    else:
        tail.visible = False
        tail.axis = vector(0, 0, 0)
    
    xJ_anim = a_J * np.cos(omega_J * t[i])
    yJ_anim = a_J * np.sin(omega_J * t[i])
    jupiter.pos = vector(xJ_anim, yJ_anim, 0)