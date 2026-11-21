# -*- coding: utf-8 -*-
"""
3D Projectile Dynamics with Magnus Effect and Quadratic Air Drag.
Solves the coupled system of ODEs using explicit Forward Euler integration,
comparing trajectories with and without aerodynamic drag across spin regimes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

v0 = 70              # Initial velocity (m/s)
dt = 0.01            # Time step (s)
launch_angle = 15    # Launch angle (degrees)
g_const = 9.81       # Gravitational acceleration (m/s^2)
magnus_const = 0.25  # Effective Magnus parameter S0*w/m (s^-1)
drag_const = 0.005   # Quadratic drag coefficient for a golf ball (m^-1)

theta = np.radians(launch_angle)  # Convert launch angle to radians

# Simulate the shot with optional aerodynamic drag
def simulate_shot(spin_type, include_drag=False):
    
    pos = np.array([0.0, 0.0, 0.0])  # Initial position
    vel = np.array([v0 * np.cos(theta), 0.0, v0 * np.sin(theta)])  # Initial velocity vector
    
    if spin_type == 'hook':
        # Curves clockwise viewed from above (to the right from behind the ball).
        # Moving in +X, curving right requires a force in -Y, so w is along -Z (-k x i = -j)
        w = np.array([0.0, 0.0, -1.0])

    elif spin_type == 'slice':
        # Curves counter-clockwise viewed from above (to the left from behind the ball).
        # Moving in +X, curving left requires a force in +Y, so w is along +Z (k x i = j)
        w = np.array([0.0, 0.0, 1.0])
        
    else:  # Reference shot without spin
        w = np.array([0.0, 0.0, 0.0])

    x = [pos[0]]
    y = [pos[1]]
    z = [pos[2]]
    
    # Simulate until the ball hits the ground (z = 0)
    while pos[2] >= 0:
        
        g = np.array([0.0, 0.0, -g_const])
        magnus = magnus_const * np.cross(w, vel)
        
        # Add quadratic air drag if include_drag = True
        if include_drag:
            speed = np.linalg.norm(vel)
            drag = -drag_const * speed * vel
        else:
            drag = np.array([0.0, 0.0, 0.0])
            
        total_acc = g + magnus + drag
        
        # Euler method
        new_pos = pos + vel * dt
        new_vel = vel + total_acc * dt
        
        pos = new_pos
        vel = new_vel
        
        x.append(pos[0])
        y.append(pos[1])
        z.append(pos[2])
        
    return x, y, z

# Simulate all 3 spin cases with and without air drag
x_h, y_h, z_h = simulate_shot('hook', include_drag=False)
x_s, y_s, z_s = simulate_shot('slice', include_drag=False)
x_ns, y_ns, z_ns = simulate_shot('no spin', include_drag=False)

x_h_d, y_h_d, z_h_d = simulate_shot('hook', include_drag=True)
x_s_d, y_s_d, z_s_d = simulate_shot('slice', include_drag=True)
x_ns_d, y_ns_d, z_ns_d = simulate_shot('no spin', include_drag=True)

# Quantitative comparison of lateral deviation and horizontal range
print("LATERAL DEVIATION COMPARISON WITH AND WITHOUT AIR DRAG")
print(f"Lateral deviation with hook (No drag):   {y_h[-1]:.2f} m")
print(f"Lateral deviation with hook (With drag): {y_h_d[-1]:.2f} m")
print(f"Lateral deviation with slice (No drag):  {y_s[-1]:.2f} m")
print(f"Lateral deviation with slice (With drag):{y_s_d[-1]:.2f} m")
print(" ")
print("HORIZONTAL RANGE COMPARISON WITH AND WITHOUT AIR DRAG")
print(f"Range with hook (No drag):               {x_h[-1]:.2f} m")
print(f"Range with hook (With drag):             {x_h_d[-1]:.2f} m")
print(f"Range with slice (No drag):              {x_s[-1]:.2f} m")
print(f"Range with slice (With drag):            {x_s_d[-1]:.2f} m")

# Plotting
fig = plt.figure(figsize=(12, 5))

# 3D Plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(x_h, y_h, z_h, label='Hook without drag (Clockwise)', color='red')
ax1.plot(x_s, y_s, z_s, label='Slice without drag (Counter-clockwise)', color='blue')
ax1.plot(x_ns, y_ns, z_ns, label='No spin without drag', color='gray', linestyle='--')
ax1.plot(x_h_d, y_h_d, z_h_d, label='Hook with drag (Clockwise)', color='orange')
ax1.plot(x_s_d, y_s_d, z_s_d, label='Slice with drag (Counter-clockwise)', color='purple')
ax1.plot(x_ns_d, y_ns_d, z_ns_d, label='No spin with drag', color='black', linestyle='--')
ax1.set_xlabel('Distance X (m)')
ax1.set_ylabel('Lateral Deviation Y (m)')
ax1.set_zlabel('Height Z (m)')
ax1.set_title('3D Shot Trajectory With and Without Drag')
ax1.legend()

# 2D Top-Down Plot
ax2 = fig.add_subplot(122)
ax2.plot(x_h, y_h, label='Hook without drag', color='red')
ax2.plot(x_s, y_s, label='Slice without drag', color='blue')
ax2.plot(x_ns, y_ns, label='No spin without drag', color='gray', linestyle='--')
ax2.plot(x_h_d, y_h_d, label='Hook with drag', color='orange')
ax2.plot(x_s_d, y_s_d, label='Slice with drag', color='purple')
ax2.plot(x_ns_d, y_ns_d, label='No spin with drag', color='black', linestyle='--')
ax2.set_xlabel('Distance X (m)')
ax2.set_ylabel('Lateral Deviation Y (m)')
ax2.set_title('Top-Down View (XY Plane)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()

# 3D Animation
fig_anim = plt.figure()
ax3 = fig_anim.add_subplot(111, projection='3d')

# Fix axis limits to keep the camera steady
ax3.set_xlim(0, max(x_h))
ax3.set_ylim(min(y_h), max(y_s))
ax3.set_zlim(0, max(z_h))
ax3.set_xlabel('X (m)'); ax3.set_ylabel('Y (m)'); ax3.set_zlabel('Z (m)')
ax3.set_title('3D Animation: Hook & Slice With and Without Drag')

line_hook, = ax3.plot([], [], [], color='red', label='Hook without drag')
line_slice, = ax3.plot([], [], [], color='blue', label='Slice without drag')
line_hook_d, = ax3.plot([], [], [], color='orange', label='Hook with drag')
line_slice_d, = ax3.plot([], [], [], color='purple', label='Slice with drag')
ax3.legend()

# Frame update function
def update(i):
    line_hook.set_data(np.array(x_h[:i]), np.array(y_h[:i]))
    line_hook.set_3d_properties(np.array(z_h[:i]))
    
    line_slice.set_data(np.array(x_s[:i]), np.array(y_s[:i]))
    line_slice.set_3d_properties(np.array(z_s[:i]))
    
    i_hook_d = min(i, len(x_h_d) - 1)
    line_hook_d.set_data(np.array(x_h_d[:i_hook_d]), np.array(y_h_d[:i_hook_d]))
    line_hook_d.set_3d_properties(np.array(z_h_d[:i_hook_d]))
    
    i_slice_d = min(i, len(x_s_d) - 1)
    line_slice_d.set_data(np.array(x_s_d[:i_slice_d]), np.array(y_s_d[:i_slice_d]))
    line_slice_d.set_3d_properties(np.array(z_s_d[:i_slice_d]))
    
    return line_hook, line_slice, line_hook_d, line_slice_d

total_frames = min(len(x_h), len(x_s))
ani = FuncAnimation(fig_anim, update, frames=range(0, total_frames, 15), interval=30, blit=False)

plt.show()