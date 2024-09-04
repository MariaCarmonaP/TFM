"""Ploting PSO evolution"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm


# Example data for particle positions (replace this with your actual data)
ITER = 20
N_PARTICLES = 25
x_data = np.random.randn(ITER, N_PARTICLES)
z_data = np.random.randn(ITER, N_PARTICLES)
y_data = x_data + z_data

# Create a grid to plot the cost function surface
x = np.linspace(-5, 5, 100)
z = np.linspace(-5, 5, 100)
X, Z = np.meshgrid(x, z)
Y = cost_function(X, Z)

# Create the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Initialize the surface plot
surf = ax.plot_surface(X, Z, Y, cmap=cm.viridis, alpha=0.6, edgecolor='none')

# Scatter plot for particles
particles, = ax.plot([], [], [], 'ro', markersize=5)

# Set up the axes
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(0, 1)

ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('Cost (Y)')

# Animation update function
def update(num):
    ax.view_init(elev=30, azim=num * 360 / ITER)
    particles.set_data(x_data[num], z_data[num])
    particles.set_3d_properties(y_data[num], 'z')
    return particles,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=ITER, interval=100, blit=False)

# Save as GIF
ani.save('pso_evolution.gif', writer='imagemagick', fps=10)

plt.show()