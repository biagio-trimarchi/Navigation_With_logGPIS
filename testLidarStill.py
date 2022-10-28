from dis import dis
from math import inf, log, sqrt
from matern32GP import GaussianProcess as GP
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge        # To draw wedge (Lidar fov)
from matplotlib import cm                   # Color map

# LIDAR DATA AND FUNCTIONS
min_distance = 0.2          # Minimum Lidar Distance 
max_distance = 4.0          # Maximum Lidar DistancetestLidarStill copy

fov_range = np.pi/3.0       # Field of view

def lidar(p, theta):
    # Simulate Lidar and return both distance lectures and point on obstacles

    ray = np.linspace(min_distance, max_distance, 100)                      # Test distances for raycasting
    angles = np.linspace(-fov_range + theta, fov_range + theta, 30)        # Test angles for raycasting

    readings = np.ones(angles.shape)*max_distance
    boundary_points = []

    i = 0
    for phi in angles:
        for dist in ray:
            test_point = p + dist*np.array([np.cos(phi), np.sin(phi)])
            if distance(obstacles, test_point) < 0.01:
                readings[i] = dist
                boundary_points.append(test_point)
                break
        i = i + 1
    return readings, boundary_points

def draw_fov(p, theta, ax):
    # Add Lidar Field of View to existing plot
    left = 180.0/np.pi * (theta + fov_range)
    right = 180.0/np.pi * (theta - fov_range)
    fov = Wedge((p[0], p[1]), max_distance, right, left, color="r", alpha=0.3)
    not_fov = Wedge((p[0], p[1]), min_distance, right, left, color="k", alpha=1.0)
    ax.add_artist(fov)          # Field of view
    ax.add_artist(not_fov)      # Too close


# OBSTACLES DATA AND FUNCTIONS

# Obstacles
obstacles = []
obstacles.append( {'center': np.array([2.0, 2.0]), 'radius': 1.0} )
obstacles.append( {'center': np.array([4.0, 4.0]), 'radius': 0.5} )

def distance(obstacles, p):
    # Distance from obstacles
    d_min = inf
    for obstacle in obstacles:
        d = np.linalg.norm(p - obstacle['center']) - obstacle['radius']
        if d < d_min:
            d_min = d
    return d_min


# MAIN
# Agent Data
p = np.array([4.0, 1.0])          # Position
theta = 2*np.pi/4.0                 # Orientation

# Lidar
readings, points = lidar(p, theta)

# GP
EDFgp = GP(2)
for point in points:
    EDFgp.addSample(point, 1)
if EDFgp.params.N_samples > 0:
    EDFgp.train()
else:
    print("Error, no points")
    exit()

# Data for plots
xlist = np.linspace(-1.0, 5.0, 100)
ylist = np.linspace(-1.0, 5.0, 100)
X, Y = np.meshgrid(xlist, ylist)
Zdist = np.zeros((X.shape))
Zgp = np.zeros((X.shape))

i = 0
for xx in xlist:
    j = 0
    for yy in ylist:
        point = np.array([xx, yy])
        Zdist[i][j] = distance(obstacles, point)
        Zgp[i][j] = - log(EDFgp.posteriorMean(point)) / EDFgp.params.L
        j = j + 1
    i = i + 1
del i, j

# Plot 
# 2D
fig, axs = plt.subplots(1, 2)
cp = axs[0].contourf(X, Y, Zdist)               # Plot EDF
fig.colorbar(cp)                               # Add a colorbar to a plot
for point in points:
    axs[0].plot(point[0], point[1], 'o')        # Plot sample points
draw_fov(p, theta, axs[1])                      # Draw Lidar
axs[0].set_title('EDF')
axs[0].set_xlabel('x (m)')
axs[0].set_ylabel('y (m)')

cp = axs[1].contourf(X, Y, Zgp)
for point in points:
    axs[1].plot(point[0], point[1], 'o')
axs[1].set_title('Estimate')
axs[1].set_xlabel('x (m)')
axs[1].set_ylabel('y (m)')

# 3D
fig, axs = plt.subplots(1, 2, subplot_kw={"projection": "3d"})

# EDF
surf = axs[0].plot_surface(X, Y, Zdist, cmap=cm.coolwarm,linewidth=0, antialiased=False)  # Plot EDF
axs[0].set_title('EDF')
axs[0].set_xlabel('x (m)')
axs[0].set_ylabel('y (m)')

# Estimate
surf = axs[1].plot_surface(X, Y, Zgp, cmap=cm.coolwarm,linewidth=0, antialiased=False)  # Plot EDF
axs[1].set_title('Estimate')
axs[1].set_xlabel('x (m)')
axs[1].set_ylabel('y (m)')

# Color bar
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()