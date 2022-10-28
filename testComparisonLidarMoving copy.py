from dis import dis
from math import inf, log, sqrt
from matern32GP import GaussianProcess as GP
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge        # To draw wedge (Lidar fov)

# LIDAR DATA AND FUNCTIONS
min_distance = 0.2          # Minimum Lidar Distance 
max_distance = 4.0          # Maximum Lidar Distance

fov_range = np.pi/2.0       # Field of view

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
C0 = np.array([2.0, 2.0])           # Center of motion
radius = 2.0                        # Radius of motion        
theta0 = np.pi/2                    # Initial Orientation

# GP
logGPIS = GP(2)        # Log gaussian implicit surface
edfGP = GP(2)          # EDF measured from single samples


# Circular motion
theta = np.linspace(0, 2*np.pi, 60)
for th in theta:
    print("Theta = ", th)
    p = C0 + radius * np.array([ np.cos(th), np.sin(th)])

    # Lidar
    readings, points = lidar(p, theta0 + th)

    # Update Log GP Implicit Surface
    for point in points:
        if logGPIS.params.N_samples == 0:
            logGPIS.addSample(point, 1)
            continue
        new = True
        for tr_point in logGPIS.data_x.T:
            if np.linalg.norm(tr_point - point) < 0.2:
                new = False
        if new:
            logGPIS.addSample(point, 1)

    # Update EDF GP
    


print('Travel finished')
print("Training...")
EDFgp.train()
print("Trained")

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
fig, axs = plt.subplots(1, 2)
cp = axs[0].contourf(X, Y, Zdist)               # Plot EDF
fig.colorbar(cp)                               # Add a colorbar to a plot
for point in EDFgp.data_x.T:
    axs[0].plot(point[0], point[1], 'o')        # Plot sample points
# draw_fov(p, theta, axs[0])                      # Draw Lidar
axs[0].set_title('Filled Contours Plot')
axs[0].set_xlabel('x (m)')
axs[0].set_ylabel('y (m)')

cp = axs[1].contourf(X, Y, Zgp)
for point in EDFgp.data_x.T:
    axs[1].plot(point[0], point[1], 'o')
axs[1].set_title('Filled Contours Plot')
axs[1].set_xlabel('x (m)')
axs[1].set_ylabel('y (m)')

plt.show()