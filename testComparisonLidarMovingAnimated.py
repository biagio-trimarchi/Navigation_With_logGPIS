from math import inf, log, sqrt
from matern32GP import GaussianProcess as GP
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge        # To draw wedge (Lidar fov)
from celluloid import Camera                # Camera for animated plots

# LIDAR DATA AND FUNCTIONS
min_distance = 0.2          # Minimum Lidar Distance 
max_distance = 2.0          # Maximum Lidar Distance

fov_range = np.pi/3.0       # Field of view (half angular span)

def lidar(p, theta):
    # Simulate Lidar and return both distance lectures and point on obstacles
    # p : position of the agent
    # theta : orientation of the agent (angle from lidar direction and x axis, counterclock-wise)

    ray = np.linspace(min_distance, max_distance, 100)                  # Testing distances for raycasting
    angles = np.linspace(-fov_range + theta, fov_range + theta, 30)     # Angles in the field of view

    readings = np.ones(angles.shape)*max_distance                       # Allocate array to store readings (and initialize all cells to max_distance)
    boundary_points = []                                                # Allocate list to store detected boundary points

    i = 0                                                               # Auxiliary counter for angular direction
    for phi in angles:                                                  # Loop trough each angle in the field of view
        for dist in ray:                                                # Loop from min_distance to max_distance
            test_point = p + dist*np.array([np.cos(phi), np.sin(phi)])  # Compute current test point
            if distance(obstacles, test_point) < 0.01:                  # If the distance from the obstacles is less then a threshold
                readings[i] = dist                                      # Store the distance in the readings array
                boundary_points.append(test_point)                      # and append the new boundary point to the list
                break                                                   # and break innermost loop (go to next angle)
        i = i + 1                                                       # next angle
    return readings, boundary_points                                    # Return the readings and the detected boundary points

def draw_fov(p, theta, ax):
    # Add Lidar Field of View to existing plot
    # p : position of the agent
    # theta : orientation of the agent (angle from lidar direction and x axis, counterclock-wise)
    # ax : plot on which draw the fov

    left = 180.0/np.pi * (theta + fov_range)                                    # Left side of the cone
    right = 180.0/np.pi * (theta - fov_range)                                   # Rigth side of the cone
    fov = Wedge((p[0], p[1]), max_distance, right, left, color="r", alpha=0.3)      # Build patch 
    not_fov = Wedge((p[0], p[1]), min_distance, right, left, color="k", alpha=1.0)  # Smaller patch of unseeable points
    ax.add_artist(fov)          # Add field of view
    ax.add_artist(not_fov)      # Remove too close points


# OBSTACLES DATA AND FUNCTIONS

# Obstacles
obstacles = []      # List of obstacles, for now circle defined by center and radius
obstacles.append( {'center': np.array([2.0, 2.0]), 'radius': 1.0} )
obstacles.append( {'center': np.array([4.0, 4.0]), 'radius': 0.5} )

def distance(obstacles, p):
    # Distance from obstacles (only circles)
    d_min = inf                 # initialize minimum distance (infinite distance)
    for obstacle in obstacles:  # loop trough each obstacle
        d = np.linalg.norm(p - obstacle['center']) - obstacle['radius'] # Compute distance
        if d < d_min:           # If distance from this obstacle is the minimum on
            d_min = d           # Update minimum distance
    return d_min                # Return minimum distance


# MAIN
# In this simulation, the agent circumnavigate an obstacle and collect samples to train 
# a log-GPIS and a pointwise EDF estimate. We then plot and compare the results

# Data for plots
xlist = np.linspace(-1.0, 5.0, 20)      # x axis values
ylist = np.linspace(-1.0, 5.0, 20)      # y axis values
X, Y = np.meshgrid(xlist, ylist)        # Mesh grid for plot
Zdist = np.zeros((X.shape))             # EDF grid
Zgp1 = np.zeros((X.shape))              # log GPIS grid
Zgp2 = np.zeros((X.shape))              # GP pointwise grid

gridsize = (1, 3)                       # Grid of the figure 
fig = plt.figure(figsize=(12, 12))       # Setuo figure
ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=1, rowspan=1)          # Real EDF plot
ax2 = plt.subplot2grid(gridsize, (0, 1), colspan=1, rowspan=1)          # log GPIS plot
ax3 = plt.subplot2grid(gridsize, (0, 2), colspan=1, rowspan=1)          # Pointwise GP plot        

camera = Camera(fig)    # Camera for animation
color_flag = True

# Agent Data
C0 = np.array([2.0, 2.0])           # Center of motion
radius = 2.0                        # Radius of motion        
theta0 = np.pi/2                    # Initial Orientation

# GP
lambda_whittle = 1.5                                # Length scale of Whittle Kernal
logGPIS = GP(2)                                     # Log gaussian implicit surface
logGPIS.params.L = sqrt(2 * 3/2) / lambda_whittle   # Length scale of Matern 3_2 (See article, euristic choice)
edfGP = GP(2)                                       # EDF measured from single samples
edfGP.params.L = 0.25                                # Length scale for the edgGP


# Circular motion
theta = np.linspace(0, 2*np.pi, 60)                 # Angular positions along the chosen circumference
for th in theta:                                    # Loop trough the circumference
    print("Theta = ", th)                         # Debug
    p = C0 + radius * np.array([ np.cos(th), np.sin(th)])   # Actual position

    # Lidar
    readings, points = lidar(p, theta0 + th)        # Simulate Lidar (theta0 + th is the orientation of the agent)

    # Update Log GP Implicit Surface
    for point in points:                            # Loop trough all the detected points
        if logGPIS.params.N_samples == 0:           # If the GP has no samples 
            logGPIS.addSample(point, 1)             # Add sample
            continue                                # Go to next detected point
        new = True                                  # Assume the detected point is "far enough" from all the samples points of the GP
        for tr_point in logGPIS.data_x.T:           # Loop trough all the sample point of the GP
            tr_point = tr_point * logGPIS.params.L  # Scale back the sample point (The implemented GP internally scales the sample points)
            if np.linalg.norm(tr_point - point) < 0.25:  # If the points are "near"
                new = False                             # Flag the point as already seen
        if new:                                     # If the point is new
            logGPIS.addSample(point, 1)             # Add sample
    logGPIS.train()             # Train GP

    # Update EDF GP
    if edfGP.params.N_samples == 0:             # If the GP has no samples
        edfGP.addSample(p, min(min(readings), max_distance))    # Add sample
    else:                                       # otherwise
        new = True                              # Assume the detected point is "far enough" from all the samples points of the GP      
        for tr_point in edfGP.data_x.T:         # Loop trough all the sample point of the GP
            tr_point = tr_point * edfGP.params.L    # Scale back the sample point (The implemented GP internally scales the sample points)
            if np.linalg.norm(tr_point - p) < 0.25:  # If the points are "near"
                new = False                         # Flag the point as already seen
        if new:                                     # If the point is new
            edfGP.addSample(p, min(min(readings), max_distance))    # Add sample
    edfGP.train()               # Train GP


    i = 0                                   # x grid cell
    for xx in xlist:                        # Loop trough x
        j = 0                               # y grid cell
        for yy in ylist:                    # Loop trough y
            point = np.array([xx, yy])      # Store grid point
            Zdist[j][i] = distance(obstacles, point)                                # Compute EDF
            Zgp1[j][i] = - log(logGPIS.posteriorMean(point)) / logGPIS.params.L     # Compute log GPIS
            Zgp2[j][i] = edfGP.posteriorMean(point)                                 # Compute GP pointwise
            j = j + 1                       # Next y grid
        i = i + 1                           # Next x grid
    del i, j                                # Deallocate memory

    # Plots
    max_value = max(Zdist.max(), Zgp1.max(), Zgp2.max())   # Min value for colorbar
    min_value = min(Zdist.min(), Zgp1.min(), Zgp2.min())   # Max value for colorbar

    # EDF plot
    cp = ax1.contourf(X, Y, Zdist, vmin=min_value, vmax=max_value)          # Color plot
    draw_fov(p, theta0+th, ax3)                                             # fov
    ax1.set_title('EDF')                                                    # Title
    ax1.set_xlabel('x (m)')                                                 # x label
    ax1.set_ylabel('y (m)')                                                 # y label

    # log GPIS plot
    ax2.contourf(X, Y, Zgp1, vmin=min_value, vmax=max_value)                # Color plot
    draw_fov(p, theta0+th, ax2)                                             # fov
    for point in logGPIS.data_x.T:                                          # Loop trough the data in GP
        point = point * logGPIS.params.L                                    # Scale back the data
        ax2.plot(point[0], point[1], 'o')                                   # Plot sample point
    ax2.set_title('log GPIS')                                               # Title
    ax2.set_xlabel('x (m)')                                                 # x label
    ax2.set_ylabel('y (m)')                                                 # y label

    # Pointwise GP plot
    ax3.contourf(X, Y, Zgp2, vmin=min_value, vmax=max_value)                # Color plot
    draw_fov(p, theta0+th, ax3)                                             # fov
    for point in edfGP.data_x.T:                                            # Loop trough the data in GP
        point = point * edfGP.params.L                                      # Scale back the data
        ax3.plot(point[0], point[1], 'o')                                   # Plot sample point
    ax3.set_title('GP pointwise')                                           # Title
    ax3.set_xlabel('x (m)')                                                 # x label
    ax3.set_ylabel('y (m)')                                                 # y label

    # Colorbar
    # cb = plt.colorbar(cp, ax = [ax1, ax2, ax3], location = 'bottom')             # Colorbar
    camera.snap()

print("Creating animation")
animation = camera.animate()
# animation.save('animation.gif', writer='imagemagick')
animation.save('animation.mp4')

print("Done")