# Lidar utilities
# This library simulate the behaviour of the lidar

# Import libraries
import obstacles                            # Obstacles library
import numpy as np                          # Linear algebra library
import matplotlib.pyplot as plt             # Plot library
from matplotlib.patches import Wedge        # To draw wedge (Lidar fov)

# Lidar data
min_distance = 0.2          # Minimum Lidar Distance 
max_distance = 2.0          # Maximum Lidar Distance

fov_range = np.pi/1.0                               # Field of view (half angular span)
resolution_distance = 100                           # Number of tested point on each ray
resolution_angle = int(fov_range * 180 / np.pi)     # Number of readed angles (default = 1 each sessagesimal degree) 

# Lidar functions
def read(p, theta):
    # Simulate Lidar and return both distance lectures and point on obstacles
    # p : position of the agent
    # theta : orientation of the agent (angle from lidar direction and x axis, counterclock-wise)

    ray = np.linspace(min_distance, max_distance, resolution_distance)              # Testing distances for raycasting
    angles = np.linspace(-fov_range + theta, fov_range + theta, resolution_angle)   # Angles in the field of view

    readings = np.ones(angles.shape)*max_distance                       # Allocate array to store readings (and initialize all cells to max_distance)
    boundary_points = []                                                # Allocate list to store detected boundary points

    i = 0                                                               # Auxiliary counter for angular direction
    for phi in angles:                                                  # Loop trough each angle in the field of view
        for dist in ray:                                                # Loop from min_distance to max_distance
            test_point = p + dist*np.array([np.cos(phi), np.sin(phi)])  # Compute current test point
            if obstacles.minDistance(test_point) < 0.01:               # If the distance from the obstacles is less then a threshold
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

    left = 180.0/np.pi * (theta + fov_range)                                        # Left side of the cone
    right = 180.0/np.pi * (theta - fov_range)                                       # Rigth side of the cone
    fov = Wedge((p[0], p[1]), max_distance, right, left, color="r", alpha=0.3)      # Build patch 
    not_fov = Wedge((p[0], p[1]), min_distance, right, left, color="k", alpha=1.0)  # Smaller patch of unseeable points
    ax.add_artist(fov)                                                              # Add field of view
    ax.add_artist(not_fov)                                                          # Remove too close points