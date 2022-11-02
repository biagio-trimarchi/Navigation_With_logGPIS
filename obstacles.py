# Obstacles utilities 
# For now just circles 
# TO DOs: - Implement JGK for convex shape EDF computations

# Import libraries
import math                         # Import some math utilities
import numpy as np                  # Linear algebra library
import matplotlib.pyplot as plt

# Obstacles list
obstacles = []                                                              # List of obstacles, for now circle defined by center and radius
obstacles.append( {'center': np.array([2.0, 2.5]), 'radius': 1.0} )         # Obstacle
obstacles.append( {'center': np.array([4.0, 3.0]), 'radius': 0.5} )         # Obstacle

def minDistance(p):
    # Distance from obstacles (only circles)
    d_min = math.inf                                                        # initialize minimum distance (infinite distance)
    for obstacle in obstacles:                                              # loop trough each obstacle
        d = np.linalg.norm(p - obstacle['center']) - obstacle['radius']     # Compute distance
        if d < d_min:                                                       # If distance from this obstacle is the minimum on
            d_min = d                                                       # Update minimum distance
    return d_min                                                            # Return minimum distance

def distance(p, obstacle):
    d = np.linalg.norm(p - obstacle['center']) - obstacle['radius']         # Compute distance
    return d

def allDistances(p):
    distances = []
    for obstacle in obstacles:
        d = np.linalg.norm(p - obstacle['center']) - obstacle['radius']     # Compute distance
        distances.append(d)                                                 # Append distance
    return distances

def plot(ax):
    circles = [plt.Circle(obs['center'], obs['radius'], color='k') for obs in obstacles]    # Circle patches
    for circle in circles:                                                                  # Loop
        ax.add_patch(circle)                                                                # Add patch to axes
