# Obstacles utilities 
# For now just circles 
# TO DOs: - Implement JGK for convex shape EDF computations

# Import libraries
import math                         # Import some math utilities
import numpy as np                  # Linear algebra library
import matplotlib.pyplot as plt     # Plot library

# Obstacles list
obstacles = []      # List of obstacles, for now circle defined by center and radius
obstacles.append( {'center': np.array([2.0, 2.0]), 'radius': 1.0} )
obstacles.append( {'center': np.array([4.0, 4.0]), 'radius': 0.5} )

def distance(obstacles, p):
    # Distance from obstacles (only circles)
    d_min = math.inf                 # initialize minimum distance (infinite distance)
    for obstacle in obstacles:  # loop trough each obstacle
        d = np.linalg.norm(p - obstacle['center']) - obstacle['radius'] # Compute distance
        if d < d_min:           # If distance from this obstacle is the minimum on
            d_min = d           # Update minimum distance
    return d_min                # Return minimum distance
