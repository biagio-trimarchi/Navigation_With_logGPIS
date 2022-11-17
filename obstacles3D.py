# Obstacles utilities 
# For now just circles 
# TO DOs: - Implement JGK for convex shape EDF computations

# Import libraries
import math                         # Import some math utilities
import numpy as np                  # Linear algebra library
import matplotlib.pyplot as plt

# Obstacles list
obstacles = []                                                                   # List of obstacles, for now circle defined by center and radius
obstacles.append( {'center': np.array([2.0, 3.0, 1.0]), 'radius': 1.0} )         # Obstacle
obstacles.append( {'center': np.array([4.0, 2.5, 2.0]), 'radius': 0.5} )         # Obstacle
obstacles.append( {'center': np.array([2.0, 0.0, 1.5]), 'radius': 1.5} )         # Obstacle


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
    u = np.linspace(0, np.pi, 30)
    v = np.linspace(0, 2 * np.pi, 30)

    x = np.outer(np.sin(u), np.sin(v))
    y = np.outer(np.sin(u), np.cos(v))
    z = np.outer(np.cos(u), np.ones_like(v))
    for obs in obstacles:
        ax.plot_wireframe(  x*obs['radius'] + obs['center'][0], 
                            y*obs['radius'] + obs['center'][1],
                            z*obs['radius'] + obs['center'][2])

if __name__ == '__main__':
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plot(ax)
    plt.show()