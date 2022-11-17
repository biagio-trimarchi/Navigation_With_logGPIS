# Lidar utilities
# This library simulate the behaviour of the lidar

# Note: The sphere is parameterized as
#                                       x = r cos(psi) cos(theta)
#                                       y = r cos(psi) sin(theta)
#                                       z = r sin(psi) 
#                               with 
#                                       theta = [-pi, pi]       (Angle formed by the projection of the point with the x-axis in the xy-pleane) 
#                                       psi   = [-pi/2, pi/2]   (Angle formed by the point with the z-axis, computed from the xy-plane)



# Import libraries
import obstacles3D                          # Obstacles library
import numpy as np                          # Linear algebra library
import matplotlib.pyplot as plt             # Plot library
from matplotlib.patches import Wedge        # To draw wedge (Lidar fov)

# Lidar data
min_distance = 0.2          # Minimum Lidar Distance 
max_distance = 2.0          # Maximum Lidar Distance

fov_range_xy = np.pi/1.0                                    # Field of view (half angular span xy plane)
fov_range_z = np.pi/2.0                                     # Field of view (half angular span z axis)
resolution_distance = 100                                   # Number of tested point on each ray
resolution_angle_xy = int(fov_range_xy * 180 / np.pi/10)     # Number of readed angles (default = 1 each sessagesimal degree) 
resolution_angle_z = int(fov_range_z * 180 / np.pi/10)       # Number of readed angles (default = 1 each sessagesimal degree) 

# Lidar functions
def read(p, theta, psi):
    # Simulate Lidar and return both distance lectures and point on obstacles
    # p : position of the agent
    # theta : orientation of the agent (angle from lidar direction and x axis, counterclock-wise)

    ray = np.linspace(min_distance, max_distance, resolution_distance)                              # Testing distances for raycasting
    angles_theta = np.linspace(-fov_range_xy + theta, fov_range_xy + theta, resolution_angle_xy)    # Angles in the xy field of view
    angles_psi = np.linspace(-fov_range_z + psi, fov_range_z + psi, resolution_angle_z)             # Angles in the z field of view

    readings = np.ones((angles_theta.size, angles_psi.size))*max_distance           # Allocate array to store readings (and initialize all cells to max_distance)
    boundary_points = []                                                            # Allocate list to store detected boundary points

    i = 0                                                           # Auxiliary counter for theta angular direction
    for theta in angles_theta:                                      # Loop trough each theta angle in the field of view
        j = 0                                                           # Auxiliary counter for psi angular direction
        for psi in angles_psi:                                          # Loop trough each psi angle in the field of view
            for dist in ray:                                                        # Loop from min_distance to max_distance
                test_point = p + dist*np.array(
                                        [np.cos(psi)*np.cos(theta),
                                         np.cos(psi)*np.sin(theta),
                                         np.sin(psi)])                              # Compute current test point
                if obstacles3D.minDistance(test_point) < 0.01:                        # If the distance from the obstacles is less then a threshold
                    readings[i] = dist                                              # Store the distance in the readings array
                    boundary_points.append(test_point)                              # and append the new boundary point to the list
                    break
            j = j+1                                                         # and break innermost loop (go to next psi angle)
        i = i + 1                                                       # next thehta angle
    return readings, boundary_points                                # Return the readings and the detected boundary points