from dis import dis
from math import inf, log, sqrt
from matern32GP import GaussianProcess as GP
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm                   # Color map

def distance(obstacles, p):
    d_min = inf
    for obstacle in obstacles:
        d = np.linalg.norm(p - obstacle['center']) - obstacle['radius']
        if d < d_min:
            d_min = d
    return d_min

obstacles = []
obstacles.append( {'center': np.array([2.0, 2.0]), 'radius': 1.0} )
obstacles.append( {'center': np.array([4.0, 4.0]), 'radius': 0.5} )

# Debug distance function
# print(distance(obstacles, np.array([4.0, 3.0])))

EDFgp = GP(2)
angles = np.linspace(0, 2*np.pi, 5)
training_points =  []
for theta in angles:
    point = obstacles[0]['center'] + np.array([obstacles[0]['radius'] * np.cos(theta), obstacles[0]['radius'] * np.sin(theta)])
    EDFgp.addSample(point, 1)
    training_points.append(point)

    point = obstacles[1]['center'] + np.array([obstacles[1]['radius'] * np.cos(theta), obstacles[1]['radius'] * np.sin(theta)])
    EDFgp.addSample(point, 1)
    training_points.append(point)

EDFgp.train()

print(EDFgp.data_x)
for x in EDFgp.data_x.T:
    print(x)

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
fig, ax = plt.subplots(1, 3)
cp = ax[0].contourf(X, Y, Zdist)
fig.colorbar(cp, ax[2]) # Add a colorbar to a plot
ax[0].set_title('Filled Contours Plot')
ax[0].set_xlabel('x (m)')
ax[0].set_ylabel('y (m)')

cp = ax[1].contourf(X, Y, Zgp)
for point in training_points:
    ax[1].plot(point[0], point[1], 'o')
ax[1].set_title('Filled Contours Plot')
ax[1].set_xlabel('x (m)')
ax[1].set_ylabel('y (m)')

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