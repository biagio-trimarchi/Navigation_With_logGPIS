from dis import dis
from math import inf, log, sqrt
from matern32GP import GaussianProcess as GP
import numpy as np
import matplotlib.pyplot as plt

def distance(obstacles, p):
    d_min = inf
    for obstacle in obstacles:
        d = np.linalg.norm(p - obstacle['center']) - obstacle['radius']
        if d < d_min:
            d_min = d
    return d_min

obstacles = []
obstacles.append( {'center': np.array([2.0, 2.0]), 'radius': 0.5} )
obstacles.append( {'center': np.array([4.0, 4.0]), 'radius': 0.5} )

# Debug distance function
# print(distance(obstacles, np.array([4.0, 3.0])))


lambda_whittle = 5
EDFgp = GP(2)
EDFgp.params.L = sqrt(2 * 3/2) / lambda_whittle
theta = 2.0*np.pi/10.0

training_points =  []
point = obstacles[0]['center'] + np.array([obstacles[0]['radius'] * np.cos(theta), obstacles[0]['radius'] * np.sin(theta)])
EDFgp.addSample(point, 1)
training_points.append(point)

point = obstacles[0]['center'] + np.array([obstacles[0]['radius'] * np.cos(0), obstacles[0]['radius'] * np.sin(0)])
EDFgp.addSample(point, 1)
training_points.append(point)

point = obstacles[0]['center'] + np.array([obstacles[0]['radius'] * np.cos(-theta), obstacles[0]['radius'] * np.sin(-theta)])
EDFgp.addSample(point, 1)
training_points.append(point)

point = obstacles[0]['center'] + np.array([obstacles[0]['radius'] * np.cos(-2*theta), obstacles[0]['radius'] * np.sin(-2*theta)])
EDFgp.addSample(point, 1)
training_points.append(point)

point = obstacles[1]['center'] + np.array([obstacles[1]['radius'] * np.cos(-np.pi/2), obstacles[1]['radius'] * np.sin(-np.pi/2)])
EDFgp.addSample(point, 1)
training_points.append(point)

point = obstacles[1]['center'] + np.array([obstacles[1]['radius'] * np.cos(-np.pi/2+theta), obstacles[1]['radius'] * np.sin(-np.pi/2+theta)])
EDFgp.addSample(point, 1)
training_points.append(point)

point = obstacles[1]['center'] + np.array([obstacles[1]['radius'] * np.cos(-np.pi/2 - theta), obstacles[1]['radius'] * np.sin(-np.pi/2 - theta)])
EDFgp.addSample(point, 1)
training_points.append(point)

EDFgp.train()
print(-log(EDFgp.posteriorMean(point)))

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
        Zdist[j][i] = distance(obstacles, point)
        Zgp[j][i] = - log(EDFgp.posteriorMean(point)) / lambda_whittle
        j = j + 1
    i = i + 1
del i, j

# Plot
fig, ax = plt.subplots()
cp = ax.contourf(X, Y, Zdist)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')

fig, ax = plt.subplots()
cp = ax.contourf(X, Y, Zgp)
fig.colorbar(cp) # Add a colorbar to a plot
for point in training_points:
    ax.plot(point[0], point[1], 'o')
ax.set_title('Filled Contours Plot')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')

plt.show()
