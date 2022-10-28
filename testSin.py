from matern32GP import GaussianProcess as GP
import numpy as np
import matplotlib.pyplot as plt

# Initialize GP
gp = GP(1)

# Test Sin Function
x_interval = np.linspace(-np.pi, np.pi/2, num = 50)
y_interval = np.sin(x_interval)

# Train points
x_train = np.linspace(-3*np.pi/4, np.pi/4, num = 10)
y_train = np.sin(x_train)
for x in x_train:
    gp.addSample(x, np.sin(x))

gp.train()
y_gaussian = np.zeros(x_interval.shape)
i = 0
for x in x_interval:
    y_gaussian[i] = gp.posteriorMean(x)
    i = i+1
del i

# Visualize Result

fig, ax = plt.subplots()
ax.plot(x_interval, y_interval)
ax.plot(x_interval, y_gaussian)
ax.plot(x_train, y_train, 'o')
plt.show()