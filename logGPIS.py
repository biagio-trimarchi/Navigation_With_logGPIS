# Library implementing the log GPIS
# (just a wrapper for some computations)

# Libraries
import math
from matern32GP import GaussianProcess as GP
import numpy as np

lambda_whittle = 1.5                                        # Length scale of Whittle Kernal
regressor = GP(2)                                           # Log gaussian implicit surface
regressor.params.L = math.sqrt(2 * 3/2) / lambda_whittle    # Length scale of Matern 3_2 (See article, euristic choice)
resolution = 0.2                                            # Resolution of state space (two point are considered the same if their distance is less than the resolution)

def addSample(p):
    # Add sample to log GPIS
    regressor.addSample(p, 1.0)

def train():
    regressor.train()

def d(p):
    # Compute estimate distance 
    return - math.log(regressor.posteriorMean(p)) / regressor.params.L

def gradd(p):
    # Compute gradient of estimated distance field
    dd = d(p)
    gradd = regressor.gradientPosterionMean(p)
    return -gradd / (lambda_whittle * dd)

def getSamplesNumber():
    return regressor.params.N_samples

def checkCollected(p):
    for tr_point in regressor.data_x.T:
        tr_point = tr_point * regressor.params.L        # Scale back the sample point (The implemented GP internally scales the sample points)
        if np.linalg.norm(tr_point - p) < resolution:         # If the points are "near"
            return True                                 # Return True
            
    return False                                        # Otherwise return False
