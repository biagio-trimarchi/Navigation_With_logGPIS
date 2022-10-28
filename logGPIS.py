# Library implementing the log GPIS
# (just a wrapper for some computations)

# Libraries
import math
from matern32GP import GaussianProcess as GP

lambda_whittle = 1.5                                        # Length scale of Whittle Kernal
regressor = GP(2)                                           # Log gaussian implicit surface
regressor.params.L = math.sqrt(2 * 3/2) / lambda_whittle    # Length scale of Matern 3_2 (See article, euristic choice)

def addSample(p):
    regressor.addSample(p, 1.0)

def d(p):
    return - math.log(regressor.posteriorMean(p)) / regressor.params.L
