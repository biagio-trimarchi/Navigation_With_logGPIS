# Little scripts to test the Control Barrier Function behaviour 
# with a simple control law using different proxy for the 
# barrier function
# In particular: 
#               - Ground truth (EDF)
#               - log GPIS     
#               - Pointwise Gaussian Process
# The agent dynamics is a simple single integrator controlled
# in velocity

# Libraries
import obstacles                                # Obstacle library
import lidar                                    # Lidar library
import numpy as np                              # Linear algebra library
import logGPIS                                  # Log GPIS library
from matern32GP import GaussianProcess as GP    # Gaussian Process with Matern Kernel

# Main
def main():
    pass

# Run main when the script is executed
if __name__ == '__main__':
    main()