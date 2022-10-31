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
import agentDynamics
import obstacles                                # Obstacle library
import lidar                                    # Lidar library
import numpy as np                              # Linear algebra library
import logGPIS                                  # Log GPIS library
from matern32GP import GaussianProcess as GP    # Gaussian Process with Matern Kernel
from quadprog import solve_qp                   # Solve quadratic programming
import matplotlib.pyplot as plt                 # Plot library
from matplotlib.patches import Wedge            # To draw wedge (Lidar fov))

class Simulation:
    def __init__(self):
        self.initialize()   # Initialize simulation

    # initialize
    def initialize(self):
        # Simulation Data
        self.T = 20.0        # Total simulation time
        self.dt = 0.01        # Sampling period
        self.t = 0           # Actual time

        # Agent data
        self.p0 = np.array([0.0, 0.0])          # Initial position
        self.pGoal = np.array([4.0, 4.0])       # Final goal
        self.pStory = []                        # List of positions
        self.p = self.p0                        # Actual position
        agentDynamics.AGENT_TYPE = 'SINGLE_INTEGRATOR_2D'       # Agent type
        agentDynamics.initialize_agent()

        # Lidar parameters
        lidar.fov_range = np.pi         # fov range
        lidar.min_distance = 0.1        # Minimum sensing distance
        lidar.max_distance = 1.0        # Minimum sensing distance 

        # Controller data
        self.kp = 1.5                    # Position gain

        # Plot variables
        self.uNom = np.zeros((2, ))      # Actual nominal input
        self.u = np.zeros((2, ))         # Actual input
        self.uNomStory = []              # List of nominal inputs
        self.uStory = []                 # List of inputs
        self.time_stamps = []            # List of time stamps

        # Control barrier function data
        self.alpha = 2.0                                        # Class K multiplier
        self.BARRIER_TYPE = 'LOG_GPIS'                          # Barrier function type ('GROUND_TRUTH', 'LOG_GPIS', 'POINTWISE')
        
        # GP
        self.edfGP = GP(2)                                      # Pointwise estimator
        logGPIS.resolution = 0.05

    # controller
    def controller(self):
        self.uNom = -self.kp * (self.p - self.pGoal)

    # safetyFilter
    def safetyFilter(self):
        # Build matrices
        # Return u

        if self.BARRIER_TYPE == 'GROUND_TRUTH':
            alpha_h = self.alpha * ( np.array(obstacles.allDistances(self.p)) )
            gradH = np.block([[ (self.p - obs['center']).reshape((1, 2)) / obstacles.distance(self.p, obs)] for obs in obstacles.obstacles ])
            self.u = solve_qp(np.eye(2), self.uNom, gradH.T, -alpha_h)[0]
        elif self.BARRIER_TYPE == 'POINTWISE':
            pass
        elif self.BARRIER_TYPE == 'LOG_GPIS':
            if logGPIS.getSamplesNumber() > 0:
                alpha_h = np.array([self.alpha * logGPIS.d(self.p)])
                gradH = logGPIS.gradd(self.p)
                self.u = solve_qp(np.eye(2), self.uNom, gradH.T, -alpha_h)[0]
            else:
                self.u = self.uNom
        else:
            self.u = self.uNom

    # run
    def run(self):
        while self.t < self.T:
            # Update agent state
            self.controller()                                                                   # Compute nominal control law
            self.safetyFilter()                                                                 # Filter control input
            self.p = self.p + agentDynamics.dynamics(self.p, self.u).flatten() * self.dt        # Update agent state

            # Simulate Lidar
            # Lidar
            readings, points = lidar.read(self.p, 0)        # Simulate Lidar

            train = False
            # Update Log GP Implicit Surface
            for point in points:                            # Loop trough all the detected points
                if logGPIS.getSamplesNumber() == 0:         # If the GP has no samples 
                    logGPIS.addSample(point)                # Add sample
                    train = True
                    continue                                # Go to next detected point
                new = True                                  # Assume the detected point is "far enough" from all the samples points of the GP
                if logGPIS.checkCollected(point):          # If already collected
                        new = False                         # Flag the point as already seen
                if new:                                     # If the point is new
                    logGPIS.addSample(point)                # Add sample
                    train = True
            if train:
                logGPIS.train()                             # Train GP
                print('train')


            # Advance simulation time
            self.t = self.t + self.dt

        # Update history variables
            self.pStory.append(self.p)              # Position
            self.uStory.append(self.u)              # Actual control law
            self.uNomStory.append(self.uNom)        # Nominal control law
            self.time_stamps.append(self.t)         # Time instants

            # DEBIG
            print(self.t)

    def plot(self):
        # Environment and trjectory
        fig_environment, ax_environment = plt.subplots()
        ax_environment.plot([item[0] for item in self.pStory], [item[1] for item in self.pStory], label='Agent trajectory')
        ax_environment.set_title('Environment')
        ax_environment.set_xlabel('x (m)')
        ax_environment.set_ylabel('y (m)')
        obstacles.plot(ax_environment)
        ax_environment.legend()

        plt.show()

# main
def main():
    sim = Simulation()      # Instatiate simulation

    sim.initialize()        # Initialize simulation
    sim.run()               # Run simulation
    sim.plot()

# Run main when the script is executed
if __name__ == '__main__':
    main()