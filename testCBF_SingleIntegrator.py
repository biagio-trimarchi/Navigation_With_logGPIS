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
        self.T = 10.0        # Total simulation time
        self.dt = 0.01        # Sampling period
        self.t = 0           # Actual time

        # Agent data
        self.p0 = np.array([0.0, 0.0])          # Initial position
        self.pGoal = np.array([4.0, 4.0])       # Final goal
        self.pStory = []                        # List of positions
        self.p = self.p0                        # Actual position
        agentDynamics.initialize_agent()

        # Controller data
        self.kp = 0.5                    # Position gain

        # Plot variables
        self.uNom = np.zeros((2, ))      # Actual nominal input
        self.u = np.zeros((2, ))         # Actual input
        self.uNomStory = []              # List of nominal inputs
        self.uStory = []                 # List of inputs
        self.time_stamps = []            # List of time stamps

        # Control barrier function data
        self.alpha = 2.0                                         # Class K multiplier
        self.BARRIER_TYPE = 'GROUND_TRUTH'                       # Barrier function type ('GROUND_TRUTH', 'LOG_GPIS', 'POINTWISE')
        agentDynamics.AGENT_TYPE = 'SINGLE_INTEGRATOR_2D'        # Agent type
        self.edfGP = GP(2)                                       # Pointwise estimator


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
        elif self.BARRIER_TYPE == 'LOG_GPIS':
            pass
        elif self.BARRIER_TYPE == 'POINTWISE':
            pass
        else:
            self.u = self.uNom

    # run
    def run(self):
        while self.t < self.T:
            # Update agent state
            self.controller()                                                                   # Compute nominal control law
            self.safetyFilter()                                                                 # Filter control input
            self.p = self.p + agentDynamics.dynamics(self.p, self.u).flatten() * self.dt        # Update agent state

            # Update GPs

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