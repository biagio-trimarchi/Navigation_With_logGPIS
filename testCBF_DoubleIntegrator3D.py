# Little scripts to test the Control Barrier Function behaviour 
# with a simple control law using different proxy for the 
# barrier function
# In particular: 
#               - Ground truth (EDF)
#               - log GPIS     
# The agent dynamics is a simple double integrator controlled
# in acceleration

# Libraries
import agentDynamics                            # Agent Dynamic library
import obstacles3D                              # Obstacle library
import lidar3D                                  # Lidar library
import numpy as np                              # Linear algebra library
import logGPIS                                  # Log GPIS library
from quadprog import solve_qp                   # Solve quadratic programming
import matplotlib.pyplot as plt                 # Plot library
from matplotlib.patches import Wedge            # To draw wedge (Lidar fov))
from celluloid import Camera                    # Camera for animated plots              
from tqdm import tqdm                           # Loading bar

class Simulation:
    def __init__(self):
        self.initialize()       # Initialize simulation

    # initialize
    def initialize(self):
        # Simulation Data
        self.T = 20.0          # Total simulation time
        self.dt = 0.01         # Sampling period
        self.t = 0.0           # Actual time

        # Agent data
        self.p0 = np.array([2.0, -3.0, 2.0])     # Initial position
        self.v0 = np.array([0.0, 0.0, 0.0])     # Initial velocity
        self.pGoal = np.array([3.0, 4.0, 1.0])  # Final goal
        self.pStory = [self.p0]                 # List of positions
        self.vStory = [self.v0]                 # List of velocities
        self.p = self.p0                        # Actual position
        self.v = self.v0                        # Actual velocity
        self.x = np.concatenate((self.p, self.v))   # Actual state
        
        agentDynamics.AGENT_TYPE = 'DOUBLE_INTEGRATOR_3D'       # Agent type
        agentDynamics.initialize_agent()

        # Lidar parameters
        lidar3D.fov_range_xy = np.pi        # fov range
        lidar3D.fov_range_z = np.pi/2.0     # fov range
        lidar3D.min_distance = 0.1          # Minimum sensing distance
        lidar3D.max_distance = 2.0          # Minimum sensing distance 

        # Controller data
        self.kp = 1.0                   # Position gain
        self.kv = 0.7                   # Velocity gain

        # Plot variables
        self.uNom = np.zeros((3, ))                 # Actual nominal input
        self.u = np.zeros((3, ))                    # Actual input
        self.uNomStory = []                         # List of nominal inputs
        self.uStory = []                            # List of inputs
        self.time_stamps = [self.t]                 # List of time stamps
        self.minDistStory = [lidar3D.max_distance]  # List of minimum distance values

        # Control barrier function data
        self.alpha1 = 1.5                               # Class K multiplier
        self.alpha2 = 2.0                               # Class K multiplier
        self.kh = self.alpha1*self.alpha2               # h multiplier
        self.klfh = self.alpha1 + self.alpha2           # L_f h multiplier

        self.BARRIER_TYPE = 'LOG_GPIS'                  # Barrier function type: 
                                                            # 'GROUND_TRUTH', 
                                                            # 'LOG_GPIS',
        self.safeDist = 0.2                             # Safe distance
        
        # GP
        logGPIS.resolution = 0.2                    # Resolution of the state space

    # controller
    def controller(self):
        self.uNom = -self.kp * (self.p - self.pGoal) - self.kv * self.v     # PD controller

    # safetyFilter
    def safetyFilter(self):
        # Build matrices
        # Solve quadratic programming
        # Return u

        # Ground truth barrier function
        if self.BARRIER_TYPE == 'GROUND_TRUTH':
            grad_h = np.block([
                    [ np.concatenate( ((self.p - obs['center']) / obstacles3D.distance(self.p, obs), np.zeros((3,)) ))] for obs in obstacles3D.obstacles 
                ])
            hess_h = [
                    np.block([ 
                        [np.eye(3), np.zeros((3,3))],
                        [np.zeros((3,3)), np.zeros((3,3))]
                     ]) for obs in obstacles3D.obstacles
                ]
            lie_f_h = grad_h @ agentDynamics.f(self.x).flatten()
            lie_f2_h = np.zeros( (len(obstacles3D.obstacles), ))
            lie_gf_h = np.zeros( (len(obstacles3D.obstacles), 3))
            i = 0
            for obs in obstacles3D.obstacles:
                lie_f2_h[i] = agentDynamics.f(self.x).T @ hess_h[i] @ agentDynamics.f(self.x) + (grad_h[i][:]).T @ agentDynamics.df(self.x) @ agentDynamics.f(self.x)
                lie_gf_h[i][:] = agentDynamics.f(self.x).T @ hess_h[i] @ agentDynamics.g(self.x) + (grad_h[i][:]).T @ agentDynamics.df(self.x) @ agentDynamics.g(self.x)
                i = i+1

            alpha_h = self.kh * ( np.array(obstacles3D.allDistances(self.p)) - self.safeDist)
            alpha_lfh = self.klfh * lie_f_h
            self.u = solve_qp(np.eye(3), self.uNom, lie_gf_h.T, -alpha_h - alpha_lfh - lie_f2_h)[0]

        # log-GPIS based barrier function
        elif self.BARRIER_TYPE == 'LOG_GPIS':
            if logGPIS.getSamplesNumber() > 0:
                grad_h = (np.concatenate( (logGPIS.gradd(self.p).flatten(), np.zeros((3,))) )).reshape((1,6))
                hess_h = np.block( [ 
                        [logGPIS.hessd(self.p), np.zeros((3,3))],
                        [np.zeros((3,3)), np.zeros((3,3))]
                    ]
                )
                lie_f_h = grad_h @ agentDynamics.f(self.x)
                lie_f_h = lie_f_h.flatten()
                lie_f2_h = agentDynamics.f(self.x).T @ hess_h @ agentDynamics.f(self.x) + grad_h @ agentDynamics.df(self.x) @ agentDynamics.f(self.x)
                lie_f2_h = lie_f2_h.flatten()
                lie_gf_h = agentDynamics.f(self.x).T @ hess_h @ agentDynamics.g(self.x) + grad_h @ agentDynamics.df(self.x) @ agentDynamics.g(self.x)
                alpha_h = self.kh * (logGPIS.d(self.p) - self.safeDist)
                alpha_lfh = self.klfh * lie_f_h

                self.u = solve_qp(np.eye(3), self.uNom, lie_gf_h.T, -alpha_h - alpha_lfh - lie_f2_h)[0]

            else:
                self.u = self.uNom

        # No filter    
        else:
            self.u = self.uNom

    # run
    def run(self):
        tspan = np.arange(0, self.T, self.dt)
        for tt in tqdm(tspan):
            # Controller
            self.controller()                                                                   # Compute nominal control law
            self.safetyFilter()                                                                 # Filter control input
            
            # Dynamic update
            self.x = self.x + agentDynamics.dynamics(self.x, self.u).flatten() * self.dt        # Update agent state
            self.p = self.x[0:3]                                                                # Extract actual position from state
            self.v = self.x[3:]                                                                 # Extract actual velocity from state

            # Simulate Lidar
            self.readings, self.points = lidar3D.read(self.p, 0, 0)                                  # Simulate Lidar

            # Update GP if used
            # log GPIS
            if self.BARRIER_TYPE == 'LOG_GPIS':
                train = False
                # Update Log GP Implicit Surface
                for point in self.points:                       # Loop trough all the detected points
                    if logGPIS.getSamplesNumber() == 0:         # If the GP has no samples 
                        logGPIS.addSample(point)                # Add sample
                        train = True
                        continue                                # Go to next detected point
                    new = True                                  # Assume the detected point is "far enough" from all the samples points of the GP
                    if logGPIS.checkCollected(point):           # If already collected
                            new = False                         # Flag the point as already seen
                    if new:                                     # If the point is new
                        logGPIS.addSample(point)                # Add sample
                        train = True
                if train:
                    logGPIS.train()                                     # Train GP
                    print('New samples collected: log-GPIS trained')    # Debug

            # Advance simulation    
            self.t = self.t + self.dt               # Advance simulation time

            # Update history variables
            self.pStory.append(self.p)                                  # Position
            self.uStory.append(self.u)                                  # Actual control law
            self.uNomStory.append(self.uNom)                            # Nominal control law
            self.time_stamps.append(self.t)                             # Time instants
            self.minDistStory.append(obstacles3D.minDistance(self.p))     # Minimum distance

            # DEBIG
            # print(self.t, self.edfGP.posteriorMean(self.p).flatten(), obstacles.minDistance(self.p)) # Debig
            print(self.t)

    def plot(self):
        # Environment and trajectory
        fig_environment = plt.figure()
        ax_environment = plt.axes(projection='3d')
        ax_environment.plot([item[0] for item in self.pStory], 
                            [item[1] for item in self.pStory],
                            [item[2] for item in self.pStory],
                             label='Agent trajectory', color='k')
        ax_environment.set_title('Environment')
        ax_environment.set_xlabel('x (m)')
        ax_environment.set_ylabel('y (m)')
        ax_environment.set_zlabel('z (m)')
        obstacles3D.plot(ax_environment)
        ax_environment.legend()

        plt.show()

# main
def main():
    sim = Simulation()      # Instantiate simulation
    
    sim.initialize()        # Initialize simulation
    sim.run()               # Run simulation
    sim.plot()              # Plot results

# Run main when the script is executed
if __name__ == '__main__':
    main()