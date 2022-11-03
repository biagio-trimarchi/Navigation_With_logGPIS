# Little scripts to test the Control Barrier Function behaviour 
# with a simple control law using different proxy for the 
# barrier function
# In particular: 
#               - Ground truth (EDF)
#               - log GPIS     
#               - Pointwise Gaussian Process
# The agent dynamics is a simple double integrator controlled
# in acceleration

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
from celluloid import Camera                    # Camera for animated plots              


class Simulation:
    def __init__(self):
        self.initialize()   # Initialize simulation

    # initialize
    def initialize(self):
        # Simulation Data
        self.T = 10.0        # Total simulation time
        self.dt = 0.1      # Sampling period
        self.t = 0          # Actual time

        # Agent data
        self.p0 = np.array([0.0, 0.0])          # Initial position
        self.v0 = np.array([0.0, 0.0])          # Initial velocity
        self.pGoal = np.array([4.0, 4.0])       # Final goal
        self.pStory = [self.p0]                 # List of positions
        self.vStory = [self.v0]                 # List of velocities
        self.p = self.p0                        # Actual position
        self.v = self.v0                        # Actual velocity
        self.x = np.concatenate((self.p, self.v))   # Actual state
        
        agentDynamics.AGENT_TYPE = 'DOUBLE_INTEGRATOR_2D'       # Agent type
        agentDynamics.initialize_agent()

        # Lidar parameters
        lidar.fov_range = np.pi         # fov range
        lidar.min_distance = 0.1        # Minimum sensing distance
        lidar.max_distance = 2.0        # Minimum sensing distance 

        # Controller data
        self.kp = 0.8                   # Position gain
        self.kv = 1.2                   # Velocity gain

        # Plot variables
        self.uNom = np.zeros((2, ))      # Actual nominal input
        self.u = np.zeros((2, ))         # Actual input
        self.uNomStory = []              # List of nominal inputs
        self.uStory = []                 # List of inputs
        self.time_stamps = []            # List of time stamps

        # Control barrier function data
        self.alpha1 = 2.0                                        # Class K multiplier
        self.alpha2 = 2.0                                        # Class K multiplier
        self.kh = self.alpha1*self.alpha2
        self.klfh = self.alpha1 + self.alpha2

        self.BARRIER_TYPE = 'POINTWISE'                          # Barrier function type ('GROUND_TRUTH', 'LOG_GPIS', 'POINTWISE')
        self.safeDist = 0.2                                     # Safe distance
        
        # GP
        self.edfGP = GP(2)                                      # Pointwise estimator
        self.edfGP.params.sigma_err = 0.01                      # Sensor noise variance
        self.edfGP.params.L = 0.3                               # GP pointwise length scale
        logGPIS.resolution = 0.2                               # Resolution of the state space

        # Animation
        if self.BARRIER_TYPE == 'LOG_GPIS':
            self.logGPISanimationSetup()
        if self.BARRIER_TYPE == 'POINTWISE':
            self.edfGPanimationSetup()

    # controller
    def controller(self):
        self.uNom = -self.kp * (self.p - self.pGoal) - self.kv * self.v     # PD contoller

    # safetyFilter
    def safetyFilter(self):
        # Build matrices
        # Return u

        if self.BARRIER_TYPE == 'GROUND_TRUTH':
            grad_h = np.block([
                    [ np.concatenate( ((self.p - obs['center']) / obstacles.distance(self.p, obs), np.zeros((2,)) ))] for obs in obstacles.obstacles 
                ])
            hess_h = [
                    np.block([ 
                        [np.eye(2), np.zeros((2,2))],
                        [np.zeros((2,2)), np.zeros((2,2))]
                     ]) for obs in obstacles.obstacles
                ]
            lie_f_h = grad_h @ agentDynamics.f(self.x).flatten()
            lie_f2_h = np.zeros( (len(obstacles.obstacles), ))
            lie_gf_h = np.zeros( (len(obstacles.obstacles), 2))
            i = 0
            for obs in obstacles.obstacles:
                lie_f2_h[i] = agentDynamics.f(self.x).T @ hess_h[i] @ agentDynamics.f(self.x) + (grad_h[i][:]).T @ agentDynamics.df(self.x) @ agentDynamics.f(self.x)
                lie_gf_h[i][:] = agentDynamics.f(self.x).T @ hess_h[i] @ agentDynamics.g(self.x) + (grad_h[i][:]).T @ agentDynamics.df(self.x) @ agentDynamics.g(self.x)
                i = i+1

            alpha_h = self.kh * ( np.array(obstacles.allDistances(self.p)) - self.safeDist)
            alpha_lfh = self.klfh * lie_f_h
            self.u = solve_qp(np.eye(2), self.uNom, lie_gf_h.T, -alpha_h - alpha_lfh - lie_f2_h)[0]
            # self.u = self.uNom
        elif self.BARRIER_TYPE == 'LOG_GPIS':
            if logGPIS.getSamplesNumber() > 0:
                grad_h = (np.concatenate( (logGPIS.gradd(self.p).flatten(), np.zeros((2,))) )).reshape((1,4))
                hess_h = np.block( [ 
                        [logGPIS.hessd(self.p), np.zeros((2,2))],
                        [np.zeros((2,2)), np.zeros((2,2))]
                    ]
                )
                lie_f_h = grad_h @ agentDynamics.f(self.x)
                lie_f_h = lie_f_h.flatten()
                lie_f2_h = agentDynamics.f(self.x).T @ hess_h @ agentDynamics.f(self.x) + grad_h @ agentDynamics.df(self.x) @ agentDynamics.f(self.x)
                lie_f2_h = lie_f2_h.flatten()
                lie_gf_h = agentDynamics.f(self.x).T @ hess_h @ agentDynamics.g(self.x) + grad_h @ agentDynamics.df(self.x) @ agentDynamics.g(self.x)
                alpha_h = self.kh * (logGPIS.d(self.p) - self.safeDist)
                alpha_lfh = self.klfh * lie_f_h

                self.u = solve_qp(np.eye(2), self.uNom, lie_gf_h.T, -alpha_h - alpha_lfh - lie_f2_h)[0]

            else:
                self.u = self.uNom
        elif self.BARRIER_TYPE == 'POINTWISE':
            if self.edfGP.params.N_samples > 0:
                grad_h = (np.concatenate( (self.edfGP.gradientPosterionMean(self.p).flatten(), np.zeros((2,))) )).reshape((1,4))
                hess_h = np.block( [ 
                        [self.edfGP.hessianPosteriorMean(self.p) , np.zeros((2,2))],
                        [np.zeros((2,2)), np.zeros((2,2))]
                    ]
                )
                lie_f_h = grad_h
                lie_f_h = grad_h @ agentDynamics.f(self.x)
                lie_f_h = lie_f_h.flatten()
                lie_f2_h = agentDynamics.f(self.x).T @ hess_h @ agentDynamics.f(self.x) + grad_h @ agentDynamics.df(self.x) @ agentDynamics.f(self.x)
                lie_f2_h = lie_f2_h.flatten()
                lie_gf_h = agentDynamics.f(self.x).T @ hess_h @ agentDynamics.g(self.x) + grad_h @ agentDynamics.df(self.x) @ agentDynamics.g(self.x)
                alpha_h = self.kh * (self.edfGP.posteriorMean(self.p) - self.safeDist).flatten()
                alpha_lfh = self.klfh * lie_f_h

                self.u = solve_qp(np.eye(2), self.uNom, lie_gf_h.T, -alpha_h - alpha_lfh - lie_f2_h)[0]
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
            self.x = self.x + agentDynamics.dynamics(self.x, self.u).flatten() * self.dt        # Update agent state
            self.p = self.x[0:2]
            self.v = self.x[2:]

            # Simulate Lidar
            # Lidar
            readings, points = lidar.read(self.p, 0)            # Simulate Lidar

            # log GPIS
            if self.BARRIER_TYPE == 'LOG_GPIS':
                train = False
                # Update Log GP Implicit Surface
                for point in points:                            # Loop trough all the detected points
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
                
                self.logGPISanimationAddFrame()
            
            # Pointwise GP
            if self.BARRIER_TYPE == 'POINTWISE':
                train = False
                if self.edfGP.params.N_samples == 0:                                            # If the GP has no samples
                    self.edfGP.addSample(self.p, min(min(readings), lidar.max_distance))        # Add sample
                    train = True
                else:                                                                           # otherwise
                    new = True                                                                  # Assume the detected point is "far enough" from all the samples points of the GP      
                    for tr_point in self.edfGP.data_x.T:                                        # Loop trough all the sample point of the GP
                        tr_point = tr_point * self.edfGP.params.L                               # Scale back the sample point (The implemented GP internally scales the sample points)
                        if np.linalg.norm(tr_point - self.p) < logGPIS.resolution:              # If the points are "near"
                            new = False                                                         # Flag the point as already seen
                    if new:                                                                     # If the point is new
                        self.edfGP.addSample(self.p, min(min(readings), lidar.max_distance))    # Add sample
                        train = True
                if train:
                    self.edfGP.train()                                      # Train GP
                    print('New samples collected: pointwise GP trained')    # Debug
                
                self.edfGPanimationAddFrame()


            # Advance simulation time
            self.t = self.t + self.dt

        # Update history variables
            self.pStory.append(self.p)              # Position
            self.uStory.append(self.u)              # Actual control law
            self.uNomStory.append(self.uNom)        # Nominal control law
            self.time_stamps.append(self.t)         # Time instants

            # DEBIG
            # print(self.t, self.edfGP.posteriorMean(self.p).flatten(), obstacles.minDistance(self.p)) # Debig
            print(self.t)

    def plot(self):
        # Environment and trajectory
        fig_environment, ax_environment = plt.subplots()
        ax_environment.plot([item[0] for item in self.pStory], [item[1] for item in self.pStory], label='Agent trajectory')
        ax_environment.set_title('Environment')
        ax_environment.set_xlabel('x (m)')
        ax_environment.set_ylabel('y (m)')
        obstacles.plot(ax_environment)
        ax_environment.legend()

        plt.show()
    
    def logGPISanimationSetup(self):
        # Setup figure and axes for animation
        self.fig_logGPISanimation, self.ax_logGPISanimation = plt.subplots()
        self.camera_logGPISanimation = Camera(self.fig_logGPISanimation)

    def logGPISanimationAddFrame(self):
        xlist = np.linspace(-1.0, 5.0, 100)      # x axis values
        ylist = np.linspace(-1.0, 5.0, 100)      # y axis values
        X, Y = np.meshgrid(xlist, ylist)        # Mesh grid for plot
        ZdistSafe = np.zeros((X.shape))         # EDF safe set grid
        ZgpSafe = np.zeros((X.shape))           # log GPIS safe set grid
        # Safe set computation
        i = 0                                   # x grid cell
        for xx in xlist:                        # Loop trough x
            j = 0                               # y grid cell
            for yy in ylist:                    # Loop trough y
                point = np.array([xx, yy])      # Store grid point
                if obstacles.minDistance(point) < self.safeDist:    # If point is not safe
                    ZdistSafe[j][i] = 1.0                           # Flag it
                if logGPIS.getSamplesNumber() > 0:                           
                    if logGPIS.d(point) < self.safeDist:            # If point is supposed not safe
                        ZgpSafe[j][i] = 1.0                         # Flag it
                j = j + 1                       # Next y grid
            i = i + 1                           # Next x grid
        
        # self.ax_logGPISanimation.clear()        # Clear axes for drawing
        self.ax_logGPISanimation.pcolormesh(X, Y, ZgpSafe, cmap='binary')
        self.ax_logGPISanimation.pcolormesh(X, Y, ZdistSafe, cmap='autumn', alpha=0.2)
        self.ax_logGPISanimation.plot(self.p[0], self.p[1], 'o', color='k')
        self.ax_logGPISanimation.set_xlabel('x (m)')
        self.ax_logGPISanimation.set_xlabel('y (m)')
        
        self.camera_logGPISanimation.snap()

    def logGPISanimationSave(self):
        print("Creating animation")
        animation = self.camera_logGPISanimation.animate()
        animation.save('logGPISsafeSetExpansion.mp4')
        print("Done")
    
    def edfGPanimationSetup(self):
        # Setup figure and axes for animation
        self.fig_edfGPanimation, self.ax_edfGPanimation = plt.subplots()
        self.camera_edfGPanimation = Camera(self.fig_edfGPanimation)

    def edfGPanimationAddFrame(self):
        xlist = np.linspace(-1.0, 5.0, 100)      # x axis values
        ylist = np.linspace(-1.0, 5.0, 100)      # y axis values
        X, Y = np.meshgrid(xlist, ylist)        # Mesh grid for plot
        ZdistSafe = np.zeros((X.shape))         # EDF safe set grid
        ZgpSafe = np.zeros((X.shape))           # log GPIS safe set grid
        # Safe set computation
        i = 0                                   # x grid cell
        for xx in xlist:                        # Loop trough x
            j = 0                               # y grid cell
            for yy in ylist:                    # Loop trough y
                point = np.array([xx, yy])      # Store grid point
                if obstacles.minDistance(point) < self.safeDist:    # If point is not safe
                    ZdistSafe[j][i] = 1.0                           # Flag it
                if self.edfGP.params.N_samples > 0:                           
                    if self.edfGP.posteriorMean(point) < self.safeDist:            # If point is supposed not safe
                        ZgpSafe[j][i] = 1.0                         # Flag it
                j = j + 1                       # Next y grid
            i = i + 1                           # Next x grid
        
        # self.ax_logGPISanimation.clear()        # Clear axes for drawing
        self.ax_edfGPanimation.pcolormesh(X, Y, ZgpSafe, cmap='binary')
        self.ax_edfGPanimation.pcolormesh(X, Y, ZdistSafe, cmap='autumn', alpha=0.2)
        self.ax_edfGPanimation.plot(self.p[0], self.p[1], 'o', color='k')
        self.ax_edfGPanimation.set_xlabel('x (m)')
        self.ax_edfGPanimation.set_xlabel('y (m)')
        
        self.camera_edfGPanimation.snap()

    def edfGPanimationSave(self):
        print("Creating animation")
        animation = self.camera_edfGPanimation.animate()
        animation.save('edfGPsafeSetExpansion.mp4')
        print("Done")

    def animation(self):
        if self.BARRIER_TYPE == 'LOG_GPIS':
            self.logGPISanimationSave()
        if self.BARRIER_TYPE == 'POINTWISE':
            self.edfGPanimationSave()

# main
def main():
    sim = Simulation()      # Instantiate simulation
    
    sim.initialize()        # Initialize simulation
    sim.run()               # Run simulation
    sim.plot()              # Plot results

    if sim.BARRIER_TYPE != 'GROUND_TRUTH':
        sim.animation()         # Save animations

# Run main when the script is executed
if __name__ == '__main__':
    main()