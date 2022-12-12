# Little script to test the drone dynamics

# Libraries
import agentDynamics as drone               # Agent library
import agentUtilities                       # Some useful function related to the agent
import trajectoryUtilities as trajectory    # Trajectory
import numpy as np                          # Linear algebra library
import matplotlib.pyplot as plt             # Plot library
import matplotlib.animation as animation    # Animation library
from tqdm import tqdm                       # Loading bar
from scipy.integrate import solve_ivp       # Solve initial value problems
from quadprog import solve_qp               # Solve quadratic programming (https://github.com/quadprog/quadprog)


# Simulation
class Simulation:
    def __init__(self):
        self.initialize()
    
    def initialize(self):
        # Simulation data
        self.t = 0.0

        # Agent
        drone.AGENT_TYPE == 'UAV'       # Set agent type
        drone.initialize_agent()        # Initialize agent parameters
        self.x0 = np.zeros((12,))       # Agent initial state
        self.x0[0] = 5.0
        self.x0[2] = -5.0

        self.x = self.x0.copy()         # Agent state

        # Trajectory 
        start = np.array([0.0, 0.0, 0.0])       # Initial position
        goal = np.array([5.0, 5.0, 3.0])        # Goal position
        P = []                                  # Control point of the trajectory
        P.append(start.copy())                  # First point
        P.append(start.copy())                  # Second point
        P.append(start.copy())                  # Third point
        P.append(goal.copy())                   # Fourth point
        P.append(goal.copy())                   # Fifth point    
        P.append(goal.copy())                   # Sixth point

        dP = []                             # Control points of the velocity profile
        for i in range(len(P)-1):           # Loop trough control points of the trajectory
            dP.append(P[i+1] - P[i])            # Compute contol point of the velocity profile
        
        ddP = []                            # Control points of the acceleration profile
        for i in range(len(dP)-1):          # Loop trough control points of the velocity profile
            ddP.append(dP[i+1] - dP[i])         # Compute contol point of the acceleration profile

    def differentialFlatnessInversion(self):
        self.pDes = np.zeros((3, ))     # Desired position
        self.vDes = np.zeros((3, ))     # Desired velocity
        self.RDes = np.eye(3)           # Desired attitude (rotation matrix)
        self.wDes = np.zeros((3, ))     # Desired angular velocity

    def controller(self):
        # Control input
        self.u = np.zeros((4,))         # Control input

        # Extract drone parameters
        m = drone.params['m']           # Mass
        J = drone.params['I']           # Inertia matrix

        # Drone state
        p = self.x[0:3]                                                     # Position
        v = self.x[3:6]                                                     # Velocity
        angles = self.x[6:9]                                                # Euler angles
        w = self.x[9:]                                                      # Angular velocity
        R = agentUtilities.worldToBody(angles[0], angles[1], angles[2])     # Rotation matrix associated to actual attitude

        ### Virtual position controller
        
        # Virtual dynamics (Double integrator)
        f_p = np.block([
            [v.reshape((3,1))],
            [np.zeros((3,1))]
         ])                                     # Autonomous flow

        g_p = np.block([ 
            [np.zeros((3,3))],
            [np.eye(3)]
        ]) * 1/m                                # Forces flow

        # Rotational dynamics (Drone attitude dynamic)
        f_R = drone.f(self.x)[6:]               # Autonomous flow
        g_R = drone.g(self.x)[6:,1:]            # Forced flow

        # Controller parameters
        eps1 = 1.0                              # Cross error term
        k1 = 1 / (4*m) * eps1 ** 2 + 5.0        # Position gain
        eta1 = 5.0                              # Class-K multiplier

        # Tracking errors
        e_p = p - self.pDes# trajectory.bezierCurve(self.t, self.P)        # Posiiton error
        e_v = v - self.vDes# trajectory.bezierCurve(self.t, self.dP)       # Velocity error

        # Lyapunov function gradient
        grad_pV_p = k1*e_p + eps1*e_v
        grad_vV_p = m*e_v + eps1*e_p
        gradV_p = np.block([ [grad_pV_p, grad_vV_p ] ])

        # Min-norm controller terms 
        V_p = 1/2 * m * e_v @ e_v + 1/2 * k1 * e_p @ e_p + eps1 * e_p @ e_v     # Lyapunov function
        lie_f_V_p = (gradV_p @ f_p).flatten()                                   # Lie derivative along f
        lie_g_V_p = gradV_p @ g_p                                               # Lie derivative along g

        # Min-norm controller
        print(-lie_g_V_p.T)
        print(lie_f_V_p + eta1*V_p)
        v = solve_qp(np.eye(3), np.zeros(3,), lie_g_V_p.T, -(lie_f_V_p + eta1*V_p))[0]      # Compute virtual control input
        print(lie_g_V_p @ v + lie_f_V_p + eta1*V_p)
        self.u[1] = v @ R @ np.array([0, 0, 1])                                         # Compute force

        ### Attitude controller
        # Controller parameters
        eps2 = 5.0                              # Cross error term
        k2 = 1 / (4*m) * eps1 ** 2 + 25.0       # Position gain
        eta2 = 10.0                             # Class-K multiplier

        # Tracking errors
        e_R = 1/2 * agentUtilities.veeMap(self.RDes.T @ R - R.T @ self.RDes)        # Rotation error
        e_w = w - R.T @ self.RDes @ w                                               # Angular velocity error

        # Lyapunov function gradient
        grad_RV_R = k2*e_R + eps2*e_w
        grad_wV_R = J@e_w + eps2*e_R
        gradV_R = np.block([ [grad_RV_R, grad_wV_R ] ])

        # Min-norm controller terms 
        V_R = 1/2 * e_w @ J @ e_w + 1/2 * k2 * e_R @ e_R + eps2 * e_R @ e_w         # Lyapunov function
        lie_f_V_R = (gradV_R @ f_R).flatten()                                                   # Lie derivative along f
        lie_g_V_R = gradV_R @ g_R                                                   # Lie derivative along g

        # Min-norm controller
        self.u[1:] = solve_qp(np.eye(3), np.zeros(3,), -lie_g_V_R.T, lie_f_V_R + eta2*V_R)[0]     # Compute torque

    
    def plot(self):
        # Plot drone actual pose and attitude
        
        # Position
        position = self.x[0:3]          # Drone position

        # Attitude
        l = 1.0                         # Auxiliary variable for plotting
        xAxis = np.array([l, 0, 0])     # Drone x-body axis (Body Frame)
        yAxis = np.array([0, l, 0])     # Drone y-body axis (Body Frame)
        zAxis = np.array([0, 0, l])     # Drone z-body axis (Body Frame)

        phi, theta, psi = self.x[6], self.x[7], self.x[8]           # Rotation angles
        rotation = agentUtilities.worldToBody(phi, theta, psi).T    # Rotation matrix

        xAxis = rotation @ xAxis        # Drone x-body axis (World frame)
        yAxis = rotation @ yAxis        # Drone y-body axis (World frame)
        zAxis = rotation @ zAxis        # Drone z-body axis (World frame)

        # Plot
        figureDrone = plt.figure()
        axDrone = plt.axes(projection='3d')

        axDrone.plot3D(
                        [position[0] - xAxis[0]/2, position[0] + xAxis[0]/2],
                        [position[1] - xAxis[1]/2, position[1] + xAxis[1]/2], 
                        [position[2] - xAxis[2]/2, position[2] + xAxis[2]/2], 
                        color = 'r')                                            # Plot x-axis
        
        axDrone.plot3D(
                        [position[0] - yAxis[0]/2, position[0] + yAxis[0]/2],
                        [position[1] - yAxis[1]/2, position[1] + yAxis[1]/2], 
                        [position[2] - yAxis[2]/2, position[2] + yAxis[2]/2], 
                        color = 'r')                                            # Plot y-axis
        
        axDrone.plot3D(
                        [position[0], position[0] + zAxis[0]/4], 
                        [position[1], position[1] + zAxis[1]/4], 
                        [position[2], position[2] + zAxis[2]/4], 
                        color = 'g')                                            # Plot z-axis

        plt.show()

    def animateRotation(self):
        figureDrone, axDrone = plt.figure(), plt.axes(projection='3d')

        xAx, = axDrone.plot3D([], [], [], color = 'r')
        yAx, = axDrone.plot3D([], [], [], color = 'r')
        zAx, = axDrone.plot3D([], [], [], color = 'g')

        def updateAxes(k, xAx, yAx, zAx):
            self.x[6] = k*0.1

            # Position
            position = self.x[0:3]          # Drone position

            # Attitude
            l = 1.0                         # Auxiliary variable for plotting
            xAxis = np.array([l, 0, 0])     # Drone x-body axis (Body Frame)
            yAxis = np.array([0, l, 0])     # Drone y-body axis (Body Frame)
            zAxis = np.array([0, 0, l])     # Drone z-body axis (Body Frame)

            phi, theta, psi = self.x[6], self.x[7], self.x[8]           # Rotation angles
            rotation = agentUtilities.worldToBody(phi, theta, psi).T    # Rotation matrix

            xAxis = rotation @ xAxis        # Drone x-body axis (World frame)
            yAxis = rotation @ yAxis        # Drone y-body axis (World frame)
            zAxis = rotation @ zAxis        # Drone z-body axis (World frame)
            
            xAx.set_data(
                                [position[0] - xAxis[0]/2, position[0] + xAxis[0]/2],
                                [position[1] - xAxis[1]/2, position[1] + xAxis[1]/2])       # Plot x-axis
            xAx.set_3d_properties([position[2] - xAxis[2]/2, position[2] + xAxis[2]/2])     # Plot x-axis
        
            yAx.set_data(
                                [position[0] - yAxis[0]/2, position[0] + yAxis[0]/2],
                                [position[1] - yAxis[1]/2, position[1] + yAxis[1]/2])       # Plot y-axis
            yAx.set_3d_properties([position[2] - yAxis[2]/2, position[2] + yAxis[2]/2])     # Plot y-axis
        
            zAx.set_data(
                                [position[0], position[0] + zAxis[0]/4], 
                                [position[1], position[1] + zAxis[1]/4])                    # Plot z-axis 
            zAx.set_3d_properties([position[2], position[2] + zAxis[2]/4])                  # Plot z-axis

        anim = animation.FuncAnimation(figureDrone, updateAxes, 50, fargs=(xAx, yAx, zAx))

        plt.show()

    def animate(self):
        figureDrone, axDrone = plt.figure(), plt.axes(projection='3d')

        xAx, = axDrone.plot3D([], [], [], color = 'r')
        yAx, = axDrone.plot3D([], [], [], color = 'r')
        zAx, = axDrone.plot3D([], [], [], color = 'g')

        axDrone.set_xlim(-6, 6)
        axDrone.set_ylim(-6, 6)
        axDrone.set_zlim(-6, 6)


        def updateAxes(k, xAx, yAx, zAx):
            x = self.xStory[:, k].flatten()

            # Position
            position = x[0:3]          # Drone position

            # Attitude
            l = 1.0                         # Auxiliary variable for plotting
            xAxis = np.array([l, 0, 0])     # Drone x-body axis (Body Frame)
            yAxis = np.array([0, l, 0])     # Drone y-body axis (Body Frame)
            zAxis = np.array([0, 0, l])     # Drone z-body axis (Body Frame)

            phi, theta, psi = x[6], x[7], x[8]           # Rotation angles
            rotation = agentUtilities.worldToBody(phi, theta, psi).T    # Rotation matrix

            xAxis = rotation @ xAxis        # Drone x-body axis (World frame)
            yAxis = rotation @ yAxis        # Drone y-body axis (World frame)
            zAxis = rotation @ zAxis        # Drone z-body axis (World frame)
            
            xAx.set_data(
                                [position[0] - xAxis[0]/2, position[0] + xAxis[0]/2],
                                [position[1] - xAxis[1]/2, position[1] + xAxis[1]/2])       # Plot x-axis
            xAx.set_3d_properties([position[2] - xAxis[2]/2, position[2] + xAxis[2]/2])     # Plot x-axis
        
            yAx.set_data(
                                [position[0] - yAxis[0]/2, position[0] + yAxis[0]/2],
                                [position[1] - yAxis[1]/2, position[1] + yAxis[1]/2])       # Plot y-axis
            yAx.set_3d_properties([position[2] - yAxis[2]/2, position[2] + yAxis[2]/2])     # Plot y-axis
        
            zAx.set_data(
                                [position[0], position[0] + zAxis[0]/4], 
                                [position[1], position[1] + zAxis[1]/4])                    # Plot z-axis 
            zAx.set_3d_properties([position[2], position[2] + zAxis[2]/4])                  # Plot z-axis

        anim = animation.FuncAnimation(figureDrone, updateAxes, 100, fargs=(xAx, yAx, zAx))
        plt.show()

    def controlledDynamics(self, t, x):
        self.x = x.flatten()
        self.differentialFlatnessInversion()
        self.controller()
        print(self.u)
        return drone.dynamics(self.x, self.u).flatten()

    def run(self):
        sol = solve_ivp(self.controlledDynamics, [0, 10], self.x0, t_eval=[0.1*t for t in range(100)])
        self.tStamp = sol.t
        self.xStory = sol.y
        print(self.xStory[0, :])
        print(self.xStory[1, :])
        print(self.xStory[2, :])


        

# Main 
def main():
    sim = Simulation()
    # sim.initialize()
    # sim.plot()
    sim.run()
    sim.animate()

if __name__ == '__main__':
    main()
