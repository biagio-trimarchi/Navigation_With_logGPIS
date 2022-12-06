# Little script to test the dron dynamics

# Libraries
import agentDynamics as drone               # Agent library
import agentUtilities                       # Some useful function related to the agent
import numpy as np                          # Linear algebra library
import matplotlib.pyplot as plt             # Plot library
import matplotlib.animation as animation    # Animation library
from tqdm import tqdm                       # Loading bar

# Simulation
class Simulation:
    def __init__(self):
        self.initialize()
    
    def initialize(self):
        
        # Agent
        drone.AGENT_TYPE == 'UAV'       # Set agent type
        drone.initialize_agent()        # Initialize agent parameters
        self.x = np.zeros((12,))        # Agent state
        self.x[8] = 0

        # Trajectory 
        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([5.0, 5.0, 3.0])
        P = []                      # Control point of the trajectory
        P.append(start.copy())      # First point
        P.append(start.copy())      # Second point
        P.append(start.copy())      # Third point
        P.append(goal.copy())       # Fourth point
        P.append(goal.copy())       # Fifth point    
        P.append(goal.copy())       # Sixth point

        dP = []                             # Control points of the velocity profile
        for i in range(len(P)-1):           # Loop trough control points of the trajectory
            dP.append(P[i+1] - P[i])            # Compute contol point of the velocity profile
        
        ddP = []                            # Control points of the acceleration profile
        for i in range(len(dP)-1):          # Loop trough control points of the velocity profile
            ddP.append(dP[i+1] - dP[i])         # Compute contol point of the acceleration profile

    def controller(self):
        pass
    
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

    def animate(self):
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
                                [position[1] - xAxis[1]/2, position[1] + xAxis[1]/2])       # Plot x axis
            xAx.set_3d_properties([position[2] - xAxis[2]/2, position[2] + xAxis[2]/2])     # Plot x axis
        
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



# Main 
def main():
    sim = Simulation()
    # sim.initialize()
    sim.plot()
    sim.animate()

if __name__ == '__main__':
    main()
