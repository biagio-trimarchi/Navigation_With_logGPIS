# Library for agent dynamics

import numpy as np              # Linear algebra library
from agentUtilities import *    # Some useful functions

# Agent data
AGENT_TYPE = 'UAV'    # Agent Type:
#                                       - 'SINGLE_INTEGRATOR_2D'
#                                       - 'DOUBLE_INTEGRATOR_2D'
#                                       - 'SINGLE_INTEGRATOR_3D'
#                                       - 'DOUBLE_INTEGRATOR_3D'
#                                       - 'UAV'

# For the UAV agent, the state vector is as follows:
#   x = [x  y  z  u  v  w  phi  theta  psi  p  q  r]
#        0  1  2  3  4  5    6      7    8  9 10 11

params = {}                             # Dictionary containing some frequently used parameters of the selected type of agent

def initialize_agent():
    # Initialize the agent parameters depending on the agent type
    if AGENT_TYPE == 'SINGLE_INTEGRATOR_2D':            # Single integrator 2D
        A = np.zeros((2,2))
        B = np.eye(2)
        params['A'] = A
        params['B'] = B
        return 0

    elif AGENT_TYPE == 'SINGLE_INTEGRATOR_3D':         # Single integrator 3D
        A = np.zeros((3,3))
        B = np.eye(3)
        params['A'] = A
        params['B'] = B
        return 0

    elif AGENT_TYPE == 'DOUBLE_INTEGRATOR_2D':          # Double integrator 2D
        A = np.block([
                [np.zeros((2,2)), np.eye(2)], 
                [np.zeros((2,2)), np.zeros((2,2))]
            ])
        B = np.block([ 
                    [np.zeros((2,2))], 
                    [np.eye(2)] 
            ])
        params['A'] = A
        params['B'] = B
        return 0

    elif AGENT_TYPE == 'DOUBLE_INTEGRATOR_3D':          # Double integrator 3D
        A = np.block([
                [np.zeros((3,3)), np.eye(3)], 
                [np.zeros((3,3)), np.zeros((3,3))]
            ])
        B = np.block([ 
                    [np.zeros((3,3))], 
                    [np.eye(3)] 
            ])
        params['A'] = A
        params['B'] = B
        return 0
    
    elif AGENT_TYPE == 'UAV':
        params['m'] = 1.0               # Mass
        params['I'] = 3*np.eye(3)       # Inertia moment
        params['g'] = 9.81              # Gravity

        return 0
    
    # If the type is invalid, raise error
    print("ERROR: invalid AGENT_TYPE")
    return -1

def f(x):
    if AGENT_TYPE == 'SINGLE_INTEGRATOR_2D':    # Single integrator autonomous flow
        return 0
    elif AGENT_TYPE == 'SINGLE_INTEGRATOR_3D':  # Single integrator autonomous flow
        return 0
    elif AGENT_TYPE == 'DOUBLE_INTEGRATOR_2D':  # Double integrator autonomous flow
        x = x.reshape((x.size ,1))              # Reshape vector to avoid runtime errors
        return params['A'] @ x
    elif AGENT_TYPE == 'DOUBLE_INTEGRATOR_3D':  # Double integrator autonomous flow
        x = x.reshape((x.size ,1))              # Reshape vector to avoid runtime errors
        return params['A'] @ x
    elif AGENT_TYPE == 'UAV':                   # Quadrotor
        f = np.zeros((12, ))
        
        # Translational part
        x = x.flatten()
        f[0:3] = x[3:6]                         # Translational velocity
        f[5] = -params['g']                     # Translational acceleration

        # Rotational part
        f[6:9] = np.linalg.inv(Rvel(x[6], x[7], x[8])) @ x[9:]                         # Angular velocity
        f[9:] = -np.linalg.inv(params['I']) @ np.cross(x[9:], params['I'] @ x[9:])     # Angular acceleration 

        return f.reshape(12, 1)

def df(x):
    if AGENT_TYPE == 'SINGLE_INTEGRATOR_2D':    # Single integrator autonomous flow (Jacobian)
        return 0
    elif AGENT_TYPE == 'SINGLE_INTEGRATOR_3D':  # Single integrator autonomous flow (Jacobian)
        return 0
    elif AGENT_TYPE == 'DOUBLE_INTEGRATOR_2D':  # Double integrator autonomous flow (Jacobian)
        return params['A']
    elif AGENT_TYPE == 'DOUBLE_INTEGRATOR_3D':  # Double integrator autonomous flow (Jacobian)
        return params['A']
    elif AGENT_TYPE == 'UAV':                   # Quadrotor
        df = np.zeros((12, 12))

        # Not sure if implemented correctly
        # Check it

        # Translational
        df[0:3, 3:6] = np.eye(3)
        df[6:9, 9:] = np.eye(3)

        # Rotational
        w = x[9:]
        aux = hatMap(w) @ params['I'] - hatMap( params['I'] @ w )

        df[6:9, 9:] = np.eye(3)
        df[9:, 9:] = np.linalg.inv(params['I'])@aux

        return df
        

def g(x):
    if AGENT_TYPE == 'SINGLE_INTEGRATOR_2D':    # Single integrator forced flow
        return params['B']
    elif AGENT_TYPE == 'SINGLE_INTEGRATOR_3D':  # Single integrator forced flow
        return params['B']
    elif AGENT_TYPE == 'DOUBLE_INTEGRATOR_2D':  # Double integrator forced flow
        return params['B']
    elif AGENT_TYPE == 'DOUBLE_INTEGRATOR_3D':  # Double integrator forced flow
        return params['B']
    elif AGENT_TYPE == 'UAV':                   # Quadrotor
        x = x.flatten()
        g = np.zeros((12, 4))

        # Translational part
        g[3:6, 0] = (1 / params['m']) * worldToBody(x[3], x[4], x[5]) @ (np.array([0, 0, 1]))
        
        # Rotational part
        g[9:, 1:] = np.linalg.inv(params["I"])

        return g

def dynamics(x, u):
    x = x.reshape((x.size, 1))      # Reshape vector to avoid runtime errors
    u = u.reshape((u.size, 1))      # Reshape vector to avoid runtime errors
    return f(x) + g(x)@u            # Return flow

if __name__ == '__main__':
    AGENT_TYPE == 'UAV'
    initialize_agent()
    x = np.zeros((12, ))
    x[9] = 0.5
    x[10] = 0.1
    x[11] = 0.5
    print(f(x))
    print(g(x))
    print(df(x))