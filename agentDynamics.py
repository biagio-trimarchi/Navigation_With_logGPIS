# Library for agent dynamics

import numpy as np      # Linear algebra library

# Agent data
AGENT_TYPE = 'SINGLE_INTEGRATOR_2D'    # Agent Type:
#                                       - 'SINGLE_INTEGRATOR_2D'
#                                       - 'DOUBLE_INTEGRATOR_2D'

params = {}                             # Dictionary containing some frequently used parameters of the selected type of agent

def initialize_agent():
    # Initialize the agent parameters depending on the agent type
    if AGENT_TYPE == 'SINGLE_INTEGRATOR_2D':            # Single integrator
        A = np.zeros((2,2))
        B = np.eye(2)
        params['A'] = A
        params['B'] = B
        return 0

    elif AGENT_TYPE == 'DOUBLE_INTEGRATOR_2D':          # Double integrator
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
    
    # If the type is invalid, raise error
    print("ERROR: invalid AGENT_TYPE")
    return -1

def f(x):
    x = x.reshape((x.size ,1))                  # Reshape vector to avoid runtime errors
    if AGENT_TYPE == 'SINGLE_INTEGRATOR_2D':    # Single integrator autonomous flow
        return 0
    elif AGENT_TYPE == 'DOUBLE_INTEGRATOR_2D':  # Double integrator autonomous flow
        return params['A'] @ x

def df(x):
    if AGENT_TYPE == 'SINGLE_INTEGRATOR_2D':    # Single integrator autonomous flow (Jacobian)
        return 0
    elif AGENT_TYPE == 'DOUBLE_INTEGRATOR_2D':  # Double integrator autonomous flow (Jacobian)
        return params['A']

def g(x):
    if AGENT_TYPE == 'SINGLE_INTEGRATOR_2D':    # Single integrator forced flow
        return params['B']
    elif AGENT_TYPE == 'DOUBLE_INTEGRATOR_2D':  # Double integrator forced flow
        return params['B']

def dynamics(x, u):
    x = x.reshape((x.size, 1))      # Reshape vector to avoid runtime errors
    u = u.reshape((u.size, 1))      # Reshape vector to avoid runtime errors
    return f(x) + g(x)@u            # Return flow