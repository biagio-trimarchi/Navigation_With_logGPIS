# Library for agent dynamics

import numpy as np      # Linear algebra library

# Agent data
AGENT_TYPE = 'SINGLE_INTEGRATOR_2D'    # Agent Type:
#                                       - 'SINGLE_INTEGRATOR_2D'
#                                       - 'DOUBLE_INTEGRATOR_2D'

params = {}                             # Dictionary containing some frequently used parameters of the selected type of agent

def initialize_agent():
    if AGENT_TYPE == 'SINGLE_INTEGRATOR_2D':
        A = np.zeros((2,2))
        B = np.eye(2)
        params['A'] = A
        params['B'] = B
        return 0

    elif AGENT_TYPE == 'DOUBLE_INTEGRATOR_2D':
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
    x = x.reshape((x.size ,1))
    if AGENT_TYPE == 'SINGLE_INTEGRATOR_2D':
        return 0
    elif AGENT_TYPE == 'DOUBLE_INTEGRATOR_2D':
        return params['A'] @ x

def df(x):
    if AGENT_TYPE == 'SINGLE_INTEGRATOR_2D':
        return 0
    elif AGENT_TYPE == 'DOUBLE_INTEGRATOR_2D':
        return params['A']

def g(x):
    if AGENT_TYPE == 'SINGLE_INTEGRATOR_2D':
        return params['B']
    elif AGENT_TYPE == 'DOUBLE_INTEGRATOR_2D':
        return params['B']