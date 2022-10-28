# Library for agent dynamics

import numpy as np      # Linear algebra library

# Agent data
AGENT_TYPE = 'SINGLE_INTEGRATOR_2D'    # Agent Type:
#                                       - 'SINGLE_INTEGRATOR_2D'
#                                       - 'DOUBLE_INTEGRATOR_2D'

params = {}                             # Dictionary containing some frequently used parameters of the selected type of agent

def initialize_agent():
    if AGENT_TYPE == 'SINGLE_INTEGRATOR_2D':
        return 0
    
    # If the type is invalid, raise error
    print("ERROR: invalid AGENT_TYPE")
    return -1

def f(x):
    if AGENT_TYPE == 'SINGLE_INTEGRATOR_2D':
        return 0

def df(x):
    if AGENT_TYPE == 'SINGLE_INTEGRATOR_2D':
        return 0

def g(x):
    if AGENT_TYPE == 'SINGLE_INTEGRATOR_2D':
        return 0