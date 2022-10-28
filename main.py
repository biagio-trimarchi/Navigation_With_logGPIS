from cmath import sqrt
from matern32GP import GaussianProcess as GP
import numpy as np
import matplotlib.pyplot as plt

class Obstacle:
    def __init__(self):
        pass

class Triangle(Obstacle):
    def __init__(self, vertices):
        super().__init__()

        self.vertices = vertices
    
    def draw():
        pass

    def distance(self, p):
        pass
                