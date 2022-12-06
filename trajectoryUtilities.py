import numpy as np      # Linear algebra library
import math as mt       # Math library

# For some properties about Bernstein basis and Bézier polynomial see: 
# https://en.wikipedia.org/wiki/B%C3%A9zier_curve

def bernsteinBasis(t, n, k):
    # Compute the k^th Bernstein polynomial of order n
    return mt.comb(n, k) * ( t ** k ) * ( (1-t) ** (n-k) )

def bezierCurve(t, P):
    # Compute Bezier curve at time t (t in [0, 1]) with P as control points

    b = 0                                           # Store sum
    n = len(P)-1                                    # Order of the curve
    for k in range(n+1):                            # Loop trough each control point
        b = b + P[k] * bernsteinBasis(t, n, k)          # Convex combination of control points weigthed with Bernstein basis
    return b