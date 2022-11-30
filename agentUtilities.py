import numpy as np
import math as mt
# Look at https://repository.upenn.edu/cgi/viewcontent.cgi?article=1705&context=edissertations
# and https://ieeexplore.ieee.org/document/5980409 for more details about the various quantities
# and the convention for the rotation matrix
# The code in inspired by https://github.com/yrlu/quadrotor


def hatMap(w):
    # The hat map build a screw symmetric matric from a 3x1 vector
    R = [
        [      0,  -w[2],      w[1] ],
        [   w[2],      0,     -w[0] ],
        [  -w[1],   w[0],         0 ]
        ]
    R = np.array(R)
    return R

def veeMap(R):  
    # The vee map, the inverse of the hat map, return the 3x1 vector that correspond to the skew symmetric matrix
    w = np.zeros((3,))
    w[0] = R[2, 1]
    w[1] = R[0, 2]
    w[2] = R[1, 0]

    return w

def worldToBody(phi, theta, psi):
    # Rotation matrix, world to body using Roll/Pitch/Yaw angles 
    # https://en.wikipedia.org/wiki/Euler_angles (ZXY Tait–Bryan angles)
    R = [
            [np.cos(psi)*np.cos(theta) - np.sin(phi)*np.sin(psi)*np.sin(theta), -np.cos(phi)*np.sin(psi), np.cos(psi)*np.sin(theta) + np.cos(theta)*np.sin(phi)*np.sin(psi)],
            [np.cos(theta)*np.sin(psi) + np.cos(psi)*np.sin(phi)*np.sin(theta), np.cos(phi)*np.cos(psi), np.sin(psi)*np.sin(theta) - np.cos(psi)*np.cos(theta)*np.sin(phi)],
            [-np.cos(phi)*np.sin(theta), np.sin(phi), np.cos(phi)*np.cos(theta)]
         ]
    R = np.array(R)
    return R

def RtoRPY(R):
    # Extract roll, pitch, and yaw from rotation matrix (ZXY Euler angles)
    # https://en.wikipedia.org/wiki/Euler_angles (ZXY Tait–Bryan angles)
    psi = np.arctan2(-R[0, 1], R[1, 1])
    phi = np.arctan2(R[2,1], np.sqrt( 1-R[2,1]**2 ) )
    theta = np.arctan2(-R[2,0], R[2,2])

    return phi, theta, psi

def Rvel(phi, theta, psi):
    # Velocities coonversion matrix, it transform 
    # the angular rate of change of the ZYX angles in 
    # angular velocities about the body reference frame axis
    R = [
            [  np.cos(theta),  0,    -np.cos(phi)*np.sin(theta) ],
            [              0,  1,                   np.sin(phi) ],
            [  np.sin(theta),  0,     np.cos(phi)*np.cos(theta) ]
        ]
    
    R = np.array(R)
    return R

if __name__ == '__main__':
    r, p, y = 0.1, 0.2, 1.5
    R = worldToBody(r, p, y)
    print(R)
    print(RtoRPY(R))

    print(Rvel(r, p, y))