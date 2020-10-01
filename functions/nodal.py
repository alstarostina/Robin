import numpy as np
import math as math
from numba import jit

@jit(nopython=True)
def radius(x,y):
    output = np.sqrt(x*x+y*y)
    return output

@jit(nopython=True)
def phi(x,y):
    output = math.atan2(x,y)
    return output

@jit(nopython=True)
def polar(x, y): #x and y neded to be of type np.array
    assert len(x) == len(y), 'x and y coordinates do not have the same size'
    vert = len(x)
    angles = np.zeros(vert)
    radii = np.zeros(vert)
    for j in range(0, vert):  # now vertices from mesh are translated into the polar
        angles[j] = phi(x[j], y[j])
        radii[j] = radius(x[j], y[j])
    return angles, radii

def distance_to_fixed_point(origin, coord):
    """
     :param origin: 1x2 np array
     :param coord: has [x y] shape where x and y all the coordinates
     :return: distanses is np.array with all the distances between origin point and nodes from the mesh (coord param)
    """
    vert = len(coord)
    test = coord - origin * np.ones((vert, 2))
    distances = np.zeros(vert)
    for j in range(0, vert):
        distances[j] = np.linalg.norm(test[j])

    return distances