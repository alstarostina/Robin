import numpy as np
from scipy import *
import scipy.special as ss
import scipy.spatial as sp


sigma = 1

def matern_kernel(x, nu):
    # TODO: Find a better tolerance here!
    return 1. if abs(x) < 1e-15 else (x**nu)*ss.kv(nu,x)/(math.gamma(nu)*(2**(nu-1)))


def matern_cov(x,y,kappa,nu):
    # if x == y: 
    # if np.isclose(x, y):
    # TODO: Find a better tolerance here
    if np.linalg.norm(x-y) <= 1e-15:
        C = 1
    else:
        C = (sigma**2)*matern_kernel(kappa*np.linalg.norm(x-y),nu)
    return C


