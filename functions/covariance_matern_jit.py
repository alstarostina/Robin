from numba import jit
import numpy as np
from scipy import *
import scipy.special as ss
import scipy.spatial as sp
from .numba_special import gamma, kv


sigma = 1


@jit(nopython=True)
def matern_kernel(x, nu):
    # TODO: Find a better tolerance here!
    return 1. if abs(x) < 1e-15 else (x**nu)*kv(nu,x)/(gamma(nu)*(2**(nu-1)))


@jit(nopython=True)
def matern_cov(x,y,kappa,nu):
    # if x == y:
    # if np.isclose(x, y):
    # TODO: Find a better tolerance here
    if np.linalg.norm(x-y) <= 1e-15:
        C = 1
    else:
        C = (sigma**2)*matern_kernel(kappa*np.linalg.norm(x-y),nu)
    return C


