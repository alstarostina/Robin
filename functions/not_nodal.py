import dolfin as df
import scipy.special as ss
import numpy as np
import math as math
from scipy import optimize
from numba import jit
from numba import njit, prange

from . import covariance_matern
from . import covariance_matern_jit


def alpha_function(x, n, beta):
    output = beta * ss.jv(n, x) + ss.jvp(n, x)
    return output

def alpha_function_new(x, n, beta, R):
    output = beta * ss.jv(n, x) + (x / R) * ss.jvp(n, x)
    return output


def true(origin, x, y, kappa, nu):
    vert = len(x)
    true_expression = np.zeros(vert)
    for p in prange(0, vert):
        true_expression[p] = covariance_matern.matern_cov(origin, np.array([x[p], y[p]]), kappa, nu)

    return true_expression

@jit(nopython=False)
def true_cov_mat(x, y, kappa, nu):
    vert = len(x)
    true_expression = np.zeros((vert, vert))
    for p in range(0,vert):
        for t in range(0, vert):
            true_expression[p][t] = covariance_matern_jit.matern_cov(np.array([x[t], y[t]]), np.array([x[p], y[p]]), kappa, nu)
    return true_expression

def compute_positive_robin_roots(beta, n, roots):
    '''
    :param beta: f = beta*J_n(x) + dJ_n(x)/dn = 0 this is how Robin is posed here
    :param n: order of J_n
    :param roots: number of roots
    :return: np array with all the roots of f
    '''
    bessel_roots = ss.jn_zeros(n,roots)
    der_bessel_roots = ss.jnp_zeros(n,roots)
    result = np.zeros(roots)
    for k in range(0,roots):
        if (n == 0 and k == 0):
            result[k] = optimize.bisect(alpha_function, 0, bessel_roots[k], args=(n,beta))
        else:
            if (n == 0):
                result[k] = optimize.bisect(alpha_function, min(der_bessel_roots[k-1], bessel_roots[k]), max(der_bessel_roots[k-1], bessel_roots[k]), args=(n,beta))
            else:
                result[k] = optimize.bisect(alpha_function, min(der_bessel_roots[k], bessel_roots[k]), max(der_bessel_roots[k], bessel_roots[k]), args=(n,beta))
    return result

def compute_all_robin_roots(beta, n, roots):
    '''
    :param beta: f = beta*J_n(x) + dJ_n(x)/dn = 0 this is how Robin is posed here
    :param n: order of J_n
    :param roots: number of roots
    :return: np array with all the roots of f
    '''
    bessel_roots = ss.jn_zeros(n,roots)
    der_bessel_roots = ss.jnp_zeros(n,roots)
    result = np.zeros(roots)
    for k in range(0,roots):
        if (n == 0 and k == 0):
                result[k] = optimize.bisect(alpha_function, 0, bessel_roots[k], args=(n,beta))

        else:
            if (n == 0):
                result[k] = optimize.bisect(alpha_function, min(der_bessel_roots[k-1], bessel_roots[k]), max(der_bessel_roots[k-1], bessel_roots[k]), args=(n,beta))
            else:
                if (alpha_function(0,n,beta) == 0):
                    result[0] = 0
                    result[k] = optimize.bisect(alpha_function, min(der_bessel_roots[k - 1], bessel_roots[k - 1]),
                                                max(der_bessel_roots[k - 1], bessel_roots[k - 1]), args=(n, beta))
                else:
                    result[k] = optimize.bisect(alpha_function, min(der_bessel_roots[k], bessel_roots[k]), max(der_bessel_roots[k], bessel_roots[k]), args=(n,beta))
    return result

def compute_positive_robin_roots_new(beta, n, R, roots):
    '''
    :param beta: f = beta*J_n(x) + dJ_n(x)/dn = 0 this is how Robin is posed here
    :param n: order of J_n
    :param roots: number of roots
    :return: np array with all the roots of f
    '''
    bessel_roots = ss.jn_zeros(n,roots)
    der_bessel_roots = ss.jnp_zeros(n,roots)
    result = np.zeros(roots)
    for k in range(0,roots):
        if (n == 0 and k == 0):
            result[k] = optimize.bisect(alpha_function_new, 0, bessel_roots[k], args=(n,beta,R))
        else:
            if (n == 0):
                result[k] = optimize.bisect(alpha_function_new, min(der_bessel_roots[k-1], bessel_roots[k]), max(der_bessel_roots[k-1], bessel_roots[k]), args=(n,beta,R))
            else:
                result[k] = optimize.bisect(alpha_function_new, min(der_bessel_roots[k], bessel_roots[k]), max(der_bessel_roots[k], bessel_roots[k]), args=(n,beta,R))
    return result

def compute_all_robin_roots_new(beta, n, R, roots):
    '''
    :param beta: f = beta*J_n(x) + dJ_n(x)/dn = 0 this is how Robin is posed here
    :param n: order of J_n
    :param roots: number of roots
    :return: np array with all the roots of f
    '''
    bessel_roots = ss.jn_zeros(n,roots)
    der_bessel_roots = ss.jnp_zeros(n,roots)
    result = np.zeros(roots)
    for k in range(0,roots):
        if (n == 0 and k == 0):
                result[k] = optimize.bisect(alpha_function_new, 0, bessel_roots[k], args=(n,beta,R))

        else:
            if (n == 0):
                result[k] = optimize.bisect(alpha_function_new, min(der_bessel_roots[k-1], bessel_roots[k]), max(der_bessel_roots[k-1], bessel_roots[k]), args=(n,beta,R))
            else:
                if (alpha_function_new(0,n,beta,R) == 0):
                    result[0] = 0
                    result[k] = optimize.bisect(alpha_function_new, min(der_bessel_roots[k - 1], bessel_roots[k - 1]),
                                                max(der_bessel_roots[k - 1], bessel_roots[k - 1]), args=(n,beta,R))
                else:
                    result[k] = optimize.bisect(alpha_function_new, min(der_bessel_roots[k], bessel_roots[k]), max(der_bessel_roots[k], bessel_roots[k]), args=(n,beta,R))
    return result

def get_alpha_nk(boundary, n, k, bessel_roots, der_bessel_roots, robin_roots, beta):
    if boundary == 'Neumann':
        if (k == 0):
            return 0
        else:
            return der_bessel_roots[k-1]

    if boundary == 'Dirichlet':
        return bessel_roots[k]

    if boundary == 'Robin': #this looks this way because I spend much time looking at the plots but I'm pretty much sure
        return robin_roots[k]

    assert False, 'unknown boundary condition!'

def get_eigenvalue(alpha_nk, R, delta, kappa):
    return 1 + (alpha_nk) ** 2 / ((R + delta) * kappa) ** 2

##################### NORM CALCULATION ###########################
def get_norm_over_phi(n, l):
    assert 0 < l < 3, 'l equals 1 or 2!'

    if (l == 1):
        if (n == 0):
            norm_over_phi = 2 * math.pi
        else:
            norm_over_phi = np.sin(4 * math.pi * n) / (4 * n) + math.pi
    else:
        if (n == 0):
            norm_over_phi = 0
        else:
            norm_over_phi = (math.pi - np.sin(4 * math.pi * n) / (4 * n))

    return norm_over_phi

def get_norm_over_r(R, delta, alpha, n):
    if (alpha == 0):
        over_r = (0.5 * ((R + delta) ** 2) )*(n == 0)
    else:
        over_r = 0.5 * (((R + delta) / alpha) ** 2) * alpha * \
             (alpha * (ss.jv(n, alpha)) ** 2 - 2 * n * ss.jv(n + 1, alpha) * ss.jv(n, alpha) + alpha * (ss.jv(n + 1, alpha)) ** 2)
    return over_r
#################################################################################################

def u_nkl_not_normed(angles, radii, n, l, alpha, R, delta):
    vert = len(angles)
    u_nkl = np.zeros(vert)
    for j in range(0, vert):
        scale_ln = 0 # this is just to make sure that I initialize this
        scale_ln = scale_ln + np.cos(n * angles[j]) * (l == 1) + np.sin(n * angles[j]) * (l > 1 and n!= 0)
        u_nkl[j] = ss.jv(n, alpha * radii[j] /( R + delta))*scale_ln

    return u_nkl

def normalize(u, norm_squared):
    if norm_squared != 0:
        u_norm = u / np.sqrt(norm_squared)
    else:
        u_norm = u
    return u_norm


@njit(parallel=True)
def etimated_cov_mat(vert, eta, eigenvalue_nk, alphalapl, u_nkl_norm):
    expressionAcc = np.zeros((vert, vert))
    for m in prange(0, vert):
        for n in range(0, vert):
            expressionAcc[m][n] = expressionAcc[m][n] + (eta ** 2) * (eigenvalue_nk ** (-alphalapl)) * u_nkl_norm[m] * u_nkl_norm[n]
    return expressionAcc
