from timeit import default_timer as timer
from numba import jit
import dolfin as df
import scipy.special as ss
import numpy as np
import math as math

from mshr.cpp import Circle, generate_mesh
from scipy import optimize
import matplotlib.pyplot as plt
from functions import covariance_matern_jit

############### SET THE PDE PARAMETERS ################################
nu = 50
rho = 1

kappa = df.sqrt(2 * nu) / rho
alphalapl = nu + 1./2.
sigma = 1
white_noise_scale = kappa ** 2 * np.sqrt(sigma ** 2 * np.sqrt(4 * math.pi) * math.gamma(nu + 0.5) / (kappa * math.gamma(nu)))
eta = white_noise_scale / (kappa ** 2)
###########################################################################

####### FIXED PARAMETERS FOR APPROXIMATION ################
# 1. DOMAIN: LENGTH L AND WINDOW SIZES THAT WE TEST
L = 1
vert = 50 #number of vertices
number_of_tests = 8
max_window_size = 2 * rho
deltas = np.arange(number_of_tests) * (1 / (number_of_tests - 1)) * max_window_size

# 2. ROBIN BOUNDARY set beta for the equation beta*u + (du/dn) = 0
beta = kappa
beta_opt = 0.9959183673469388

#how many terms in the sum?
number_of_eigens = 5000
########################################################################################

@jit(nopython=True)
def loop2(length, eta, eigenvalue_nk, alphalapl, u_nkl_norm):
    expressionAcc = np.zeros((length, length))
    for m in range(0, length):
        for n in range(0, length):
            expressionAcc[m][n] = expressionAcc[m][n] + (eta ** 2) * (eigenvalue_nk ** (-alphalapl)) * u_nkl_norm[m] * u_nkl_norm[n]
    return expressionAcc

@jit(nopython=False)
def loop3(x, y, kappa, nu, length):
    true_expression = np.zeros((length, length))
    for pp in range(0,length):
        for tt in range(0, length):
            true_expression[pp][tt] = covariance_matern_jit.matern_cov(np.array([x[tt], y[tt]]), np.array([x[pp], y[pp]]), kappa, nu)
    return true_expression


results_robin = np.zeros(number_of_tests)
results_robin_opt = np.zeros(number_of_tests)
results_dir = np.zeros(number_of_tests)
results_neu = np.zeros(number_of_tests)

def alpha_function(x,beta, L):
    output = 2 * x * np.cos(x) / (beta * L) + (1 - ((x / (beta * L)) ** 2)) * np.sin(x)
    return output

@jit(nopython=True)
def loop2(length, eta, eigenvalue_nk, alphalapl, u_nkl_norm):
    expressionAcc = np.zeros((length, length))
    for m in range(0, length):
        for n in range(0, length):
            expressionAcc[m][n] = expressionAcc[m][n] + (eta ** 2) * (eigenvalue_nk ** (-alphalapl)) * u_nkl_norm[m] * u_nkl_norm[n]
    return expressionAcc

for i in range(0, number_of_tests):  # ENTER MAIN LOOP OVER WINDOW SIZES
    delta = deltas[i]

    expression_robin = np.zeros((vert, vert))
    expression_robin_opt = np.zeros((vert, vert))

    nodes = np.linspace(delta*0.5, L + (delta*0.5), vert)
    true = loop3(nodes, np.zeros(vert), kappa, nu, vert)

    for n in range(0, number_of_eigens):
        alpha = optimize.bisect(alpha_function, n * math.pi + 0.001 * (n == 0), (n + 1) * math.pi, args=(beta, L+delta))
        eigenvalue_robin = 1 + (alpha / (kappa * (L+ delta))) ** 2
        eigenfunction_robin = np.sin(alpha * nodes/ (L+ delta)) + (alpha / (beta * (L+ delta))) * np.cos(alpha * nodes/ (L+ delta))
        normalization_constant = ((alpha ** 2 + 2 * beta * (L+ delta) + (beta * (L+ delta)) ** 2) / (2 * (L+ delta) * (beta ** 2))) ** (0.5)

        eigenfunction = eigenfunction_robin / normalization_constant
        expression_robin = expression_robin + loop2(vert, eta, eigenvalue_robin, alphalapl, eigenfunction)

        alpha_opt = optimize.bisect(alpha_function, n * math.pi + 0.001 * (n == 0), (n + 1) * math.pi,
                                args=(beta_opt, L + delta))
        eigenvalue_robin_opt = 1 + (alpha_opt / (kappa * (L + delta))) ** 2
        eigenfunction_robin_opt = np.sin(alpha_opt * nodes / (L + delta)) + (alpha_opt / (beta_opt * (L + delta))) * np.cos(
            alpha_opt * nodes / (L + delta))
        normalization_constant_opt = ((alpha_opt ** 2 + 2 * beta_opt * (L + delta) + (beta_opt * (L + delta)) ** 2) / (
                    2 * (L + delta) * (beta_opt ** 2))) ** (0.5)

        eigenfunction_opt = eigenfunction_robin_opt / normalization_constant_opt
        expression_robin_opt = expression_robin_opt + loop2(vert, eta, eigenvalue_robin_opt, alphalapl, eigenfunction_opt)

    error_matrix = np.absolute(true - expression_robin)
    result = error_matrix.max()
    results_robin[i] = result

    error_matrix_opt = np.absolute(true - expression_robin_opt)
    result_opt = error_matrix_opt.max()
    results_robin_opt[i] = result_opt
plt.semilogy(deltas, results_robin, label='Robin',marker='o')
plt.semilogy(deltas, results_robin_opt, label='Robin wth optimal parameter',marker='o')
plt.xlabel(r'(window size, $\rho =$ %.2f, $\nu =$ %.2f)' % (rho, nu),
           fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

