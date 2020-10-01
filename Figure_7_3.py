import dolfin as df
import numpy as np
from fenics import *
from scipy import *
from scipy import optimize
import matplotlib.pyplot as plt
from numba import jit
from functions import covariance_matern_jit

#SET UP SPDE PARAMETERS
nu = 1
rhos = [0.1, 0.2, 0.4, 0.6, 0.8, 1] #rho values for the testing
sigma = 1

# SET UP THE 1D DOMAIN D
L = 1 #length of the domain
length = 50 #number of nodes
nodes = np.linspace(0, L, length) #set the nodes itself

#SET UP VALUES OF BETA
number_of_tests = 50
max_beta_over_kappa = 1
min_beta_over_kappa = 0.01
betas_over_k = np.arange(number_of_tests) * (1 / (number_of_tests - 1)) * (max_beta_over_kappa - min_beta_over_kappa) + min_beta_over_kappa

#reserve the arrays to store the results
error_matrix = np.zeros((length, length))
results_robin = np.zeros((number_of_tests, np.size(rhos)))

def alpha_function(x,beta): #function to obtain alpha_n (Equation 5.5)
    output = 2 * x * np.cos(x) / (beta * L) + (1 - ((x / (beta * L)) ** 2)) * np.sin(x)
    return output

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

for l in range(0, np.size(rhos)):
    # SET UP THE SPDE PARAMETERS THAT ARE DEPENDENT ON RHO
    rho = rhos[l]
    kappa = df.sqrt(2 * nu) / rho
    alphalapl = nu + 1. / 2.
    white_noise_scale = kappa ** 2 * np.sqrt(sigma ** 2 * np.sqrt(4 * pi) * math.gamma(nu + 0.5) / (kappa * math.gamma(nu)))
    eta = white_noise_scale / (kappa ** 2)
    # compute the value of true covariance
    true_expression = loop3(nodes, np.zeros(length), kappa, nu, length)
    if rho < 0.6:
        number_of_eigens = 40000
    else:
        number_of_eigens = 2000
    for p in range(0,number_of_tests):
        beta = betas_over_k[p]*kappa
        print(beta)
        expression = np.zeros((length, length))
        for k in range(0, number_of_eigens):
            if beta == 0:
                eigenvalue = 1 + (pi * k / (kappa * L))**2
                eigenfunction = np.cos(pi*k*nodes/L)
                if k == 0:
                    normalization_constant = 2*pi
                else:
                    normalization_constant = ((L*(2*pi*k + np.sin(2*pi*k)))/(4*pi*k)) ** (0.5)
            else:
                alpha = optimize.bisect(alpha_function, k * pi + 0.001 * (k == 0), (k + 1) * pi, args=(beta))
                eigenvalue = 1 + (alpha / (kappa * L)) ** 2
                eigenfunction = np.sin(alpha * nodes/ L) + (alpha / (beta * L)) * np.cos(alpha * nodes/ L)
                normalization_constant = ((alpha ** 2 + 2 * beta * L + (beta * L) ** 2) / (2 * L * (beta ** 2))) ** (0.5)

            eigenfunction = eigenfunction / normalization_constant
            expression = expression + loop2(length, eta, eigenvalue, alphalapl, eigenfunction)
            error_matrix = np.absolute(true_expression - expression)
            results_robin[p,l] = error_matrix.max()
    plt.semilogy(betas_over_k, results_robin[:, l], label=r'error Robin $\rho =$ %.2f' % rho)

plt.semilogy(betas_over_k, np.ones(number_of_tests), label = 'error Dirichlet')
plt.ylabel(r'1D, $L = %.1f $' %L)
plt.xlabel(r'($\beta\ \kappa$-values from %.2f to %.2f, $\nu =$ %.2f)' % (betas_over_k[0], betas_over_k[number_of_tests-1], nu),
           fontsize=14)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()