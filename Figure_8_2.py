from timeit import default_timer as timer
from numba import jit
import dolfin as df
import scipy.special as ss
import numpy as np
import math as math

from mshr.cpp import Circle, generate_mesh
from scipy import optimize
import matplotlib.pyplot as plt
from functions import nodal, not_nodal
############### SET THE PDE PARAMETERS ################################
nus = [0.25, 0.5, 0.7, 1, 5, 20, 50]
rho = 1
sigma = 1
###########################################################################



####### FIXED PARAMETERS FOR APPROXIMATION ################
# 1. DOMAIN: RADIUS AND WINDOW SIZES THAT WE TEST
R = 0.5
number_of_tests = 30
max_beta = 1
min_beta = 0
betas = np.arange(number_of_tests) * (1 / (number_of_tests - 1)) * (max_beta - min_beta) + min_beta

delta = 0
# 2. ROBIN BOUNDARY set beta for the equation beta*u + (du/dn) = 0
 # BETA HERE IS FIXED WE TEST WINDOW SIZE

# 4. from GEOMETRICAL STRUCTURE OF LAPLACIAN EIGENFUNCTIONS by D. S. GREBENKOV
# AND B.-T. NGUYEN, Section 3.2:
# u_nkl are the eigenfunctions; l is bound to be 1 or 2 but n and k can be chosen
roots = 20  # choose k
param_n = 30  # choose n

#######################################################################################

################ GET NODES ON THE CIRCLE TO WORK WITH ##################################
domain = Circle(df.Point(0, 0), R)
mesh = generate_mesh(domain, 12)  # change the second parameter to get more nodes
# df.plit(mesh)

coord = mesh.coordinates()
vert = coord.shape[0]  # number of vertices
y = coord[:, 1]
x = coord[:, 0]
########################################################################################
angles, radii = nodal.polar(x, y)  # GET POLAR COORDINATES

#true = not_nodal.true_cov_mat(x, y, kappa, nu)  # GET THE TRUE MATRIX

results_robin = np.zeros((number_of_tests, np.size(nus)))

####WARNING! SciPy routine jn_zeros() hangs  without any error message when trying to compute zeros for these orders:
# more on this: https://github.com/scipy/scipy/issues/11994
banned_list_for_n = [231, 244, 281, 288, 347, 348, 357, 405, 406, 419, 437,
                     505, 506, 507, 570, 582, 591, 643, 644, 655, 658, 679,
                     706, 713, 722, 752, 756, 757, 764, 775, 793, 796, 811,
                     820, 840, 855, 875, 886, 916, 942, 948, 966]
# These n are saved in extra list and the loop just skips them
for m in range(0, np.size(nus)):
    nu = nus[m]
    print('nu = %.2f' % nu)
    alphalapl = nu + 1.
    kappa = np.sqrt(2 * nu) / rho
    eta = np.sqrt(sigma ** 2 * 4 * math.pi * math.gamma(nu + 1) / ((kappa ** 2) * math.gamma(nu)))
    true = not_nodal.true_cov_mat(x, y, kappa, nu)  # GET THE TRUE MATRIX

    for i in range(0, number_of_tests):  # ENTER MAIN LOOP OVER WINDOW SIZES

        beta = betas[i]
        print('beta = %.2f' % beta)
        expression_robin = np.zeros((vert, vert))

        for n in range(0, param_n):
            if (n in banned_list_for_n):
                continue  # skip the step if n is in the banned list
            print(n, i)

            a = ss.jnp_zeros(n, roots)  # TODO a,b,c independent of window (alpha_nk also) how to compute them only once?
            b = ss.jn_zeros(n, roots)
            if (beta == 0):
                c = a
            else:
                c = not_nodal.compute_all_robin_roots_new(beta, n, R, roots)

            for k in range(0, roots):
                if (beta == 0):
                    alpha_nk_robin = not_nodal.get_alpha_nk('Neumann', n, k, b, a, c, beta)
                else:
                    alpha_nk_robin = not_nodal.get_alpha_nk('Robin', n, k, b, a, c, beta)


                eigenvalue_nk_robin = not_nodal.get_eigenvalue(alpha_nk_robin, R, delta, kappa)


                for l in range(1, 3):
                    norm_over_phi = not_nodal.get_norm_over_phi(n, l)

                    norm_over_r_robin = not_nodal.get_norm_over_r(R, delta, alpha_nk_robin, n)

                    norm_robin = norm_over_r_robin * norm_over_phi

                    ####################### ESTIMATION OF U_NKL THEMSELVES#################################
                    u_nkl_robin = not_nodal.u_nkl_not_normed(angles, radii, n, l, alpha_nk_robin, R, delta)

                    u_nkl_robin_norm = not_nodal.normalize(u_nkl_robin, norm_robin)

                    expression_robin = expression_robin + not_nodal.etimated_cov_mat(vert, eta, eigenvalue_nk_robin,
                                                                                     alphalapl,
                                                                                     u_nkl_robin_norm)


        error_matrix_robin = np.absolute(true - expression_robin)

        result_robin = error_matrix_robin.max()

        results_robin[i,m] = result_robin

        ind = np.unravel_index(np.argmax(error_matrix_robin, axis=None), error_matrix_robin.shape)
        print(result_robin, 'maximum error robin', coord[ind[0]],
              coord[ind[1]])  # coord[ind[0]] coord[ind[1]] gives back vertices that have the largest error
    plt.semilogy(betas, results_robin[:,m], label = r'error Robin $\nu =$ %.2f' %nu)

plt.semilogy(betas, np.ones(number_of_tests), label = 'error Dirichlet')
plt.ylabel(r'circular domain, $R = %.1f $' %R)
plt.xlabel(r'($\beta$-values from %.2f to %.2f, $\rho =$ %.1f)' % (betas[0], betas[number_of_tests-1], rho),
           fontsize=14)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
