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
nu = 1
rho = 1
kappa = df.sqrt(2 * nu) / rho
alphalapl = nu + 1.
sigma = 1
eta = np.sqrt(sigma ** 2 * 4 * math.pi * math.gamma(nu + 1) / ((kappa ** 2) * math.gamma(nu)))
###########################################################################

####### FIXED PARAMETERS FOR APPROXIMATION ################
# 1. DOMAIN: RADIUS AND WINDOW SIZES THAT WE TEST
R = 0.5
number_of_tests = 3
max_window_size = 0.7 * rho
deltas = np.arange(number_of_tests) * (1 / (number_of_tests - 1)) * max_window_size
# 2. ROBIN BOUNDARY set beta for the equation beta*u + (du/dn) = 0
beta = kappa #1.226613804099113  # BETA HERE IS FIXED WE TEST WINDOW SIZE
print(kappa)
beta_opt = 1.2*kappa #8.2 #1.2*kappa #4.91 # #5 #1.226613804099113 #8.2 # #1.226613804099113
print('beta opt %f' %beta_opt)
# 4. from GEOMETRICAL STRUCTURE OF LAPLACIAN EIGENFUNCTIONS by D. S. GREBENKOV
# AND B.-T. NGUYEN, Section 3.2:
# u_nkl are the eigenfunctions; l is bound to be 1 or 2 but n and k can be chosen
roots = 30  # choose k
param_n = 30  # choose n


################ GET NODES ON THE CIRCLE TO WORK WITH ##################################
domain = Circle(df.Point(0, 0), R)
mesh = generate_mesh(domain, 12)  # change the second parameter to get more nodes
#df.plot(mesh)

coord = mesh.coordinates()
vert = coord.shape[0]  # number of vertices
y = coord[:, 1]
x = coord[:, 0]
########################################################################################
angles, radii = nodal.polar(x, y)  # GET POLAR COORDINATES

true = not_nodal.true_cov_mat(x, y, kappa, nu)  # GET THE TRUE MATRIX

results_robin = np.zeros(number_of_tests)
results_robin_opt = np.zeros(number_of_tests)
results_dir = np.zeros(number_of_tests)
results_neu = np.zeros(number_of_tests)

####WARNING! SciPy routine jn_zeros() hangs  without any error message when trying to compute zeros for these orders:
# more on this: https://github.com/scipy/scipy/issues/11994
banned_list_for_n = [231, 244, 281, 288, 347, 348, 357, 405, 406, 419, 437,
                     505, 506, 507, 570, 582, 591, 643, 644, 655, 658, 679,
                     706, 713, 722, 752, 756, 757, 764, 775, 793, 796, 811,
                     820, 840, 855, 875, 886, 916, 942, 948, 966]
# These n are saved in extra list and the loop just skips them

for i in range(0, number_of_tests):  # ENTER MAIN LOOP OVER WINDOW SIZES
    delta = deltas[i] * 0.5  # half of window size will be added to R

    expression_robin = np.zeros((vert, vert))
    expression_robin_opt = np.zeros((vert, vert))
    expression_dirichlet = np.zeros((vert,vert))
    expression_neumann = np.zeros((vert, vert))

    for n in range(0, param_n):
        if (n in banned_list_for_n):
            continue  # skip the step if n is in the banned list
        print(n, i)

        a = ss.jnp_zeros(n, roots)  # TODO a,b,c independent of window (alpha_nk also) how to compute them only once?
        b = ss.jn_zeros(n, roots)
        c = not_nodal.compute_all_robin_roots_new(beta, n, R, roots)
        c_opt = not_nodal.compute_all_robin_roots_new(beta_opt, n, R, roots)

        for k in range(0, roots):
            alpha_nk_robin = not_nodal.get_alpha_nk('Robin', n, k, b, a, c, beta)
            alpha_nk_robin_opt = not_nodal.get_alpha_nk('Robin', n, k, b, a, c_opt, beta_opt)
            alpha_nk_dirichlet = not_nodal.get_alpha_nk('Dirichlet', n, k, b, a, c_opt, beta_opt)
            alpha_nk_neumann = not_nodal.get_alpha_nk('Neumann', n, k, b, a, c_opt, beta_opt)

            eigenvalue_nk_robin = not_nodal.get_eigenvalue(alpha_nk_robin, R, delta, kappa)
            eigenvalue_nk_robin_opt = not_nodal.get_eigenvalue(alpha_nk_robin_opt, R, delta, kappa)
            eigenvalue_nk_dirichlet = not_nodal.get_eigenvalue(alpha_nk_dirichlet, R, delta, kappa)
            eigenvalue_nk_neumann = not_nodal.get_eigenvalue(alpha_nk_neumann, R, delta, kappa)

            for l in range(1, 3):
                norm_over_phi = not_nodal.get_norm_over_phi(n, l)

                norm_over_r_robin = not_nodal.get_norm_over_r(R, delta, alpha_nk_robin, n)
                norm_over_r_robin_opt = not_nodal.get_norm_over_r(R, delta, alpha_nk_robin_opt, n)
                norm_over_r_dirichlet = not_nodal.get_norm_over_r(R, delta, alpha_nk_dirichlet, n)
                norm_over_r_neumann = not_nodal.get_norm_over_r(R, delta, alpha_nk_neumann, n)

                norm_robin = norm_over_r_robin * norm_over_phi
                norm_robin_opt = norm_over_r_robin_opt * norm_over_phi
                norm_dirichlet = norm_over_r_dirichlet * norm_over_phi
                norm_neumann = norm_over_r_neumann * norm_over_phi

                ####################### ESTIMATION OF U_NKL THEMSELVES#################################
                u_nkl_robin = not_nodal.u_nkl_not_normed(angles, radii, n, l, alpha_nk_robin, R, delta)
                u_nkl_robin_opt = not_nodal.u_nkl_not_normed(angles, radii, n, l, alpha_nk_robin_opt, R, delta)
                u_nkl_neumann = not_nodal.u_nkl_not_normed(angles, radii, n, l, alpha_nk_neumann, R, delta)
                u_nkl_dirichlet = not_nodal.u_nkl_not_normed(angles, radii, n, l, alpha_nk_dirichlet, R, delta)

                u_nkl_robin_norm = not_nodal.normalize(u_nkl_robin, norm_robin)
                u_nkl_robin_norm_opt = not_nodal.normalize(u_nkl_robin_opt, norm_robin_opt)
                u_nkl_dir_norm = not_nodal.normalize(u_nkl_dirichlet, norm_dirichlet)
                u_nkl_neu_norm = not_nodal.normalize(u_nkl_neumann, norm_neumann)

                expression_robin = expression_robin + not_nodal.etimated_cov_mat(vert, eta, eigenvalue_nk_robin,
                                                                                 alphalapl,
                                                                                 u_nkl_robin_norm)
                expression_robin_opt = expression_robin_opt + not_nodal.etimated_cov_mat(vert, eta, eigenvalue_nk_robin_opt,
                                                                                 alphalapl,
                                                                                 u_nkl_robin_norm_opt)
                expression_dirichlet = expression_dirichlet + not_nodal.etimated_cov_mat(vert, eta,
                                                                                         eigenvalue_nk_dirichlet,
                                                                                         alphalapl, u_nkl_dir_norm)
                expression_neumann = expression_neumann + not_nodal.etimated_cov_mat(vert, eta, eigenvalue_nk_neumann,
                                                                                     alphalapl, u_nkl_neu_norm)

    error_matrix_robin = np.absolute(true - expression_robin)
    error_matrix_robin_opt = np.absolute(true - expression_robin_opt)
    error_matrix_dir = np.absolute(true - expression_dirichlet)
    error_matrix_neu = np.absolute(true - expression_neumann)

    result_robin = error_matrix_robin.max()
    result_robin_opt = error_matrix_robin_opt.max()
    result_dir = error_matrix_dir.max()
    result_neu = error_matrix_neu.max()

    results_robin[i] = result_robin
    results_robin_opt[i] = result_robin_opt
    results_dir[i] = result_dir
    results_neu[i] = result_neu

    ind = np.unravel_index(np.argmax(error_matrix_robin, axis=None), error_matrix_robin.shape)
    ind_opt = np.unravel_index(np.argmax(error_matrix_robin_opt, axis=None), error_matrix_robin_opt.shape)
    print(result_robin, 'maximum error Robin', coord[ind[0]],
          coord[ind[1]])  # coord[ind[0]] coord[ind[1]] gives back vertices that have the largest error
    print(result_robin_opt, 'maximum error Robin with optimal beta', coord[ind_opt[0]],
          coord[ind_opt[1]])
    print(result_dir, 'maximum error Dirichlet')
    print(result_neu, 'maximum error Neumann')

plt.semilogy(deltas, results_robin, label='Robin',marker='o')
plt.semilogy(deltas, results_robin_opt, label='Robin wth optimal parameter',marker='o')
plt.semilogy(deltas, results_dir, label='Dirichlet', marker='o')
plt.semilogy(deltas, results_neu, label='Neumann', marker='o')
plt.ylabel('maximum error on the circle with radius = %.1f' %R)
plt.xlabel('(window size, rho = %.2f, nu = %.2f)' % (rho, nu),
           fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

