import dolfin as df
import scipy.special as ss
import numpy as np
import math as math

from mshr.cpp import Circle, generate_mesh
from scipy import optimize
import matplotlib.pyplot as plt
from functions import covariance_matern, nodal, not_nodal

############### SET THE PARAMETERS ################################
nu = 1
l = 3
rho =  l * df.sqrt(8 * nu)
kappa = 1. / l
alphalapl = nu + 1.
sigma = 1
eta = np.sqrt(sigma ** 2 * 4 * math.pi * math.gamma(nu + 1) / ((kappa**2) * math.gamma(nu)))
###########################################################################

####### FIXED PARAMETERS FOR APPROXIMATION ################
#1. DOMAIN: RADIUS AND WINDOW
delta = 0
R = 10
#2. ROBIN BOUNDARY set beta for the equation beta*u + (du/dn) = 0
beta = 1. / (1.42 * l)

#3. Reference point for autocorrelation function for recreation Of Fig. 4
#  from "WHITTLE-MATÉRN PRIORS FOR BAYESIAN STATISTICAL
# INVERSION WITH APPLICATIONS IN ELECTRICAL
# IMPEDANCE TOMOGRAPHY by Roininen Huttunen Lasanen
origin_x = 7.85
origin_y = 0
origin = np.array([origin_x, origin_y])

#4. from GEOMETRICAL STRUCTURE OF LAPLACIAN EIGENFUNCTIONS by D. S. GREBENKOV
# AND B.-T. NGUYEN, Section 3.2:
#u_nkl are the eigenfunctions; l is bound to be 1 or 2 but n and k can be chosen
roots = 20 #choose k
param_n = 30 #choose n
#######################################################################################

################ GET NODES ON THE CIRCLE TO WORK WITH ##################################
domain = Circle(df.Point(0, 0), R)
mesh = generate_mesh(domain, 20) #change the second parameter to get more nodes

coord = mesh.coordinates()
vert = coord.shape[0]  # number of vertices
y = coord[:, 1]
x = coord[:, 0]
########################################################################################

##################### GET POLAR COORDINATES AND DISTANCE TO REF POINT###################
angles, radii = nodal.polar(x,y)
phi_origin, radius_origin = nodal.polar(np.array([origin_x]), np.array([origin_y]))
distances = nodal.distance_to_fixed_point(origin, coord)
#########################################################################################

true_expression = not_nodal.true(origin, x, y, kappa, nu)


###################STORAGE FOR COVARIANCE FUNCTIONS##################
expression = np.zeros(vert)
expression_dir = np.zeros(vert)
expression_neu = np.zeros(vert)
for n in range(0,param_n):
    print(n) #just to see how it is going
    a = ss.jnp_zeros(n, roots)
    b = ss.jn_zeros(n, roots)
    c = not_nodal.compute_all_robin_roots_new(beta, n, R, roots)
    for k in range(0,roots):
        alpha_nk_robin = not_nodal.get_alpha_nk('Robin', n, k, b, a, c, beta)
        alpha_nk_dirichlet = not_nodal.get_alpha_nk('Dirichlet', n, k, b, a, c, beta)
        alpha_nk_neumann = not_nodal.get_alpha_nk('Neumann', n, k, b, a, c, beta)

        eigenvalue_nk_robin = not_nodal.get_eigenvalue(alpha_nk_robin, R, delta, kappa)
        eigenvalue_nk_dirichlet = not_nodal.get_eigenvalue(alpha_nk_dirichlet, R, delta, kappa)
        eigenvalue_nk_neumann = not_nodal.get_eigenvalue(alpha_nk_neumann, R, delta, kappa)

        for l in range(1,3):

            norm_over_phi = not_nodal.get_norm_over_phi(n, l)

            norm_over_r_robin = not_nodal.get_norm_over_r(R, delta, alpha_nk_robin, n)
            norm_over_r_dirichlet = not_nodal.get_norm_over_r(R, delta, alpha_nk_dirichlet, n)
            norm_over_r_neumann = not_nodal.get_norm_over_r(R, delta, alpha_nk_neumann, n)

            norm_robin = norm_over_r_robin * norm_over_phi
            norm_dirichlet = norm_over_r_dirichlet * norm_over_phi
            norm_neumann = norm_over_r_neumann * norm_over_phi

####################### ESTIMATION OF U_NKL THEMSELVES#################################
            u_nkl_robin = not_nodal.u_nkl_not_normed(angles, radii, n, l, alpha_nk_robin, R, delta)
            u_nkl_neumann = not_nodal.u_nkl_not_normed(angles, radii, n, l, alpha_nk_neumann, R, delta)
            u_nkl_dirichlet = not_nodal.u_nkl_not_normed(angles, radii, n, l, alpha_nk_dirichlet, R, delta)

            origin_u_nkl_robin = not_nodal.u_nkl_not_normed(np.array([phi_origin]), np.array([radius_origin]), n, l, alpha_nk_robin, R, delta)
            origin_u_nkl_dirichlet = not_nodal.u_nkl_not_normed(np.array([phi_origin]), np.array([radius_origin]), n, l, alpha_nk_dirichlet, R, delta)
            origin_u_nkl_neumann = not_nodal.u_nkl_not_normed(np.array([phi_origin]), np.array([radius_origin]), n, l, alpha_nk_neumann, R, delta)

            u_nkl_robin_norm = not_nodal.normalize(u_nkl_robin, norm_robin)
            origin_u_nkl_robin_norm = not_nodal.normalize(origin_u_nkl_robin, norm_robin)

            u_nkl_dir_norm = not_nodal.normalize(u_nkl_dirichlet, norm_dirichlet)
            origin_u_nkl_dir_norm = not_nodal.normalize(origin_u_nkl_dirichlet, norm_dirichlet)

            u_nkl_neu_norm = not_nodal.normalize(u_nkl_neumann, norm_neumann)
            origin_u_nkl_neu_norm = not_nodal.normalize(origin_u_nkl_neumann, norm_neumann)

            #here I update the covariance for each computed u_nkl
            for m in range(0, vert):
                expression[m] = np.add(expression[m],
                                              (eta ** 2) * (eigenvalue_nk_robin ** (-alphalapl )) * origin_u_nkl_robin_norm *
                                               u_nkl_robin_norm[m])

                expression_dir[m] = np.add(expression_dir[m],
                                       (eta ** 2) * (eigenvalue_nk_dirichlet ** (-alphalapl)) * origin_u_nkl_dir_norm *
                                       u_nkl_dir_norm[m])

                expression_neu[m] = np.add(expression_neu[m],
                                        (eta ** 2) * (eigenvalue_nk_neumann ** (-alphalapl)) * origin_u_nkl_neu_norm *
                                        u_nkl_neu_norm[m])

plt.scatter(distances, expression, alpha=0.8, s=10, label='Robin')
plt.scatter(distances, expression_dir, alpha=0.8, s=10, label='Dirichlet')
plt.scatter(distances, expression_neu, alpha=0.8, s=10, label='Neumann')
plt.scatter(distances, true_expression, alpha=0.8, s=10, label='true')
plt.legend()
plt.xlabel('distance')
plt.title('test')
plt.show()


