"""
Comparison of Segmentation using
Wasserstein Distance and Optimal Transport Map
of Gaussian Processes
"""
import numpy as np
import scipy.io
import scipy.linalg
import pickle
from matplotlib import pyplot as plt
from wgpot import GP_W_barycenter, Wasserstein_GP, logmap, expmap
from utils import Plot_GP
import time

def compareSegs(file1, file2, plot = False):

    file_1 = open(file1, 'rb')
    gp_list = pickle.load(file_1)
    file_1.close()
    
    file_2 = open(file2, 'rb')
    gp2_list = pickle.load(file_2)
    file_2.close()
    
    # Notice: Visualize all the GPs
    #  Plot all the GPs
    if plot== True:
        fig = plt.figure(1)
        mean_alpha = 0.1
        var_alpha = 0.02
        timeSpace = np.arange(1, 51).reshape(-1, 1)
        for index, gp in enumerate(gp_list):
            mu, K= gp
            if index == 0:
                Plot_GP(plt, timeSpace, mu, K, 'b', mean_alpha, var_alpha, 'GPs')
            else:
                Plot_GP(plt, timeSpace, mu, K, 'b', mean_alpha, var_alpha)
                
        for index2, gp2 in enumerate(gp2_list):
            mu2, K2= gp2
            if index2 == 0:
                Plot_GP(plt, timeSpace, mu2, K2, 'g', mean_alpha, var_alpha, 'GPs2')
            else:
                Plot_GP(plt, timeSpace, mu2, K2, 'g', mean_alpha, var_alpha)
            # break
        plt.xlabel('Time')
        plt.ylabel('Position')
    
    # Notice: Compute the Wasserstein distance of two GPs
    """ gp_0 = gp_list[0]
    gp_1 = gp_list[1]
    wd_gp = Wasserstein_GP(gp_0, gp_1)
    wd_gp2 = Wasserstein_GP(gp_1, gp_0)
    print('The Wasserstein distance of two GPs is ', wd_gp, wd_gp2) """
    
    #* Compute the Wasserstein Barycenter of this set of GPs to compare the segments
    mu_bc, K_bc = GP_W_barycenter(gp_list)
    mu2_bc, K2_bc = GP_W_barycenter(gp2_list)
    val = Wasserstein_GP([mu_bc,K_bc], [mu2_bc,K2_bc])
    print("Valor del Wassersteins entre dos: ", val)
    if plot == True:
        fig = plt.figure(1)
        Plot_GP(plt, timeSpace, mu_bc, K_bc, 'r', 1, 0.5, 'Barycenter')
        Plot_GP(plt, timeSpace, mu2_bc, K2_bc, 'c', 1, 0.5, 'Barycenter2')
        plt.legend()
        plt.title('Wasserstein Barycenter of the segments (red and cyan)')
        plt.show()
        plt.savefig('data/barycenter_result.png', bbox_inches='tight')

    #*Obtain the optimal transport map between two GPs
    

    #?Obtain the push forward of GPs. It's the elements on the principal geodesic
    v_mu, v_T = logmap(mu_bc, K_bc, mu2_bc, K2_bc)
    coste_cum = 0
    
    resolution_points = [0.2, 0.4, 0.6, 0.8, 1.0]

    """ for t in resolution_points:
        # Calcular el desplazamiento en el punto t
        v_mu_t = t * v_mu
        v_T_t = t * v_T
        
        # Aplicar el mapeo exponencial para obtener el nuevo GP en el punto t
        q_mu, q_K = expmap(mu_bc, K_bc, v_mu_t, v_T_t)
        
        # Calcular el coste de Wasserstein desde GP_1 a q (punto en la geodésica)
        wd_gp_to_q = Wasserstein_GP([mu2_bc, K2_bc], [q_mu, q_K])
        # Calcular el coste de Wasserstein desde q a GP_0
        wd_q_to_0 = Wasserstein_GP([q_mu, q_K], [mu_bc, K_bc])
        
        print(f"Coste de movimiento desde GP_1 a punto intermedio q: {wd_gp_to_q}")
        print(f"Coste de movimiento desde punto intermedio q a GP_0: {wd_q_to_0}")
        
        # Acumular el coste total
        #coste_cum += abs(wd_gp_to_q - wd_q_to_0) """
    
    for t in resolution_points: #* The resolution of the Geodesic can be modified by the user
        v_mu_t = t * v_mu
        v_T_t = t * v_T
        q_mu, q_K = expmap(mu2_bc, K2_bc, v_mu_t, v_T_t)
        wd_gpCum = Wasserstein_GP([q_mu,q_K], [mu2_bc,K2_bc])
        wd_gpCum1 = Wasserstein_GP( [mu2_bc,K2_bc],[q_mu,q_K])
        wd_gpCum2 = Wasserstein_GP([q_mu,q_K], [mu_bc,K_bc])
        print("Valor 1:", wd_gpCum)
        """ print("Valor 2:", wd_gpCum1)
        print("Valor 3:", wd_gpCum2) """
        coste_cum = coste_cum + wd_gpCum
    #print("Cumulative cost: ", coste_cum)
    
    if plot == True:
        fig = plt.figure(2)
        for j in [0.2, 0.4, 0.6, 0.8]:
            Plot_GP(plt, timeSpace, q_mu, q_K, 'orange', 0.5, 0.25, 'geodesic t=' + str(j))
        Plot_GP(plt, timeSpace, mu_bc, K_bc, 'r', 1, 0.5, 'GP_0')
        Plot_GP(plt, timeSpace, mu2_bc, K2_bc, 'b', 1, 0.5, 'GP_1')
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.legend()
        plt.savefig('data/transport_result.png', bbox_inches='tight')
        plt.show()
    return coste_cum