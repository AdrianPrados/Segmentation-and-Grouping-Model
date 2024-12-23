import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import matplotlib.pyplot as plt
import os
import time
import pickle


def save_gp_data(data, file_name):
    """
    Save GP data to a pickle file inside the GaussianPkl folder.
    """
    folder_path = 'GaussianPkl'
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
        print(f"Data saved to {file_path}")

def plot_gp(gp_x, gp_y, gp_z, t, x, y,z,size):
    """
    Plot the Gaussian Processes for both x and y dimensions.
    Parameters:
    - gp_x: Gaussian Process for x dimension
    - gp_y: Gaussian Process for y dimension
    - t: Original time vector (non-normalized)
    - x: Original x coordinates (non-normalized)
    - y: Original y coordinates (non-normalized)
    """
    # Create a grid of time points for the plot
    t_pred = np.linspace(min(t), max(t), size).reshape(-1, 1)
    
    # Predict GP mean and covariance for x dimension
    mu_x_pred, sigma_x_pred = gp_x.predict(t_pred, return_std=True)
    
    # Predict GP mean and covariance for y dimension
    mu_y_pred, sigma_y_pred = gp_y.predict(t_pred, return_std=True)
    
    if gp_z!=None:
        mu_z_pred, sigma_z_pred = gp_z.predict(t_pred, return_std=True)
    
    
    # Plot for dimension X
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 3, 1)
    plt.plot(t, x, 'kx', label='Original X data')
    plt.plot(t, mu_x_pred, 'b', label='GP Mean X')
    plt.fill_between(t, 
                     mu_x_pred - 1.96 * sigma_x_pred, 
                     mu_x_pred + 1.96 * sigma_x_pred, 
                    alpha=0.2, color='blue', label='95% Confidence Interval')
    plt.title('Gaussian Process - X Dimension')
    plt.xlabel('Time')
    plt.ylabel('X')
    plt.legend()

    # Plot for dimension Y
    plt.subplot(1, 3, 2)
    plt.plot(t, y, 'kx', label='Original Y data')
    plt.plot(t, mu_y_pred, 'r', label='GP Mean Y')
    plt.fill_between(t, 
                     mu_y_pred - 1.96 * sigma_y_pred, 
                     mu_y_pred + 1.96 * sigma_y_pred, 
                    alpha=0.2, color='red', label='95% Confidence Interval')
    plt.title('Gaussian Process - Y Dimension')
    plt.xlabel('Time')
    plt.ylabel('Y')
    plt.legend()
    
    """ plt.subplot(1, 2, 2)
    plt.plot(x, y, 'kx', label='Original movement')
    plt.plot(mu_x_pred, mu_y_pred, 'r', label='GP Mean Z')
    plt.title('Gaussian Process - Z Dimension')
    plt.xlabel('Time')
    plt.ylabel('Z')
    plt.legend() """
    
    if gp_z!=None:
        # Plot for dimension Y
        plt.subplot(1, 3, 3)
        plt.plot(t, z, 'kx', label='Original Z data')
        plt.plot(t, mu_z_pred, 'r', label='GP Mean Z')
        plt.fill_between(t, 
                        mu_z_pred - 1.96 * sigma_z_pred, 
                        mu_z_pred + 1.96 * sigma_z_pred, 
                        alpha=0.2, color='red', label='95% Confidence Interval')
        plt.title('Gaussian Process - Z Dimension')
        plt.xlabel('Time')
        plt.ylabel('Z')
        plt.legend()

    plt.tight_layout()
    plt.show()

def createGaussian(segment,demos,len_segment,dim,m,visualize=False):
    j = 0
    data_x = []
    data_y = []
    data_z = []
    for i in range(demos):
        x_o=segment[j:j+int(len_segment/demos),0] #* multiply by 800 only to scale small data
        y_o=segment[j:j+int(len_segment/demos),1]
        if dim==3:
            z_o=segment[j:j+int(len_segment/demos),2]
        
        x_min, x_max = np.min(x_o), np.max(x_o)
        y_min, y_max = np.min(y_o), np.max(y_o)

        x_normalized = (x_o - x_min) / (x_max - x_min)
        y_normalized = (y_o - y_min) / (y_max - y_min)

        #* Reducing the number of points for the GP
        idx_red = np.linspace(0, len(x_o) - 1, 50, dtype=int)

        x = x_normalized[idx_red]
        y = y_normalized[idx_red]
        if dim==3:
            z_min, z_max = np.min(z_o), np.max(z_o)
            z_normalized = (z_o - z_min) / (z_max - z_min)
            z = z_normalized[idx_red]
            #z = z_o[idx_red]
        t = np.linspace(0, 1, len(y))
        
        #* kernel definition
        kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) #C(1.0, (1e-3, 1e3))

        #* Gaussian Process for X
        gp_x = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gp_x.fit(t.reshape(-1, 1), x)
        
        mu_x = gp_x.predict(t.reshape(-1, 1))
        K_x = gp_x.kernel_(t.reshape(-1, 1), t.reshape(-1, 1)) + gp_x.alpha * np.eye(len(t))+ 1e-6 * np.eye(len(t))

        #* Gaussian Process for Y
        gp_y = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gp_y.fit(t.reshape(-1, 1), y)
        
        mu_y = gp_y.predict(t.reshape(-1, 1))
        K_y = gp_y.kernel_(t.reshape(-1, 1), t.reshape(-1, 1)) + gp_y.alpha * np.eye(len(t))+ 1e-6 * np.eye(len(t))
        
        if dim==3:
            #* Gaussian Process for Z
            gp_z = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
            gp_z.fit(t.reshape(-1, 1), z)
            
            mu_z = gp_z.predict(t.reshape(-1, 1))
            K_z = gp_z.kernel_(t.reshape(-1, 1), t.reshape(-1, 1)) + gp_z.alpha * np.eye(len(t))+ 1e-6 * np.eye(len(t))

        data_x.append((mu_x.reshape(-1,1), K_x))
        data_y.append((mu_y.reshape(-1,1), K_y))
        if dim==3:
            data_z.append((mu_z.reshape(-1,1), K_z))

        if visualize == True:
            if dim==3:
                plot_gp(gp_x, gp_y, gp_z, t, x, y, z,len(y))
            elif dim==2:
                plot_gp(gp_x, gp_y,None, t, x, y,None,len(y))
        j=j+int(len_segment/demos)
    save_gp_data(data_x, f'gpx_data_segment_{m}.pkl')
    save_gp_data(data_y, f'gpy_data_segment_{m}.pkl')
    if dim==3:
        save_gp_data(data_z, f'gpz_data_segment_{m}.pkl')
        print()
    



