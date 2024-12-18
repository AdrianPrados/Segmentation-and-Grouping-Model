from shapesimilarity import shape_similarity, shape_similarity3D
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

from wgpot import GP_W_barycenter


def cost(file1x=None, file1y=None,file1z=None, file2x=None, file2y=None,file2z=None, plot = False):
    """ file1x = f'GaussianPkl/gpx_data_segment_2.pkl'
    file1y = f'GaussianPkl/gpy_data_segment_2.pkl'
    file2x = f'GaussianPkl/gpx_data_segment_4.pkl'
    file2y = f'GaussianPkl/gpy_data_segment_4.pkl' """

    file_1x = open(file1x, 'rb')
    gp_list1x = pickle.load(file_1x)
    file_1x.close()

    file_1y = open(file1y, 'rb')
    gp_list1y = pickle.load(file_1y)
    file_1y.close()

    file_2x = open(file2x, 'rb')
    gp_list2x = pickle.load(file_2x)
    file_2x.close()

    file_2y = open(file2y, 'rb')
    gp_list2y = pickle.load(file_2y)
    file_2y.close()
    
    mux_bc, Kx_bc = GP_W_barycenter(gp_list1x)
    muy_bc, Ky_bc = GP_W_barycenter(gp_list1y)
    mu2x_bc, K2x_bc = GP_W_barycenter(gp_list2x)
    mu2y_bc, K2y_bc = GP_W_barycenter(gp_list2y)
    
    if file1z == None:
        shape1 = np.column_stack((mux_bc, muy_bc))
        shape2 = np.column_stack((mu2x_bc, mu2y_bc))
    
    elif file1z != None:
        file_1z = open(file1z, 'rb')
        gp_list1z = pickle.load(file_1z)
        file_1z.close()
        file_2z = open(file2z, 'rb')
        gp_list2z = pickle.load(file_2z)
        file_2z.close()
        
        muz_bc, Kz_bc = GP_W_barycenter(gp_list1z)
        mu2z_bc, K2z_bc = GP_W_barycenter(gp_list2z)
        
        shape1 = np.column_stack((mux_bc, muy_bc, muz_bc))
        shape2 = np.column_stack((mu2x_bc, mu2y_bc, mu2z_bc))

    if file1z != None:
        similarity = shape_similarity3D(shape1, shape2,checkRotation=False) #* checkRotation=True allows the method to don't take into account the orientation of the movements #* checkRotation=True allows the method to don't take into account the orientation of the movements
        if plot==True:
            plotShape3D(shape1, shape2, similarity)
        return similarity, shape1, shape2
    elif file1z == None:
        similarity = shape_similarity(shape1, shape2,checkRotation=True) #* checkRotation=True allows the method to don't take into account the orientation of the movements 
        if plot == True:
            plotShape(shape1, shape2, similarity)
        return similarity, shape1, shape2
    

def plotShape(shape1, shape2, similarity):
    plt.plot(shape1[:,0], shape1[:,1], linewidth=2.0)
    plt.plot(shape2[:,0], shape2[:,1], linewidth=2.0)
    if similarity > 0.5:
        plt.title(f'Similarity is: {similarity:.2f}, considered as same primitives', fontsize=14, fontweight='bold', color='green')
    else:
        plt.title(f'Similarity is: {similarity:.2f}, considered as different primitives', fontsize=14, fontweight='bold', color='red')
    plt.show()
    
def plotShape3D(shape1, shape2, similarity):
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    print(shape1)
    
    # Plot shape1
    ax.plot(shape1[:, 0], shape1[:, 1], shape1[:, 2], label='Shape 1', linewidth=2.0, color='blue')
    # Plot shape2
    ax.plot(shape2[:, 0], shape2[:, 1], shape2[:, 2], label='Shape 2', linewidth=2.0, color='orange')
    
    # Adding labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    
    # Setting the title with similarity information
    if similarity > 0.5:
        ax.set_title(f'Similarity is: {similarity:.2f}, considered as same primitives', fontsize=14, fontweight='bold', color='green')
    else:
        ax.set_title(f'Similarity is: {similarity:.2f}, considered as different primitives', fontsize=14, fontweight='bold', color='red')
    
    ax.legend()
    plt.show()