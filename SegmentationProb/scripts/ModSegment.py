import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.cluster import DBSCAN

from utils import *
import drawData as scr2
import downsampling as ds
import time as t

from sklearn.mixture import GaussianMixture
from scipy.signal import find_peaks
import os



import matplotlib as mpl
mpl.rc('font', size=18)

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k','r', 'g', 'b', 'c', 'm', 'y', 'k','r', 'g', 'b', 'c', 'm', 'y', 'k','r', 'g', 'b', 'c', 'm', 'y', 'k','r', 'g', 'b', 'c', 'm', 'y', 'k','r', 'g', 'b', 'c', 'm', 'y', 'k'  ]
    

def normal(x, mean=0, stdev=1):
    """
    Calculates the value of the normal distribution at a given point.
    Parameters:
    - x: The point at which to evaluate the normal distribution.
    - mean: The mean of the normal distribution (default: 0).
    - stdev: The standard deviation of the normal distribution (default: 1).
    Returns:
    The value of the normal distribution at the given point.
    """
    return (1 / (stdev * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mean) / stdev)**2)
    

def calc_time_deriv(time, data):
    """
    Calculate the time derivative of the given data.
    Parameters:
    - time (array-like): Array of time values.
    - data (array-like): Array of data values.
    Returns:
    - deriv (ndarray): Array of time derivatives of the data.
    """
    n_pts, n_dims = np.shape(data)
    deriv = np.zeros((n_pts-1, n_dims))
    for i in range(n_pts-1):
        deriv[i] = (data[i + 1] - data[i]) / (time[i + 1] - time[i])
    return deriv

def calc_jerk_in_time(time, position):
    """
    Calculate the jerk in time for a given position.
    Parameters:
    - time (float): The time value.
    - position (float): The position value.
    Returns:
    - float: The jerk value.
    """
    
    data = position
    for _ in range(3):
        data = calc_time_deriv(time, data)
    return data

def calc_acceleration(time, position):
    """
    Calculate the acceleration in time for a given position.
    Parameters:
    - time (float): The time value.
    - position (float): The position value.
    Returns:
    - float: The acceleration value.
    """
    
    data = position
    for _ in range(2):
        data = calc_time_deriv(time, data)
    return data



def moving_average(data, window_size):
    """
    Calculates the moving average of a given data array.
    Parameters:
    - data (array-like): The input data array.
    - window_size (int): The size of the moving window.
    Returns:
    - avg_data (ndarray): The array containing the moving averages.
    """
    
    avg_data = []
    for i in range(len(data) - window_size):
        avg_data.append(np.mean(data[i:i+window_size]))
    return np.array(avg_data)

def detect_brutal_changes(data, change_threshold, segment_size,acceleration):
    """
    Detects abrupt changes in jerk and segments the data accordingly.
    
    Parameters:
    - data (array-like): The input jerk data.
    - change_threshold (float): The threshold for detecting abrupt changes.
    - segment_size (int): The minimum number of data points to be considered a segment.
    
    Returns:
    - segments (list): A list of indices marking the start of each segment.
    """
    jerk_diff = np.abs(np.diff(data))
    print("Jerk diff",jerk_diff)
    segments = []
    start_index = 0
    data_prev = acceleration[0]
    for i in range (len(data)-1):
        if jerk_diff[i] - start_index >= 10e-2 or (np.sign(acceleration[i]) != np.sign(data_prev)):# or (jerk_diff[i] - start_index) != abs(jerk_diff[i] - start_index):
            segments.append(i)
        start_index = jerk_diff[i]
        data_prev = acceleration[i]
    if len(segments) == 0:
        segments.append(0)
    else:
        segments.append(i+1) #* Add the last point

    print("Segmentos detectados:", segments)
    return segments


def count_thresh(data, threshold, segment_size, grace_threshold,acceleration):
    """
    Counts the number of segments in the data that exceed the threshold for a given segment size and grace threshold.
    Parameters:
    - data (array-like): The input data array.
    - threshold (float): The threshold over which data must reach to be considered a changepoint.
    - segment_size (int): The number of samples that data must stay above the threshold to be considered a segment.
    - grace_threshold (int): The number of samples that data can dip below the threshold without being considered a new changepoint.
    Returns:
    - segments (list): A list of indices where the segments start.
    """

    segments = [0]
    count = 0
    grace_count = 0
    grace_threshold = 10
    data_prev = acceleration[0]
    for i in range(len(data)):
        print("Data en i",data[i])
        print("Threshold",threshold)
        if data[i] >= threshold and (np.sign(acceleration[i]) != np.sign(data_prev)):
            segments.append(i)
        else:
            if grace_count > grace_threshold:
                count = 0
            else:
                grace_count = grace_count + 1
                count = count + 1
        data_prev = acceleration[i]
    print("Segmentos en thresh",segments)
    
    """ if count >= segment_size:
    print("Holaaaa")
    segments.append(int(data[i]))
    if segments[-1] + count != i:
        print("Agregando segmento")
        segments.append(i - count)
    count = count + 1
    grace_count = 0 """
    
    return segments

def normalize_time_series(time, data, target_length=100):
    """
    Normaliza la serie temporal y los datos para que tengan una longitud común.
    
    :param time: Array con los tiempos originales.
    :param data: Array con los datos originales.
    :param target_length: Longitud deseada para la serie temporal normalizada.
    :return: Tiempo y datos normalizados.
    """
    # Crear un nuevo eje temporal con la longitud deseada
    new_time = np.linspace(time[0], time[-1], target_length)
    
    # Interpolar los datos para ajustarlos a la nueva longitud
    new_data = np.zeros((target_length, data.shape[1]))
    for i in range(data.shape[1]):
        new_data[:, i] = np.interp(new_time, time, data[:, i])
    
    return new_time, new_data

def signed_norm(vector):
    norm = np.linalg.norm(vector, axis=1)  
    sign = np.sign(vector[:, 0])  
    return norm * sign 


def segment(time, data, base_thresh=1000, segment_size=10, window_size=64, grace_thresh=32, plot=False, mode="variations"):
    """
    Detects changepoints in the given data based on the jerk values.
    Parameters:
    - time (array-like): The time values corresponding to the data.
    - data (array-like): The data to be segmented.
    - base_thresh (int, optional): The threshold value for segment detection. Defaults to 1000.
    - segment_size (int, optional): The size of each segment. Defaults to 256.
    - window_size (int, optional): The size of the moving average window. Defaults to 64.
    - grace_thresh (int, optional): The grace threshold for segment detection. Defaults to 32.
    - plot (bool, optional): Whether to plot the detected changepoints. Defaults to False.
    Returns:
    - segments (list): A list of indices indicating the detected changepoints in the data.
    """
    
    # Verificar que time y data tengan la misma longitud
    if len(time) != len(data):
        raise ValueError("El tamaño del array de tiempo no coincide con el tamaño del array de datos.")
    
    # Comprobar si existen valores NaN
    if np.isnan(time).any() or np.isnan(data).any():
        raise ValueError("Existen valores NaN en los datos o en el tiempo.")
    
    # Verificar que los parámetros son válidos
    if base_thresh < 0 or segment_size <= 0 or window_size <= 0 or grace_thresh < 0:
        raise ValueError("Los valores de los parámetros no son válidos.")
    
    jerk = calc_jerk_in_time(time, data)
    total_jerk = signed_norm(jerk)
    avg_jerk = moving_average(total_jerk, window_size)
    norm_avg_jerk = avg_jerk / np.max(avg_jerk)
    
    #*Acceleration
    acceleration = calc_acceleration(time, data)
    total_acceleration = signed_norm(acceleration)
    avg_acceleration = moving_average(total_acceleration, window_size)
    norm_avg_acceleration = avg_acceleration / np.max(avg_acceleration)
    
    #* Two different types of segmentation
    if mode == "variations":
        segments = detect_brutal_changes(norm_avg_jerk, base_thresh, segment_size,norm_avg_acceleration)
    elif mode == "threshold":
        segments = count_thresh(norm_avg_jerk, base_thresh, segment_size, grace_thresh,norm_avg_acceleration)
    else:
        raise ValueError("Invalid Mode")
    if plot:
        # Crear el gráfico
        plt.figure(figsize=(12, 6))
        plt.plot(norm_avg_jerk, label='Norm Avg Jerk', color='blue')
        plt.scatter(segments, norm_avg_jerk[segments], color='red', label='Keypoints', zorder=5)  # Puntos de segmentos
        plt.title('Normalized Avg Jerk Values with Marked Segments')
        plt.xlabel('Index')
        plt.ylabel('Norm Avg Jerk')
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Línea horizontal en y=0
        plt.legend()
        plt.grid()
        plt.show()
        
        # Crear el gráfico
        plt.figure(figsize=(12, 6))
        plt.plot(norm_avg_acceleration, label='Norm Avg Acceleration', color='blue')
        plt.scatter(segments, norm_avg_acceleration[segments], color='red', label='Keypoints', zorder=5)  # Puntos de segmentos
        plt.title('Normalized Avg Acceleration Values with Marked Segments')
        plt.xlabel('Index')
        plt.ylabel('Norm Avg Acceleration')
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Línea horizontal en y=0
        plt.legend()
        plt.grid()
        plt.show()
    
    for i in range(1, len(segments)):
        segments[i] = segments[i] + window_size // 2

    if plot:
        fig = plt.figure(figsize=(7, 6))
        plt.title('Changepoint Detection')
        color_ind = 0
        for i in range(len(norm_avg_jerk)):
            plt.plot(i, norm_avg_jerk[i], colors[color_ind % len(colors)] + '.', ms=12)
            if color_ind < len(segments):
                if (segments[color_ind] == i):
                    color_ind = color_ind + 1
        plt.xlabel('Time')
        plt.ylabel('Jerk')
        plt.show()
        
    return segments

def calc_segment_prob(segment_list, data_len, window_size, plot=True):
    """
    Calculate the segment probabilities for a given list of segments.
    Parameters:
    - segment_list (list): List of segment points.
    - data_len (int): Length of the data.
    - window_size (float): Size of the window.
    - plot (bool, optional): Whether to plot the probabilities. Defaults to False.
    Returns:
    - probabilities (ndarray): Array of segment probabilities.
    """
    
    probabilities = np.ones((data_len,))
    for segment_point in segment_list:
        probabilities = probabilities + normal(np.linspace(0, 1, data_len), mean=segment_point / data_len, stdev=window_size / data_len)
    probabilities = probabilities / np.sum(probabilities)
    if plot:
        fig = plt.figure(figsize=(7, 6))
        plt.title('Segment ' + str(segment_list) + ' Probabilities')
        plt.plot(probabilities, lw=5)
        plt.xlabel('Time')
        plt.ylabel('Keypoint Probability')
        plt.show()
    return probabilities


def calc_prob_from_segments(list_of_list_of_segments, data_len, window_size, plot=False):
    """
    Calculate the probabilities of segments from a list of lists of segments.
    Parameters:
    - list_of_list_of_segments (list): A list of lists of segments.
    - data_len (int): The length of the data.
    - window_size (int): The size of the window.
    - plot (bool, optional): Whether to plot the combined probabilities. Defaults to False.
    Returns:
    - probabilities (ndarray): An array of probabilities.
    """
    
    probabilities = np.ones((data_len,))
    for segment_list in list_of_list_of_segments:
        segment_probabilities = calc_segment_prob(segment_list[1:], data_len, window_size, plot=True)
        print("Segment probabilities",segment_probabilities)
        probabilities = probabilities * segment_probabilities
    probabilities = probabilities / np.sum(probabilities)
    if plot==True:
        plt.figure()
        plt.title('Combined Probabilities Prueba')
        plt.plot(probabilities)
        plt.show()
    return probabilities



def probabilistically_combine(list_of_list_of_segments, data_len, window_size, n_samples=10, n_pass=2, plot=False):
    """
    Combines segments probabilistically to generate keypoints.
    Args:
        list_of_list_of_segments (list): A list of lists containing segments.
        data_len (int): The length of the data.
        window_size (int): The size of the window for grouping keypoints.
        n_samples (int, optional): The number of keypoints to sample. Defaults to 10.
        n_pass (int, optional): The minimum number of keypoints in a group to consider it as a final keypoint. Defaults to 2.
        plot (bool, optional): Whether to plot the probabilities. Defaults to False.
    Returns:
        numpy.ndarray: An array containing the keypoints.
    """
    
    probabilities = calc_prob_from_segments(list_of_list_of_segments, data_len, window_size, plot)
    #probabilities /= np.sum(probabilities)
    """ print("Lista de probabilidades: ",list_of_list_of_segments)
    print(probabilities) """
    """ keypoints = np.random.choice(data_len, size=n_samples, replace=True, p=probabilities)
    print('Chosen Keypoints con el caso previo')
    print(keypoints) """
    
    peaks, _= find_peaks(
    probabilities)
    """ peaks2, properties2 = find_peaks(
    probabilities)
    peaks = np.append(peaks,peaks2) """
    print(peaks)
    plt.figure(figsize=(10, 5))
    plt.plot(probabilities, label="Probabilidades", marker="o", linestyle="-")
    plt.scatter(peaks, probabilities[peaks], color="red", label="Peaks", zorder=5)
    plt.title("Peaks in probabilities distributions")
    plt.xlabel("Index")
    plt.ylabel("Probabilities")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()
    
    
    keypoints=peaks

    sorted_keys = np.sort(keypoints)
    sorted_keys = np.insert(sorted_keys, 0, 0)
    sorted_keys = np.append(sorted_keys, data_len)
    print('Sorted Keypoints',sorted_keys)
    final_keys = []
    cur_key = 0
    cur_key_group = []
    window_size = 290
    n_pass=2
    for i in range(len(sorted_keys)):
        if sorted_keys[i] <= cur_key + window_size:
            cur_key_group.append(sorted_keys[i])
        else:
            if len(cur_key_group) >= n_pass:
                final_keys.append(int(np.mean(cur_key_group)))
                print("Valor de final keys",final_keys)
            cur_key = sorted_keys[i]
            cur_key_group = [cur_key]
    if len(cur_key_group) >= n_pass:
        final_keys.append(int(np.mean(cur_key_group)))
    #* Evaluate the first and last group
    keypoints = np.insert(final_keys, 0, int(0))
    keypoints = np.append(keypoints, int(data_len))
    if abs(keypoints[0] - keypoints[1]) < window_size:
        print(abs(keypoints[0] - keypoints[1]))
        keypoints=np.delete(keypoints,1)
    if abs(keypoints[-1] - keypoints[-2]) < window_size:
        keypoints=np.delete(keypoints,-2)
    
    print('Final Keypoints Obtained')
    print(keypoints)
    #keypoints = sorted_keys
    
    
    
    
    return keypoints

def full_segmentation(time, list_of_data, base_thresh=1000, segment_size=256, window_size=64, grace_thresh=32, n_samples=10, n_pass=2, plot=False):
    """
    Perform full segmentation on a list of data.
    Args:
        time (array-like): The time values.
        list_of_data (list): A list of data to be segmented.
        base_thresh (int, optional): The base threshold value. Defaults to 1000.
        segment_size (int, optional): The size of each segment. Defaults to 256.
        window_size (int, optional): The size of the sliding window. Defaults to 64.
        grace_thresh (int, optional): The grace threshold value. Defaults to 32.
        n_samples (int, optional): The number of samples to use for probabilistic combination. Defaults to 10.
        n_pass (int, optional): The number of passes for probabilistic combination. Defaults to 2.
        plot (bool, optional): Whether to plot the segments. Defaults to False.
    Returns:
        list: The segmented data.
    """
    
    list_of_segments = [segment(time, data, base_thresh=base_thresh, segment_size=segment_size, window_size=window_size, grace_thresh=grace_thresh, plot=plot) for data in list_of_data]
    segments = probabilistically_combine(list_of_segments, len(time), window_size, n_samples=n_samples, n_pass=n_pass, plot=plot)
    return segments


def main3d(i):
    """
    This function performs 3D segmentation and visualization of robot data.
    It reads robot data from a specified file, performs segmentation on different data streams,
    and probabilistically combines the segments. Finally, it visualizes the trajectory in a 3D plot.
    """
    
    seed = 440773
    np.random.seed(seed)
    fname = f'../h5 files/vaso_data_{i+1}.h5'
    joint_data, tf_data, wrench_data, gripper_data = read_robot_data(fname)
    
    
    joint_time = joint_data[0][:, 0] + joint_data[0][:, 1] * (10.0**-9)
    joint_pos = np.unwrap(joint_data[1], axis=0)
    
    traj_time = tf_data[0][:, 0] + tf_data[0][:, 1] * (10.0**-9)
    traj_pos = tf_data[1]
    #t.sleep(1000)
    
    wrench_time = wrench_data[0][:, 0] + wrench_data[0][:, 1] * (10.0**-9)
    wrench_frc = wrench_data[1]
    
    gripper_time = gripper_data[0][:, 0] + gripper_data[0][:, 1] * (10.0**-9)
    gripper_pos = gripper_data[1]
    
    traj_pos, ds_inds = ds.DouglasPeuckerPoints2(traj_pos, 1000)
    
    
    joint_time = joint_time[ds_inds]
    joint_pos = joint_pos[ds_inds, :]
    traj_time = traj_time[ds_inds]
    wrench_time = wrench_time[ds_inds]
    wrench_frc = wrench_frc[ds_inds, :]
    gripper_time = gripper_time[ds_inds]
    gripper_pos = gripper_pos[ds_inds]
    
    print('Joint Positions')
    thresh = 0.35
    ssize = 32
    wsize = 64
    gthresh = 32
    joint_segments = segment(joint_time, joint_pos, base_thresh=thresh, segment_size=ssize, window_size=wsize, grace_thresh=gthresh, plot=False)
    
    print('Trajectory')
    thresh = 0.25
    ssize = 32
    wsize = 64
    gthresh = 32
    traj_segments = segment(traj_time, traj_pos, base_thresh=thresh, segment_size=ssize, window_size=wsize, grace_thresh=gthresh, plot=False)
    
    print('Wrench Force')
    thresh = 0.15
    ssize = 32
    wsize = 64
    gthresh = 32
    frc_segments = segment(wrench_time, wrench_frc, base_thresh=thresh, segment_size=ssize, window_size=wsize, grace_thresh=gthresh, plot=False)
    
    print('Gripper')
    thresh = 0.25
    ssize = 32
    wsize = 64
    gthresh = 32
    gripper_segments = segment(gripper_time, gripper_pos, base_thresh=thresh, segment_size=ssize, window_size=wsize, grace_thresh=gthresh, plot=False)
    
    #segments = probabilistically_combine([traj_segments], len(traj_pos), 1, n_samples=50, n_pass=2, plot=True)
    segments = [traj_segments]
    return segments,traj_pos

#Example process using a 2D trajectory with a single data stream
def main2d(i):
    np.random.seed(6)
    [[sm_t, sm_x, sm_y], [unsm_t, unsm_x, unsm_y], [norm_t, norm_x, norm_y]] = scr2.read_demo_h5('gal.h5', i) #* the second value indicates which demo to read
    norm_y = -norm_y
    demo = np.hstack((np.reshape(norm_x, (len(norm_x), 1)), np.reshape(norm_y, (len(norm_y), 1))))
    thresh = 0.16
    ssize = 32
    wsize = 64
    seg_initial = segment(norm_t, demo, base_thresh=thresh, segment_size=ssize, window_size=wsize, plot=False, mode="variations")

    
    return seg_initial,demo

def plot_all_demos_segmented(demo_indices, unified_segments):
    colors = plt.cm.get_cmap('hsv', 20)  # Generar una colormap con colores únicos
    plt.figure(figsize=(10, 10))

    for demo_index in demo_indices:
        # Leer la demostración desde el archivo .h5
        [[sm_t, sm_x, sm_y], [unsm_t, unsm_x, unsm_y], [norm_t, norm_x, norm_y]] = scr2.read_demo_h5('gal.h5', demo_index)
        norm_y = -norm_y
        demo = np.hstack((np.reshape(norm_x, (len(norm_x), 1)), np.reshape(norm_y, (len(norm_y), 1))))

        # Pintar la demostración
        plt.plot(demo[:, 0], demo[:, 1], label=f'Demostración {demo_index}', color="grey")

        # Pintar los segmentos unificados
        for i in range(len(unified_segments) - 1):
            plt.plot(demo[unified_segments[i]:unified_segments[i + 1], 0],
                    demo[unified_segments[i]:unified_segments[i + 1], 1],
                    lw=3, color=colors(i),label="Segment" if demo_index == 0 else "")

    plt.title('Todas las Demostraciones con Segmentación Unificada')
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='lower center')
    plt.show()
    
if __name__ == '__main__':
    all_segments = []
    all_demos = []
    segment_data =[]
    mode = 2 #* Mode of execution, it can be 2D or 3D
    
    #?Segmentation in 3D
    if mode == 3:
        for i in range(3): #* Number of demos
            print(f"Demo {i}")
            segments,demo = main3d(i=i)
            print(segments)
            all_segments.extend(segments) #* use extend if you use multimodal demonstrations
            all_demos.append(demo)
            print(f"Segmentos {i}:", segments)
        # Concatenar todos los segmentos en un solo array
        #all_segments_flat = np.concatenate(all_segments).reshape(-1, 1)
        segments = probabilistically_combine(all_segments, len(demo), 1, n_samples=50, n_pass=2, plot=True)
        #segments = all_segments
        print('Final Segments')
        # Figura para visualizar los segmentos
        plt.rcParams['figure.figsize'] = (9, 7)
        fig = plt.figure()
        fig.suptitle('Trajectory')
        ax = plt.axes(projection='3d')
        for j in range(len(all_demos)):
            demo = all_demos[j]
            
            # Aquí guardamos los segmentos de esta demo
            demo_segments = []
            
            for i in range(len(segments)-1):
                segment = demo[segments[i]:segments[i+1], :]
                demo_segments.append(segment)
                
                #* Draw each segments
                
                ax.plot3D(demo[segments[i]:segments[i+1], 0], demo[segments[i]:segments[i+1], 1], demo[segments[i]:segments[i+1], 2],c=colors[i], label="Segment " + str(i+1) if j == 0 else "", lw=5)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                # First remove fill
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                # Now set color to white (or whatever is "invisible")
                ax.xaxis.pane.set_edgecolor('w')
                ax.yaxis.pane.set_edgecolor('w')
                ax.zaxis.pane.set_edgecolor('w')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
                ax.legend()
                plt.tight_layout()
            
            segment_data.append(demo_segments)

        plt.show()
        
        
    #? Segmentation in 2D
    elif mode == 2:
        for i in range(3): #* Number of demos
            print(f"Demo {i}")
            segments,demo = main2d(i=i)
            all_segments.append(segments)
            all_demos.append(demo)
            print(f"Segmentos {i}:", segments)
        # Concatenar todos los segmentos en un solo array
        #all_segments_flat = np.concatenate(all_segments).reshape(-1, 1)
        segments = probabilistically_combine(all_segments, len(demo), 1, n_samples=3, n_pass=2, plot=True)
        #segments = all_segments
        print('Final Segments')
        # Figura para visualizar los segmentos
        fig = plt.figure(figsize=(6, 6))
        for j in range(len(all_demos)):
            demo = all_demos[j]
            demo_segments = []
            
            for i in range(len(segments)-1):
                segment = demo[segments[i]:segments[i+1], :]
                demo_segments.append(segment)
                
                # Dibujar cada segmento con su color
                plt.plot(demo[segments[i]:segments[i+1], 0], demo[segments[i]:segments[i+1], 1], lw=5, c=colors[i], label="Segment " + str(i+1) if j == 0 else "")
                plt.xticks([])
                plt.yticks([])
                plt.legend(loc='lower center')
            
            # Guardar los segmentos de esta demo en la lista segment_data
            segment_data.append(demo_segments)

        plt.show()
    
    # Nombre de la carpeta donde guardaremos los archivos de los segmentos
    folder_name = 'SegmentsFolder'

    # Crear la carpeta si no existe
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Suponiendo que all_demos es una lista de matrices, donde cada matriz tiene las coordenadas (x, y, z).
    # Y que segments es una lista de índices que define cómo se dividen las demostraciones en segmentos.

    # Variable para contar segmentos
    segment_count = 0


    for i in range(len(segments) - 1):
        segmentedData = []
        for j in range(len(all_demos)):
            
            # Extraer el segmento de la demo actual
            segment = all_demos[j][segments[i]:segments[i + 1]]
            segmentedData.append(segment)
            # Definir el nombre del archivo para este segmento
        segmentedData = np.concatenate(segmentedData, axis=0)    
        filename = f'segmento{i+1}.txt'  # Segmento se numera de acuerdo a la cuenta
        
        # Ruta del archivo
        file_path = os.path.join(folder_name, filename)
        
        # Abrir el archivo en modo escritura
        with open(file_path, 'w') as f:
            # Escribir las coordenadas x, y, z en el archivo
            np.savetxt(f, segmentedData, comments='', fmt='%.6f')  # Ajustar formato si es necesario
            
        print(f'Segmento guardado: {file_path}')

    
    
