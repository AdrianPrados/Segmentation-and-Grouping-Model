import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from utils import *
import scripts.drawData as scr2
import downsampling as ds
import time as t

import matplotlib as mpl
mpl.rc('font', size=18)

colors = ['r', 'g', 'b', 'c', 'm', 'y']
    

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


def count_thresh(data, threshold, segment_size, grace_threshold):
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
    for i in range(len(data)):
        if data[i] >= threshold:
            if count >= segment_size:
                if segments[-1] + count != i:
                    segments.append(i - count)
            count = count + 1
            grace_count = 0
        else:
            if grace_count > grace_threshold:
                count = 0
            else:
                grace_count = grace_count + 1
                count = count + 1
    print(segments)
    return segments


def segment(time, data, base_thresh=1000, segment_size=256, window_size=64, grace_thresh=32, plot=False):
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
    
    jerk = calc_jerk_in_time(time, data)
    total_jerk = np.linalg.norm(jerk, axis=1)
    avg_jerk = moving_average(total_jerk, window_size)
    norm_avg_jerk = avg_jerk / np.max(avg_jerk)
    segments = count_thresh(norm_avg_jerk, base_thresh, segment_size, grace_thresh)
    print("Segments:", segments)
    print("All: ", norm_avg_jerk)
    print("Especifico: ", norm_avg_jerk[segments[0]])
    #t.sleep(10000)
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

def calc_segment_prob(segment_list, data_len, window_size, plot=False):
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
        segment_probabilities = calc_segment_prob(segment_list[1:], data_len, window_size, plot)
        probabilities = probabilities * segment_probabilities
    probabilities = probabilities / np.sum(probabilities)
    if plot:
        plt.figure()
        plt.title('Combined Probabilities')
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
    keypoints = np.random.choice(data_len, size=n_samples, replace=True, p=probabilities)
    print('Chosen Keypoints')
    print(keypoints)
    sorted_keys = np.sort(keypoints)
    final_keys = []
    cur_key = 0
    cur_key_group = []
    for i in range(len(sorted_keys)):
        print("Sorted keys en i",sorted_keys[i])
        print("Cur key",cur_key)
        print("Window size",window_size)
        if sorted_keys[i] <= cur_key + window_size:
            cur_key_group.append(sorted_keys[i])
        else:
            if len(cur_key_group) >= n_pass:
                final_keys.append(int(np.mean(cur_key_group)))
                print("Final keys",final_keys)
            cur_key = sorted_keys[i]
            cur_key_group = [cur_key]
    if len(cur_key_group) >= n_pass:
        final_keys.append(int(np.mean(cur_key_group)))
    print('Unique Sorted Keypoints')
    print(final_keys)
    keypoints = np.insert(final_keys, 0, 0)
    keypoints = np.append(keypoints, data_len)
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


def main3d():
    """
    This function performs 3D segmentation and visualization of robot data.
    It reads robot data from a specified file, performs segmentation on different data streams,
    and probabilistically combines the segments. Finally, it visualizes the trajectory in a 3D plot.
    """
    
    seed = 440773
    np.random.seed(seed)
    fname = '../h5 files/three_button_pressing_demo.h5'
    joint_data, tf_data, wrench_data, gripper_data = read_robot_data(fname)
    
    joint_time = joint_data[0][:, 0] + joint_data[0][:, 1] * (10.0**-9)
    joint_pos = np.unwrap(joint_data[1], axis=0)
    
    traj_time = tf_data[0][:, 0] + tf_data[0][:, 1] * (10.0**-9)
    traj_pos = tf_data[1]
    
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
    thresh = 0.2
    ssize = 64
    wsize = 64
    gthresh = 4
    joint_segments = segment(joint_time, joint_pos, base_thresh=thresh, segment_size=ssize, window_size=wsize, grace_thresh=gthresh, plot=False)
    
    print('Trajectory')
    thresh = 0.25
    ssize = 64
    wsize = 64
    gthresh = 4
    traj_segments = segment(traj_time, traj_pos, base_thresh=thresh, segment_size=ssize, window_size=wsize, grace_thresh=gthresh, plot=False)
    
    print('Wrench Force')
    thresh = 0.15
    ssize = 64
    wsize = 64
    gthresh = 4
    frc_segments = segment(wrench_time, wrench_frc, base_thresh=thresh, segment_size=ssize, window_size=wsize, grace_thresh=gthresh, plot=False)
    
    print('Gripper')
    thresh = 0.1
    ssize = 64
    wsize = 64
    gthresh = 4
    gripper_segments = segment(gripper_time, gripper_pos, base_thresh=thresh, segment_size=ssize, window_size=wsize, grace_thresh=gthresh, plot=False)
    
    segments = probabilistically_combine([joint_segments, traj_segments, frc_segments, gripper_segments], len(traj_pos), wsize, n_samples=20, n_pass=3, plot=True)
    
    plt.rcParams['figure.figsize'] = (9, 7)
    fig = plt.figure()
    fig.suptitle('Trajectory')
    ax = plt.axes(projection='3d')
    for i in range(len(segments)-1):
        ax.plot3D(traj_pos[segments[i]:segments[i+1], 0], traj_pos[segments[i]:segments[i+1], 1], traj_pos[segments[i]:segments[i+1], 2], label="Segment " + str(i+1), lw=5)
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
    plt.show()

#Example process using a 2D trajectory with a single data stream
def main2d():
    np.random.seed(6)
    [[sm_t, sm_x, sm_y], [unsm_t, unsm_x, unsm_y], [norm_t, norm_x, norm_y]] = scr2.read_demo_h5('AA.h5', 0) #* the second value indicates which demo to read
    norm_y = -norm_y
    demo = np.hstack((np.reshape(norm_x, (len(norm_x), 1)), np.reshape(norm_y, (len(norm_y), 1))))
    
    thresh = 0.99
    ssize = 16 
    wsize = 16
    seg_initial = segment(norm_t, demo, base_thresh=thresh, segment_size=ssize, window_size=wsize, plot=True)
    
    segments = probabilistically_combine([seg_initial], len(demo), wsize, n_samples=10, n_pass=2, plot=True)
    print('Final Segments')
    print(segments)
    
    fig = plt.figure(figsize=(6, 6))
    for i in range(len(segments)-1):
        plt.plot(demo[segments[i]:segments[i+1], 0], demo[segments[i]:segments[i+1], 1], lw=5, c=colors[i+1], label="Segment " + str(i+1))
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='lower center')
    plt.show()
    
if __name__ == '__main__':
    """ print("Example of 2D trajectory")
    main2d() """
    print("Example of 3D trajectory")
    main3d()