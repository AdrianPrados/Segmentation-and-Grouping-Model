from createGP import *
from compareSegs import *
from costMatrix import *
import itertools
import time
from scipy.interpolate import interp1d
import numpy as np
from fastdtw import fastdtw
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

#! Example of use for the generation of a Library of Primitives movements from 0.
#! Once the library is created you can compare the data against each of the groups

#? The method works with values normalized and not normalized


#* Folder name where the segments are stored
folder_name = 'Letras' #Change for your folder name
all_segments = []
demos = 3


#* Read all the segments from the folder
for filename in os.listdir(folder_name):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_name, filename)
        segment = np.loadtxt(file_path)
        all_segments.append(segment)
all_segments = np.array(all_segments)


dimension = all_segments[0].shape[1] #* Dimension of the data it can be 2D or 3D
#! Create the Gaussian processes for each segment
for i in range(all_segments.shape[0]):
    segment = all_segments[i]
    createGaussian(segment=segment,demos=demos,len_segment=segment.shape[0],dim=dimension,m=i,visualize=False)
#! Comparison between segments
#* First we evaluate if there is a value that is not 0 in the cost matrix
#* If there is a value that is greater than 0.5, we calculate the cost of movement for each of the segments
seg = list(range(all_segments.shape[0]))
combi = list(itertools.combinations(seg, 2))
listsimilars = []
for (seg1, seg2) in combi:
    print(f'Comparing segment {seg1} against segment {seg2}')
    finalCost = 0
    n= seg1
    m= seg2
    file1x = f'GaussianPkl/gpx_data_segment_{n}.pkl'
    file1y = f'GaussianPkl/gpy_data_segment_{n}.pkl'
    file2x = f'GaussianPkl/gpx_data_segment_{m}.pkl'
    file2y = f'GaussianPkl/gpy_data_segment_{m}.pkl'
    if dimension == 3:
        file1z = f'GaussianPkl/gpz_data_segment_{n}.pkl'
        file2z = f'GaussianPkl/gpz_data_segment_{m}.pkl'
        value,shape1,shape2 = cost(file1x=file1x, file1y=file1y,file1z=file1z, file2x=file2x, file2y=file2y,file2z=file2z, plot = False)
    elif dimension == 2:
        value,shape1,shape2 = cost(file1x=file1x, file1y=file1y, file2x=file2x, file2y=file2y, plot = False)
    print("Value of the cost matrix: ", value)
    costeFinal = 0
    if dimension==2:
        if value > 0.25: #Can be adapted to the value that you want
            for i in ["x","y"]:
                print(i)
                file1 = f'GaussianPkl/gp{i}_data_segment_{n}.pkl'
                file2 = f'GaussianPkl/gp{i}_data_segment_{m}.pkl'
                coste = compareSegs(file1, file2, plot = False)
                costeFinal += coste
                print("Hello:",costeFinal)
            print("Energy cost: ", costeFinal)
        if (costeFinal/10) > 1.0:
            finalCost = value/costeFinal
        else:
            finalCost = value #* (costeFinal/10)
    elif dimension==3:
        if value > 0.25: #Can be adapted to the value that you want
            for i in ["x","y","z"]:
                print(i)
                file1 = f'GaussianPkl/gp{i}_data_segment_{n}.pkl'
                file2 = f'GaussianPkl/gp{i}_data_segment_{m}.pkl'
                coste = compareSegs(file1, file2, plot = False)
                costeFinal += coste
            print("Energy cost: ", costeFinal)
        if (costeFinal/10) > 1.0:
            finalCost = value/costeFinal
        else:
            finalCost = value #* (costeFinal/10)
    if finalCost > 0.5:
        listsimilars.append((n,m))
    print("Final cost: ", finalCost)
    if dimension == 2:
        plotShape(shape1, shape2, finalCost)
    elif dimension == 3:
        plotShape3D(shape1, shape2, finalCost)
print("Matching values: ", listsimilars)

#! Now we generate the Library of Primitives 
#! (this is done only when you don't have any initial library, if you have it, you can just compare against it)

# Create the MovementLibrary folder if it doesn't exist
output_folder = 'MovementLibrary'
os.makedirs(output_folder, exist_ok=True)


# Create a dictionary to store the connections
connections = {}

# Fill the dictionary with the connections
for (a, b) in listsimilars:
    if a not in connections:
        connections[a] = set()
    if b not in connections:
        connections[b] = set()
    connections[a].add(b)
    connections[b].add(a)

# Find connected components
visited = set()
groups = []

def dfs(node, group):
    stack = [node]
    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            group.append(current)
            if current in connections:
                for neighbor in connections[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)

# Perform DFS to find all groups
for segment in connections:
    if segment not in visited:
        group = []
        dfs(segment, group)
        groups.append(tuple(sorted(group)))  # Save the sorted group as a tuple

print("Groups of related segments:", groups)

# Save the segments
# 1. Save segments that are not in any group
all_indices = set(range(len(all_segments)))
grouped_indices = set(itertools.chain(*groups))  # Indices that are in groups
ungrouped_indices = all_indices - grouped_indices  # Indices that are not in groups

# Save ungrouped segments
for index in ungrouped_indices:
    file_path = os.path.join(output_folder, f'movement_{index}.txt')
    np.savetxt(file_path, all_segments[index])
    print(f'Segment {index} saved in {file_path}')

# 2. Save only the first segment of each group
for group in groups:
    first_segment = group[0]  # Only save the first segment of the group
    file_path = os.path.join(output_folder, f'movement_{first_segment}.txt')
    np.savetxt(file_path, all_segments[first_segment])
    print(f'First segment of group {group} (segment {first_segment}) saved in {file_path}')


#! Now we plot the original values with the same colors if they are from the same group of segments
if dimension == 2:
    colors = plt.cm.get_cmap('tab10', len(os.listdir(folder_name)))
    fig = plt.figure(figsize=(6, 6))
    colorIdx = []
    #* Read all the segments from the folder
    for idx, filename in enumerate(os.listdir(folder_name)):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_name, filename)
            segment = np.loadtxt(file_path)
        length = int(segment.shape[0]/demos)
        color = colors(idx)
        colorIdx.append(color)
        for group in groups:
            if idx in group:
                print("Group: ", group)
                val = group.index(idx)
                print("Position in the group: ", val)
                color = colorIdx[group[0]]
                colorIdx[idx]=color
        for i in range(demos):
            plt.plot(segment[i*length:(i+1)*length, 0], segment[i*length:(i+1)*length, 1], lw=5,c=color, label="Segment " + str(i+1))
    #print("List of colors: ", colorIdx)
    #plt.legend()
    plt.show()
elif dimension == 3:
    # Colors for each segment
    colors = plt.cm.get_cmap('tab10', len(os.listdir(folder_name)))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    colorIdx = []

    # Read all the segments from the folder and plot them
    for idx, filename in enumerate(os.listdir(folder_name)):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_name, filename)
            segment = np.loadtxt(file_path)

            # Divide the data based on the number of demonstrations
            length = int(segment.shape[0] / demos)
            color = colors(idx)
            colorIdx.append(color)

            # Determine the color based on the group
            for group in groups:
                if idx in group:
                    print("Group: ", group)
                    val = group.index(idx)
                    print("Position in the group: ", val)
                    color = colorIdx[group[0]]
                    colorIdx[idx] = color

            # Plot each demonstration of the segment
            for i in range(demos):
                ax.plot(
                    segment[i * length:(i + 1) * length, 0],  # X
                    segment[i * length:(i + 1) * length, 1],  # Y
                    segment[i * length:(i + 1) * length, 2],  # Z
                    lw=3,
                    c=color,
                    label=f"Segment {i + 1}" if idx == 0 else None  # Labels only once
                )

    # Configure axes and show the plot
    ax.set_title("Segments in 3D")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
