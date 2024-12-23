import numpy as np

### Different downsampling methods for trajectories

## Naive downsampling (selects every n points)

# function to downsample a 1 dimensional trajectory to n points
# arguments
# traj: nxd vector, where n is number of points and d is number of dims
# n (optional): the number of points in the downsampled trajectory. Default is 100.
# returns the trajectory downsampled to n points
def downsample_traj(traj, n=100):
    n_pts, n_dims = np.shape(traj)
    npts = np.linspace(0, n_pts - 1, n)
    out = np.zeros((n, n_dims))
    for i in range(n):
        out[i][:] = traj[int(npts[i])][:]
    return out

## Distance-based downsampling
#downsample a trajectory of n points to m points based on distance between those points

#function to get the total distance of a n x d trajectory
#arguments
#traj: nxd vector, where n is number of points and d is number of dims
#returns the total distance of traj, calculated using euclidean distance  
def get_traj_dist(traj):
    dist = 0.
    for n in range(len(traj) - 1):
        dist = dist + np.linalg.norm(traj[n + 1] - traj[n])
    #if (DEBUG):
    #    print('Traj total dist: %f' % (dist))
    return dist
    
#downsample to a certain number of points
def db_downsample(traj, new_len):
    (n_pts, n_dims) = np.shape(traj)
    total_dist = get_traj_dist(traj)
    interval_len = total_dist / (new_len - 1)
    sum_len = 0.0
    out_traj = np.zeros((new_len, n_dims))
    ind = 0
    for n in range(n_pts - 1):
        if (sum_len >= 0.0):
            out_traj[ind, :] = traj[n, :]
            ind += 1
            sum_len -= interval_len
        sum_len += np.linalg.norm(traj[n + 1] - traj[n])
    out_traj[-1, :] = traj[-1, :]
    return out_traj
    
#downsample to a certain number of points, return indeces
def db_downsample_inds(traj, new_len):
    (n_pts, n_dims) = np.shape(traj)
    total_dist = get_traj_dist(traj)
    interval_len = total_dist / (new_len - 1)
    sum_len = 0.0
    out_traj = np.zeros((new_len, n_dims))
    out_inds = []
    ind = 0
    for n in range(n_pts - 1):
        if (sum_len >= 0.0):
            out_inds.append(n)
            out_traj[ind, :] = traj[n, :]
            ind += 1
            sum_len -= interval_len
        sum_len += np.linalg.norm(traj[n + 1] - traj[n])
    out_traj[-1, :] = traj[-1, :]
    return out_traj, out_inds
    
#downsample to a certain distance between points
def db_downsample_dist(traj, seg_len):
    (n_pts, n_dims) = np.shape(traj)
    interval_len = seg_len
    sum_len = 0.0
    out_traj = np.zeros((n_pts, n_dims))
    ind = 0
    for n in range(n_pts - 1):
        if (sum_len >= 0.0):
            out_traj[ind, :] = traj[n, :]
            ind += 1
            sum_len -= interval_len
        sum_len += np.linalg.norm(traj[n + 1] - traj[n])
    out_traj[ind, :] = traj[-1, :]
    ind += 1
    out_traj = out_traj[0:ind]
    return out_traj

## Douglas-Peucker Downsampling
#iteratively or recursively adds points farthest away from current downsampled set

#find the perpendicular distance between a point and a line formed by 2 points
def perpendicularDistance(pp, p1, p2):
    #find distance from pp to line p1p2
    # vector formulation from: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    n = p2 - p1
    n = n / np.linalg.norm(n)
    return np.linalg.norm( (p1 - pp) - (np.dot((p1 - pp), n) * n) )

#recursive Douglas-Peucker method with every point within epsilon distance from current downsampling
def DouglasPeucker(PointList, epsilon):
    # Find the point with the maximum distance
    dmax = 0
    index = 0
    (n_pts, n_dims) = np.shape(PointList)
    for i in range(1, n_pts):
        d = perpendicularDistance(PointList[i], PointList[0], PointList[n_pts - 1]) 
        if (d > dmax):
            index = i
            dmax = d
            
    # If max distance is greater than epsilon, recursively simplify
    if (dmax > epsilon):
        # Recursive call
        recResults1 = DouglasPeucker(PointList[0:index], epsilon)
        recResults2 = DouglasPeucker(PointList[index - 1:], epsilon)

        # Build the result list
        ResultList = np.vstack((recResults1, recResults2))
    else:
        ResultList = np.vstack((PointList[0], PointList[n_pts - 1]))
    # Return the result
    return ResultList
    
#iterative Douglas-Peucker method with every point within epsilon distance from current downsampling (python has shallow recursion depth, the iterative method ensures no errors, although is slower)
def DouglasPeuckerIterative(PointList, epsilon):
    (n_pts, n_dims) = np.shape(PointList)
    above_eps = False
    ResultList = np.vstack((PointList[0], PointList[n_pts-1]))
    inds = [0, n_pts-1]
    while not above_eps:
        above_eps = True
        # Find the point with the maximum distance for each segment
        for seg in range(len(inds) - 1):
            dmax = 0
            index = 0
            for i in range(inds[seg], inds[seg+1]):
                #print([i, index])
                d = perpendicularDistance(PointList[i], ResultList[seg], ResultList[seg + 1]) 
                if (d > dmax):
                    index = i - 1 #this is to fix some indexing error
                    dmax = d
            if (dmax > epsilon):
                above_eps = False
                #ResultList.insert(PointList[index, :].copy(), seg + 1)
                ResultList = np.insert(ResultList, seg + 1, PointList[index, :], axis=0)
                inds.insert(seg + 1, index)
    # Return the result
    return ResultList

#iterative method, stops at a certain number of points instead of epsilon distance (can be interpreted as epsilon=dist of last point added)
def DouglasPeuckerPoints(PointList, num_points):
    (n_pts, n_dims) = np.shape(PointList)
    ResultList = np.vstack((PointList[0], PointList[n_pts-1]))
    inds = [0, n_pts-1]
    while len(inds) < num_points:
        dmax = 0
        index = 0
        segnum = 0
        for seg in range(len(inds) - 1):
            for i in range(inds[seg], inds[seg+1]):
                d = perpendicularDistance(PointList[i], ResultList[seg], ResultList[seg + 1]) 
                if (d > dmax):
                    index = i
                    dmax = d
                    segnum = seg
        ResultList = np.insert(ResultList, segnum + 1, PointList[index, :], axis=0)
        inds.insert(segnum + 1, index)
    # Return the result
    return ResultList
    
#same as previous function, returns indeces as well as downsampled points
def DouglasPeuckerPoints2(PointList, num_points):
    (n_pts, n_dims) = np.shape(PointList)
    ResultList = np.vstack((PointList[0], PointList[n_pts-1]))
    inds = [0, n_pts-1]
    while len(inds) < num_points:
        dmax = 0
        index = 0
        segnum = 0
        for seg in range(len(inds) - 1):
            for i in range(inds[seg], inds[seg+1]):
                d = perpendicularDistance(PointList[i], ResultList[seg], ResultList[seg + 1]) 
                if (d > dmax):
                    index = i
                    dmax = d
                    segnum = seg
        ResultList = np.insert(ResultList, segnum + 1, PointList[index, :], axis=0)
        inds.insert(segnum + 1, index)
    # Return the result
    return ResultList, inds
    
if __name__ == '__main__':
    from utils import get_lasa_trajn
    import matplotlib.pyplot as plt
    [x, y] = get_lasa_trajn('Leaf_1')

    traj = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1))))
    
    n = 8
    
    traj_naive = downsample_traj(traj, n)
    traj_db = db_downsample(traj, n)
    traj_dp = DouglasPeuckerPoints(traj, n)
    
    fig = plt.figure()
    demo, = plt.plot(traj[:, 0], traj[:, 1], 'k-', lw=3, ms=5, label='Demonstration')
    naive, = plt.plot(traj_naive[:, 0], traj_naive[:, 1], 'm.-', lw=2, ms=13, label='Naive')
    db, = plt.plot(traj_db[:, 0], traj_db[:, 1], 'gx-', lw=2, ms=13, mew=3, label='Distance-based')
    dp, = plt.plot(traj_dp[:, 0], traj_dp[:, 1], 'y*-', lw=2, ms=13, label='Douglas-Peucker')
    plt.xticks([])
    plt.yticks([])
    plt.legend(fontsize='x-large', loc='best', bbox_to_anchor=(0.5, 0.5))
    plt.show()