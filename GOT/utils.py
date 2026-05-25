"""
WGPOT
Wasserstein Distance and Optimal Transport Map
of Gaussian Processes

shape_similarity3D implemented
"""

import numpy as np
import scipy.io
import scipy.linalg
import math

from shapesimilarity.procrustesanalysis import procrustes_normalize_curve
from shapesimilarity.frechetdistance import frechet_distance
from shapesimilarity.geometry import curve_length


def Plot_GP(plt, X, mu, K, color, mean_alpha=1, var_alpha=0.5, label=None):

    if label:
        plt.plot(X, mu, c=color, alpha=mean_alpha, label=label)
    else:
        plt.plot(X, mu, c=color, alpha=mean_alpha)
    mu = mu[:, 0]
    s2 = np.diag(K)
    s = np.sqrt(s2)
    upper = mu + s
    lower = mu - s
    plt.fill_between(X.T[0, :], upper, lower, color=color, alpha=var_alpha)


# Notice: Read the data from original mat file
def read_all_gps(mat_address='data/exampleData.mat'):
    mat = scipy.io.loadmat(mat_address)
    days = mat['days']
    vanavara_gps = mat['Vanavara_GPs']
    num_of_GP = vanavara_gps.shape[1]

    gp_list = []
    for i in range(num_of_GP):
        gp_list.append((vanavara_gps[0, i][0, 0], vanavara_gps[0, i][0, 1]))

    return gp_list, days


def _kabsch_rotation(P, Q):
    """
    Calcula la mejor rotación R que alinea P con Q.
    P y Q deben tener la misma forma: (N, 3)
    """
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Evitar reflexión (queremos rotación propia)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def shape_similarity3D(shape1, shape2, checkRotation=True):
    """
    Versión 3D de shape_similarity.
    shape1, shape2: arrays (N, 3) y (M, 3)
    """
    shape1 = np.asarray(shape1, dtype=float)
    shape2 = np.asarray(shape2, dtype=float)

    if shape1.ndim != 2 or shape2.ndim != 2 or shape1.shape[1] != 3 or shape2.shape[1] != 3:
        raise ValueError("shape_similarity3D espera arrays de forma (N, 3) y (M, 3).")

    # Normalización tipo Procrustes, igual que en 2D
    procrustes_normalized_curve1 = procrustes_normalize_curve(shape1)
    procrustes_normalized_curve2 = procrustes_normalize_curve(shape2)

    geo_avg_curve_len = math.sqrt(
        curve_length(procrustes_normalized_curve1) *
        curve_length(procrustes_normalized_curve2)
    )

    # En 3D no hay un único ángulo, así que usamos una rotación rígida 3D
    candidates = [procrustes_normalized_curve1]

    if checkRotation:
        R = _kabsch_rotation(procrustes_normalized_curve1, procrustes_normalized_curve2)
        rotated_curve1 = procrustes_normalized_curve1 @ R
        candidates.append(rotated_curve1)

    min_frechet_distance = float("inf")
    for candidate in candidates:
        d = frechet_distance(candidate, procrustes_normalized_curve2)
        if d < min_frechet_distance:
            min_frechet_distance = d

    result = max(1 - min_frechet_distance / (geo_avg_curve_len / math.sqrt(2)), 0)
    return round(result, 4)