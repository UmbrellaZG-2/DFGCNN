import math
import torch
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn import Module
from sklearn.cluster import KMeans
from utils import apply_gaussian_membership_functions, select_max_membership


def Kmeans_init(A, F):
    kmeans = KMeans(n_clusters=F, random_state=0, init='k-means++', n_init=10)
    A_detached = A.detach()
    A_numpy = A_detached.cpu().numpy()
    kmeans.fit(A_numpy)
    mean = kmeans.cluster_centers_
    std = np.zeros((F, A.shape[1]))
    for i in range(F):
        cluster_points = A_numpy[kmeans.labels_ == i]
        if len(cluster_points) > 1:
            std[i, :] = np.std(cluster_points, axis=0)+1e-5
        else:
            std[i, :] = 1
    mean_tensor = torch.tensor(mean, dtype=torch.float32)
    std_tensor = torch.tensor(std, dtype=torch.float32)
    return mean_tensor, std_tensor


class FGCNNLayer(Module):
    def __init__(self, fussy):
        super(FGCNNLayer, self).__init__()
        self.F = fussy

    def forward(self, x):
        means, stds = Kmeans_init(x, self.F)
        m_ship = apply_gaussian_membership_functions(x, means, stds, self.F)
        m_ship = select_max_membership(m_ship)
        return m_ship







