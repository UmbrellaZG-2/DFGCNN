import torch
import os
import random
import numpy as np
import scipy.sparse as sp
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from torch.nn import Module
from torch.nn.parameter import Parameter


def seed_everything(seed=616):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))

    if np.where(rowsum == 0)[0].shape[0] != 0:
        indices = np.where(rowsum == 0)[0]
        for i in indices:
            rowsum[i] = float('inf')

    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    if np.where(rowsum == 0)[0].shape[0] != 0:
        indices = np.where(rowsum == 0)[0]
        for i in indices:
            rowsum[i] = float('inf')
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


class Gaussianmembership(Module):
    def __init__(self, mean, std):
        super(Gaussianmembership, self).__init__()
        self.mean = Parameter(torch.FloatTensor(mean))
        self.std = Parameter(torch.FloatTensor(std))

    def forward(self, x):
        res = torch.exp(-((x - self.mean) ** 2) / (2 * self.std ** 2))
        return res.squeeze()


def apply_gaussian_membership_functions(X, num_means, num_std, num_fussy):
    means_list = [num_means[i] for i in range(num_fussy)]
    stds_list = [num_std[i] for i in range(num_fussy)]
    num_samples = X.shape[0]
    num_features = X.shape[1]
    membership_matrix = torch.zeros((num_samples, num_features, num_fussy))
    for k in range(num_fussy):
        get_mem = Gaussianmembership(means_list[k], stds_list[k])
        membership_matrix[:, :, k] = get_mem(X)

    return membership_matrix

def select_max_membership(membership_matrix):
    return torch.max(membership_matrix, dim=2)[0]

def load(name):
    if name == 'cora':
        dataset = CoraGraphDataset(raw_dir='./data/cora')
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()

    graph = dataset[0]

    train_mask = graph.ndata.pop('train_mask')
    test_mask = graph.ndata.pop('test_mask')

    feat = graph.ndata.pop('feat')
    labels = graph.ndata.pop('label')

    return graph, feat, labels, train_mask, test_mask

