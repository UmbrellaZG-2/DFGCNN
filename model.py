import torch
from torch.nn.modules.module import Module
from layer import FGCNNLayer
from dgl.nn.pytorch import GraphConv as GCNConv


class DFGCNN(Module):
    def __init__(self, num_in, d1, d2, fussy):
        super(DFGCNN, self).__init__()
        # 第一层FGCNN
        self.fc1_1 = GCNConv(num_in, d1)
        self.fc1_2 = FGCNNLayer(fussy)

        # 第二层FGCNN
        self.fc2_1 = GCNConv(d1, d2)
        self.fc2_2 = FGCNNLayer(fussy)

    def forward(self, x, adj):
        x1_1 = self.fc1_1(adj, x)
        x1_2 = self.fc1_2(x1_1)
        x1_3 = torch.mul(x1_1, x1_2)

        x2_1 = self.fc2_1(adj, x1_3)
        x2_2 = self.fc2_2(x2_1)
        x2_3 = torch.mul(x2_1, x2_2)

        return x2_3

