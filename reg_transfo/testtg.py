import torch
from torch_geometric.data import Data

batchsize = 4
num_nodes = 10
feature_dim = 16
graph = torch.rand(batchsize, num_nodes, feature_dim)
pos = torch.rand(batchsize, num_nodes, 2)
d = Data(x = graph, pos = pos)
print(d)
