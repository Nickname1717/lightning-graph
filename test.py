import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

dataset = Planetoid(root='data', name='Cora')


print("Number of samples in the dataset:", len(dataset))

batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

for batch in data_loader:
    print("Batch size:", batch.num_graphs)
    print(batch)
    break

