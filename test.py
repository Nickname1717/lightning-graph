import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split

# 加载 Planetoid 数据集
dataset = Planetoid(root='/path/to/dataset', name='Cora')

# 划分数据集
indices = list(range(len(dataset)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

# 创建训练集和验证集的子图
train_dataset = dataset[train_indices]
val_dataset = dataset[val_indices]

# 创建 DataLoader 加载数据
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)