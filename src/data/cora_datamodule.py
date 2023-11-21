import torch
from torch_geometric.data import Batch
from torch_geometric.datasets import Planetoid
from torch.utils.data import random_split
from torch_geometric.loader import DataListLoader
from torch_geometric.loader.dense_data_loader import collate_fn
from torch_geometric.transforms import NormalizeFeatures
from src.data.components import custom_preprocess
from lightning import LightningDataModule
from torch_geometric.loader.dense_data_loader import collate_fn
from torch.utils.data import ConcatDataset, Dataset, random_split
from src.data.components import proprecess_addone
from src.data.components import proprecess_addtwo
from torch_geometric.data import DataLoader


class CoraDataModule(LightningDataModule):
    def __init__(self, data_dir='data', batch_size=64, num_workers=15,proprecess='addone'):
        super(CoraDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = NormalizeFeatures()
        self.proprecess = proprecess

    #数据准备
    def setup(self, stage=None):
        dataset = Planetoid(root=self.data_dir, name='Cora', transform=self.transform)
        #preprecess,根据函数名初始化
        dataset = custom_preprocess(dataset, self.proprecess)
        self.train_dataset=dataset
    #dataloader返回
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    def val_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)





if __name__ == "__main__":

    datamodel = CoraDataModule()
    datamodel.setup()
    print(datamodel.train_dataloader)
