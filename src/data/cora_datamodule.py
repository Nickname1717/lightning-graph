import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid
from torch.utils.data import random_split
from torch_geometric.transforms import NormalizeFeatures

from lightning import LightningDataModule

class CoraDataModule(LightningDataModule):
    def __init__(self, data_dir='data', batch_size=64, num_workers=4, validation_ratio=0.1, test_ratio=0.1):
        super(CoraDataModule, self).__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio

        self.transform = NormalizeFeatures()

    def prepare_data(self):
        # This method is used to download and preprocess the dataset.
        # Since we are using the Planetoid dataset, it is not necessary to implement anything here.
        Planetoid(root=self.data_dir, name='Cora', transform=self.transform)

    def setup(self, stage=None):
        # This method is used to split the dataset into training, validation, and test sets.
        dataset = Planetoid(root=self.data_dir, name='Cora', transform=self.transform)
        num_data = len(dataset.x)

        # Calculate the number of samples for validation and test sets
        num_val = int(self.validation_ratio * num_data)
        num_test = int(self.test_ratio * num_data)

        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [num_data - num_val - num_test, num_val, num_test])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)




if __name__ == "__main__":
    _ = CoraDataModule().setup()
