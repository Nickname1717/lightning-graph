from typing import Any, Dict, Optional, Tuple
from torch_geometric.data import Batch
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch_geometric.data import Data

class DiffusionDataModule(LightningDataModule):
    def __init__(
        self,
        num_nodes,
        num_features,
        num_edge_features,
        batch_size: int = 1,

    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.num_features = num_features
        self.batch_size = batch_size
        self.num_edge_features=num_edge_features
    # 数据的初始化
    def setup(self, stage: Optional[str] = None) -> None:


        x = torch.zeros((self.num_nodes, self.num_features), dtype=torch.float)  # Initialize with all zeros
        random_indices = torch.randint(0, self.num_features, (self.num_nodes,))
        x[torch.arange(self.num_nodes), random_indices] = 1.0

        edges = torch.tensor([[0, 1, 2, 3],
                              [1, 2, 3, 0]], dtype=torch.long)

        edge_features = torch.eye(self.num_edge_features)[edges[0]]

        adjacency_matrix = torch.zeros((self.num_nodes, self.num_nodes, self.num_edge_features), dtype=torch.float)
        adjacency_matrix[edges[0], edges[1], :] = edge_features
        adjacency_matrix[edges[1], edges[0], :] = edge_features

        graph_data = Data(x=x, edge_index=edges, edge_attr=adjacency_matrix)
        self.graph_data = graph_data


    #包装成dataloader返回
    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader([self.graph_data], batch_size=None)




if __name__ == "__main__":
    _ = DiffusionDataModule()
