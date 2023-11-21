from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

from src.models.components import  BackboneCombinedModel, HeadCombinedModel


class CoraLitModule(LightningModule):
    def __init__(
        self,
        gcn,
        gat,
        mlp,
        loss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:

        super().__init__()
        self.save_hyperparameters(logger=False)
        #初始化model中的各个模型及函数
        self.gcn=gcn
        self.gat=gat
        self.mlp=mlp
        self.criterion = loss
        self.optimizer=optimizer
        self.scheduler=scheduler

        self.train_acc = Accuracy(task="multiclass", num_classes=7)
        self.val_acc = Accuracy(task="multiclass", num_classes=7)
        self.test_acc = Accuracy(task="multiclass", num_classes=7)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()
    def forward(self, data):
        gcn_output=self.gcn(data.x,data.edge_index)
        gat_output=self.gat(gcn_output,data.edge_index)
        mlp_output=self.mlp(gat_output)
        return mlp_output

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
    def model_step(self, data):
        x, y, train_mask,val_mask, edge_index = data.x, data.y, data.train_mask,data.val_mask, data.edge_index
        logits = self.forward(data)
        loss = self.criterion(logits[train_mask], y[train_mask])
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y
    def training_step(self,data):
        loss, preds, targets = self.model_step(data)
        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        # return loss or backpropagation will fail
        return loss


    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = CoraLitModule(None, None, None, None,None,None)
