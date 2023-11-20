from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
class CoraLitModule(LightningModule):
    def __init__(
        self,
        net1,
        net2,
        decoder,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:

        super().__init__()


        self.save_hyperparameters(logger=False)

        self.net1=net1
        self.net2=net2
        self.decoder=decoder
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()


        self.train_acc = Accuracy(task="multiclass", num_classes=7)
        self.val_acc = Accuracy(task="multiclass", num_classes=7)
        self.test_acc = Accuracy(task="multiclass", num_classes=7)


        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor,edge_index:torch.Tensor):

        gcn_output=self.net1(x,edge_index)
        gat_output=self.net2(gcn_output,edge_index)

        return self.decoder(gat_output)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, train_mask,val_mask, edge_index = batch.x, batch.y, batch.train_mask,batch.val_mask, batch.edge_index



        logits = self.forward(x, edge_index)

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

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass


    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
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
