from typing import Any, Dict, Tuple
import time
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.utils import diffusion_utils
import torch.nn.functional as F

from src.utils.noise_schedule import PredefinedNoiseScheduleDiscrete, DiscreteUniformTransition


class DiffusionModel(LightningModule):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        net: torch.nn.Module,
        diffusion_noise_schedule,
        diffusion_steps
    ) -> None:

        super().__init__()

        self.save_hyperparameters(logger=False)
        self.diffusion_noise_schedule=diffusion_noise_schedule
        self.T=diffusion_steps
        self.net = net

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(diffusion_noise_schedule,
                                                              timesteps=diffusion_steps)
        self.transition_model = DiscreteUniformTransition(x_classes=4, e_classes=5)

    #前向传播
    def forward(self,X,E) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return X,E

    def on_train_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()

    #训练过程
    def training_step(self, data, i) -> torch.Tensor:
        X, E = data.x, data.edge_attr
        print("\n--------diffusion_noise_schedule:"+self.diffusion_noise_schedule+"-----------")
        print("OriginX：")
        print(X)
        noisy_data = self.apply_noise(X, E)
        print("diffusionX:")
        print(noisy_data.float())
        loss = self.calculate_loss(X, E)
        return loss

    def calculate_loss(self, X, E) -> torch.Tensor:
        # Perform the forward pass and calculate the loss
        logits = self.forward(X, E)

        # Replace the following line with your actual loss calculation
        loss = torch.tensor(1.0, requires_grad=True)

        return loss
        # noisy_data = self.apply_noise(X, E)

    def apply_noise(self, X, E):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(1, 1), device=X.device).float()[0, 0]
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        # assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        # assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE)

        X_t = F.one_hot(sampled_t.X, num_classes=4)
        # E_t = F.one_hot(sampled_t.E, num_classes=1)
        # assert (X.shape == X_t.shape) and (E.shape == E_t.shape)



        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': X_t, 'E_t':E}
        return X_t


    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass
    #优化器返回
    def configure_optimizers(self) -> Dict[str, Any]:

        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        return {"optimizer": optimizer}




if __name__ == "__main__":
    _ = DiffusionModel()
