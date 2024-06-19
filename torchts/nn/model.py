from abc import abstractmethod
from typing import Optional, Callable, Dict
from torch.optim import Adam
import torch.nn.functional as F
import lightning as L
from lightning.pytorch import Trainer, seed_everything
from torch.utils.data import DataLoader, TensorDataset


"""
The lightningModule has many convenient methods, but the core ones are the following:
1. __init__ and setup()
2. forward()
3. train_step()
4. validation_step()
5. test_step()
6. predict_step()
7. configure_optimizers()
"""


class TimeSeriesModel(L.LightningModule):
    """Base class for all TorchTS models.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        optimizer_kwargs (dict): Arguments for the optimizer
        criterion: Loss function
        criterion_kwargs (dict): Arguments for the loss function
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        scheduler_kwargs (dict): Arguments for the scheduler
        scaler (torchts.utils.scaler.Scaler): Scaler
    """

    def __init__(
        self,
        optimizer=Adam,
        optimizer_kwargs=None,  # includes learning rate
        criterion=F.mse_loss,
        criterion_kwargs=None,
        scheduler=None,
        scheduler_kwargs=None,
        scaler=None,
    ):
        self.criterion = criterion
        self.criterion_args = criterion_kwargs
        self.scaler = scaler

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        # a good practice to call super().__init__() at the end of the __init__ method to ensure that
        # any initialization logic in the parent class is executed correctly.
        super().__init__()

    def fit(self, x, y, max_epochs=10, batch_size=128):
        """Fits model to the given dataset.

        Args:
            x (torch.Tensor): Input dataset
            y (torch.Tensor): Output dataset
            max_epochs (int): Number of training epochs
            batch_size (int): Batch size for torch.utils.dataset.DataLoader
        """
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        trainer = Trainer(max_epochs=max_epochs)
        trainer.fit(self, loader)

    def prepare_batch(self, batch):
        return batch

    def _step(self, batch, batch_idx, num_batches):
        """

        Args:
            batch: Output of the torch.utils.dataset.DataLoader
            batch_idx: Integer displaying index of this batch
            dataset: Data set to use

        Returns: loss for the batch
        """
        x, y = self.prepare_batch(batch)

        if self.training:
            batches_seen = batch_idx + self.current_epoch * num_batches
        else:
            batches_seen = batch_idx

        pred = self(x, y, batches_seen)

        if self.scaler is not None:
            y = self.scaler.inverse_transform(y)
            pred = self.scaler.inverse_transform(pred)

        if self.criterion_args is not None:
            loss = self.criterion(pred, y, **self.criterion_args)
        else:
            loss = self.criterion(pred, y)

        return loss

    def training_step(self, batch, batch_idx):
        """Trains model for one step.

        Args:
            batch (torch.Tensor): Output of the torch.utils.dataset.DataLoader
            batch_idx (int): Integer displaying index of this batch
        """
        train_loss = self._step(batch, batch_idx, len(self.trainer.train_dataloader))
        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        """Validates model for one step.

        Args:
            batch (torch.Tensor): Output of the torch.utils.dataset.DataLoader
            batch_idx (int): Integer displaying index of this batch
        """
        val_loss = self._step(batch, batch_idx, 0)
        self.log("val_loss", val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        """Tests model for one step.

        Args:
            batch (torch.Tensor): Output of the torch.utils.dataset.DataLoader
            batch_idx (int): Integer displaying index of this batch
        """
        test_loss = self._step(batch, batch_idx, 0)
        self.log("test_loss", test_loss)
        return test_loss

    @abstractmethod
    def forward(self, *args):
    # def forward(self, x, y=None, batches_seen=None):
        """Forward pass.

        Args:
            x (torch.Tensor): Input dataset

        Returns:
            torch.Tensor: Predicted dataset
        """

    def predict(self, x):
        """Runs model inference.

        Args:
            x (torch.Tensor): Input dataset

        Returns:
            torch.Tensor: Predicted dataset
        """
        return self(x).detach()

    def predict_step(self, x):
        return self(x).detach()

    def configure_optimizers(self):
        """Configure optimizer.

        Returns:
            torch.optim.Optimizer: Optimizer

        6 options
        - single optimizer
        - list or tuple of optimizers
        - Two lists - the first list has multiple optimizers, the second has multiple LR scheduler
        - dictionary
        - None - run without optimizers
        """
        optimizer = self.optimizer(self.parameters())

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
            return [optimizer], [scheduler]

        return optimizer

