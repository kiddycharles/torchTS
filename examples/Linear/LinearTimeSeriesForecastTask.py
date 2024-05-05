import torch
import torch.nn.functional as F
import torch.optim
import torchmetrics
from torchts.nn.model import TimeSeriesModel


class TimeSeriesForecastTask(TimeSeriesModel):
    def __init__(
            self,
            model: torch.nn.Module,
            seq_len: int,
            label_len: int,
            pred_len: int,
            variate: str,
            padding: int = 0,
            loss: str = "mse",
            learning_rate: float = 0.0001,
            lr_scheduler: str = "exponential",
            inverse_scaling: bool = False,
            scaler=None,
            **kwargs
    ):
        super(TimeSeriesForecastTask, self).__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        metrics = torchmetrics.MetricCollection([torchmetrics.MeanSquaredError(), torchmetrics.MeanAbsoluteError()])
        self.val_metrics = metrics.clone(prefix="Val_")
        self.test_metrics = metrics.clone(prefix="Test_")
        self.scaler = scaler

    def forward(self, batch_x):
        outputs = self.model(batch_x)
        return outputs

    def shared_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        outputs = self(batch_x)
        f_dim = -1 if self.hparams.variate == "mu" else 0
        batch_y = batch_y[:, -self.model.pred_len:, f_dim:]
        return outputs, batch_y

    def training_step(self, batch, batch_idx):
        outputs, batch_y = self.shared_step(batch, batch_idx)
        loss = self.loss(outputs, batch_y)
        self.log("Train_Loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs, batch_y = self.shared_step(batch, batch_idx)
        loss = self.loss(outputs, batch_y)
        self.log("Val_Loss", loss)
        if self.hparams.inverse_scaling and self.scaler is not None:
            outputs = self.scaler.inverse_transform(outputs)
            batch_y = self.scaler.inverse_transform(batch_y)
        outputs = outputs.reshape(-1)
        batch_y = batch_y.reshape(-1)
        self.val_metrics(outputs, batch_y)
        self.log_dict(self.val_metrics)
        return {"outputs": outputs, "targets": batch_y}

    def test_step(self, batch, batch_idx):
        outputs, batch_y = self.shared_step(batch, batch_idx)
        if self.hparams.inverse_scaling and self.scaler is not None:
            outputs = self.scaler.inverse_transform(outputs)
            batch_y = self.scaler.inverse_transform(batch_y)
        outputs = outputs.reshape(-1)
        batch_y = batch_y.reshape(-1)
        self.test_metrics(outputs, batch_y)
        self.log_dict(self.test_metrics)
        return {"outputs": outputs, "targets": batch_y}

    def on_fit_start(self):
        if self.hparams.inverse_scaling and self.scaler is not None:
            if self.scaler.device == torch.device("cpu"):
                self.scaler.to(self.device)

    def on_test_start(self):
        if self.hparams.inverse_scaling and self.scaler is not None:
            if self.scaler.device == torch.device("cpu"):
                self.scaler.to(self.device)

    def loss(self, outputs, targets, **kwargs):
        if self.hparams.loss == "mse":
            return F.mse_loss(outputs, targets)
        raise RuntimeError("The loss function {self.hparams.loss} is not implemented.")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        if self.hparams.lr_scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
        elif self.hparams.lr_scheduler == "two_step_exp":

            def two_step_exp(epoch):
                if epoch % 4 == 2:
                    return 0.5
                if epoch % 4 == 0:
                    return 0.2
                return 1.0

            scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=[two_step_exp])
        else:
            raise RuntimeError("The scheduler {self.hparams.lr_scheduler} is not implemented.")
        return [optimizer], [scheduler]