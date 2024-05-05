import lightning as L
import torchts.utils.logging as logging
from torchts.utils.data.datamodules import ETTDataModule, YahooFinanceDataModule
from torchts.utils.cli import CustomLightningCLI
from InformerForecastTask import InformerForecastTask
from lightning.pytorch.callbacks import RichProgressBar
import time


# def main():
#     --config configs/ETTh1/multivariate/pred_len_24.yaml
#     current_time = time.time()
#     logging.format_logger(L.pytorch._logger)
#     cli = CustomLightningCLI(InformerForecastTask, ETTDataModule, run=False)  # noqa: F841
#     cli.trainer.fit(cli.model, datamodule=cli.datamodule)
#     cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
#     print("Done training: Time elapsed: {:.2f}s".format(time.time() - current_time))


def main():
    current_time = time.time()
    logging.format_logger(L.pytorch._logger)
    cli = CustomLightningCLI(InformerForecastTask, YahooFinanceDataModule, run=False)  # noqa: F841
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
    print("Done training: Time elapsed: {:.2f}s".format(time.time() - current_time))


if __name__ == "__main__":
    main()
