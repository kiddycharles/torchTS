import lightning as L
import torchts.utils.logging as logging
from torchts.utils.data.datamodules import ETTDataModule, YahooFinanceDataModule
from torchts.utils.cli import CustomLightningCLI
from TransformerForecastTask import TransformerForecastTask
from lightning.pytorch.callbacks import RichProgressBar


def main():
    logging.format_logger(L.pytorch._logger)
    cli = CustomLightningCLI(InformerForecastTask, YahooFinanceDataModule, run=False)  # noqa: F841
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
