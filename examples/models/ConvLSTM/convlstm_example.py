import sys

# Add the directory containing the module to the Python path
sys.path.append('/home/juy022/torchTS')

from torchts.nn.models.ConvLSTM import Seq2Seq, Seq2SeqParams, WeightsInitializer
from torchts.utils.moving_mnist import MovingMNISTDataLoaders
import time
from lightning.pytorch import Trainer
from lightning.pytorch import loggers as pl_loggers
import torch


def main():
    ###
    # Common Params
    ###
    artifact_dir = "./tmp"
    input_seq_length = 10
    train_batch_size = 32
    validation_bath_size = 16
    ###
    # Setup Pipeline
    ###
    model_params: Seq2SeqParams = {
        "input_seq_length": input_seq_length,
        "num_layers": 2,
        "num_kernels": 64,
        "return_sequences": False,
        "convlstm_params": {
            "in_channels": 1,
            "out_channels": 1,
            "kernel_size": (3, 3),
            "padding": "same",
            "activation": "relu",
            "frame_size": (64, 64),
            "weights_initializer": WeightsInitializer.He,
        },
    }

    model = Seq2Seq(**model_params)
    #
    # training_params: TrainingParams = {
    #     "epochs": 1,
    #     "loss_criterion": nn.BCELoss(reduction="sum"),
    #     "accuracy_criterion": nn.L1Loss(),
    #     "optimizer": Adam(model.parameters(), lr=1e-4),
    #     "early_stopping": EarlyStopping(
    #         patience=30,
    #         verbose=True,
    #         delta=0.0001,
    #     ),
    #     "metrics_filename": "metrics.csv",
    # }

    print("Loading dataset ...")
    data_loaders = MovingMNISTDataLoaders(
        train_batch_size=train_batch_size,
        validation_batch_size=validation_bath_size,
        input_frames=model_params["input_seq_length"],
        label_frames=1,
        split_ratios=[0.7, 0.299, 0.001],
    )

    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    trainer = Trainer(max_epochs=100, logger=tb_logger, accelerator='auto')
    start = time.time()

    trainer.fit(model, data_loaders.train_dataloader)
    print("Training time taken %f" % (time.time() - start), flush=True)
    # see performance for validation dataset
    trainer.validate(model, data_loaders.validation_dataloader)

    # see performance for test dataset
    trainer.test(model, data_loaders.test_dataloader)

    torch.save(model, 'saved_model.pth')

    model.eval()
    with torch.no_grad():
        for inputs, target in data_loaders.test_dataloader:
            output = model(inputs)
            print(output.shape)
            print(target.shape)


if __name__ == "__main__":
    main()
