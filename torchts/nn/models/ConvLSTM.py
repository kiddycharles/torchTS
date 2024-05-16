import os
import torchvision
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
import lightning as pl
from torchts.nn.model import TimeSeriesModel


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Initialize the ConvLSTM cell.

        :param input_dim: int
                Number of channels of input tensor
        :param hidden_dim: int
                Number of channels of hidden tensor
        :param kernel_size: (int, int)
                Size of the convolutional kernel
        :param bias: bool
                Whether to use bias in convolutional layers or not
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class Seq2SeqConvLSTM(nn.Module):
    def __init__(self, nf, in_channel):
        super().__init__()

        """ 
        Overall Architecture
        - Encoder
            # ConvLSTM
            # Hidden Representation from the Encoder
        - Decoder
            # ConvLSTM (Takes in Hidden Representation from the Encoder)
            # 3D CNN  (Produces Regression Predictions for Our Model)
        """

        self.encoder_layer_1_convlstm = ConvLSTMCell(input_dim=in_channel,
                                                     hidden_dim=nf,
                                                     kernel_size=(3, 3),
                                                     bias=True)

        self.encoder_layer_2_convlstm = ConvLSTMCell(input_dim=nf,
                                                     hidden_dim=nf,
                                                     kernel_size=(3, 3),
                                                     bias=True)

        self.decoder_layer_1_convlstm = ConvLSTMCell(input_dim=nf,
                                                     hidden_dim=nf,
                                                     kernel_size=(3, 3),
                                                     bias=True)

        self.decoder_layer_2_convlstm = ConvLSTMCell(input_dim=nf,
                                                     hidden_dim=nf,
                                                     kernel_size=(3, 3),
                                                     bias=True)

        self.decoder_CNN = nn.Conv3d(in_channels=nf,
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))

    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        outputs = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_layer_1_convlstm(input_tensor=x[:, t, :, :],
                                                     cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_layer_2_convlstm(input_tensor=h_t,
                                                       cur_state=[h_t2,
                                                                  c_t2])  # we could concat to provide skip conn here

        # encoder_vector
        encoder_vector = h_t2

        # decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_layer_1_convlstm(input_tensor=encoder_vector,
                                                       cur_state=[h_t3,
                                                                  c_t3])  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_layer_2_convlstm(input_tensor=h_t3,
                                                       cur_state=[h_t4,
                                                                  c_t4])  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)

        return outputs

    def forward(self, x, future_seq=0):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
            :param future_seq:
            :param x:
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t = self.encoder_layer_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_layer_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_layer_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_layer_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)

        return outputs


def test_end(outputs):
    # OPTIONAL
    avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    tensorboard_logs = {'test_loss': avg_loss}
    return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}


class MovingMNISTConvLSTM(TimeSeriesModel):
    def __init__(self, opt, model=None, **kwargs):
        super().__init__(**kwargs)

        self.model = model
        #
        # logging config
        self.log_images = True
        self.opt = opt
        # Training config
        self.criterion = torch.nn.MSELoss()
        self.n_steps_past = opt.n_steps_past
        self.n_steps_ahead = opt.n_steps_ahead

    def create_video(self, x, y_hat, y):
        # predictions with input for illustration purposes
        preds = torch.cat([x.cpu(), y_hat.unsqueeze(2).cpu()], dim=1)[0]

        # entire input and ground truth
        y_plot = torch.cat([x.cpu(), y.unsqueeze(2).cpu()], dim=1)[0]

        # error (l2 norm) plot between pred and ground truth
        difference = (torch.pow(y_hat[0] - y[0], 2)).detach().cpu()
        zeros = torch.zeros(difference.shape)
        difference_plot = torch.cat([zeros.cpu().unsqueeze(0), difference.unsqueeze(0).cpu()], dim=1)[
            0].unsqueeze(1)

        # concat all images
        final_image = torch.cat([preds, y_plot, difference_plot], dim=0)

        # make them into a single grid image file
        grid = torchvision.utils.make_grid(final_image, nrow=self.n_steps_past + self.n_steps_ahead)

        return grid

    def forward(self, x):
        x = x.to(device='mps')

        output = self.model(x, future_seq=self.n_steps_ahead)

        return output

    def training_step(self, batch, batch_idx):
        x, y = batch[:, 0:self.n_steps_past, :, :, :], batch[:, self.n_steps_past:, :, :, :]
        x = x.permute(0, 1, 4, 2, 3)
        y = y.squeeze()

        y_hat = self.forward(x).squeeze()  # is squeeze necessary?

        loss = self.criterion(y_hat, y)

        # save learning_rate
        lr_saved = self.trainer.optimizers[0].param_groups[-1]['lr']
        lr_saved = torch.scalar_tensor(lr_saved).to(device='mps')

        # save predicted images every 250 global_step
        if self.log_images:
            if self.global_step % 250 == 0:
                final_image = self.create_video(x, y_hat, y)

                self.logger.experiment.add_image(
                    'epoch_' + str(self.current_epoch) + '_step' + str(self.global_step) + '_generated_images',
                    final_image, 0)
                plt.close()

        tensorboard_logs = {'train_mse_loss': loss,
                            'learning_rate': lr_saved}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return {"val_loss": self.criterion(y_hat, y)}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': self.criterion(y_hat, y)}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.opt.lr, betas=(self.opt.beta_1, self.opt.beta_2))
