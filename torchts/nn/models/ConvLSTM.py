import logging
from typing import Optional, Tuple, TypedDict, Union
from typing_extensions import NotRequired
from enum import Enum
import torch
from torch import nn
from torchts.nn.model import TimeSeriesModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class WeightsInitializer(str, Enum):
    Zeros = "zeros"
    He = "he"
    Xavier = "xavier"


class ConvLSTMParams(TypedDict):
    in_channels: int
    out_channels: int
    kernel_size: Union[int, Tuple]
    padding: Union[int, Tuple, str]
    activation: str
    frame_size: Tuple[int, int]
    weights_initializer: NotRequired[WeightsInitializer]


class Seq2SeqParams(TypedDict):
    input_seq_length: int
    label_seq_length: NotRequired[Optional[int]]
    num_layers: int
    num_kernels: int
    return_sequences: NotRequired[bool]
    convlstm_params: ConvLSTMParams


class BaseConvLSTMCell(nn.Module):
    """The ConvLSTM Cell implementation."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple],
            padding: Union[int, Tuple, str],
            activation: str,
            frame_size: Tuple,
            weights_initializer: WeightsInitializer = WeightsInitializer.Zeros,
    ) -> None:
        """

        Args:
            in_channels (int): Number of channels of input tensor.
            out_channels (int): Number of channels of output tensor
            kernel_size (Union[int, Tuple]): Size of the convolution kernel.
            padding (padding (Union[int, Tuple, str]): 'same', 'valid' or (int, int)
            activation (str): Name of activation function
            frame_size (Tuple): height and width
        """
        super().__init__()
        self.activation = self.__activation(activation)
        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels,
            out_channels=4 * out_channels,
            kernel_size=kernel_size,
            padding=padding,
            device=DEVICE,
        ).to(DEVICE)

        self.W_ci = nn.parameter.Parameter(
            torch.zeros(out_channels, *frame_size, dtype=torch.float)
        ).to(DEVICE)
        self.W_co = nn.parameter.Parameter(
            torch.zeros(out_channels, *frame_size, dtype=torch.float)
        ).to(DEVICE)
        self.W_cf = nn.parameter.Parameter(
            torch.zeros(out_channels, *frame_size, dtype=torch.float)
        ).to(DEVICE)
        self.__initialize_weights(weights_initializer)

    def __activation(self, activation: str) -> nn.Module:
        if activation == "tanh":
            return nn.Tanh()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "leakyRelu":
            return nn.LeakyReLU()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def __initialize_weights(self, initializer: WeightsInitializer):
        if initializer == WeightsInitializer.Zeros:
            return

        elif initializer == WeightsInitializer.He:
            nn.init.kaiming_normal_(self.W_ci, mode="fan_in", nonlinearity="leaky_relu")
            nn.init.kaiming_normal_(self.W_co, mode="fan_in", nonlinearity="leaky_relu")
            nn.init.kaiming_normal_(self.W_cf, mode="fan_in", nonlinearity="leaky_relu")
            return

        elif initializer == WeightsInitializer.Xavier:
            nn.init.xavier_normal_(self.W_ci, gain=1.0)
            nn.init.xavier_normal_(self.W_co, gain=1.0)
            nn.init.xavier_normal_(self.W_cf, gain=1.0)
            return
        else:
            raise ValueError(f"Invalid weights Initializer: {initializer}")

    def forward(
            self, X: torch.Tensor, prev_h: torch.Tensor, prev_cell: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        new_h, new_cell = self.convlstm_cell(X, prev_h, prev_cell)
        return new_h, new_cell

    def convlstm_cell(
            self, X: torch.Tensor, prev_h: torch.Tensor, prev_cell: torch.Tensor
    ):
        """ConvLSTM cell calculation.

        Args:
            X (torch.Tensor): input dataset.
            h_prev (torch.Tensor): previous hidden state.
            c_prev (torch.Tensor): previous cell state.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (current_hidden_state, current_cell_state)
        """
        X = X.to(prev_h.device)  # Convert X to the same device as prev_h
        conv_output = self.conv(torch.cat([X, prev_h], dim=1))

        i_conv, f_conv, c_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * prev_cell)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * prev_cell)

        # Current cell output (state)
        C = forget_gate * prev_cell + input_gate * self.activation(c_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C)

        # Current hidden state
        H = output_gate * self.activation(C)

        return H.to(DEVICE), C.to(DEVICE)


class ConvLSTM(nn.Module):
    """The ConvLSTM implementation (Shi et al., 2015)."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple],
            padding: Union[int, Tuple, str],
            activation: str,
            frame_size: Tuple,
            weights_initializer: WeightsInitializer = WeightsInitializer.Zeros,
    ) -> None:
        """

        Args:
            in_channels (int): input channel.
            out_channels (int): output channel.
            kernel_size (Union[int, Tuple]): The size of convolution kernel.
            padding (Union[int, Tuple, str]): Should be in ['same', 'valid' or (int, int)]
            activation (str): Name of activation function.
            frame_size (Tuple): height and width.
            weights_initializer (Optional[str]): Weight initializers of ['zeros', 'he', 'xavier'].
        """
        super().__init__()

        self.ConvLSTMCell = BaseConvLSTMCell(
            in_channels,
            out_channels,
            kernel_size,
            padding,
            activation,
            frame_size,
            weights_initializer,
        )

        self.out_channels = out_channels

    def forward(
            self,
            X: torch.Tensor,
            h: Optional[torch.Tensor] = None,
            cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """

        Args:
            X (torch.Tensor): tensor with the shape of (batch_size, num_channels, seq_len, height, width)

        Returns:
            torch.Tensor: tensor with the same shape of X
            :param X:
            :param cell:
            :param h:
        """
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(
            (batch_size, self.out_channels, seq_len, height, width)
        ).to(DEVICE)

        # Initialize hidden state
        h = torch.zeros((batch_size, self.out_channels, height, width)).to(DEVICE)

        # Initialize cell input
        cell = torch.zeros((batch_size, self.out_channels, height, width)).to(DEVICE)

        # Unroll over time steps
        for time_step in range(seq_len):
            h, cell = self.ConvLSTMCell(X[:, :, time_step], h, cell)
            output[:, :, time_step] = h  # type: ignore

        return output


class Seq2Seq(TimeSeriesModel):
    """The sequence to sequence model implementation using ConvLSTM."""

    def __init__(
            self,
            input_seq_length: int,
            num_layers: int,
            num_kernels: int,
            convlstm_params: ConvLSTMParams,
            label_seq_length: Optional[int] = None,
            return_sequences: bool = False,
    ) -> None:
        """

        Args:
            input_seq_length (int): Number of input frames.
            label_seq_length (Optional[int]): Number of label frames.
            num_layers (int): Number of ConvLSTM layers.
            num_kernels (int): Number of kernels.
            return_sequences (int): If True, the model predict the next frames that is the same length of inputs. If False, the model predicts only one next frame or the frames given by `label_seq_length`.
        """
        super().__init__()
        self.input_seq_length = input_seq_length
        self.label_seq_length = label_seq_length
        if label_seq_length is not None and return_sequences is True:
            logger.warning(
                "the `label_seq_length` is ignored because `return_sequences` is set to True."
            )
        self.num_layers = num_layers
        self.num_kernels = num_kernels
        self.return_sequences = return_sequences
        self.in_channels = convlstm_params["in_channels"]
        self.kernel_size = convlstm_params["kernel_size"]
        self.padding = convlstm_params["padding"]
        self.activation = convlstm_params["activation"]
        self.frame_size = convlstm_params["frame_size"]
        self.out_channels = convlstm_params["out_channels"]
        self.weights_initializer = convlstm_params["weights_initializer"]

        self.sequential = nn.Sequential()

        # Add first layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1",
            ConvLSTM(
                in_channels=self.in_channels,
                out_channels=self.num_kernels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                activation=self.activation,
                frame_size=self.frame_size,
                weights_initializer=self.weights_initializer,
            ),
        )

        self.sequential.add_module(
            "layernorm1",
            nn.LayerNorm([self.num_kernels, self.input_seq_length, *self.frame_size]),
        )

        # Add the rest of the layers
        for layer_idx in range(2, self.num_layers + 1):
            self.sequential.add_module(
                f"convlstm{layer_idx}",
                ConvLSTM(
                    in_channels=self.num_kernels,
                    out_channels=self.num_kernels,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    activation=self.activation,
                    frame_size=self.frame_size,
                    weights_initializer=self.weights_initializer,
                ),
            )

            self.sequential.add_module(
                f"layernorm{layer_idx}",
                nn.LayerNorm(
                    [self.num_kernels, self.input_seq_length, *self.frame_size]
                ),
            )

        self.sequential.add_module(
            "conv3d",
            nn.Conv3d(
                in_channels=self.num_kernels,
                out_channels=self.out_channels,
                kernel_size=(3, 3, 3),
                padding="same",
            ),
        )

        self.sequential.add_module("sigmoid", nn.Sigmoid())

    def forward(self, X: torch.Tensor):
        # Forward propagation through all the layers
        output = self.sequential(X)

        if self.return_sequences is True:
            return output

        if self.label_seq_length:
            return output[:, :, : self.label_seq_length, ...]

        return output[:, :, -1:, ...]

    def training_step(self, batch, batch_idx):
        # Print the batch to understand its structure
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    #
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True)
        return loss

    def predict_step(self, x):
        return self(x).detach()


class SelfAttentionMemory(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()

        # attention for hidden layer
        self.query_h = nn.Conv2d(input_dim, hidden_dim, 1, padding="same")
        self.key_h = nn.Conv2d(input_dim, hidden_dim, 1, padding="same")
        self.value_h = nn.Conv2d(input_dim, input_dim, 1, padding="same")
        self.z_h = nn.Conv2d(input_dim, input_dim, 1, padding="same")

        # attention for memory layer
        self.key_m = nn.Conv2d(input_dim, hidden_dim, 1, padding="same")
        self.value_m = nn.Conv2d(input_dim, input_dim, 1, padding="same")
        self.z_m = nn.Conv2d(input_dim, input_dim, 1, padding="same")

        # weights of concat channels of h Zh and Zm.
        self.w_z = nn.Conv2d(input_dim * 2, input_dim * 2, 1, padding="same")

        # weights of concat channels of Z and h.
        self.w = nn.Conv2d(input_dim * 3, input_dim * 3, 1, padding="same")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, h, m) -> Tuple:
        """
        Return:
            Tuple(tirch.Tensor, torch.Tensor): new Hidden layer and new memory module.
        """
        batch_size, _, H, W = h.shape
        # hidden attention
        k_h = self.key_h(h)
        q_h = self.query_h(h)
        v_h = self.value_h(h)

        k_h = k_h.view(batch_size, self.hidden_dim, H * W)
        q_h = q_h.view(batch_size, self.hidden_dim, H * W).transpose(1, 2)
        v_h = v_h.view(batch_size, self.input_dim, H * W)

        attention_h = torch.softmax(
            torch.bmm(q_h, k_h), dim=-1
        )  # The shape is (batch_size, H*W, H*W)
        z_h = torch.matmul(attention_h, v_h.permute(0, 2, 1))
        z_h = z_h.transpose(1, 2).view(batch_size, self.input_dim, H, W)
        z_h = self.z_h(z_h)

        # memory attention
        k_m = self.key_m(m)
        v_m = self.value_m(m)

        k_m = k_m.view(batch_size, self.hidden_dim, H * W)
        v_m = v_m.view(batch_size, self.input_dim, H * W)

        attention_m = torch.softmax(torch.bmm(q_h, k_m), dim=-1)
        z_m = torch.matmul(attention_m, v_m.permute(0, 2, 1))
        z_m = z_m.transpose(1, 2).view(batch_size, self.input_dim, H, W)
        z_m = self.z_m(z_m)

        # channel concat of Zh and Zm.
        Z = torch.cat([z_h, z_m], dim=1)
        Z = self.w_z(Z)

        # channel concat of Z and h
        W = torch.cat([Z, h], dim=1)
        W = self.w(W)

        # mi_conv: Wm;zi * Z + Wm;hi * Ht + bm;i
        # mg_conv: Wm;zg * Z + Wm;hg * Ht + bm;g
        # mo_conv: Wm;zo * Z + Wm;ho * Ht + bm;o
        mi_conv, mg_conv, mo_conv = torch.chunk(W, chunks=3, dim=1)
        input_gate = torch.sigmoid(mi_conv)
        g = torch.tanh(mg_conv)
        new_M = (1 - input_gate) * m + input_gate * g
        output_gate = torch.sigmoid(mo_conv)
        new_H = output_gate * new_M

        return new_H, new_M, attention_h


class SAMConvLSTMCell(nn.Module):
    def __init__(
            self,
            attention_hidden_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple],
            padding: Union[int, Tuple, str],
            activation: str,
            frame_size: Tuple,
            weights_initializer: WeightsInitializer = WeightsInitializer.Zeros,
    ) -> None:
        super().__init__()
        self.convlstm_cell = BaseConvLSTMCell(
            in_channels,
            out_channels,
            kernel_size,
            padding,
            activation,
            frame_size,
            weights_initializer,
        )
        self.attention_memory = SelfAttentionMemory(out_channels, attention_hidden_dims)

    def forward(
            self,
            X: torch.Tensor,
            prev_h: torch.Tensor,
            prev_cell: torch.Tensor,
            prev_memory: torch.Tensor,
    ) -> Tuple:
        new_h, new_cell = self.convlstm_cell(X, prev_h, prev_cell)
        new_h, new_memory, attention_h = self.attention_memory(new_h, prev_memory)
        return (
            new_h.to(DEVICE),
            new_cell.to(DEVICE),
            new_memory.to(DEVICE),
            attention_h.to(DEVICE),
        )


class SAMConvLSTMParams(TypedDict):
    attention_hidden_dims: int
    convlstm_params: ConvLSTMParams


class SAMConvLSTM(nn.Module):
    def __init__(self, attention_hidden_dims: int, convlstm_params: ConvLSTMParams):
        super().__init__()
        self.attention_hidden_dims = attention_hidden_dims
        self.in_channels = convlstm_params["in_channels"]
        self.kernel_size = convlstm_params["kernel_size"]
        self.padding = convlstm_params["padding"]
        self.activation = convlstm_params["activation"]
        self.frame_size = convlstm_params["frame_size"]
        self.out_channels = convlstm_params["out_channels"]
        self.weights_initializer = convlstm_params["weights_initializer"]

        self.sam_convlstm_cell = SAMConvLSTMCell(
            attention_hidden_dims, **convlstm_params
        )

        self._attention_scores: Optional[torch.Tensor] = None

    @property
    def attention_scores(self) -> Optional[torch.Tensor]:
        return self._attention_scores

    def forward(
            self,
            X: torch.Tensor,
            h: Optional[torch.Tensor] = None,
            cell: Optional[torch.Tensor] = None,
            memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, _, seq_len, height, width = X.size()

        # NOTE: Cannot store all attention scores because of memory. So only store attention map of the center.
        # And the same attention score are applied to each channel.
        self._attention_scores = torch.zeros(
            (batch_size, seq_len, height * width), device=DEVICE
        )

        if h is None:
            h = torch.zeros(
                (batch_size, self.out_channels, height, width), device=DEVICE
            )

        if cell is None:
            cell = torch.zeros(
                (batch_size, self.out_channels, height, width), device=DEVICE
            )

        if memory is None:
            memory = torch.zeros(
                (batch_size, self.out_channels, height, width), device=DEVICE
            )

        output = torch.zeros(
            (batch_size, self.out_channels, seq_len, height, width), device=DEVICE
        )

        for time_step in range(seq_len):
            h, cell, memory, attention_h = self.sam_convlstm_cell(
                X[:, :, time_step], h, cell, memory
            )

            output[:, :, time_step] = h  # type: ignore
            # Save attention maps of the center point because storing
            # the full `attention_h` is difficult because of the lot of memory usage.
            # `attention_h` shape is (batch_size, height*width, height*width)
            self._attention_scores[:, time_step] = attention_h[
                                                   :, attention_h.size(0) // 2
                                                   ]

        return output


class SAMSeq2SeqParams(TypedDict):
    attention_hidden_dims: int
    input_seq_length: int
    num_layers: int
    num_kernels: int
    return_sequences: NotRequired[bool]
    convlstm_params: ConvLSTMParams


class SAMSeq2Seq(TimeSeriesModel):
    def __init__(
            self,
            attention_hidden_dims: int,
            input_seq_length: int,
            num_layers: int,
            num_kernels: int,
            convlstm_params: ConvLSTMParams,
            return_sequences: bool = False,
    ):
        """

        Args:
            attention_hidden_dims (int): Number of attention hidden layers.
            input_seq_length (int): Number of input frames.
            num_layers (int): Number of ConvLSTM layers.
            num_kernels (int): Number of kernels.
            return_sequences (int): If True, the model predict the next frames that is the same length of inputs. If False, the model predicts only one next frame.
            convlstm_params (ConvLSTMParams): Parameters for ConvLSTM module.
        """
        super().__init__()
        self.attention_hidden_dims = attention_hidden_dims
        self.input_seq_length = input_seq_length
        self.num_layers = num_layers
        self.num_kernels = num_kernels
        self.return_sequences = return_sequences
        self.in_channels = convlstm_params["in_channels"]
        self.kernel_size = convlstm_params["kernel_size"]
        self.padding = convlstm_params["padding"]
        self.activation = convlstm_params["activation"]
        self.frame_size = convlstm_params["frame_size"]
        self.out_channels = convlstm_params["out_channels"]
        self.weights_initializer = convlstm_params["weights_initializer"]

        self.sequential = nn.Sequential()

        self.sequential.add_module(
            "sam-convlstm1",
            SAMConvLSTM(
                attention_hidden_dims=self.attention_hidden_dims,
                convlstm_params={
                    "in_channels": self.in_channels,
                    "out_channels": self.num_kernels,
                    "kernel_size": self.kernel_size,
                    "padding": self.padding,
                    "activation": self.activation,
                    "frame_size": self.frame_size,
                    "weights_initializer": self.weights_initializer,
                },
            ),
        )

        self.sequential.add_module(
            "layernorm1",
            nn.LayerNorm([num_kernels, self.input_seq_length, *self.frame_size]),
        )

        for layer_idx in range(2, num_layers + 1):
            self.sequential.add_module(
                f"sam-convlstm{layer_idx}",
                SAMConvLSTM(
                    attention_hidden_dims=self.attention_hidden_dims,
                    convlstm_params={
                        "in_channels": self.num_kernels,
                        "out_channels": self.num_kernels,
                        "kernel_size": self.kernel_size,
                        "padding": self.padding,
                        "activation": self.activation,
                        "frame_size": self.frame_size,
                        "weights_initializer": self.weights_initializer,
                    },
                ),
            )
            self.sequential.add_module(
                f"layernorm{layer_idx}",
                nn.LayerNorm([num_kernels, self.input_seq_length, *self.frame_size]),
            )

        self.sequential.add_module(
            "conv3d",
            nn.Conv3d(
                in_channels=self.num_kernels,
                out_channels=self.out_channels,
                kernel_size=(3, 3, 3),
                padding="same",
            ),
        )

        self.sequential.add_module("sigmoid", nn.Sigmoid())

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = self.sequential(X)

        if self.return_sequences is True:
            return output

        return output[:, :, -1:, :, :]

    def get_attention_maps(self):
        # get all sa_convlstm module
        sam_convlstm_modules = [
            (name, module)
            for name, module in self.named_modules()
            if module.__class__.__name__ == "SAMConvLSTM"
        ]
        return {
            name: module.attention_scores for name, module in sam_convlstm_modules
        }  # attention scores shape is (batch_size, seq_length, height * width)
