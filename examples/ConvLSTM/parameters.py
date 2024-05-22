from collections import OrderedDict
from torchts.nn.models.ConvLSTM import ConvGRU_cell, ConvLSTM_cell


convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],

    [
        ConvLSTM_cell(shape=(64, 64), input_channels=16, filter_size=5, num_features=64),
        ConvLSTM_cell(shape=(32, 32), input_channels=64, filter_size=5, num_features=96),
        ConvLSTM_cell(shape=(16, 16), input_channels=96, filter_size=5, num_features=96)
    ]
]

convlstm_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({
            'conv3_leaky_1': [64, 16, 3, 1, 1],
            'conv4_leaky_1': [16, 1, 1, 1, 0]
        }),
    ],

    [
        ConvLSTM_cell(shape=(16, 16), input_channels=96, filter_size=5, num_features=96),
        ConvLSTM_cell(shape=(32, 32), input_channels=96, filter_size=5, num_features=96),
        ConvLSTM_cell(shape=(64, 64), input_channels=96, filter_size=5, num_features=64),
    ]
]

convgru_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],

    [
        ConvGRU_cell(shape=(64, 64), input_channels=16, filter_size=5, num_features=64),
        ConvGRU_cell(shape=(32, 32), input_channels=64, filter_size=5, num_features=96),
        ConvGRU_cell(shape=(16, 16), input_channels=96, filter_size=5, num_features=96)
    ]
]

convgru_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({
            'conv3_leaky_1': [64, 16, 3, 1, 1],
            'conv4_leaky_1': [16, 1, 1, 1, 0]
        }),
    ],

    [
        ConvGRU_cell(shape=(16, 16), input_channels=96, filter_size=5, num_features=96),
        ConvGRU_cell(shape=(32, 32), input_channels=96, filter_size=5, num_features=96),
        ConvGRU_cell(shape=(64, 64), input_channels=96, filter_size=5, num_features=64),
    ]
]