import os
import warnings

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from torchts.data_provider.data_factory import data_provider
from torchts.nn.models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, iTransformer, Reformer
from torchts.utils.tools import visual

warnings.filterwarnings('ignore')


class LightningExpMain(L.LightningModule):
    def __init__(self, args):
        super(LightningExpMain, self).__init__()
        self.args = args
        self.model = self._build_model()
        self.criterion = nn.MSELoss()
        self.scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        self.preds = []
        self.trues = []

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'LongSequenceTimeForecasting': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'iTransformer': iTransformer
        }
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def setup(self, stage=None):
        # Data setup can be done here
        self.train_data, self.train_loader = self._get_data('train')
        if not self.args.train_only:
            self.vali_data, self.vali_loader = self._get_data('val')
            self.test_data, self.test_loader = self._get_data('test')

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return optimizer

    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        if 'Linear' in self.args.model:
            return self.model(batch_x)
        else:
            if self.args.output_attention:
                return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    def training_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x, batch_y, batch_x_mark, batch_y_mark = (
            batch_x.float().to(self.device),
            batch_y.float().to(self.device),
            batch_x_mark.float().to(self.device),
            batch_y_mark.float().to(self.device),
        )
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = self.criterion(outputs, batch_y)
        else:
            outputs = self.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            loss = self.criterion(outputs, batch_y)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x, batch_y, batch_x_mark, batch_y_mark = (
            batch_x.float().to(self.device),
            batch_y.float().to(self.device),
            batch_x_mark.float().to(self.device),
            batch_y_mark.float().to(self.device),
        )
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = self.criterion(outputs, batch_y)
        else:
            outputs = self.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            loss = self.criterion(outputs, batch_y)

        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x, batch_y, batch_x_mark, batch_y_mark = (
            batch_x.float().to(self.device),
            batch_y.float().to(self.device),
            batch_x_mark.float().to(self.device),
            batch_y_mark.float().to(self.device),
        )
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = self.criterion(outputs, batch_y)
        else:
            outputs = self.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            loss = self.criterion(outputs, batch_y)

        # Store predictions and ground truth for plotting
        self.preds.append(outputs.detach().cpu().numpy())
        self.trues.append(batch_y.detach().cpu().numpy())

        self.log('test_loss', loss)
        return loss

    def on_test_end(self):
        # Concatenate predictions and ground truth
        preds = np.concatenate(self.preds, axis=0)
        trues = np.concatenate(self.trues, axis=0)

        # Plot the results
        folder_path = './test_results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for i in range(min(len(preds), 10)):  # plot the first 10 predictions
            gt = np.concatenate((trues[i, :, -1], preds[i, :, -1]), axis=0)
            pd = np.concatenate((trues[i, :, -1], preds[i, :, -1]), axis=0)
            visual(gt, pd, os.path.join(folder_path, f'{i}.pdf'))

    def predict_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x, batch_y, batch_x_mark, batch_y_mark = (
            batch_x.float().to(self.device),
            batch_y.float(),
            batch_x_mark.float().to(self.device),
            batch_y_mark.float().to(self.device),
        )
        dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            outputs = self.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        return outputs.detach().cpu().numpy()

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True,
                          num_workers=self.args.num_workers)

    def val_dataloader(self):
        return DataLoader(self.vali_data, batch_size=self.args.batch_size, shuffle=False,
                          num_workers=self.args.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.args.batch_size, shuffle=False,
                          num_workers=self.args.num_workers)

    def predict_dataloader(self):
        pred_data, pred_loader = self._get_data(flag='pred')
        return pred_loader
