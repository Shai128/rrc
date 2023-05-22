import itertools
from typing import List

import torch

from src import utils
from src.data_utils.data_sample.data_sample import LabeledDataSample
from src.data_utils.data_sample.regression_data_sample import RegressionDataSample, LabeledRegressionDataSample
from src.data_utils.datasets.dataset import Dataset
from src.model_prediction.model_prediction import ModelPrediction
from src.model_prediction.qr_model_prediction import QRModelPrediction
from src.models.abstract_models.online_learning_model import OnlineLearningModel
from src.models.architectures.base_model import BaseModel
from src.models.architectures.lstm_model import LSTMModel


def batch_pinball_loss(quantile_level, quantile, y):
    diff = quantile - y.squeeze()
    mask = (diff.ge(0).float() - quantile_level).detach()

    return (mask * diff).mean(dim=0)


class OnlineQR(OnlineLearningModel):

    def __init__(self, x_dim, y_dim, device, lstm_hidden_size, lstm_layers, lstm_in_layers,
                 lstm_out_layers, alpha, non_linearity='lrelu', dropout=0.1, lr=1e-3, wd=0.,
                 extrapolate_quantiles=False, backward_size=3, train_all_q=False, batch_size=256):
        super().__init__()
        self.lr = lr
        self.wd = wd
        tau_dim = 1
        self.x_feature_extractor = LSTMModel(x_dim, y_dim=y_dim, out_dim=lstm_out_layers[-1],
                                             lstm_hidden_size=lstm_hidden_size, lstm_layers=lstm_layers,
                                             lstm_in_layers=lstm_in_layers, lstm_out_layers=lstm_out_layers[:-1],
                                             dropout=dropout, non_linearity=non_linearity).to(device)

        self.quantile_estimator = BaseModel(lstm_out_layers[-1] + tau_dim, y_dim, [32], dropout=dropout,
                                            batch_norm=False, non_linearity=non_linearity).to(device)

        self.models = [self.quantile_estimator, self.x_feature_extractor]
        params = list(itertools.chain(*[list(model.parameters()) for model in self.models]))
        self.optimizers = [torch.optim.Adam(params, lr=lr, weight_decay=wd)]

        self.backward_size = backward_size  # args.backward_size
        self.train_all_q = train_all_q  # args.train_all_q
        self.device = device
        self.alpha = alpha
        self.lr = lr
        self.wd = wd
        self.extrapolate_quantiles = extrapolate_quantiles
        self.lstm_hd = lstm_hidden_size
        self.lstm_nl = lstm_layers
        self.lstm_in_hd = lstm_in_layers
        self.lstm_out_hd = lstm_out_layers
        self.batch_size = batch_size

    def predict(self, data_sample: RegressionDataSample, alpha=None) -> QRModelPrediction:
        self.eval()
        if alpha is None:
            alpha = self.alpha
        x = data_sample.x
        previous_ys = data_sample.previous_ys
        alpha_rep = torch.ones((x.shape[0], 1), device=x.device) * alpha
        if self.train_all_q:
            _, inverse_cdf = self.get_quantile_function(x, previous_ys,
                                                        extrapolate_quantiles=self.extrapolate_quantiles)
            q_low = inverse_cdf(alpha_rep / 2)
            q_high = inverse_cdf(1 - alpha_rep / 2)
            intervals = torch.stack([q_low, q_high]).T
            return QRModelPrediction(intervals.squeeze())
        else:
            intervals = self.estimate_quantiles(x, previous_ys, torch.Tensor([alpha / 2, 1 - alpha / 2]).to(x.device))
            return QRModelPrediction(intervals.squeeze())

    def online_fit(self, labeled_data_sample: LabeledRegressionDataSample, **kwargs):
        self.train()
        all_x = labeled_data_sample.all_x
        all_y = labeled_data_sample.all_y
        all_previous_ys = labeled_data_sample.all_previous_ys
        all_x.required_grad = True
        all_previous_ys.required_grad = True

        batch_size = min(all_x.shape[0], self.batch_size)
        loss = self.pinball_loss(all_x[-batch_size:], all_y[-batch_size:, -1], all_previous_ys[-batch_size:])

        for optimizer in self.optimizers:
            optimizer.zero_grad()
            if loss.requires_grad:
                loss.backward()
            optimizer.step()
        return loss

    def pinball_loss(self, x, y, previous_ys):
        if self.train_all_q:
            quantile_levels = self.get_learned_quantile_levels()
        else:
            alpha = self.alpha
            quantile_levels = torch.Tensor([alpha / 2, 1 - alpha / 2]).to(x.device)

        quantiles = self.estimate_quantiles(x, previous_ys, quantile_levels=quantile_levels)
        quantile_levels_rep = quantile_levels.unsqueeze(0).repeat(x.shape[0], 1).flatten(0, 1)
        assert len(y.shape) == 2
        y_rep = y.repeat(1, quantile_levels.shape[0]).flatten(0, 1)
        pinball_loss = batch_pinball_loss(quantile_levels_rep, quantiles.flatten(0, 1), y_rep)
        return pinball_loss

    def forward(self, x, previous_ys, quantile_levels):
        if previous_ys.shape[0] == 1 and x.shape[0] > 1:
            previous_ys = previous_ys.repeat(x.shape[0], 1, 1)
        x_and_time_extraction = self.x_feature_extractor(x, previous_ys)
        y_rec = self.quantile_estimator(torch.cat([quantile_levels.unsqueeze(-1), x_and_time_extraction], dim=-1))

        return y_rec

    def get_learned_quantile_levels(self):
        if self.train_all_q:
            return torch.arange(0.02, 0.99, 0.005, device=self.device)
        else:
            return torch.Tensor([self.tau / 2, 1 - self.tau / 2]).to(self.device)

    def estimate_quantiles(self, x, previous_ys, quantile_levels):
        quantile_levels_rep = quantile_levels.unsqueeze(0).repeat(x.shape[0], 1).flatten(0, 1)
        x_rep = x.unsqueeze(1).repeat(1, quantile_levels.shape[0], 1, 1).flatten(0, 1)
        previous_ys_rep = previous_ys.unsqueeze(1).repeat(1, quantile_levels.shape[0], 1, 1).flatten(0, 1)
        unflatten = torch.nn.Unflatten(dim=0, unflattened_size=(x.shape[0], quantile_levels.shape[0]))
        quantiles = unflatten(self.forward(x_rep, previous_ys_rep, quantile_levels_rep)).squeeze(-1)
        return quantiles

    def get_quantile_function(self, x, previous_ys, extrapolate_quantiles=False):
        quantile_levels, _ = self.get_learned_quantile_levels().sort()
        quantiles = self.estimate_quantiles(x, previous_ys, quantile_levels)
        quantile_levels = quantile_levels.detach().squeeze()
        quantiles = quantiles.detach()
        quantile_functions = utils.batch_estim_dist(quantiles, quantile_levels, self.dataset.y_scaled_min,
                                                    self.dataset.y_scaled_max,
                                                    smooth_tails=True, tau=0.01,
                                                    extrapolate_quantiles=extrapolate_quantiles)
        return quantile_functions

    def offline_train_aux(self, dataset: Dataset, training_timestamps: List[int], epochs: int, **kwargs) -> None:
        if len(training_timestamps) == 0:
            return
        raise NotImplementedError()

    def load_state(self, dataset_name: str, epoch: int):
        raise NotImplementedError()

    def save_state(self, dataset_name: str, epoch: int):
        raise NotImplementedError()

    @property
    def name(self):
        return f"qr_all_q={int(self.train_all_q)}_lstm_hd={self.lstm_hd}_lstm_nl={self.lstm_nl}lstm_in_hd={self.lstm_in_hd}_lstm_out_hd={self.lstm_out_hd}_lr={self.lr}"

    def plot_losses(self):
        pass
