import torch
from torch import nn

from src.data_utils.data_scaler import DataScaler
from src.utils import optimizer_to


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.scaler = DataScaler()
        self.optimizers = []
        self.models = []

    def initialize_scalers(self, x_train, y_train):
        self.scaler.initialize_scalers(x_train, y_train)
        scaled_y_train = self.scaler.scale_y(y_train)
        self.y_min = torch.min(scaled_y_train).item()
        self.y_max = torch.max(scaled_y_train).item()

    def plot_losses(self):
        pass

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, value):
        self._models = value

    @property
    def optimizers(self):
        return self._optimizers

    @optimizers.setter
    def optimizers(self, value):
        self._optimizers = value

    def train(self, mode: bool = True):
        super(Model, self).train(mode)
        for model in self.models:
            model.train(mode)

    def eval(self):
        super(Model, self).eval()
        for model in self.models:
            model.eval()

    def to(self, device):
        super(Model, ).to(device)
        for optimizer in self.optimizers:
            optimizer_to(optimizer, device)
        return self

