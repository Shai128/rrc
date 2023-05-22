import abc

import torch
from torch import nn

from src.data_utils.data_sample.i2i_data_sample import I2IDataSample, LabeledI2IDataSample
from src.model_prediction.i2i_model_prediction import I2IUQModelPrediction
from src.models.abstract_models.online_learning_model import OnlineLearningModel
from src.models.image_to_image_models.depth_models.Leres.lib.models.multi_depth_model_auxiv2 import ModelOptimizer
from src.utils import load_ckpt, save_ckpt, has_state


def unsqueeze_tensors_in_dict(data: dict, device):
    new_data = {}
    for key in data.keys():
        if torch.is_tensor(data[key]):
            new_data[key] = data[key].unsqueeze(0).to(device)
        else:
            new_data[key] = data[key]
    return new_data


class OnlineI2ILearningModel(OnlineLearningModel, abc.ABC):
    def __init__(self, base_save_path, **kwargs):
        OnlineLearningModel.__init__(self, **kwargs)
        self.base_save_path = base_save_path
        self.losses = []

    def load_state(self, dataset_name: str, epoch: int):
        print(f"loading model: {self.model_save_name} for dataset {dataset_name} at epoch {epoch}")
        return load_ckpt(self.base_save_path, dataset_name, self.model_save_name, self.network,
                         optimizer=self.optimizer.optimizer, scheduler=self.scheduler, epoch=epoch)

    def save_state(self, dataset_name: str, epoch: int):
        save_ckpt(self.base_save_path, dataset_name, self.model_save_name, epoch, self.network,
                  optimizer=self.optimizer.optimizer, scheduler=self.scheduler)

    def has_state(self, dataset_name: str, epoch: int) -> bool:
        return has_state(self.base_save_path, dataset_name, self.model_save_name, epoch)

    def online_fit(self, labeled_data_sample: LabeledI2IDataSample, **kwargs):
        self.fit(labeled_data_sample, **kwargs)

    def fit(self, labeled_data_sample: LabeledI2IDataSample, **kwargs):
        loss = self.loss(labeled_data_sample, **kwargs)
        loss_dict = {'total_loss': loss}
        self.optimizer.optim(loss_dict)
        self.losses += [loss.cpu().item()]
        return loss

    @abc.abstractmethod
    def loss(self, data: LabeledI2IDataSample, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def optimizer(self) -> ModelOptimizer:
        pass

    @property
    @abc.abstractmethod
    def scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        pass

    @property
    @abc.abstractmethod
    def network(self) -> nn.Module:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    def model_save_name(self) -> str:
        return self.name


# # Image to Image Time-Series Model
class OnlineI2IUQModel(OnlineLearningModel, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abc.abstractmethod
    def predict(self, data_sample: I2IDataSample) -> I2IUQModelPrediction:
        pass

    @abc.abstractmethod
    def online_fit(self, labeled_data_sample: LabeledI2IDataSample, **kwargs):
        pass

