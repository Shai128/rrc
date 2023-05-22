import abc
from abc import ABC
from typing import List

from src.data_utils.data_sample.data_sample import DataSample, LabeledDataSample
from src.data_utils.datasets.dataset import Dataset
from src.model_prediction.model_prediction import ModelPrediction
from src.models.abstract_models.model import Model


class OnlineLearningModel(Model, ABC):

    def __init__(self, **kwargs):
        Model.__init__(self)

    @abc.abstractmethod
    def online_fit(self, labeled_data_sample: LabeledDataSample, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, data_sample: DataSample) -> ModelPrediction:
        pass

    def offline_train(self, dataset : Dataset, training_timestamps: List[int], epochs: int, **kwargs) -> None:
        if len(training_timestamps) == 0:
            return
        dataset_name = dataset.dataset_name
        if self.has_state(dataset_name, epochs):
            self.load_state(dataset_name, epochs)
            print(f"successfully loaded model {self.name} for data {dataset.dataset_name} at epoch {epochs}")
            return
        else:
            epochs_to_train = epochs
            for e in range(epochs, 0, -20):
                if self.has_state(dataset_name, e):
                    self.load_state(dataset_name, epoch=e)
                    epochs_to_train = epochs - e
                    print(f"successfully loaded model {self.name} for data {dataset.dataset_name} at epoch {e}")
                    break
            print("warning. did not find offline trained state. starting offline fit")
            self.offline_train_aux(dataset, training_timestamps, epochs_to_train, **kwargs)
            self.save_state(dataset_name, epochs)

    def offline_train_aux(self, dataset : Dataset, training_timestamps: List[int], epochs: int, **kwargs) -> None:
        pass

    def load_state(self, dataset_name: str, epoch: int):
        pass

    def save_state(self, dataset_name: str, epoch: int):
        pass

    def has_state(self, dataset_name: str, epoch: int) -> bool:
        return True

    @property
    @abc.abstractmethod
    def name(self):
        pass
