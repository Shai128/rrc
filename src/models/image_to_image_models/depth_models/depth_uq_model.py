import copy
from typing import List
import os

import numpy as np
from tqdm import tqdm

from src.data_utils.data_sample.data_sample import LabeledDataSample
from src.data_utils.data_sample.depth_data_sample import DepthDataSample, LabeledDepthDataSample, \
    DepthStdRegressionDataSample, LabeledDepthStdRegressionDataSample
from src.data_utils.datasets.dataset import Dataset
from src.data_utils.datasets.depth_dataset import DepthDataset
from src.model_prediction.i2i_model_prediction import I2IUQModelPrediction
from src.models.image_to_image_models.abstract_models.i2i_std_regressor import BaselineStdRegressor, I2IStdRegressor
from src.models.image_to_image_models.abstract_models.online_i2i_model import OnlineI2IUQModel
from src.models.image_to_image_models.depth_models.depth_mean_regressor import DepthMeanRegressor
from src.models.image_to_image_models.depth_models.depth_std_regressor import StdRegressionByAlignedPreviousResiduals, \
    ResidualMagnitudeRegressor


def get_i2i_uq_model(std_method: str, backbone: str, device, base_save_path: str) -> I2IStdRegressor:
    model_params = {
                    'base_save_path': base_save_path,
                    'backbone': backbone,
                    'device': device,
                    'loss_batch_size': 4096
                    }
    if std_method == 'baseline':
        return BaselineStdRegressor(**model_params)
    elif std_method == 'residual_magnitude':
        return ResidualMagnitudeRegressor(**model_params)
    elif std_method == 'previous_residual_with_alignment':
        return StdRegressionByAlignedPreviousResiduals(**model_params)
    else:
        raise NotImplementedError(f"invalid uq method: {std_method}")


class DepthUQModel(OnlineI2IUQModel):

    def __init__(self, trained_mean_regressor_path, backbone, std_method, device, **kwargs):
        super().__init__(**kwargs)
        if not os.path.exists(trained_mean_regressor_path):
            print(f"Warning: the mean depth regressor weights were not found in path: {trained_mean_regressor_path}")
        self.mean_estimator = DepthMeanRegressor(trained_mean_regressor_path, backbone, device, **kwargs)
        self.std_estimator = get_i2i_uq_model(std_method, backbone, device, **kwargs)
        self._name = f"depth_bb={backbone}_std={std_method}"

    def predict(self, data_sample: DepthDataSample) -> I2IUQModelPrediction:
        self.eval()
        mean_estimate = self.mean_estimator.predict(data_sample)
        sample_for_std_regressor = DepthStdRegressionDataSample(mean_estimate=mean_estimate,
                                                                **data_sample.__dict__)
        std_prediction = self.std_estimator.predict(sample_for_std_regressor)
        return I2IUQModelPrediction(mean_estimate, std_prediction)

    def online_fit(self, labeled_data_sample: LabeledDepthDataSample,
                   labeled_inference_data_sample: LabeledDepthDataSample = None, **kwargs):
        assert labeled_inference_data_sample is not None
        self.train()
        self.mean_estimator.online_fit(labeled_data_sample)

        train_sample_mean_estimate = self.mean_estimator.predict(labeled_data_sample)
        train_sample_for_std_regressor = LabeledDepthStdRegressionDataSample(mean_estimate=train_sample_mean_estimate,
                                                                             **labeled_data_sample.__dict__)
        inference_sample_mean_estimate = self.mean_estimator.predict(labeled_inference_data_sample)
        inference_sample_for_std_regressor = LabeledDepthStdRegressionDataSample(
            mean_estimate=inference_sample_mean_estimate,
            **labeled_inference_data_sample.__dict__)
        self.std_estimator.online_fit(train_sample_for_std_regressor,
                                      labeled_inference_data_sample=inference_sample_for_std_regressor)

    def train(self, mode: bool = True):
        self.mean_estimator.train(mode)
        self.std_estimator.train(mode)

    def eval(self):
        self.mean_estimator.eval()
        self.std_estimator.eval()

    # TODO: complete offline fit
    def offline_train_aux(self, dataset : DepthDataset, training_timestamps: List[int], epochs: int, **kwargs) -> None:
        training_timestamps = copy.deepcopy(training_timestamps)
        for e in (range(epochs)):
            self.train()
            np.random.shuffle(training_timestamps)
            for i in tqdm(training_timestamps):
                train_sample = dataset.get_train_sample(i)
                self.mean_estimator.fit(train_sample)

                train_sample_mean_estimate = self.mean_estimator.predict(train_sample)
                std_train_sample = LabeledDepthStdRegressionDataSample(mean_estimate=train_sample_mean_estimate,
                                                                       **train_sample.__dict__)
                self.std_estimator.fit(std_train_sample)

            print("epoch: ", e)

            # if e % 20 == 0 and e > 0:
            #     self.save_state(dataset_name, e * len(train_idx), e)
            #     self.uq_model.save_state(dataset_name, e * len(train_idx), e)

    def load_state(self, dataset_name: str, epoch: int):
        self.mean_estimator.load_state(dataset_name, epoch)
        self.std_estimator.load_state(dataset_name, epoch)

    def save_state(self, dataset_name: str, epoch: int):
        self.mean_estimator.save_state(dataset_name, epoch)
        self.std_estimator.save_state(dataset_name, epoch)

    def has_state(self, dataset_name: str, epoch: int) -> bool:
        return self.mean_estimator.has_state(dataset_name, epoch) and self.std_estimator.has_state(dataset_name, epoch)

    @property
    def name(self):
        return self._name
