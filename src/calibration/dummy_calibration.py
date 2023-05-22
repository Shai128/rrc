import torch

from src.calibration.online_calibration import OnlineCalibration
from src.model_prediction.i2i_model_prediction import I2IUQModelPrediction
from src.uncertainty_quantifiers.prediction_interval import PredictionInterval
from src.data_utils.data_sample.data_sample import DataSample, LabeledDataSample
from src.model_prediction.model_prediction import ModelPrediction
from src.model_prediction.qr_model_prediction import QRModelPrediction
from src.uncertainty_quantifiers.prediction_intervals_matrix import PredictionIntervalsMatrix


class DummyCalibration(OnlineCalibration):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

    def calibrate(self, data_sample: DataSample, model_prediction: QRModelPrediction, **kwargs) -> PredictionInterval:
        return PredictionInterval(model_prediction.intervals)

    def update(self, labeled_sample: LabeledDataSample, model_prediction: ModelPrediction,
               calibrated_uncertainty_set: PredictionInterval, timestamp: int, **kwargs):
        pass

    @property
    def name(self) -> str:
        return "uncalibrated"


class DummyI2ICalibration(OnlineCalibration):

    def __init__(self, lambda_hat :int=1, **kwargs):
        super().__init__(**kwargs)
        self.lambda_hat = lambda_hat
        pass

    def calibrate(self, data_sample: DataSample, model_prediction: I2IUQModelPrediction, **kwargs) -> PredictionIntervalsMatrix:
        estimated_mean, l, u = model_prediction.estimated_mean.mean_estimate, model_prediction.lower_std, model_prediction.upper_std
        estimated_mean = estimated_mean.cpu()
        if torch.is_tensor(l):
            l = l.cpu()
        if torch.is_tensor(u):
            u = u.cpu()
        shape = estimated_mean.squeeze().shape + (2,)
        calibrated_interval = torch.zeros(*shape)

        calibrated_interval[..., 0] = estimated_mean.clone() - self.lambda_hat * l
        calibrated_interval[..., 1] = estimated_mean.clone() + self.lambda_hat * u
        return PredictionIntervalsMatrix(calibrated_interval, estimated_mean)

    def update(self, labeled_sample: LabeledDataSample, model_prediction: ModelPrediction,
               calibrated_uncertainty_set: PredictionInterval, timestamp: int, **kwargs):
        pass

    @property
    def name(self) -> str:
        return f"uncalibrated_lambda={self.lambda_hat}"
