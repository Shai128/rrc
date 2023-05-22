from typing import List

import torch
from src.calibration.online_calibration import OnlineCalibration
from src.uncertainty_quantifiers.prediction_interval import PredictionInterval
from src.data_utils.data_sample.data_sample import DataSample, LabeledDataSample
from src.losses import MiscoverageLoss
from src.model_prediction.qr_model_prediction import QRModelPrediction


class ACICalibration(OnlineCalibration):

    def __init__(self, gamma: float, alpha: float, calibration_set_size: int, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha_t = alpha
        self.alpha = alpha
        self.Q_t = 0
        self.miscoverage_loss = MiscoverageLoss()
        self.calibration_set_samples = []
        self.calibration_set_predictions = []
        self.calibration_set_size = calibration_set_size
        self.alpha_t_history = []
        self.Q_t_history = []

    @staticmethod
    def compute_q_t(samples: List[LabeledDataSample], predictions: List[QRModelPrediction], alpha_t):
        if len(samples) < 10:
            return 0
        y_cal = torch.stack([sample.y for sample in samples]).squeeze()
        y_lower = torch.stack([intervals_cal.intervals[..., 0] for intervals_cal in predictions]).to(y_cal.device)
        y_upper = torch.stack([intervals_cal.intervals[..., 1] for intervals_cal in predictions]).to(y_cal.device)
        q = 1 - alpha_t + (1 / y_cal.shape[0])
        q = min(1, max(0, q))
        Q_t = torch.quantile(torch.max(y_lower - y_cal, y_cal - y_upper), q=q).item()
        return Q_t

    def calibrate(self, data_sample: DataSample, model_prediction: QRModelPrediction, **kwargs) -> PredictionInterval:
        estimated_intervals = model_prediction.intervals
        with torch.no_grad():
            if len(estimated_intervals.shape) == 1:
                estimated_intervals = estimated_intervals.unsqueeze(0)
            calibrated_interval = estimated_intervals.clone()
            calibrated_interval[:, 0] -= self.Q_t
            calibrated_interval[:, 1] += self.Q_t
        calibrated_interval = PredictionInterval(calibrated_interval)
        return calibrated_interval

    def update(self, labeled_sample: LabeledDataSample, model_prediction: QRModelPrediction,
               calibrated_uncertainty_set: PredictionInterval, timestamp: int,
               **kwargs):
        not_in_interval = self.miscoverage_loss(calibrated_uncertainty_set, labeled_sample)
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha - not_in_interval)
        self.calibration_set_samples += [labeled_sample]
        self.calibration_set_predictions += [model_prediction]
        self.calibration_set_samples = self.calibration_set_samples[-self.calibration_set_size:]
        self.calibration_set_predictions = self.calibration_set_predictions[-self.calibration_set_size:]
        self.Q_t = ACICalibration.compute_q_t(self.calibration_set_samples, self.calibration_set_predictions,
                                              self.alpha_t)
        self.alpha_t_history.append(self.alpha_t)
        self.Q_t_history.append(self.Q_t)

    @property
    def name(self) -> str:
        return f"aci_gamma={self.gamma}_alpha_t={self.alpha}_Q_t={self.Q_t}"
