from typing import List
from src.calibration.online_calibration import OnlineCalibration
from src.uncertainty_quantifiers.uncertainty_set import UncertaintySet
from src.data_utils.data_sample.data_sample import DataSample, LabeledDataSample
from src.losses import OnlineLoss
from src.model_prediction.model_prediction import ModelPrediction
from src.calibration.set_constructing_functions.set_constructing_function import PredictionSetConstructingFunction


class RollingRiskControl(OnlineCalibration):
    def __init__(self, gamma: float, alpha: float, set_constructing_function: PredictionSetConstructingFunction,
                 loss: OnlineLoss, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.theta_t = 0
        self.alpha = alpha
        self.set_constructing_function = set_constructing_function
        self.loss = loss

    def calibrate(self, data_sample: DataSample, model_prediction: ModelPrediction, **kwargs) -> UncertaintySet:
        return self.set_constructing_function(data_sample, self.theta_t, model_prediction)

    def update(self, labeled_sample: LabeledDataSample, model_prediction: ModelPrediction,
               calibrated_uncertainty_set: UncertaintySet, timestamp: int,
               **kwargs):
        l_t = self.loss(calibrated_uncertainty_set, labeled_sample)
        self.loss.update(calibrated_uncertainty_set, labeled_sample)
        self.theta_t = self.theta_t + self.gamma * (l_t - self.alpha)
        self.set_constructing_function.update(labeled_sample, model_prediction, calibrated_uncertainty_set)

    @property
    def name(self):
        return f"rrc_gamma={self.gamma}_f={self.set_constructing_function.name}_loss={self.loss.name}"


class RollingRiskControlWithMultipleRisks(OnlineCalibration):

    def __init__(self, gammas: List[float], alphas: List[float],
                 set_constructing_function: PredictionSetConstructingFunction, losses: List[OnlineLoss], **kwargs):
        super().__init__(**kwargs)
        self.gammas = gammas
        self.alphas = alphas
        self.set_constructing_function = set_constructing_function
        self.losses = losses
        self.rrcs = [
            RollingRiskControl(gamma, alpha, set_constructing_function, loss, **kwargs) for
            gamma, alpha, loss in zip(gammas, alphas, losses)]

    def calibrate(self, data_sample: DataSample, model_prediction: ModelPrediction, **kwargs) -> UncertaintySet:
        thetas = [rrc.theta_t for rrc in self.rrcs]
        return self.set_constructing_function(data_sample, thetas, model_prediction)

    def update(self, labeled_sample: LabeledDataSample, model_prediction: ModelPrediction,
               calibrated_uncertainty_set: UncertaintySet, timestamp: int,
               **kwargs):
        for rrc in self.rrcs:
            rrc.update(labeled_sample, model_prediction, calibrated_uncertainty_set, timestamp)

    @property
    def name(self) -> str:
        return f"multi_rrc_gammas={self.gammas}_f={self.set_constructing_function.name}_losses={[loss.name for loss in self.losses]}"
