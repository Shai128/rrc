from typing import List, Dict

from src.calibration.online_calibration import OnlineCalibration
from src.data_utils.data_scaler import DataScaler
from src.uncertainty_quantifiers.uncertainty_set import UncertaintySet, UncertaintySetsCollection
from src.data_utils.data_sample.data_sample import DataSample, LabeledDataSample
from src.data_utils.datasets.dataset import Dataset
from src.main_helper import initialize_uncertainty_set_collection
from src.model_prediction.model_prediction import ModelPrediction
from src.utils import Task


class CalibrationSchemes:
    def __init__(self, calibration_schemes: List[OnlineCalibration]):
        self.calibration_schemes = calibration_schemes

    def calibrate(self, data_sample: DataSample, model_prediction: ModelPrediction, **kwargs) -> Dict[
        OnlineCalibration, UncertaintySet]:
        calibrated_uncertainty_sets = {}
        for calibration_scheme in self.calibration_schemes:
            calibrated_prediction_set = calibration_scheme.calibrate(data_sample, model_prediction, **kwargs)
            calibrated_uncertainty_sets[calibration_scheme] = calibrated_prediction_set
        return calibrated_uncertainty_sets

    def update(self, labeled_sample: LabeledDataSample, model_prediction: ModelPrediction,
               calibrated_uncertainty_sets: Dict[OnlineCalibration, UncertaintySet],
               timestamp : int,
               **kwargs):
        for calibration_scheme in self.calibration_schemes:
            calibrated_uncertainty_set = calibrated_uncertainty_sets[calibration_scheme]
            calibration_scheme.update(labeled_sample, model_prediction=model_prediction,
                                      calibrated_uncertainty_set=calibrated_uncertainty_set,
                                      timestamp=timestamp, **kwargs)

    def initialize_uncertainty_set_collections(self, dataset: Dataset, task: Task, device, sample_size: int,
                                               scaler: DataScaler) -> Dict[
        OnlineCalibration, UncertaintySetsCollection]:
        return {calibration_scheme: initialize_uncertainty_set_collection(dataset, task, device, sample_size, scaler)
                for calibration_scheme in self.calibration_schemes
                }