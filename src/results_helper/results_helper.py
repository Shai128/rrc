import os
from typing import List

import pandas as pd

from src.calibration.online_calibration import OnlineCalibration
from src.calibration.parameters_free_online_calibration import ParameterFreeOnlineCalibration
from src.data_utils.data_utils import DataPart
from src.data_utils.datasets.dataset import Dataset
from src.models.abstract_models.online_learning_model import OnlineLearningModel
from src.uncertainty_quantifiers.uncertainty_set import UncertaintySetsCollection
from src.utils import create_folder_if_it_doesnt_exist

data_part_to_save_name_map = {
    DataPart.Train: "train",
    DataPart.Validation: "validation",
    DataPart.Test: "test",
}


def get_results_save_dir(base_results_save_dir: str, dataset_name: str, model_name: str, calibration_name: str,
                         data_part: DataPart) -> str:
    method_name = f"{model_name}_{calibration_name}".replace(" ", "_")
    return os.path.join(base_results_save_dir, data_part_to_save_name_map[data_part], dataset_name, method_name)


class ResultsHelper:
    def __init__(self, base_results_save_dir, seed):
        self.base_results_save_dir = base_results_save_dir
        self.seed = seed

    def save_performance_metrics(self, uncertainty_set_collection: UncertaintySetsCollection, dataset: Dataset,
                                 timestamps: List[int], model: OnlineLearningModel,
                                 calibration_scheme: OnlineCalibration, data_part: DataPart):
        save_dir = get_results_save_dir(self.base_results_save_dir, dataset.dataset_name, model.name, calibration_scheme.name, data_part)
        results = self.compute_performance_metrics(uncertainty_set_collection, dataset, timestamps, model,
                                                   calibration_scheme)
        create_folder_if_it_doesnt_exist(save_dir)
        save_path = os.path.join(save_dir, f"seed={self.seed}.csv")
        pd.DataFrame(results, index=[self.seed]).to_csv(save_path)

    def compute_performance_metrics(self, uncertainty_set_collection: UncertaintySetsCollection, dataset: Dataset,
                                    timestamps: List[int], model: OnlineLearningModel,
                                    calibration_scheme: OnlineCalibration) -> dict:
        if isinstance(calibration_scheme, ParameterFreeOnlineCalibration):
            calibration_scheme_name = calibration_scheme.calibration_scheme.name
        else:
            calibration_scheme_name = calibration_scheme.name
        results = {'set size': len(timestamps), 'calibration_scheme': calibration_scheme_name}
        return results

