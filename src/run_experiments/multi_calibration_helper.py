from typing import List

import torch
from tqdm import tqdm

from src.calibration.calibration_schemes import CalibrationSchemes
from src.calibration.parameters_free_online_calibration import ParameterFreeOnlineCalibration
from src.data_utils.data_utils import DataPart
from src.data_utils.datasets.dataset import Dataset
from src.models.abstract_models.online_learning_model import OnlineLearningModel
from src.results_helper.results_helper import ResultsHelper
from src.utils import Task


def multiple_calibrations_online_inference(model: OnlineLearningModel, calibration_schemes: CalibrationSchemes,
                                           dataset: Dataset,
                                           timestamps: List[int],
                                           data_part: DataPart,
                                           results_handler: ResultsHelper, task: Task, device):
    uncertainty_set_collections = calibration_schemes.initialize_uncertainty_set_collections(dataset, task, device,
                                                                                             len(timestamps),
                                                                                             dataset.scaler)
    for curr_idx in tqdm(timestamps):
        inference_sample = dataset.get_inference_sample(curr_idx)
        with torch.no_grad():
            model_prediction = model.predict(inference_sample)
            calibrated_uncertainty_sets = calibration_schemes.calibrate(inference_sample, model_prediction)

            for calibration_scheme, calibrated_uncertainty_set in calibrated_uncertainty_sets.items():
                uncertainty_set_collections[calibration_scheme].add_uncertainty_sets(calibrated_uncertainty_set)

            labeled_inference_sample = dataset.get_labeled_inference_sample(curr_idx)
            calibration_schemes.update(labeled_inference_sample, model_prediction, calibrated_uncertainty_sets,
                                       timestamp=curr_idx)

        train_sample = dataset.get_train_sample(curr_idx)
        model.online_fit(train_sample, labeled_inference_data_sample=labeled_inference_sample)

    for calibration_scheme, uncertainty_set_collection in uncertainty_set_collections.items():
        results_handler.save_performance_metrics(uncertainty_set_collection, dataset, timestamps,
                                                 model, calibration_scheme, data_part)


def run_on_data_part(model: OnlineLearningModel, calibration_schemes: CalibrationSchemes,
                     dataset: Dataset,
                     train_timestamps: List[int],
                     results_handler: ResultsHelper, task: Task, device, data_part: DataPart):
    print(f"Running on {data_part.name.lower()} set...")
    multiple_calibrations_online_inference(model, calibration_schemes, dataset, train_timestamps, data_part,
                                           results_handler, task, device)


def run_on_train(model: OnlineLearningModel, calibration_schemes: CalibrationSchemes,
                 dataset: Dataset, results_handler: ResultsHelper, task: Task, device):
    train_timestamps = dataset.get_online_train_timestamps()
    run_on_data_part(model, calibration_schemes, dataset, train_timestamps, results_handler, task, device,
                     DataPart.Train)


def run_on_validation(model: OnlineLearningModel, calibration_schemes: CalibrationSchemes,
                      dataset: Dataset, results_handler: ResultsHelper, task: Task, device):
    validation_timestamps = dataset.get_validation_timestamps()
    run_on_data_part(model, calibration_schemes, dataset, validation_timestamps, results_handler, task, device,
                     DataPart.Validation)

    for calibration_scheme in calibration_schemes.calibration_schemes:
        if isinstance(calibration_scheme, ParameterFreeOnlineCalibration):
            calibration_scheme.set_hyperparameters_on_validation_set(validation_timestamps)


def run_on_test(model: OnlineLearningModel, calibration_schemes: CalibrationSchemes,
                dataset: Dataset, results_handler: ResultsHelper, task: Task, device):
    test_timestamps = dataset.get_test_timestamps()
    run_on_data_part(model, calibration_schemes, dataset, test_timestamps, results_handler, task, device,
                     DataPart.Test)


def offline_train(model: OnlineLearningModel, dataset: Dataset, training_epochs: int):
    offline_train_timestamps = dataset.get_offline_train_timestamps()
    model.offline_train(dataset, offline_train_timestamps, epochs=training_epochs)


def run_experiments(task: Task, device: torch.device, model, dataset: Dataset, calibration_schemes: CalibrationSchemes,
                   results_handler: ResultsHelper, training_epochs: int):
    offline_train(model, dataset, training_epochs)
    run_on_train(model, calibration_schemes, dataset, results_handler, task, device)
    run_on_validation(model, calibration_schemes, dataset, results_handler, task, device)
    run_on_test(model, calibration_schemes, dataset, results_handler, task, device)
