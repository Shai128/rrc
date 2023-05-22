import warnings
from sys import platform
from typing import List

import matplotlib
import torch
from tqdm import tqdm

from src.calibration.online_calibration import OnlineCalibration
from src.calibration.parameters_free_online_calibration import ParameterFreeOnlineCalibration
from src.data_utils.data_scaler import DataScaler
from src.data_utils.data_utils import DataType, DataPart
from src.data_utils.datasets.dataset import Dataset
from src.data_utils.datasets.depth_dataset import DepthDataset
from src.models.abstract_models.online_learning_model import OnlineLearningModel
from src.results_helper.results_helper import ResultsHelper
from src.uncertainty_quantifiers.prediction_interval import PredictionIntervalsCollection
from src.uncertainty_quantifiers.prediction_intervals_matrix import PredictionIntervalsMatrixCollection
from src.uncertainty_quantifiers.uncertainty_set import UncertaintySetsCollection
from src.utils import Task, set_seeds

if platform not in ['win32', 'darwin']:
    matplotlib.use('Agg')

warnings.filterwarnings("ignore")


def update_args(args):
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(device_name)
    print(f"device: {device_name}")
    args.data_type = DataType.Real if args.ds_type.lower() == 'real' else DataType.Synthetic
    set_seeds(args.seed)
    return args


def validate_args(args):
    assert args.ds_type.lower() in ['real', 'syn', 'synthetic'], "ds_type must be either 'real' or 'synthetic'"


def add_general_args(parser):
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='risk level')
    parser.add_argument('--bs', type=int, default=256,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0,
                        help='weight decay')
    parser.add_argument('--base_results_save_dir', type=str, default="./results",
                        help="results save dir")
    parser.add_argument('--use_best_hyperparams', action='store_true',
                        help='whether to use the best hyperparameters stored in hyperparameters.json file')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help="fraction of samples used for test")
    parser.add_argument('--validation_ratio', type=float, default=0.2,
                        help="fraction of samples used for validation")
    parser.add_argument('--epochs', type=float, default=60,
                        help="number of epochs for offline training")


def initialize_uncertainty_set_collection(dataset: Dataset, task: Task, device: torch.device, sample_size: int,
                                          scaler: DataScaler) -> UncertaintySetsCollection:
    if task == Task.Regression:
        return PredictionIntervalsCollection(sample_size, scaler, device)
    elif task == Task.Depth:
        assert isinstance(dataset, DepthDataset)
        return PredictionIntervalsMatrixCollection(sample_size, device, scaler, dataset.image_width,
                                                   dataset.image_height)
    else:
        raise Exception("not implemented yet")


def online_inference(model: OnlineLearningModel, calibration_scheme: OnlineCalibration,
                     dataset: Dataset, timestamps: List[int], task: Task, device) -> UncertaintySetsCollection:
    uncertainty_set_collection = initialize_uncertainty_set_collection(dataset, task, device, len(timestamps),
                                                                       dataset.scaler)
    for curr_timestamp in tqdm(timestamps):
        inference_sample = dataset.get_inference_sample(curr_timestamp)
        with torch.no_grad():
            model_prediction = model.predict(inference_sample)
            calibrated_uncertainty_set = calibration_scheme.calibrate(inference_sample, model_prediction)
            uncertainty_set_collection.add_uncertainty_sets(calibrated_uncertainty_set)

            labeled_inference_sample = dataset.get_labeled_inference_sample(curr_timestamp)
            calibration_scheme.update(labeled_inference_sample, model_prediction, calibrated_uncertainty_set,
                                      timestamp=curr_timestamp)

        train_sample = dataset.get_train_sample(curr_timestamp)
        model.online_fit(train_sample, labeled_inference_data_sample=labeled_inference_sample)

    return uncertainty_set_collection


def run_on_data_part(model: OnlineLearningModel, calibration_scheme: OnlineCalibration,
                     dataset: Dataset,
                     train_timestamps: List[int],
                     results_handler: ResultsHelper, task: Task, device, data_part: DataPart):
    print(f"Running on {data_part.name.lower()} set...")
    uncertainty_set_collection = online_inference(model, calibration_scheme, dataset, train_timestamps,
                                                  task, device)
    results_handler.save_performance_metrics(uncertainty_set_collection, dataset, train_timestamps,
                                             model, calibration_scheme, data_part)


def run_on_train(model: OnlineLearningModel, calibration_scheme: OnlineCalibration,
                 dataset: Dataset, results_handler: ResultsHelper, task: Task, device):
    train_timestamps = dataset.get_online_train_timestamps()
    run_on_data_part(model, calibration_scheme, dataset, train_timestamps, results_handler, task, device,
                     DataPart.Train)


def run_on_validation(model: OnlineLearningModel, calibration_scheme: OnlineCalibration,
                      dataset: Dataset, results_handler: ResultsHelper, task: Task, device):
    validation_timestamps = dataset.get_validation_timestamps()
    run_on_data_part(model, calibration_scheme, dataset, validation_timestamps, results_handler, task, device,
                     DataPart.Validation)

    if isinstance(calibration_scheme, ParameterFreeOnlineCalibration):
        calibration_scheme.set_hyperparameters_on_validation_set(validation_timestamps)


def run_on_test(model: OnlineLearningModel, calibration_scheme: OnlineCalibration,
                dataset: Dataset, results_handler: ResultsHelper, task: Task, device):
    test_timestamps = dataset.get_test_timestamps()
    run_on_data_part(model, calibration_scheme, dataset, test_timestamps, results_handler, task, device,
                     DataPart.Test)


def offline_train(model: OnlineLearningModel, dataset: Dataset, training_epochs: int):
    offline_train_timestamps = dataset.get_offline_train_timestamps()
    model.offline_train(dataset, offline_train_timestamps, epochs=training_epochs)


def run_experiment(task: Task, device: torch.device, model, dataset: Dataset, calibration_scheme: OnlineCalibration,
                   results_handler: ResultsHelper, training_epochs: int):
    offline_train(model, dataset, training_epochs)
    run_on_train(model, calibration_scheme, dataset, results_handler, task, device)
    run_on_validation(model, calibration_scheme, dataset, results_handler, task, device)
    run_on_test(model, calibration_scheme, dataset, results_handler, task, device)
