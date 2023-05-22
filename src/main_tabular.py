import argparse
import ast
import json
import warnings
from sys import platform

import matplotlib

from src.calibration.calibrations_factory import RollingRiskControlFactory
from src.calibration.parameters_free_online_calibration import ParameterFreeOnlineCalibration
from src.calibration.set_constructing_functions.set_constructing_function_factory import \
    PredictionIntervalConstructingFunctionWithCQRFactory
from src.calibration.stretching_functions.stretching_functions_factory import ExponentialStretchingFactory
from src.data_utils.data_utils import DataType
from src.data_utils.datasets.dataset import Dataset
from src.data_utils.get_dataset_utils import get_dataset
from src.losses import MiscoverageLoss, PinballLoss
from src.main_helper import update_args, validate_args, add_general_args, run_experiment
from src.models.online_qr import OnlineQR
from src.results_helper.regression_results_helper import RegressionResultsHelper, SyntheticDataRegressionResultsHelper, \
    RealDataRegressionResultsHelper
from src.utils import Task

if platform not in ['win32', 'darwin']:
    matplotlib.use('Agg')

warnings.filterwarnings("ignore")

def get_best_hyperparams(train_all_q, dataset_name, hyperparameters_path="hyperparameters.json"):
    hyperparameters_json = json.loads(open(hyperparameters_path, 'r').read())
    HYPERPARAMETERS_TO_USE = hyperparameters_json['hyperparameters']

    hp = HYPERPARAMETERS_TO_USE[f"train_all_q={int(train_all_q)}"][f"cal_split=0"][dataset_name]
    if type(hp['lstm_in_layers']) == str:
        hp['lstm_in_layers'] = ast.literal_eval(hp['lstm_in_layers'])
    if type(hp['lstm_out_layers']) == str:
        hp['lstm_out_layers'] = ast.literal_eval(hp['lstm_out_layers'])

    return hp

def get_best_hyperparams_from_args(args):
    try:
        hp = get_best_hyperparams(args.train_all_q, args.dataset_name)
        args = argparse.Namespace(**{**vars(args), **hp})
    except Exception as e:
        print(f"warning: didn't find best hyperparameters for dataset: {args.dataset_name} because {e}. using the default ones instead")

    return args


def parse_args_utils(args):
    validate_args(args)
    update_args(args)
    args.lstm_out_layers = ast.literal_eval(args.lstm_out_layers)
    args.lstm_in_layers = ast.literal_eval(args.lstm_in_layers)
    if args.backward_size == 0:
        args.lstm_hidden_size = args.lstm_in_layers[-1]
        args.lstm_layers = 0
    args = get_best_hyperparams_from_args(args)
    return args


def parse_args():
    parser = argparse.ArgumentParser()
    add_general_args(parser)
    parser.add_argument('--ds_type', type=str, default="Real",
                        help='type of data set. real or synthetic. REAL for real. SYN for synthetic')

    parser.add_argument('--dataset_name', type=str, default='wind',
                        help='dataset to use')
    parser.add_argument('--backward_size', type=int, default=3,
                        help='number of samples')

    parser.add_argument('--lstm_layers', type=int, default=1,
                        help='')

    parser.add_argument('--lstm_hidden_size', type=int, default=128,
                        help='')

    parser.add_argument('--lstm_in_layers', type=str, default="[32, 64]",
                        help='hidden dimensions')

    parser.add_argument('--lstm_out_layers', type=str, default="[64, 32]",
                        help='hidden dimensions')

    parser.add_argument('--non_linearity', type=str, default="lrelu",
                        help='')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='')

    parser.add_argument('--calibration_set_size', type=int, default=300,  # for ACI only
                        help="the size of the calibration set used by ACI algorithm")

    parser.add_argument('--data_path', type=str, default='datasets/real_data',
                        help='')

    parser.add_argument('--train_all_q', action='store_true',
                        help="whether to train the model to learn quantiles at all levels")
    parser.add_argument('--max_data_size', type=int, default=20000,
                        help='')
    parser.add_argument('--offline_train_ratio', type=float, default=0.,
                        help='')
    parser.add_argument('--training_epochs', type=int, default=100,
                        help='')
    parser.set_defaults(train_all_q=False)

    args = parser.parse_args()
    args = parse_args_utils(args)
    args.task = Task.Regression
    return args


def get_calibration_scheme(args):
    alpha = args.alpha
    loss = MiscoverageLoss()
    stretching_factory = ExponentialStretchingFactory()
    set_constructing_factory = PredictionIntervalConstructingFunctionWithCQRFactory(stretching_factory)
    rrc_factory = RollingRiskControlFactory(alpha, set_constructing_factory, loss)
    objective_loss = PinballLoss(alpha)
    return ParameterFreeOnlineCalibration(objective_loss, rrc_factory)


def get_model(args, dataset: Dataset):
    y_dim = dataset.y_dim
    x_dim = dataset.x_dim
    model = OnlineQR(x_dim, y_dim,
                     lstm_hidden_size=args.lstm_hidden_size,
                     lstm_layers=args.lstm_layers, lstm_in_layers=args.lstm_in_layers,
                     lstm_out_layers=args.lstm_out_layers, dropout=args.dropout,
                     lr=args.lr, wd=args.wd, device=args.device, non_linearity=args.non_linearity,
                     backward_size=args.backward_size, train_all_q=args.train_all_q,
                     alpha=args.alpha, batch_size=args.bs)
    return model


def get_results_handler(data_type: DataType, base_results_save_dir: str, seed: int) -> RegressionResultsHelper:
    if data_type == DataType.Real:
        return RealDataRegressionResultsHelper(base_results_save_dir, seed)
    else:
        return SyntheticDataRegressionResultsHelper(base_results_save_dir, seed)


def main():
    args = parse_args()
    dataset = get_dataset(args)
    model = get_model(args, dataset)
    calibration_scheme = get_calibration_scheme(args)
    results_handler = get_results_handler(dataset.data_type, args.base_results_save_dir, args.seed)
    run_experiment(args.task, args.device, model, dataset, calibration_scheme, results_handler, args.epochs)


if __name__ == '__main__':
    main()
