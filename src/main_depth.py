import argparse
import os.path
import warnings
from sys import platform

import matplotlib

from src.calibration.rolling_risk_control import RollingRiskControlWithMultipleRisks
from src.calibration.set_constructing_functions.set_constructing_function import \
    PredictionIntervalMatrixConstructingFunctionWithMeanAndStd, MaxAggregation
from src.calibration.stretching_functions.stretching_functions import ExponentialStretching
from src.data_utils.get_dataset_utils import get_depth_dataset
from src.losses import ImageMiscoverageLoss, PoorCenterCoverageLoss
from src.main_helper import update_args, validate_args, add_general_args, run_experiment
from src.models.image_to_image_models.depth_models.depth_uq_model import DepthUQModel
from src.results_helper.depth_results_helper import DepthResultsHelper
from src.utils import Task

if platform not in ['win32', 'darwin']:
    matplotlib.use('Agg')

warnings.filterwarnings("ignore")


def parse_args_utils(args):
    validate_args(args)
    update_args(args)
    if args.backbone not in ['res50', 'res101']:
        raise Exception("backbone must be either 'res50' or 'res101'")

    if args.std_method not in ['baseline', 'residual_magnitude',
                              'previous_residual_with_alignment']:
        raise Exception("std_method must be one of: 'baseline', 'residual_magnitude', 'pixelwise_qr'")
    args.task = Task.Depth
    return args


def parse_args():
    parser = argparse.ArgumentParser()
    add_general_args(parser)

    parser.add_argument('--std_method', type=str, default="baseline",
                        help="")
    parser.add_argument('--backbone', type=str, default="res101",
                        help="")
    parser.add_argument('--trained_model_path', type=str, default="../saved_models/Leres",
                        help="")
    parser.add_argument('--calibration_set_size', type=int, default=0,
                        help='')
    parser.add_argument('--base_data_path', type=str, default="datasets/real_data/depths",
                        help='')
    parser.add_argument('--base_save_path', type=str, default="../../",
                        help='base path for saving models')
    parser.add_argument('--annotations_path', type=str, default="annotations/train_annotations_onlyvideos.json",
                        help='')
    parser.add_argument('--ds_type', type=str, default="REAL",
                        help='type of data set. real or synthetic. REAL for real. SYN for synthetic')

    parser.add_argument('--dataset_name', type=str, default='KITTI',
                        help='')
    parser.add_argument('--max_data_size', type=int, default=10000,
                        help='')
    parser.add_argument('--offline_train_ratio', type=float, default=0.6,
                        help='')
    parser.add_argument('--training_epochs', type=int, default=60,
                        help='')
    args = parser.parse_args()

    args = parse_args_utils(args)

    return args


def get_model(args):
    trained_model_path = os.path.join(args.trained_model_path, f"{args.backbone}.ph")
    model = DepthUQModel(trained_model_path, args.backbone, args.std_method, args.device, base_save_path=args.base_save_path)
    return model


def get_calibration_scheme():
    set_constructing_function = PredictionIntervalMatrixConstructingFunctionWithMeanAndStd(ExponentialStretching(),
                                                                                           MaxAggregation())
    return RollingRiskControlWithMultipleRisks(gammas=[0.05, 0.01],
                                               alphas=[0.2, 0.1],
                                               set_constructing_function=set_constructing_function,
                                               losses=[ImageMiscoverageLoss(),
                                                       PoorCenterCoverageLoss(0.6)],
                                               )


def main():
    args = parse_args()
    dataset = get_depth_dataset(args)
    model = get_model(args)
    calibration_scheme = get_calibration_scheme()
    results_handler = DepthResultsHelper(args.base_results_save_dir, args.seed)
    run_experiment(args.task, args.device, model, dataset, calibration_scheme, results_handler, args.epochs)


if __name__ == '__main__':
    main()
