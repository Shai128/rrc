import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from src.data_utils.data_utils import DataType
from src.data_utils.datasets.dataset import Dataset
from src.data_utils.datasets.depth_dataset import DepthDataset
from src.data_utils.datasets.regression_dataset import RegressionDataset
from src.utils import Task


def get_real_dataset(args) -> Dataset:
    if args.task == Task.Regression:
        x, y = get_real_regression_dataset(args.dataset_name, args.data_path)
        return RegressionDataset(x, y, args.backward_size, args.max_data_size, args.offline_train_ratio,
                                 args.test_ratio, args.validation_ratio, DataType.Real, args.dataset_name, args.device)
    else:
        raise Exception("not implemented yet")  # for e.g., classification


def get_dataset(args) -> Dataset:
    if args.data_type == DataType.Real:
        return get_real_dataset(args)
    else:  # TODO: add synthetic dataset
        raise Exception("not implemented yet")


def get_depth_dataset(args) -> DepthDataset:
    return DepthDataset(args.base_data_path, args.annotations_path, args.dataset_name, args.max_data_size,
                        args.offline_train_ratio, args.test_ratio, args.validation_ratio, args.data_type, args.device)


def get_real_regression_dataset(name: str, base_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Load a dataset
    Parameters
    ----------
    name : string, dataset name
    base_path : string, e.g. "path/to/RCOL_datasets/directory/"
    Returns
    -------
    X : features (nXp)
    y : labels (n)
	"""
    if name == 'energy':
        df =  pd.read_csv(os.path.join(base_path, 'energy.csv'))
        y = np.array(df['Appliances'])
        X = df.drop(['Appliances', 'date'], axis=1)
        date = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
        X['day'] = date.dt.day
        X['month'] = date.dt.month
        X['year'] = date.dt.year
        X['hour'] = date.dt.hour
        X['minute'] = date.dt.minute
        X['day_of_week'] = date.dt.dayofweek
        X = np.array(X)

    elif name == 'tetuan_power':
        df = pd.read_csv(os.path.join(base_path, 'tetuan_power.csv'))
        y = np.array(df['Zone 1 Power Consumption'])
        X = df.drop(['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption', 'DateTime'],
                    axis=1)
        date = pd.to_datetime(df['DateTime'].apply(lambda datetime: datetime.replace(' 0:', ' 00:')),
                              format='%m/%d/%Y %H:%M')
        X['day'] = date.dt.day
        X['month'] = date.dt.month
        X['year'] = date.dt.year
        X['hour'] = date.dt.hour
        X['minute'] = date.dt.minute
        X['day_of_week'] = date.dt.dayofweek
        X = np.array(X)

    elif name == 'traffic':
        df =  pd.read_csv(os.path.join(base_path, 'traffic.csv'))
        df['holiday'].replace(df['holiday'].unique(),
                              list(range(len(df['holiday'].unique()))), inplace=True)
        df['weather_description'].replace(df['weather_description'].unique(),
                                          list(range(len(df['weather_description'].unique()))), inplace=True)
        df['weather_main'].replace(['Clear', 'Haze', 'Mist', 'Fog', 'Clouds', 'Smoke', 'Drizzle', 'Rain', 'Squall',
                                    'Thunderstorm', 'Snow'],
                                   list(range(len(df['weather_main'].unique()))), inplace=True)
        y = np.array(df['traffic_volume'])
        X = df.drop(['date_time', 'traffic_volume'], axis=1)
        date = pd.to_datetime(df['date_time'].apply(lambda datetime: datetime.replace(' 0:', ' 00:')),
                              format='%Y-%m-%d %H:%M:%S')
        X['day'] = date.dt.day
        X['month'] = date.dt.month
        X['year'] = date.dt.year
        X['hour'] = date.dt.hour
        # X['minute'] = date.dt.minute
        X['day_of_week'] = date.dt.dayofweek

        X = np.array(X)

    elif name == 'wind':
        df = pd.read_csv(os.path.join(base_path, 'wind_power.csv'))
        date = pd.to_datetime(df['dt'], format='%Y-%m-%d %H:%M:%S')
        X = df.drop(['dt', 'MW'], axis=1)
        y = np.array(df['MW'])[1:]
        X['day'] = date.dt.day
        X['month'] = date.dt.month
        X['year'] = date.dt.year
        X['minute'] = date.dt.minute
        X['hour'] = date.dt.hour
        X['day_of_week'] = date.dt.dayofweek
        X = np.array(X)[:-1]

    elif name == 'prices':
        df = pd.read_csv(os.path.join(base_path, 'Prices_2016_2019_extract.csv'))
        # 15/01/2016  4:00:00
        date = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S')
        X = df.drop(['Date', 'Spot', 'hour'], axis=1)
        y = np.array(df['Spot'])[1:]
        X['day'] = date.dt.day
        X['month'] = date.dt.month
        X['year'] = date.dt.year
        X['hour'] = date.dt.hour
        X['day_of_week'] = date.dt.dayofweek
        X = np.array(X)[:-1]
    else:
        raise Exception("invalid dataset")

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    return X, y