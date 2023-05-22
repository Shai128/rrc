import functools
import os
import random
from collections import OrderedDict
from enum import Enum
from enum import auto

import dill
import numpy as np
import torch
from torch import nn


class Task(Enum):
    Regression = auto(),
    Classification = auto()
    Depth = auto()


def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def has_state(base_path, dataset_name, model_name, epoch):
    path = get_checkpoint_path(base_path, dataset_name, model_name, epoch)
    exists = os.path.exists(path)
    print(f"state in path {path}: does exist: {exists}")
    return exists


def get_checkpoint_path(base_path, dataset_name, model_name, epoch):
    checkpoint_path = os.path.join(get_checkpoint_dir(base_path, dataset_name), f"{model_name}_epoch={epoch}.pth")
    return checkpoint_path


def get_checkpoint_dir(base_path, dataset_name):
    checkpoint_dir = os.path.join(base_path, 'saved_models', dataset_name)
    return checkpoint_dir


def save_ckpt(base_path, dataset_name, model_name, epoch, model, optimizer, scheduler):
    """Save checkpoint"""
    create_folder_if_it_doesnt_exist(get_checkpoint_dir(base_path, dataset_name))
    checkpoint_path = get_checkpoint_path(base_path, dataset_name, model_name, epoch)
    torch.save({
        'epoch': epoch,
        'scheduler': scheduler.state_dict(),
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()},
        checkpoint_path, pickle_module=dill)


def load_ckpt(base_path, dataset_name, model_name, model, epoch, optimizer, scheduler):
    """
    Load checkpoint.
    """
    checkpoint_path = get_checkpoint_path(base_path, dataset_name, model_name, epoch)

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, pickle_module=dill)
        model_state_dict_keys = model.state_dict().keys()
        checkpoint_state_dict_noprefix = strip_prefix_if_present(checkpoint['model_state_dict'], "module.")

        if all(key.startswith('module.') for key in model_state_dict_keys):
            model.module.load_state_dict(checkpoint_state_dict_noprefix)
        else:
            model.load_state_dict(checkpoint_state_dict_noprefix)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scheduler.__setattr__('last_epoch', checkpoint['step'])
        del checkpoint
        torch.cuda.empty_cache()


def create_folder_if_it_doesnt_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_current_seed():
    return np.random.randint(0, 2 ** 31)


def pearsons_corr(x, y):
    """
    computes the correlation between to samples of empirical samples
    Parameters
    ----------
    x - a vector if n samples drawn from X
    y - a vector if n samples drawn from Y
    Returns
    -------
    The empirical correlation between X and Y
    """
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))


def interp1d_func(x, x0, x1, y0, y1):
    return y0 + (x - x0) * ((y1 - y0) / (x1 - x0))


def batch_interp1d(x: torch.Tensor, y: torch.Tensor, a: float = None, b: float = None):
    if a is None or b is None:
        fill_value = 'extrapolate'
    else:
        fill_value = (a, b)

    def interp(desired_x):
        # desired_x = np.random.rand(3, 100) * 30 - 5
        desired_x = desired_x.to(x.device)
        if len(desired_x.shape) != 2 or desired_x.shape[0] != x.shape[0]:
            raise Exception(f"the shape of the input vector should be ({x.shape[0]},m), but got {desired_x.shape}")
        desired_x, _ = desired_x.sort()
        desired_x_rep = desired_x.unsqueeze(-1).repeat(1, 1, x.shape[-1] - 1)
        x_rep = x.unsqueeze(1).repeat(1, desired_x.shape[1], 1)
        relevant_idx = torch.stack(
            ((x_rep[:, :, :-1] <= desired_x_rep) & (desired_x_rep <= x_rep[:, :, 1:])).nonzero(as_tuple=True))

        x0 = x[relevant_idx[0], relevant_idx[2]]
        y0 = y[relevant_idx[0], relevant_idx[2]]
        x1 = x[relevant_idx[0], relevant_idx[2] + 1]
        y1 = y[relevant_idx[0], relevant_idx[2] + 1]
        desired_x_in_interpolation_range = desired_x[relevant_idx[0], relevant_idx[1]]
        res = torch.zeros_like(desired_x)
        res[relevant_idx[0], relevant_idx[1]] = interp1d_func(desired_x_in_interpolation_range, x0, x1, y0, y1)
        if fill_value == 'extrapolate':
            idx = (desired_x < x[:, 0, None]).nonzero(as_tuple=True)
            x0, x1 = x[idx[0], 0], x[idx[0], 1]
            y0, y1 = y[idx[0], 0], y[idx[0], 1]
            res[idx[0], idx[1]] = interp1d_func(desired_x[idx[0], idx[1]], x0, x1, y0, y1)

            idx = (desired_x > x[:, -1, None]).nonzero(as_tuple=True)
            x0, x1 = x[idx[0], -1], x[idx[0], -2]
            y0, y1 = y[idx[0], -1], y[idx[0], -2]
            res[idx[0], idx[1]] = interp1d_func(desired_x[idx[0], idx[1]], x0, x1, y0, y1)

        else:
            a, b = fill_value
            res[desired_x < x[:, 0, None]] = a
            res[desired_x > x[:, -1, None]] = b
        return res

    return interp


def batch_estim_dist(quantiles: torch.Tensor, percentiles: torch.Tensor, y_min, y_max, smooth_tails, tau,
                     extrapolate_quantiles=False):
    """ Estimate CDF from list of quantiles, with smoothing """
    device = quantiles.device
    noise = torch.rand_like(quantiles) * 1e-8
    noise_monotone, _ = torch.sort(noise)
    quantiles = quantiles + noise_monotone
    assert len(percentiles.shape) == 1 and len(quantiles.shape) == 2 and quantiles.shape[1] == percentiles.shape[0]
    percentiles = percentiles.unsqueeze(0).repeat(quantiles.shape[0], 1)

    # Smooth tails
    cdf = batch_interp1d(quantiles, percentiles, 0.0, 1.0)
    if extrapolate_quantiles:
        inv_cdf = batch_interp1d(percentiles, quantiles)
        return cdf, inv_cdf
    inv_cdf = batch_interp1d(percentiles, quantiles, y_min, y_max)

    if smooth_tails:
        # Uniform smoothing of tails
        quantiles_smooth = quantiles
        tau_lo = torch.ones(quantiles.shape[0], 1, device=device) * tau
        tau_hi = torch.ones(quantiles.shape[0], 1, device=device) * (1 - tau)
        q_lo = inv_cdf(tau_lo)
        q_hi = inv_cdf(tau_hi)
        idx_lo = torch.where(percentiles < tau_lo)[0]
        idx_hi = torch.where(percentiles > tau_hi)[0]
        if len(idx_lo) > 0:
            quantiles_smooth[idx_lo] = torch.linspace(quantiles[0], q_lo, steps=len(idx_lo), device=device)
        if len(idx_hi) > 0:
            quantiles_smooth[idx_hi] = torch.linspace(q_hi, quantiles[-1], steps=len(idx_hi), device=device)

        cdf = batch_interp1d(quantiles_smooth, percentiles, 0.0, 1.0)

    # Standardize
    breaks = torch.linspace(y_min, y_max, steps=1000, device=device).unsqueeze(0).repeat(quantiles.shape[0], 1)
    cdf_hat = cdf(breaks)
    f_hat = torch.diff(cdf_hat)
    f_hat = (f_hat + 1e-10) / (torch.sum(f_hat + 1e-10, dim=-1)).reshape((f_hat.shape[0], 1))
    cumsum = torch.cumsum(f_hat, dim=-1)
    cdf_hat = torch.cat([torch.zeros_like(cumsum)[:, 0:1], cumsum], dim=-1)
    cdf = batch_interp1d(breaks, cdf_hat, 0.0, 1.0)
    inv_cdf = batch_interp1d(cdf_hat, breaks, y_min, y_max)

    return cdf, inv_cdf


