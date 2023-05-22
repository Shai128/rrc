import numpy as np
import torch

from src import utils


def get_avg_miscoverage_streak_len(y: torch.Tensor, upper_q: torch.Tensor, lower_q: torch.Tensor) -> float:
    miscoverages = (y > upper_q) | (y < lower_q)
    streaks_lengths = []
    curr_streak_length = 0
    for i in range(len(miscoverages)):
        if miscoverages[i]:
            curr_streak_length += 1
        elif curr_streak_length > 0:
            streaks_lengths += [curr_streak_length]
            curr_streak_length = 0

    return np.mean(streaks_lengths).item()


def get_avg_miscoverage_counter(y: torch.Tensor, upper_q: torch.Tensor, lower_q: torch.Tensor) -> float:
    miscoverages = ((y > upper_q) | (y < lower_q)).int()
    streaks_lengths = torch.zeros_like(miscoverages)
    streaks_lengths[-1] = miscoverages[-1]

    for t in reversed(range(miscoverages.shape[0] - 1)):
        streaks_lengths[t] = (streaks_lengths[t + 1] + miscoverages[t]) * miscoverages[t]

    res = torch.mean(streaks_lengths.float(), dim=-1)
    if torch.numel(res) == 1:
        return res.item()
    else:
        return res.mean().item()


def corr(cov_identifier: torch.Tensor, average_len: torch.Tensor):
    return utils.pearsons_corr(cov_identifier.float(), average_len.float()).item()


def calculate_pearson_corr(y : torch.Tensor, upper_q: torch.Tensor, lower_q: torch.Tensor):
    cov_identifier = ((y <= upper_q) & (y >= lower_q)).float()
    average_len = upper_q - lower_q
    return corr(cov_identifier, average_len)


def calculate_coverage(y: torch.Tensor, upper_q: torch.Tensor, lower_q: torch.Tensor):
    return ((y <= upper_q) & (y >= lower_q)).float().mean().item() * 100


def calculate_average_length(upper_q: torch.Tensor, lower_q: torch.Tensor):
    return (upper_q - lower_q).mean().item()


def calculate_median_length(upper_q: torch.Tensor, lower_q: torch.Tensor):
    return (upper_q - lower_q).median().item()


# def hsic(cov_identifier: torch.Tensor, average_len: torch.Tensor):
#     return src.HSIC(cov_identifier.float().unsqueeze(-1), average_len.float().unsqueeze(-1)).item()


# def calculate_hsic(y, upper_q, lower_q):
#     cov_identifier = ((y <= upper_q) & (y >= lower_q)).float()
#     average_len = (upper_q - lower_q)
#     return hsic(cov_identifier, average_len)
