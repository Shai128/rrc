import copy
import itertools
import os
import traceback
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import re
import seaborn as sns

from src.utils import create_folder_if_it_doesnt_exist

metric_name_to_metric_display_name = {
    'average miscoverage streak length': 'MSL',
    'days avg. Δ-coverage': "ΔCoverage",
    "average length": "Average length (scaled)",
    "average miscoverage counter": 'Miscoverage counter',
    'coverage': "Coverage",
     'image coverage': "Image coverage",
     'image average length': "Average length",
     'center coverage': "Center coverage",
     'center average length': "Center length",
     'center poor coverage occurrences (60.0 %)': "Center failure rate"
}

calibration_name_to_display_name = {
    'parameter_free_aci': "ACI",
    'parameter_free_rrc_f=cqr_id_loss=miscoverage': "RRC",
    'parameter_free_rrc_f=cqr_exp_loss=miscoverage': "RRC (exp)",
    'parameter_free_rrc_f=cqr_error_adaptive_loss=miscoverage': "RRC (Error)",
    'parameter_free_rrc_f=cqr_id_loss=miscoverage_counter': "RRC",
    'parameter_free_rrc_f=cqr_exp_loss=miscoverage_counter': "RRC (exp)",
    'parameter_free_rrc_f=cqr_error_adaptive_loss=miscoverage_counter': "RRC (Error)",
    "parameter_free_multi_rrc_f=cqr_exp_losses=['image_miscoverage']": 'RRC',
    "parameter_free_multi_rrc_f=cqr_exp_losses=['image_miscoverage',_'poor_center_coverage']": 'RRC (multi)',
    "uncalibrated": "Naive"
}

depth_model_name_to_display_name = {
    'depth_bb=res101_std=baseline': 'const.',
    'depth_bb=res101_std=previous_residual_with_alignment': 'flow prev. resid.',
    'depth_bb=res101_std=residual_magnitude': 'resid.',

}

results_base_path = os.path.join("../results", "test")
length_metric_display_name = "Average length (scaled)"
plots_save_path = 'plots/'
create_folder_if_it_doesnt_exist(plots_save_path)
color_palette = {"RRC (Error)": 'tab:blue', "RRC (exp)": "tab:purple", "RRC": 'tab:orange', "ACI": 'tab:red', "Naive": "tab:green", "RRC (multi)": 'tab:blue'}

def read_method_results(base_path: str, dataset_name: str, calibration_scheme_name: str, seeds=20, apply_mean=True,
                        display_errors=False, model_name: str = None):
    path = os.path.join(base_path, dataset_name)
    relevant_dirs = os.listdir(path)
    if model_name is not None:
        relevant_dirs = list(filter(lambda x: model_name in x, os.listdir(path)))
    calibration_folder = list(filter(lambda x: x.endswith(calibration_scheme_name), relevant_dirs))[0]
    full_folder_path = os.path.join(base_path, dataset_name, calibration_folder)
    df = read_method_results_aux(full_folder_path, seeds, apply_mean, display_errors=display_errors)
    return df


def read_method_results_aux(folder_path, seeds=20, apply_mean=True, display_errors=False):
    df = pd.DataFrame()

    for seed in range(seeds):
        save_path = f"{folder_path}/seed={seed}.csv"
        try:
            seed_df = pd.read_csv(save_path).drop(['Unnamed: 0'], axis=1, errors='ignore')

            if 'coverage' in seed_df and abs(
                    seed_df['coverage'].item() - 0) < 0.01:
                # print(f"{folder_path}/seed={seed}.csv has 0 coverage")
                if np.isnan(seed_df['average length']).any():
                    print(
                        f"{folder_path}/seed={seed}.csv has invalid average length. the value is: {seed_df['average length'].item()}")
                    display(seed_df)
                    # print("got here")
                    continue
            if '(miscoverage streak) average length' in df.columns and \
                    np.isnan(seed_df['(miscoverage streak) average length']).any():
                print(
                    f"{folder_path}/seed={seed}.csv has invalid (miscoverage streak) average length")
                print("the value is: ", seed_df['(miscoverage streak) average length'].item())
                display(seed_df)

            df = pd.concat([df, seed_df], axis=0)
        except Exception as e:
            # print("got an exception")
            if display_errors:
                print(e)
    if len(df) == 0:
        # print(f"{folder_path} had 0 an error")
        save_path = f"{folder_path}/seed=0.csv"
        pd.read_csv(save_path).drop(['Unnamed: 0'], axis=1, errors='ignore')  # raises an exception
        raise Exception(f"could not find results in path {folder_path}")

    if apply_mean:
        df = df.apply(np.mean).to_frame().T

    return df


def get_comparison_df(seeds, dataset_names, calibration_schemes, display_errors=False):
    total_df = pd.DataFrame()
    for dataset_name in dataset_names:
        dataset_display_name = dataset_name.replace("_", " ").replace("tetuan ", "")
        for calibration_method in calibration_schemes:
            try:
                df = read_method_results(results_base_path, dataset_name, calibration_method.name, seeds,
                                         display_errors=display_errors, apply_mean=False)

                df['Method'] = calibration_name_to_display_name[calibration_method.name]
                df['Dataset'] = dataset_display_name
                df['seed'] = range(len(df))
                total_df = total_df.append(df)
            except Exception as e:
                if display_errors:
                    print("got an error: ", e)
                    traceback.print_exc()

            if len(total_df) == 0:
                continue
        if len(total_df) == 0:
            print("no results")
            return total_df
        length_metric_name = 'average length'
        curr_data_idx = total_df['Dataset'] == dataset_display_name
        mean_length = total_df[curr_data_idx][length_metric_name].mean()
        tmp_df = total_df.loc[curr_data_idx]
        tmp_df[length_metric_name] /= mean_length
        total_df.loc[curr_data_idx] = tmp_df
    return total_df


def plot_metrics(total_df, metrics, desired_coverage_level, save_dir):
    if len(total_df) == 0:
        print("empty df")
        return
    save_dir = os.path.join(plots_save_path, save_dir)
    for metric in metrics:
        print("metric: ", metric)
        metric_display_name = metric_name_to_metric_display_name[metric]
        total_df[metric_display_name] = total_df[metric]
        total_df.index = range(len(total_df))
        plt.figure(figsize=(5, 5))
        graph = sns.boxplot(y='Dataset', x=metric_display_name, hue='Method',
                            data=total_df, linewidth=2.5,
                            palette=color_palette,
                            width=0.7)
        axes = [graph.axes]

        for ax in axes:
            if metric != metrics[0]:
                ax.get_legend().remove()
            else:
                legend = ax.get_legend()
                legend.get_frame().set_alpha(None)
                legend.get_frame().set_facecolor((0, 0, 0, 0.05))
        if metric == 'coverage':
            for ax in axes:
                ax.axvline(desired_coverage_level, ls='--')
                ticks = list(filter(lambda x: 0 <= x <= 100, ax.get_xticks()))
                ticks = list(filter(lambda x: x < 90, ticks)) + [90] + list(filter(lambda x: x > 90, ticks))
                ax.set_xticks(ticks)
                if len(total_df[total_df['Method'] == 'Naive']) > 0:
                    ax.set_xlim(70, 92)
                elif total_df['coverage'].min() > 88.5 and total_df['coverage'].max() < 90.5:
                    ax.set_xlim(88.5, 90.5)
        elif metric == 'average miscoverage counter':
            if total_df[metric].min() > 0.107 and total_df[metric].max() < 0.2:
                for ax in axes:
                    ax.set_xlim(0.107, 0.2)

        if metric == 'average miscoverage streak length':
            for ax in axes:
                ax.axvline(100 / desired_coverage_level, ls='--')
        if metric == "average miscoverage counter":
            alpha = 100 - desired_coverage_level
            mc_alpha = alpha / (100 - alpha)
            print("mc_alpha: ", mc_alpha)
            for ax in axes:
                ax.axvline(mc_alpha, ls='--')
        metric_save_name = metric.replace(" ", "_").replace("Δ", "Delta")
        save_path = os.path.join(save_dir, f"{metric_save_name}.png")
        create_folder_if_it_doesnt_exist(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def horizontal_plot_metrics(total_df, metrics, desired_coverage_level, save_dir):
    if len(total_df) == 0:
        print("empty df")
        return
    save_dir = os.path.join(plots_save_path, save_dir)
    for metric in metrics:
        print("metric: ", metric)
        metric_display_name = metric_name_to_metric_display_name[metric]
        total_df[metric_display_name] = total_df[metric]
        total_df.index = range(len(total_df))
        plt.figure(figsize=(8, 4))
        graph = sns.boxplot(x='Dataset', y=metric_display_name, hue='Method',
                            data=total_df, linewidth=2.5,
                            palette=color_palette,
                            width=0.7)
        axes = [graph.axes]

        for ax in axes:
            if metric != metrics[0]:
                ax.get_legend().remove()
            else:
                legend = ax.get_legend()
                legend.get_frame().set_alpha(None)
                legend.get_frame().set_facecolor((0, 0, 0, 0.05))
        if metric == 'coverage':
            for ax in axes:
                ax.axhline(desired_coverage_level, ls='--')
                ticks = list(filter(lambda x: 0 <= x <= 100, ax.get_yticks()))
                ticks = list(filter(lambda x: x < 90, ticks)) + [90] + list(filter(lambda x: x > 90, ticks))
                ax.set_yticks(ticks)
                if len(total_df[total_df['Method'] == 'Naive']) > 0:
                    ax.set_ylim(70, 92)
                elif total_df['coverage'].min() > 88.5 and total_df['coverage'].max() < 90.5:
                    ax.set_ylim(88.5, 90.5)
        elif metric == 'average miscoverage counter':
            if total_df[metric].min() > 0.107 and total_df[metric].max() < 0.2:
                for ax in axes:
                    ax.set_ylim(0.107, 0.2)

        if metric == 'average miscoverage streak length':
            for ax in axes:
                ax.axhline(100 / desired_coverage_level, ls='--')
        if metric == "average miscoverage counter":
            alpha = 100 - desired_coverage_level
            mc_alpha = alpha / (100 - alpha)
            print("mc_alpha: ", mc_alpha)
            for ax in axes:
                ax.axhline(mc_alpha, ls='--')
        metric_save_name = metric.replace(" ", "_").replace("Δ", "Delta")
        save_path = os.path.join(save_dir, f"{metric_save_name}.png")
        create_folder_if_it_doesnt_exist(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def depth_data_plot(seeds, desired_coverage_level,
                    full_dataset_name,
                    calibration_methods,
                    model_names,
                    metrics,
                    save_dir,
                    display_errors=False,

                    **kwargs):
    save_dir = os.path.join(plots_save_path, save_dir)
    create_folder_if_it_doesnt_exist(save_dir)
    for metric in metrics:
        total_df = pd.DataFrame()
        metric_display_name = metric_name_to_metric_display_name[metric]
        print("metric: ", metric_display_name)
        for calibration_method in calibration_methods:
            calibration_method_display_name = calibration_name_to_display_name[calibration_method]
            for model_name in model_names:
                try:
                    df = read_method_results(results_base_path, full_dataset_name,
                                             calibration_scheme_name=calibration_method,
                                             seeds=seeds, apply_mean=False, display_errors=display_errors,
                                             model_name=model_name)
                    df = df.rename({col: col
                                   .replace("Estimated quantiles ", "")
                                   .replace("days ", "")
                                    for col in df.columns}, axis=1)

                    df = df[metric].to_frame()
                    df = pd.DataFrame(data={'seed': range(len(df)), metric_display_name: df[metric]})
                    df['Uncertainty Quantification Heuristic'] = depth_model_name_to_display_name[model_name]
                    df['Method'] = calibration_method_display_name

                    total_df = total_df.append(df)
                except Exception as e:
                    if display_errors:
                        print(f"model_name: {model_name} calibration_method: {calibration_method}, got an error: {e}")
                        traceback.print_exc()

        if len(total_df) > 0:
            total_df.index = range(len(total_df))
            params = {'palette': color_palette, 'data':total_df, 'linewidth': 2.5, 'width': 0.7}
            if len(model_names) == 1:
                plt.figure(figsize=(5, 4))
                graph = sns.boxplot(x='Method', y=metric_display_name, **params)
            elif len(calibration_methods) == 1:
                plt.figure(figsize=(6, 14))
                graph = sns.boxplot(x='Uncertainty Quantification Heuristic', y=metric_display_name, **params)
            else:
                plt.figure(figsize=(8, 4))
                graph = sns.boxplot(x='Uncertainty Quantification Heuristic', y=metric_display_name, hue='Method', **params)
            axes = [graph.axes]

            for ax in axes:
                if ax.get_legend() is not None:
                    if metric != metrics[0]:
                        ax.get_legend().remove()
                    else:
                        legend = ax.get_legend()
                        legend.get_frame().set_alpha(None)
                        legend.get_frame().set_facecolor((0, 0, 0, 0.1))
            if metric == 'image coverage':
                for ax in axes:
                    ax.axhline(desired_coverage_level, ls='--')
            if metric == 'center poor coverage occurrences (60.0 %)':
                if len(total_df[total_df['Method'] == 'RRC (multi)']) > 0:
                    for ax in axes:
                        ax.axhline(10, ls='--')

            metric_save_name = metric.replace(" ", "_").replace("%", "")
            save_path = os.path.join(save_dir, f"{metric_save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
