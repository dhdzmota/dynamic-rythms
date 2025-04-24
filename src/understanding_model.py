import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
from sklearn.utils import resample

import src.utils as utils

from src.model_metrics import (
    compute_brierscoreloss,
    compute_aucroc,
    compute_aucpr,
    bootstrap_func,
)

warnings.filterwarnings('ignore')

GENERAL_PATH = utils.get_general_path()
FINAL_DATA_PATH = utils.get_data_path('final')
TEMP_RESULTS_DATA_PATH = utils.get_data_path('temp_results')
GROUPED_INFO_METRICS_PATH = utils.join_paths(TEMP_RESULTS_DATA_PATH, 'grouped_info_metrics.parquet')
GENERAL_METRICS_PATH = utils.join_paths(TEMP_RESULTS_DATA_PATH, 'general_metrics.parquet')

RANDOM_SEED = utils.RANDOM_SEED
SET_COLORS_DICT = utils.SET_COLOR_DICT

# Matplotlib configurations
main_color = '#41596a'
mpl.rcParams['text.color'] = main_color
mpl.rcParams['axes.labelcolor'] = main_color
mpl.rcParams['axes.edgecolor'] = main_color
mpl.rcParams['axes.edgecolor'] = main_color
mpl.rcParams['xtick.color'] = main_color
mpl.rcParams['xtick.labelcolor'] = main_color
mpl.rcParams['ytick.color'] = main_color
mpl.rcParams['ytick.labelcolor'] = main_color


def get_final_datasets():
    x = {}
    info = {}
    for file in os.listdir(FINAL_DATA_PATH):
        if not file.startswith('.'):
            start_letter, filename_string = file.split('_')
            dataset_name = filename_string.split('.')[0]
            path = utils.join_paths(FINAL_DATA_PATH, file)
            if start_letter=='i':
                info[dataset_name] = pd.read_parquet(path)
            elif start_letter=='x':
                x[dataset_name] = pd.read_parquet(path)
    return x, info


def generate_grouped_info_metrics(info, months=6, save=True):
    if utils.check_if_filepath_exists(GROUPED_INFO_METRICS_PATH):
        print(f'Loading file from {GROUPED_INFO_METRICS_PATH}')
        grouped_info_metrics = utils.read_pickle(GROUPED_INFO_METRICS_PATH)
        return grouped_info_metrics
    date_col = 'meteorological_current_datetime_val'
    grouper = pd.Grouper(key=date_col, freq=f'{months}M')
    grouped_info_metrics = {}
    grouped_info_metrics['aucpr'] = {}
    grouped_info_metrics['aucroc'] = {}
    grouped_info_metrics['brier'] = {}
    for key in info.keys():
        grouped_info_metrics['aucpr'][key] = info[key].groupby(
            grouper
        ).apply(
            bootstrap_func, function=compute_aucpr
        )
        grouped_info_metrics['aucroc'][key] = info[key].groupby(
            grouper
        ).apply(
            bootstrap_func, function=compute_aucroc
        )
        grouped_info_metrics['brier'][key] = info[key].groupby(
            grouper
        ).apply(
            bootstrap_func, function=compute_brierscoreloss
        )
    if save:
        print(f'Saving info at {GROUPED_INFO_METRICS_PATH}')
        utils.save_as_pickle(what=grouped_info_metrics, where=GROUPED_INFO_METRICS_PATH)
    return grouped_info_metrics


def plotting__brier_loss_score(grouped_info_metrics, sets=['train', 'test', 'OOT']):
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in sets:
        yerr = (
            grouped_info_metrics['brier'][key]['err_lo'],
            grouped_info_metrics['brier'][key]['err_hi']
        )
        ax.errorbar(
            grouped_info_metrics['brier'][key].index,
            grouped_info_metrics['brier'][key]['mean'],
            yerr=yerr,
            capsize=5,
            marker='o',
            alpha=0.8,
            label=key,
            color=SET_COLORS_DICT[key]
        )
    date_min_max = [
        grouped_info_metrics['brier']['train'].index.min(),
        grouped_info_metrics['brier']['OOT'].index.max()
    ]
    ax.plot(
        date_min_max,
        [0, 0],
        label='Max. score value',
        linestyle='--',
        color=main_color,
        alpha=0.5
    )
    plt.legend()
    plt.title('Brier loss over time for different datasets.')
    plt.xlabel('Dates')
    plt.ylabel('Brier loss score')
    plt.show()


def plotting__auc_roc(grouped_info_metrics, sets=['train', 'test', 'OOT'], baseline=0.5):
    fig, ax = plt.subplots(figsize=(9, 5))
    for key in sets:
        yerr = (
            grouped_info_metrics['aucroc'][key]['err_lo'],
            grouped_info_metrics['aucroc'][key]['err_hi']
        )
        ax.errorbar(
            grouped_info_metrics['aucroc'][key].index,
            grouped_info_metrics['aucroc'][key]['mean'],
            yerr=yerr,
            capsize=5,
            marker='o',
            alpha=0.8,
            label=key,
            color=SET_COLORS_DICT[key]
        )
    date_min_max = [
        grouped_info_metrics['aucroc']['train'].index.min(),
        grouped_info_metrics['aucroc']['OOT'].index.max()
    ]
    ax.plot(
        date_min_max,
        [1, 1],
        label='Max. score value',
        linestyle='--',
        color=main_color,
        alpha=0.5

    )

    ax.plot(
        date_min_max,
        [baseline, baseline],
        label='Baseline',
        linestyle='-.',
        color=main_color,
        alpha=0.5
    )

    plt.ylim(0.45, 1.05)
    plt.legend()
    plt.title('AUC ROC over time for different datasets.')
    plt.xlabel('Dates')
    plt.ylabel('AUC ROC')
    plt.show()


def plotting__auc_pr(grouped_info_metrics, sets=['train', 'test', 'OOT'], baseline=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in sets:
        yerr = (
            grouped_info_metrics['aucpr'][key]['err_lo'],
            grouped_info_metrics['aucpr'][key]['err_hi']
        )
        ax.errorbar(
            grouped_info_metrics['aucpr'][key].index,
            grouped_info_metrics['aucpr'][key]['mean'],
            yerr=yerr,
            capsize=5,
            marker='o',
            alpha=0.8,
            label=key,
            color=SET_COLORS_DICT[key]
        )
    date_min_max = [
        grouped_info_metrics['aucpr']['train'].index.min(),
        grouped_info_metrics['aucpr']['OOT'].index.max()
    ]

    ax.plot(
        date_min_max,
        [1, 1],
        label='Max. score value',
        linestyle='--',
        color=main_color,
        alpha=0.5

    )

    ax.plot(
        date_min_max,
        [baseline, baseline],
        label='Baseline',
        linestyle='-.',
        color=main_color,
        alpha=0.5
    )

    plt.ylim(0, 1.05)
    plt.legend()
    plt.title('AUC PR over time for different datasets.')
    plt.xlabel('Dates')
    plt.ylabel('AUC PR')
    plt.show()


def generate_general_metrics(info, save=True):
    if utils.check_if_filepath_exists(GENERAL_METRICS_PATH):
        print(f'Loading file from {GENERAL_METRICS_PATH}')
        metrics = utils.read_pickle(GENERAL_METRICS_PATH)
        return metrics
    metrics = {}
    metrics['aucroc'] = {}
    metrics['aucpr'] = {}
    metrics['brier'] = {}
    for key in info.keys():
        metrics['aucroc'][key] = bootstrap_func(info[key], compute_aucroc)
        metrics['aucpr'][key] = bootstrap_func(info[key], compute_aucpr)
        metrics['brier'][key] = bootstrap_func(info[key], compute_brierscoreloss)
    if save:
        print(f'Saving info at {GENERAL_METRICS_PATH}')
        utils.save_as_pickle(what=metrics, where=GENERAL_METRICS_PATH)
    return metrics

def plot_general_metrics(metrics_dict, metric, baseline=None):
    metrics_defined = metrics_dict[metric]
    metrics_defined_df = pd.DataFrame(metrics_defined).T
    metrics_defined_df['color'] = pd.Series(SET_COLORS_DICT)
    plt.bar(
        metrics_defined_df.index,
        metrics_defined_df['mean'],
        yerr=(metrics_defined_df['err_lo'], metrics_defined_df['err_hi']),
        color=metrics_defined_df.color
    )
    for i, row in enumerate(metrics_defined_df.iterrows()):
        set_name, traits = row
        plt.text(i, traits['mean'] + 0.08*traits['mean'], s=round(traits['mean'], 3), ha='center', va='bottom')
    metrics_defined_df_max_mean = (metrics_defined_df['mean'] + 0.2*metrics_defined_df['mean']).max()
    if baseline is not None:
        metrics_defined_df['baseline'] = baseline
        plt.plot(metrics_defined_df.index, metrics_defined_df['baseline'], color='k', linestyle='--', label='Baseline')
        plt.legend()
    plt.ylim(0, metrics_defined_df_max_mean)
    plt.ylabel(f'{metric}')
    plt.xlabel(f'datasets')
    plt.title(f'General {metric.upper()} for each dataset ')
    plt.show()
    return metrics_defined_df
