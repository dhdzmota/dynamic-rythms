import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shap
import warnings

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
from sklearn.utils import resample

import src.utils as utils
import src.training_model as training_model

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
SHAP_COLOR_PALLETE = utils.SHAP_COLOR_PALLETE

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

DIFFERENCE_THRESHOLD = 0.96
WRITTEN_FEATURES_NB = 6

def shap_top_N_features(x, shap_explainer, N=10):
    shap_values_x_df = get_shap_values_df(x, shap_explainer)
    importance = shap_values_x_df.abs().sum().sort_values(ascending=False).head(N)
    return importance.index.to_list()


def get_shap_values_df(x, shap_explainer):
    shap_values_x = shap_explainer.shap_values(x)
    shap_values_x_df = pd.DataFrame(shap_values_x, columns=x.columns, index=x.index)
    return shap_values_x_df

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


def plotting_score_hours_before_outage(info, sets=['train', 'OOT'], elements=150):
    fig, ax = plt.subplots(figsize=(10,5))
    for key in sets:
        hours_to_outage_scores = info[key].groupby('hours_to_outage').agg(
            pred_mean=('pred', 'mean')
        ).sort_index().head(elements)
        ax.scatter(
            hours_to_outage_scores.index,
            hours_to_outage_scores.pred_mean,
            c=SET_COLORS_DICT[key],
            alpha=0.5,
            label=key)
        ax.plot(hours_to_outage_scores.pred_mean, color=SET_COLORS_DICT[key])
    plt.xlabel('Hours before an outage (generated by a storm)')
    plt.ylabel('Average model score')
    plt.title('Score mean value depending on the hours left so that an outage occurs.')
    plt.legend()
    plt.ylim(0, 1)
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

def get_sample(info, x, sets=['OOT']):
    samples = {}
    for key in sets:
        samples[key] = {}
        info_key = info[key]
        episode_fips_metrics = info_key.groupby('episode_fips_id').agg(
            pred_first=('pred', 'first'),
            pred_last=('pred', 'last'),
            pred_min=('pred', 'min'),
            pred_max=('pred', 'max'),
            element_nb=('time', 'count')
        )
        episode_fips_metrics['difference__last_first'] = (
                episode_fips_metrics['pred_last'] -
                episode_fips_metrics['pred_first']
        )
        sample_index = episode_fips_metrics[
            episode_fips_metrics.difference__last_first > DIFFERENCE_THRESHOLD
        ].sample(1,random_state=RANDOM_SEED).index.to_list()
        sample_info_key = info_key[info_key.episode_fips_id.isin(sample_index)]
        sample_x_key = x[key].loc[sample_info_key.index]

        samples[key]['info'] = sample_info_key
        samples[key]['x'] = sample_x_key
        return samples

def get_model_explainer():
    model = training_model.get_model()
    explainer = shap.TreeExplainer(model)
    return model, explainer


def plot_shap_importance_for_each_sample(samples):
    model, explainer = get_model_explainer()
    expected_value = explainer.expected_value

    for key in samples.keys():
        sample_x_key = samples[key]['x']
        sample_info_key = samples[key]['info']

        shap_values_sample_x_key_df = get_shap_values_df(
            x=sample_x_key,
            shap_explainer=explainer
        )
        shap_values_sample_x_key_df_abs = shap_values_sample_x_key_df.abs()
        important_features = shap_top_N_features(
            x=sample_x_key,
            shap_explainer=explainer,
            N=500
        )
        all_other_importances = shap_values_sample_x_key_df_abs.drop(
            important_features, axis=1
        ).sum(axis=1)
        shap_values_sample_x_key_df_abs_relevant = shap_values_sample_x_key_df_abs[
            important_features
        ]
        shap_values_sample_x_key_df_abs_relevant['all_other_features'] = all_other_importances
        total_sum = shap_values_sample_x_key_df_abs_relevant.T.sum()
        relevant_cols = shap_values_sample_x_key_df_abs_relevant.columns
        for col in relevant_cols:
            shap_values_sample_x_key_df_abs_relevant[col] = (
                    shap_values_sample_x_key_df_abs_relevant[col] / total_sum
            )

        last_record = utils.get_record_from_df(df=shap_values_sample_x_key_df_abs_relevant, pos=-1)
        shap_values_sample_x_key_df_abs_relevant_top5_index = last_record.sort_values(ascending=False).iloc[: WRITTEN_FEATURES_NB].index
        positions_top5_text = (last_record.cumsum() + last_record.shift(1).fillna(0).cumsum()) / 2
        top5_text = positions_top5_text.loc[shap_values_sample_x_key_df_abs_relevant_top5_index]

        first_record = utils.get_record_from_df(df=shap_values_sample_x_key_df_abs_relevant, pos=0)
        shap_values_sample_x_key_df_abs_relevant_low5_index = first_record.sort_values(ascending=False).iloc[: WRITTEN_FEATURES_NB].index
        positions_low5_text = (first_record.cumsum() + first_record.shift(1).fillna(0).cumsum()) / 2
        low5_text = positions_low5_text.loc[shap_values_sample_x_key_df_abs_relevant_low5_index]

        all_records_info = [
            shap_values_sample_x_key_df_abs_relevant[col]
            for col in shap_values_sample_x_key_df_abs_relevant.columns
        ]
        fig, ax = plt.subplots(figsize=(20,10))
        # Plot Stack, which is already normalized. These are normalized feature importances.
        ax.stackplot(
            sample_info_key.meteorological_current_datetime_val,
            all_records_info,
            colors=SHAP_COLOR_PALLETE,
            alpha=0.7,
        )
        pos_x_top = sample_info_key.meteorological_current_datetime_val.max()
        pos_x_low = sample_info_key.meteorological_current_datetime_val.min()
        tot_seconds = (pos_x_top - pos_x_low).total_seconds()
        additional_space = tot_seconds * 0.5
        additional_pos_x_top = pos_x_top + pd.Timedelta(f"{additional_space} seconds")
        additional_pos_x_low = pos_x_low - pd.Timedelta(f"{additional_space} seconds")

        # Plot text
        for element_top, element_low in zip(top5_text.index, low5_text.index):
            val_top = round(sample_x_key.iloc[-1][element_top], 2)
            pos_y = top5_text.loc[element_top]
            text = element_top
            ax.text(pos_x_top, pos_y, f' <-- {text}: ({val_top})', ha='left', va='center')
            val_low = round(sample_x_key.iloc[0][element_low], 2)
            pos_y = low5_text.loc[element_low]
            text = element_low
            ax.text(pos_x_low, pos_y, f'{text}: ({val_low}) --> ', ha='right', va='center')
        ax.plot(
            sample_info_key.meteorological_current_datetime_val,
            sample_info_key.pred,
            marker='o',
            label='Model score',
            color='k'
        )
        ax.vlines(
            sample_info_key.meteorological_current_datetime_val,
            0,
            1,
            color='k',
            linestyle='--',
            label='Temporal markers',
            alpha=0.7
        )
        plt.xlim(additional_pos_x_low, additional_pos_x_top)
        plt.legend()
        plt.title(
            'Behaviour of the model score over time for a specific storm.'
            ' With a representation of normalized feature importance changing over time.'
        )
        plt.xlabel('Date')
        plt.ylabel('Model Score')
        plt.show()

        shap.decision_plot(
            expected_value,
            shap_values_sample_x_key_df.to_numpy(),
            sample_x_key,
            link='logit',
            feature_order='importance',
            feature_display_range=slice(-1, -20, -1),
            highlight=sample_info_key['y'].astype('bool'),
        )
        plt.show()

def shap_importance_for_each_sample(x, info):
    samples = get_sample(info, x, sets=['OOT','test'])
    plot_shap_importance_for_each_sample(samples)

def general_shap_summary(x, sets=['OOT', 'test']):
    model = training_model.get_model()
    explainer = shap.TreeExplainer(model)
    for key in sets:
        x_key = x[key]
        shap_values = explainer.shap_values(x_key)
        plt.title(f'Summary Shap Values for {key} set')
        shap.summary_plot(shap_values, x_key)

def plot_shap_dependence_plots(x, sets=['OOT', 'test'], nb_important_features=5):
    model = training_model.get_model()
    explainer = shap.TreeExplainer(model)
    for key in sets:
        x_key = x[key]
        shap_values = explainer.shap_values(x_key)
        top_inds = np.argsort(-np.sum(np.abs(shap_values), 0))
        for i in range(nb_important_features):
            shap.dependence_plot(top_inds[i], shap_values, x_key)


def plot_score_distribution(info, sets=['OOT']):
    for key in sets:
        info_key = info[key]
        fig, ax = plt.subplots(figsize=(20,10))
        alpha_val = 0.4
        ax.hist(info_key['pred'][info_key['y']==0], bins=100, color='blue', alpha=alpha_val)
        ax2 = ax.twinx()
        ax2.hist(info_key['pred'][info_key['y']==1], bins=100, color='orange', alpha=alpha_val)
        plt.title(f'Score distribution for the positive class on the {key} set.')
        plt.xlabel('Model score (bins)')
        ax.set_ylabel('Count (negative class)')
        ax2.set_ylabel('Count (positive class)')


def plot_general_score_distribution_w_qs(info, sets=['OOT'], q_min=0, q_max=0):
    for key in sets:
        info_key = info[key]
        fig, ax = plt.subplots(figsize=(20,10))
        alpha_val = 0.2
        y_hist, x_hist, plot_hist = ax.hist(info_key['pred'], bins=1000, density=True)
        ax.vlines([q_min, q_max], 0, y_hist.max(), color='k', linestyle='--')
        plt.fill_between(
            [q_min, q_max],
            y_hist.max(),
            alpha=alpha_val,
            color='r',
            hatch='//',
            label='Uncertainty area'
        )
        plt.title('Score distribution with the corresponding Uncertainty Area.')
        plt.ylabel('Histogram Density')
        plt.xlabel('Model score')
        plt.legend()
        plt.show()
