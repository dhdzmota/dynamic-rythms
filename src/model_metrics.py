import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
from sklearn.utils import resample

import src.utils as utils


RANDOM_SEED = utils.RANDOM_SEED


def compute_brierscoreloss(df):
    try:
        score = brier_score_loss(df['y'], df['pred'])
    except ValueError:
        score = np.nan
    return score


def compute_aucroc(df):
    try:
        score = roc_auc_score(df['y'], df['pred'])
    except ValueError:
        score = np.nan
    return score


def compute_aucpr(df):
    try:
        precision, recall, _ = precision_recall_curve(df['y'], df['pred'])
        score = auc(recall, precision)
    except ValueError:
        score=np.nan
    return score

def bootstrap_func(df, function, n_bootstraps=50, ci=0.95, seed=RANDOM_SEED):
    bootstraped_scores = []
    rng = np.random.RandomState(seed)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, df.shape[0], df.shape[0])
        score = function(df.iloc[indices])
        bootstraped_scores.append(score)
    bs = pd.Series(bootstraped_scores)
    ci_upper = bs.quantile(ci)
    ci_lower = bs.quantile(1-ci)
    bs_mean = bs.mean()
    response = pd.Series({'mean': bs_mean, 'err_lo': bs_mean-ci_lower, 'err_hi': ci_upper-bs_mean})
    return response