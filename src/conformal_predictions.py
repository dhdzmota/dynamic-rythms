import pandas as pd
import pickle

from mapie.classification import MapieClassifier

import src.utils as utils


GENERAL_PATH = utils.get_general_path()
MAPIE_MODEL_PATH = utils.join_paths(GENERAL_PATH, 'models')
MAPIE_MODEL_FILENAME = utils.join_paths(MAPIE_MODEL_PATH, 'conformal_model.pkl')


def save_mapie_model(model):
    with open(MAPIE_MODEL_FILENAME, "wb") as f:
        pickle.dump(model, f)


def get_mapie_model():
    with open(MAPIE_MODEL_FILENAME, 'rb') as f:
        load_model = pickle.load(f)
    return load_model


def fit_mapie_classifier(model, x, y, save=True, force=False):
    if not utils.check_if_filepath_exists(MAPIE_MODEL_FILENAME) or force:
        mapie_clf = MapieClassifier(estimator=model, cv='prefit', method='lac')
        mapie_clf.fit(x, y)
        if save:
            save_mapie_model(mapie_clf)
    else:
        print(
            f'Mapie Model has already been trained and already exists'
            f', it is located at: {MAPIE_MODEL_FILENAME}'
        )
        mapie_clf = get_mapie_model()
    return mapie_clf


def conformal_prediction(mapie_clf, x, alpha=0.10):
    y_pred_label, y_predicted_sets = mapie_clf.predict(x, alpha=alpha)
    # For each sample, returns 1 if there is certainty regarding
    # the class with respect to the model and the allowed error rate, 0 if not.
    class_certainty = 1 - (y_predicted_sets.sum(axis=1).reshape(1, -1)[0] - 1)
    y_pred_label = pd.Series(y_pred_label)
    y_pred_label.loc[class_certainty==0] = -1
    y_pred_label = y_pred_label.to_numpy()
    return y_pred_label, y_predicted_sets, class_certainty


def conformal_prediction_projected(mapie_clf, x, alpha=0.10):
    y_pred_label, _, _ = conformal_prediction(mapie_clf, x, alpha=alpha)
    return y_pred_label


def get_predicted_labels_and_set(mapie_clf, x, alpha=0.10, sets=['test'], ):
    print(f'The allowed error rate is {alpha}. This means a coverage of {1-alpha}.')
    results = {}
    for key in sets:
        x_key = x[key]
        results[key] = {}
        y_pred_label, y_predicted_sets, class_certainty = conformal_prediction(mapie_clf, x_key, alpha=alpha)
        results[key]['y_pred_label'] = y_pred_label
        results[key]['y_predicted_sets'] = y_predicted_sets
        results[key]['class_certainty'] = class_certainty
    return results


def integrate_certainty_into_info_df(info, results, sets=['test']):
    for key in sets:
        info_key = info[key]
        results_key = results[key]
        info_key['class_certainty'] = results_key['class_certainty']
    return info


def get_certainty_frame(info, sets=['test']):
    certainty_frames = {}
    for key in sets:
        info_key = info[key]
        q = pd.qcut(info_key['pred'], q=1000)
        info_key['q'] = q
        certainty_frame = info_key.groupby('q').class_certainty.mean()
        certainty_frames[key] = certainty_frame
    return certainty_frames


def get_certainty_quantiles_min_max(certainty_frame):
    possible_qs = certainty_frame[certainty_frame < 1]
    if possible_qs.shape[0]:
        q_min = possible_qs.index.min().left
        q_max = possible_qs.index.max().right
        return q_min, q_max
    print('Certainty frame does not exist. '
          'Change (lower) alpha in conformal prediction to get results.')
    return 0, 0

