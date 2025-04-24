import pickle

from xgboost import XGBClassifier

import src.dataset_splitting as dataset_splitting
import src.utils as utils


RANDOM_SEED = utils.RANDOM_SEED

INTERIM_DATA_PATH = utils.get_data_path('interim')
FINAL_DATA_PATH = utils.get_data_path('final')
GENERAL_PATH = utils.get_general_path()
MODEL_PATH = utils.join_paths(GENERAL_PATH, 'models')
MODEL_FILENAME = utils.join_paths(MODEL_PATH, 'models.pkl')


NO_FEATURE_COLS = [
    'time',
    'episode_fips_id',
    'meteorological_current_datetime_val',
    'hours_to_outage',
    'outage_in_an_hour',
]

ADDITIONAL_INFO_COLUMNS = [
    'coord0',
    'coord1',
    'day_of_year',
    'month_of_year',
]

XGB_PARAMETERS = dict(
    n_estimators=1000000,
    learning_rate=0.1,
    max_depth=3,
    scale_pos_weight=None,
    use_label_encoder=False,
    eval_metric='aucpr',
    reg_lambda=1,
    alpha=1,
    subsample=0.75,
    colsample_bytree=0.75,
    early_stopping_rounds=25,
    gamma=1,
)


def read_splitted_data():
    datasets = dataset_splitting.read_datasets()
    return datasets


def get_datasets_x_y_info(datasets):
    x = {}
    y = {}
    info = {}
    for key, data in datasets.items():
        x[key] = data.drop(NO_FEATURE_COLS, axis=1)
        y[key] = data.outage_in_an_hour
        info[key] = data[NO_FEATURE_COLS + ADDITIONAL_INFO_COLUMNS]
    return x, y, info


def save_x_y_info(x, info):
    for key in x.keys():
        x_path = utils.join_paths(FINAL_DATA_PATH, f'x_{key}.parquet')
        i_path = utils.join_paths(FINAL_DATA_PATH, f'i_{key}.parquet')
        print(f'Saving x for {key} data at: {x_path}')
        x[key].to_parquet(x_path)
        print(
            f'Saving info for {key} data at: {i_path} '
            f'(target and inference are contained).'
        )
        info[key].to_parquet(i_path)


def save_model(model):
    with open(MODEL_FILENAME, "wb") as f:
        pickle.dump(model, f)


def get_model():
    with open(MODEL_FILENAME, 'rb') as f:
        load_model = pickle.load(f)
    return load_model


def model_score(model, X):
    score = model.predict_proba(X)[:, 1]
    return score


def train_model(force=False, save_data=False):
    if not utils.check_if_filepath_exists(MODEL_FILENAME) or force:
        # Read data
        datasets = read_splitted_data()
        x, y, info = get_datasets_x_y_info(datasets)

        neg = (y['train'] == 0).sum()
        pos = (y['train'] == 1).sum()
        scale_pos_weight = neg / pos

        XGB_PARAMETERS['scale_pos_weight'] = scale_pos_weight
        print(f' The model parameters are: {XGB_PARAMETERS}')

        model = XGBClassifier(**XGB_PARAMETERS)
        print('Start with training...')
        model.fit(
            x['train'],
            y['train'],
            eval_set=[(x['train'], y['train']), (x['eval'], y['eval'])],
        )
        save_model(model)

        for key in info.keys():
            info[key]['y'] = y[key]
            info[key]['pred'] = model_score(model, x[key])
        if save_data:
            print('Saving data...')
            save_x_y_info(x, info)
    else:
        print(
            f'Model has already been trained and already exists'
            f', it is located at: {MODEL_FILENAME}'
        )
        model = get_model()
    return model


if __name__ == "__main__":
    train_model(force=False)
