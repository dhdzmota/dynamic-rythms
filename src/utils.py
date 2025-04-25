import geopandas as gpd
import json
import os
import pandas as pd
import pickle
import urllib
import yaml

from io import BytesIO
from zipfile import ZipFile

RANDOM_SEED = 42
CUSTOMERS_OUT_NB = 10**3.5 # 10**3.5
STATE_ABBREVIATIONS = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
    'United States Virgin Islands': 'VI'
}

SHAP_COLOR_PALLETE = (
    '#008bfb',
    '#007bf4',
    '#3569e8',
    '#6657d9',
    '#8443c6',
    '#a21eaa',
    '#bc009f',
    '#d6008e',
    '#e9007d',
    '#f80068',
    '#ff0055',
)

SET_COLOR_DICT = {
    'train': '#f8cc62',
    'test': '#bba681',
    'eval': '#737373',
    'cal': '#41596a',
    'OOT': '#7f95a4',
}


def get_general_path():
    '''Function to get the general path'''
    file_path = os.path.dirname(os.path.abspath(__file__))
    general_path = os.path.join(file_path, '..')
    return general_path


def join_paths(*p1):
    """
    Helper function to join paths
    """
    return os.path.join(*p1)


def check_if_filepath_exists(filepath):
    """Check if the corresponding path exists."""
    exists = os.path.exists(filepath)
    return exists


def make_desired_folder(data_file_path):
    general_path = get_general_path()
    file_path = join_paths(general_path, data_file_path)
    exists = check_if_filepath_exists(file_path)
    if not exists:
        os.makedirs(file_path)
    return None


def get_data_path(name):
    '''Obtain the relative path for the data folder'''
    general_path = get_general_path()
    file_path = join_paths(general_path, 'data', name)
    return file_path


def save_dataframe(filepath, dataframe, file_format='parquet'):
    """ Saves dataframe into the desired path as the desired format.

    :param filepath:
    :param dataframe:
    :param file_format:
    :return:
    """
    if file_format == 'parquet':
        dataframe.to_parquet(filepath)
    elif file_format == 'csv':
        dataframe.to_csv(filepath)
    elif file_format == 'pickle':
        dataframe.to_pickle(filepath)
    print(f'Data was saved into `{filepath}`.')


def get_config_file():
    """Access the configuration file with the URL links"""
    general_path = get_general_path()
    yaml_path = join_paths(general_path, 'config','config.yaml')
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return config


def get_model_hyperparameters():
    """Access de hyperparameters file."""
    general_path = get_general_path()
    json_path = join_paths(general_path, 'config', 'model_params.json')
    with open(json_path, 'r') as f:
        hyperparams = json.loads(f)
    return hyperparams


def get_parameters_file():
    """Access the parameters file with the params definitions"""
    general_path = get_general_path()
    params_path = join_paths(general_path, 'config','POWER_Parameter_Manager.csv')
    params = pd.read_csv(params_path, engine='python', header=1).head(20).to_dict()['Parameter(s):']
    return params


def key_list(dictionary):
    """Get the keys of a dictionary as a list"""
    key_list_dict = list(dictionary.keys())
    return key_list_dict


def save_json_from_url_zip(url, save_data_path, verbose=False):
    """From a URL, that downloads a zip file containing json files, download the info.
    """
    print('Downloading info...')
    req = urllib.request.Request(url, headers={'User-Agent': "Magic Browser"})
    url_response = urllib.request.urlopen(req)
    zip_file = ZipFile(BytesIO(url_response.read()))
    for f in zip_file.namelist():
        if f.endswith('.json'):
            zip_file.extract(f, path=save_data_path)
            if verbose:
                print(f'Extracting {f} into {save_data_path}')
    print(f'Done with extraction into {save_data_path}.')
    return None


def save_shapefile_from_url_zip(url, save_data_path):
    """From a URL, that downloads a zip file containing shp files, download the info and
    then get the main dataframe.
    """
    if not check_if_filepath_exists(save_data_path):
        print('Downloading info...')
        req = urllib.request.Request(url, headers={'User-Agent': "Magic Browser"})
        url_response = urllib.request.urlopen(req)
        zip_file = ZipFile(BytesIO(url_response.read()))

        for f in zip_file.namelist():
            print(f)
            if f.endswith('shx') or f.endswith('shp') or f.endswith('dbf'):
                zip_file.extract(f, path=save_data_path)
                print(f'Extracting {f} into {save_data_path}')
            if f.endswith('shp'):
                file_name = f
    else:
        print('Info already exists...')
        for f in os.listdir(save_data_path):
            if f.endswith('shp'):
                file_name = f
    main_file = os.path.join(save_data_path, file_name)
    return main_file


def save_info(main_file, filepath):
    "Read a main file shp dataframe and save it into a desired path"
    if not check_if_filepath_exists(filepath):
        city_data = gpd.read_file(main_file)
        save_dataframe(filepath=filepath, dataframe=city_data)
    else:
        print(f'Information is already saved at: {filepath}')


def save_as_json(what, where):
    if not check_if_filepath_exists(where):
        with open(where, 'w') as f:
            json.dump(what, f)


def save_as_pickle(what, where):
    if not check_if_filepath_exists(where):
        with open(where, 'wb') as f:
            pickle.dump(what, f)


def read_pickle(where):
    with open(where, 'rb') as f:
        loaded_file = pickle.load(f)
        return loaded_file


def miniprocess_outage_raw_df(outages):
    print('Processing outages...')
    print('Deleting customers_out nulls...')
    outages = outages[outages.customers_out.notna()]  # Filter nan values from customers_out
    # Then we keep only the relevant outage (affecting a high amount of customers)
    print(f'Keeping relevant outages according to CUSTOMERS_OUT_NB={CUSTOMERS_OUT_NB}')
    outages = outages[outages.customers_out >= CUSTOMERS_OUT_NB]
    print('Changing run_start_time to datetime...')
    outages.run_start_time = pd.to_datetime(outages.run_start_time)  # Transform into datetime to manipulate dates
    print('Mapping state_ids...')
    outages["state_id"] = outages.state.map(STATE_ABBREVIATIONS) # Use the state abbreviations to get an ID
    print('Filling fips_code_ids...')
    outages["fips_code_id"] = outages.fips_code.astype(str).str.zfill(5)
    outages["sub_general_id"] = (outages.fips_code_id + '_' + outages.state_id)
    return outages


def get_required_outages_dfs(*years, eaglei_data_path=None):
    eaglei_data_paths = os.listdir(eaglei_data_path)
    paths = []
    for year in years:
        paths += [join_paths(eaglei_data_path,file) for file in eaglei_data_paths if str(year) in file]
    dfs = []
    for file in paths:
        print(f"Reading file: {file}.")
        outage_data = pd.read_csv(file)
        outage = miniprocess_outage_raw_df(outage_data)
        dfs.append(outage)
    print("Done reading.")
    if len(paths) > 1:
        print("Merging information.")
        outages_df = pd.concat(dfs)
        del dfs
    else:
        outages_df = dfs[0]
    print('Data is ready.')

    return outages_df


def save_pickle_model(model:any, file_name:str='outage_model.pkl')->None:
    """
    save a given model into a pickle file in the model folder

    Args:
        model (any): model to be saved.
        file_name (str, optional): name to save the model. 
                    Defaults to 'outage_model.pkl'.
    """
    general_path = get_general_path()
    model_folder = join_paths(general_path, 'models')
    os.makedirs(model_folder, exist_ok=True)
    model_path = join_paths(model_folder, file_name)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def get_record_from_df(df, pos):
    record = df.iloc[pos]
    return record
