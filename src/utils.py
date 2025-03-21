import geopandas as gpd
import os
import yaml
import pickle

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


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


def save_shapefile_from_url_zip(url, save_data_path):
    """From a URL, that downloads a zip file containing shp files, download the info and
    then get the main dataframe.
    """
    if not check_if_filepath_exists(save_data_path):
        print('Downloading info...')
        url_response = urlopen(url)
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