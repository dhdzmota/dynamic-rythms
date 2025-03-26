import json
import pandas as pd
import requests

import utils
METEOROLOGICAL_API_KEY = 'METEOROLOGICAL_API_URL'


def get_api_url(lat, lon, datetime_start, datetime_end, temporality='hourly'):
    """

    :param lat: Location Latitude
    :param lon: Location Longitude
    :param datetime_start: Start datetime (yyyy-mm-dd hh format)
    :param datetime_end: Start datetime (yyyy-mm-dd hh format)
    :param temporality: Temporality of the meteorological information (eg. Daily, Hourly)
    :return:
        url: Formated string of the API url.
    """
    config = utils.get_config_file() # Bottleneck
    parameters = utils.get_parameters_file() # Bottleneck

    start = pd.to_datetime(datetime_start).strftime('%Y%m%d')
    end = pd.to_datetime(datetime_end).strftime('%Y%m%d')
    base_url = config[METEOROLOGICAL_API_KEY]
    parameter_list = utils.key_list(parameters)
    temporality_str = f'{temporality}/point?'
    params = f"parameters={(',').join(parameter_list)}"
    community = '&community=RE'
    longitude = f"&longitude={lon}"
    latitude = f"&latitude={lat}"
    start_date = f"&start={start}"
    end_date = f"&end={end}"
    final_format = '&format=JSON'
    url = (f'{base_url}{temporality_str}{params}{community}{longitude}{latitude}'
           f'{start_date}{end_date}{final_format}')
    return url


def get_meteorological_info(url):
    text_information = requests.get(url).text
    formated_text_information = json.loads(text_information)
    return formated_text_information


def save_meteorological_information(information, path):
    utils.save_as_json(what=information, where=path)

def process():
    lat =20.66682
    lon =-103.39182
    datetime_start='2020-01-01 11:30'
    datetime_end='2020-01-01 20:31'
    temporality = 'hourly'
    url = get_api_url(
        lat=lat,
        lon=lon,
        datetime_start=datetime_start,
        datetime_end=datetime_end,
        temporality=temporality
    )
    met_info = get_meteorological_info(url)
    external_data_path = utils.get_data_path('external')
    data_path = utils.join_paths(external_data_path, 'meteorological_info', 'test.json')
    save_meteorological_information(information=met_info, path=data_path)


#if __name__ == '__main__':
#    process()
