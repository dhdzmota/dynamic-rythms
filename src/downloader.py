import utils as utils

CITY_URL_KEY = 'CITY_URL_SHAPEFILE'
COUNTY_URL_KEY = 'COUNTY_URL_SHAPEFILE'
WGS84 = "EPSG:4326"  # World Geodetic System 1984 ensemble

EXTERNAL_DATA_PATH = utils.get_data_path('external')
GENERAL_PATH = utils.get_general_path()

RAW_CITY_DATA_PATH = utils.join_paths(EXTERNAL_DATA_PATH, 'cities_raw')
RAW_COUNTY_DATA_PATH = utils.join_paths(EXTERNAL_DATA_PATH, 'counties_raw')

CITY_DATA_PATH = utils.join_paths(EXTERNAL_DATA_PATH, 'city.parquet')
COUNTY_DATA_PATH = utils.join_paths(EXTERNAL_DATA_PATH, 'county.parquet')

def download_cities():
    config = utils.get_config_file()
    url = config[CITY_URL_KEY]
    main_file = utils.save_shapefile_from_url_zip(
        url=url, save_data_path=RAW_CITY_DATA_PATH
    )
    utils.save_info(main_file, filepath=CITY_DATA_PATH)
    return None

def download_counties():
    config = utils.get_config_file()
    url = config[COUNTY_URL_KEY]
    main_file = utils.save_shapefile_from_url_zip(
        url=url, save_data_path=RAW_COUNTY_DATA_PATH
    )
    utils.save_info(main_file, filepath=COUNTY_DATA_PATH)
    return None

def download_meteorological_info():
    """Function to download meteorological data"""
    general_path = get_general_path()
    interim_path = get_data_path('interim')

    municipal_data_path = join_paths(
        general_path, interim_path, SIMPLIFIED_MUNICIPAL_DATA
    )
    municipal_data = pd.read_pickle(municipal_data_path)
    municipal_data['information_link'] = (
        municipal_data.representative_coordinate.apply(
            find_url_with_point,
            start_date=INITIAL_DATE,
            end_date=FINAL_DATE
        )
    )
    for row in municipal_data.iterrows():
        extract_and_save_meteorological_information(row)
    return None


def download():
    download_cities()
    download_counties()
    # download_meteorological_info()
    pass


if __name__ == "__main__":
    download()
