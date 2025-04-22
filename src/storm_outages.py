'''
This main script aims to generate the overlap between storms and outages.
'''

import src.utils as utils
import pandas as pd
import warnings

# Min number of customers affected.
CUSTOMERS_OUT_NB = utils.CUSTOMERS_OUT_NB

# Max timelaps that divide outages.
SEPPARATION_HOURS = 18
# Represents 15 minutes, in seconds
MIN_OUTAGE_SECONDS = 60 * 15
NB_15_MIN_IN_HOUR = 4
DAYS_AFTER_STORM_THRESHOLD = 1

GENERAL_PATH = utils.get_general_path()
RAW_DATA_PATH = utils.get_data_path('raw')
INTERIM_DATA_PATH = utils.get_data_path('interim')
DYNAMIC_RYTHMS_DATA_PATH = utils.join_paths(RAW_DATA_PATH, 'dynamic-rhythms-train-data', 'data')
EAGLEI_DATA_PATH = utils.join_paths(DYNAMIC_RYTHMS_DATA_PATH, 'eaglei_data')
STORM_EVENTS_CLEANED_PATH = utils.join_paths(INTERIM_DATA_PATH, 'storm_events_cleaned.csv')
STORM_OUTAGES = utils.join_paths(INTERIM_DATA_PATH, 'storm_outages_2014_2023.parquet')

warnings.filterwarnings('ignore')


def get_outages_index(outages_county):
    ''' This function groups outages depending on their separation

    :param outages_county:
    :return:
    '''
    outages_county = outages_county.sort_values("run_start_time")
    # Then we keep only the relevant outage (affecting a high amount of customers)
    outages_county = outages_county[outages_county.customers_out >= CUSTOMERS_OUT_NB]
    # We can define a separation of continuity to "divide" timelapses, in other words, separate outages events.
    # We calculate the difference in seconds of each outage
    outages_county['second_difference'] = outages_county.run_start_time.diff().dt.total_seconds()
    # Each time we find an interval mark greater than the separation time (defined in separation_hours),
    # we identify it as true (1), or false (0).
    separation_hours_seconds = MIN_OUTAGE_SECONDS * NB_15_MIN_IN_HOUR * (SEPPARATION_HOURS)
    outages_county['interval_mark'] = (
            outages_county.second_difference.fillna(MIN_OUTAGE_SECONDS) >= separation_hours_seconds
    ).astype(int)
    # then we do the cumulative sum to "generate an index" of same representation.
    outages_county['outage_index'] = outages_county['interval_mark'].cumsum()
    return outages_county


def process_outages(outages):
    """ Function that process outages information, creating an outage_index_resumed dataframe.

    :param outages:
    :return:
    """
    # Get the corresponding index.
    outages_index = outages.groupby('sub_general_id').apply(get_outages_index)
    # Generate an id for each index.
    outages_index['outage_index_id'] = (
            outages_index.fips_code_id + '__' + outages_index.outage_index.astype(str).str.zfill(4)
    )
    # Groupby the index, which depends on separation by time to identify if an outage belongs to the same event or to
    # another one
    outages_index_resumed = outages_index.groupby(
        'outage_index_id'
    ).agg(
        fips_code=('fips_code', 'first'),
        fips_code_id=('fips_code_id', 'first'),
        county=('county', 'first'),
        state=('state', 'first'),
        state_id=('state_id', 'first'),
        total_relevant_registers=('customers_out', 'count'),
        total_customers_out=('customers_out', 'sum'),
        run_start_time_min=('run_start_time', 'min'),
        run_start_time_max=('run_start_time', 'max'),
    ).reset_index()

    outages_index_resumed['run_start_time_max'] = (
            outages_index_resumed['run_start_time_max'] + pd.to_timedelta(900, unit='s')
    )
    outages_index_resumed['outage_duration'] = (
            outages_index_resumed.run_start_time_max - outages_index_resumed.run_start_time_min
    ).dt.total_seconds() / 3600 / 24  # We get the days of time difference
    outages_index_resumed['outage_customers_over_duration'] = (
            outages_index_resumed['total_customers_out'] / outages_index_resumed['outage_duration']
    )
    outages_index_resumed['state'] = outages_index_resumed['state'].str.lower()
    return outages_index_resumed


def process_storm_events(storm_events):
    ''' Function that processess the storm events by grouping them and finally yielding storm episodes by county.

    :param storm_events:
    :return:
    '''
    begin_datetime = (
        storm_events['BEGIN_YEARMONTH'].astype(str) +
        storm_events['BEGIN_DAY'].astype(str).str.zfill(2) +
        storm_events['BEGIN_TIME'].astype(str).str.zfill(4)
    )

    end_datetime = (
        storm_events['END_YEARMONTH'].astype(str) +
        storm_events['END_DAY'].astype(str).str.zfill(2) +
        storm_events['END_TIME'].astype(str).str.zfill(4)
    )

    storm_events['BEGIN_DATETIME'] = pd.to_datetime(begin_datetime, format='%Y%m%d%H%M')
    storm_events['END_DATETIME'] = pd.to_datetime(end_datetime, format='%Y%m%d%H%M')
    storm_events['DURATION_HOURS'] = (
            (storm_events['END_DATETIME'] - storm_events['BEGIN_DATETIME'])
    ).dt.total_seconds() / 3600

    storm_events['fips_code_id'] = storm_events.new_fips.astype(str).str.zfill(5)

    storm_episodes = storm_events.groupby(
        "EPISODE_ID"
    ).agg(
        nb_events=('EVENT_ID', 'count'),
        affected_states=('STATE', 'unique'),
        affected_states_ids=('STATE_FIPS', 'unique'),
        distinct_events=('EVENT_TYPE', 'unique'),
        fips_only_county_code_id=('CZ_FIPS', 'unique'),
        fips_code_id=('fips_code_id', 'unique'),
        touched_cz_names=('CZ_NAME', 'unique'),
        timezone=('CZ_TIMEZONE', 'unique'),
        episode_description=('EPISODE_NARRATIVE', 'first'),
        begin_datetime=('BEGIN_DATETIME', 'min'),
        end_datetime=('END_DATETIME', 'max'),
    ).reset_index()

    storm_episodes['storm_duration'] = (
                (storm_episodes.end_datetime - storm_episodes.begin_datetime).dt.total_seconds() / 3600
    ).replace(0, 0.01)
    storm_episodes['state'] = storm_episodes.affected_states.apply(lambda x: x[0]).str.lower()
    storm_events['fips_code_id'] = storm_events.new_fips.astype(str).str.zfill(5)
    storms_state_exploded = storm_episodes.explode('fips_code_id')
    storms_state_exploded['episode_fips_id'] = (
            storms_state_exploded.EPISODE_ID.astype(str)
            + '_'
            + storms_state_exploded.fips_code_id.astype(str).str.zfill(5)
    )
    storm_state_exploded_columns = [
        'EPISODE_ID',
        'fips_code_id',
        'episode_description',
        'begin_datetime',
        'end_datetime',
        'storm_duration',
        'episode_fips_id'
    ]
    storms_state_exploded = storms_state_exploded[storm_state_exploded_columns]
    return storms_state_exploded


def combining_outages_and_storms(storms_state_exploded, outages_index_resumed):
    storms_outages = storms_state_exploded.merge(
        outages_index_resumed,
        on='fips_code_id',
        how='left',
    )
    storms_outages['outage_start_minus_storm_start'] = (
        storms_outages['run_start_time_min'] - storms_outages['begin_datetime']
    ).dt.total_seconds() / 3600 / 24  # This constraint is what defines a "legan" join,

    storms_outages['outage_end_minus_storm_end'] = (
        storms_outages['run_start_time_max'] - storms_outages['end_datetime']
    ).dt.total_seconds() / 3600 / 24

    storms_outages['outage_start_minus_storm_end'] = (
        storms_outages['run_start_time_min'] - storms_outages['end_datetime']
    ).dt.total_seconds() / 3600 / 24

    storms_outages['outage_end_minus_storm_start'] = (
        storms_outages['run_start_time_max'] - storms_outages['begin_datetime']
    ).dt.total_seconds() / 3600 / 24

    storms_outages['storm_caused_outage_cond1'] = (storms_outages.outage_start_minus_storm_start >= 0)
    storms_outages['storm_caused_outage_cond2'] = (storms_outages.outage_start_minus_storm_end <= 0)
    storms_outages['storm_caused_outage_cond3'] = (
        storms_outages.outage_start_minus_storm_end.between(0, DAYS_AFTER_STORM_THRESHOLD)
    )

    storm_outages_conditions = (
            (storms_outages.storm_caused_outage_cond1 & storms_outages.storm_caused_outage_cond2) |
            (storms_outages.storm_caused_outage_cond1 & storms_outages.storm_caused_outage_cond3)
    )

    storms_outages.loc[storm_outages_conditions, 'storm_caused_outage'] = 1
    storms_outages.loc[~storm_outages_conditions, 'storm_caused_outage'] = 0
    storms_outages['episode_fips_id'] = (
            storms_outages.EPISODE_ID.astype(str) + '_' + storms_outages.fips_code_id.astype(str)
    )
    storms_caused_outages = storms_outages[storms_outages.storm_caused_outage == 1]
    storms_with_response_var = storms_state_exploded.merge(
        storms_caused_outages[
            ['storm_caused_outage',
             'episode_fips_id',
             'outage_index_id',
             'outage_start_minus_storm_start',
             'outage_end_minus_storm_end',
             'outage_start_minus_storm_end',
             'outage_end_minus_storm_start',
             'outage_duration',
             'run_start_time_min',
             'run_start_time_max',
             ]
        ],
        how='left',
        on='episode_fips_id'
    )
    storms_with_response_var.storm_caused_outage = storms_with_response_var.storm_caused_outage.fillna(0)
    return storms_with_response_var


def create_storm_caused_outage():
    # Read data
    storm_events = pd.read_csv(STORM_EVENTS_CLEANED_PATH)
    outages = utils.get_required_outages_dfs(
        2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023,
        eaglei_data_path=EAGLEI_DATA_PATH
    )
    # Process outage info
    print('Processing Outages...')
    outages_index_resumed = process_outages(outages=outages)
    # Process storms info
    print('Processing Storm events...')
    storms_state_exploded = process_storm_events(storm_events=storm_events)
    # Merge outages and storms, generate a dataframe if storm caused outage.
    print('Processing Merging Storm events and outages...')
    storms_with_response_var = combining_outages_and_storms(
        storms_state_exploded=storms_state_exploded,
        outages_index_resumed=outages_index_resumed,
    )
    print(f'Saving results into the following path: {STORM_OUTAGES}')
    storms_with_response_var.to_parquet(STORM_OUTAGES)


def execute():
    if not utils.check_if_filepath_exists(STORM_OUTAGES):
        create_storm_caused_outage()
    else:
        print(f'File already exists, it is located at: {STORM_OUTAGES}')


if __name__ == "__main__":
    execute()
