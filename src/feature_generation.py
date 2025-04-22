import pandas as pd
from pandas import DataFrame

class OutageFeatures:
    """
     Creates all features for outage Model.
    """
    def __init__(self, data:DataFrame) -> None:
        self.df = data
        self.no_work_cols = ['time', 
                             'episode_fips_id', 
                             'meteorological_current_datetime_val', 
                             'outage_in_an_hour', 
                             'hours_to_outage']
        self.work_cols = [col for col in self.df.columns
                          if col not in self.no_work_cols]

    def get_diff_features(self)->None:
        """Generates features about:
            - Absolute diference between temperature at 10 meters 
                and temperature at 2 meteres.
            - Absolute diference between wind speed at 50 meters
                and wind speed at 2 meters. 
        """
        self.df['abs_diff_between_t2m_t10m'] = self.df.apply(lambda row: abs(row['T2M'] \
                                                                        - row['T10M']), axis=1)
        self.df['abs_diff_between_ws50m_wd2m'] = self.df.apply(lambda row: abs(row['WS50M'] \
                                                                        - row['WS2M']), axis=1)

    def get_feature_previous_n_hours(self, col:str, n:int)-> None:
        """For a given column, it generates a new column
           with the values for the previous n hours. 

        Args:
            col (str): column to be calculate
            n (int): hours back to get the value 
        """
        self.df[f'{col}_{n}_hours_ago'] = self.df.groupby('episode_fips_id')[col].shift(n)

    def get_delta_featues(self, col:str)-> None:
        """For a given column it calculate:
            - Diference between the current value vs the value 
            n hours before (in this case 1, 2 and 3 hours)
            - Diference between value hours (i.e. the value 1 hour before vs the value 3 hours before)

        Args:
            col (str): column to calculate the deltas.
        """
        self.df[f'{col}_delta_one_hour'] = self.df.apply(lambda row: row[col] \
                                                        - row[f'{col}_an_hour_ago'], axis=1)
        self.df[f'{col}_delta_two_hour'] = self.df.apply(lambda row: row[col] \
                                                        - row[f'{col}_two_hours_ago'], axis=1)
        self.df[f'{col}_delta_three_hour'] = self.df.apply(lambda row: row[col] \
                                                     - row[f'{col}_three_hours_ago'], axis=1)
        self.df[f'{col}_delta_previous'] = self.df.apply(lambda row: row[f'{col}_an_hour_ago'] \
                                                        - row[f'{col}_two_hours_ago'], axis=1)
        self.df[f'{col}_delta_two_previous'] = self.df.apply(lambda row:
                                                                row[f'{col}_two_hours_ago'] \
                                                        - row[f'{col}_three_hours_ago'], axis=1)

    def get_tendency_features(self, col:str) -> None:
        """Calculate the tendency of the values with the follow logic:
           - if the value increase the tendency will be equal to 1 
           - if the value mantains the tendency will be equal to 0
           - if the value decrease the tendency will be equal to 0 

        Args:
            col (str): Column to calculate features.
        """
        self.df[f'{col}_previous_tendency'] = self.df[f'{col}_delta_previous'].apply(
                                                    lambda x: 1 if x> 0 else  (-1 if x <0 else 0))
        self.df[f'{col}_two_previous_tendency'] = self.df[f'{col}_delta_two_previous'].apply(
                                                    lambda x: 1 if x> 0 else  (-1 if x <0 else 0))
        self.df[f'{col}_current_tendency'] = self.df[f'{col}_delta_one_hour'].apply(
                                                    lambda x: 1 if x> 0 else  (-1 if x <0 else 0))

    def get_features(self)->DataFrame:
        """Create all features for the outage model. 

        Returns:
            DataFrame: dataframe with all the features calculated. 
        """
        self.df.sort_values(by=['episode_fips_id',
                                'meteorological_current_datetime_val'], 
                                inplace=True)
        self.get_diff_features()
        for col in self.work_cols:
            for ix in range(1, 3):
                self.get_feature_previous_n_hours(col, ix)
            self.get_delta_featues(col)
            self.get_tendency_features(col)

        return self.df
