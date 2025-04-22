from pandas import DataFrame

RANGE = [1, 2, 3]


class OutageFeatures:
    """
     Creates all features for outage Model.
    """
    def __init__(self, data: DataFrame) -> None:
        self.df = data
        self.no_work_cols = [
            'time',
            'episode_fips_id',
            'meteorological_current_datetime_val',
            'outage_in_an_hour',
            'hours_to_outage',
            'day_of_year',
            'hour_of_day',
            'day_of_week',
            'month_of_year',
            'coord0',
            'coord1',
            'coord2',
        ]
        self.work_cols = [
            col for col in self.df.columns if col not in self.no_work_cols
        ]

    def get_diff_features(self) -> None:
        """Generates features about:
        - Diference between temperature at 10 meters and at 2 meters.
        - Diference between Wind speed at 50 meters and at 2 meters.

        :return:
        """
        self.df['diff_between_t10m_t2m'] = self.df['T10M'] - self.df['T2M']
        self.df['diff_between_t50m_t2m'] = self.df['WS50M'] - self.df['WS2M']

    def get_feature_previous_n_hours(self, col: str, n: int) -> None:
        """For a given column, it generates a new column with the values for the previous n hours.

        :param col: str Column to be calculated
        :param n: Hours back to get the value
        :return:
        """
        self.df[f'{col}_{n}_hours_ago'] = self.df.groupby('episode_fips_id')[col].shift(n)

    def get_delta_featues(self, col: str) -> None:
        """For a given column it calculates:
        - Diference between the current value vs the value n hours before (in this case 1, 2 and 3 hours)
        - Diference between value hours (i.e. the value 1 hour before vs the value 3 hours before)

        :param col: str Column to calculate the deltas
        :return:
        """
        self.df[f'{col}_delta_one_hour'] = self.df[col] - self.df[f'{col}_1_hours_ago']
        self.df[f'{col}_delta_two_hour'] = self.df[col] - self.df[f'{col}_2_hours_ago']
        self.df[f'{col}_delta_three_hour'] = self.df[col] - self.df[f'{col}_3_hours_ago']
        self.df[f'{col}_delta_previous'] = self.df[f'{col}_1_hours_ago'] - self.df[f'{col}_2_hours_ago']
        self.df[f'{col}_delta_two_previous'] = self.df[f'{col}_2_hours_ago'] - self.df[f'{col}_3_hours_ago']

    def get_tendency_features(self, col: str) -> None:
        """Calculate the tendency of the values with the follow logic:
        - if the value increase the tendency will be equal to 1
        - if the value mantains the tendency will be equal to 0
        - if the value decrease the tendency will be equal to 0
        :param col: str Column to calculate features.
        :return:
        """
        def tendency_func(x):
            return 1 if x > 0 else (-1 if x < 0 else 0)

        self.df[f'{col}_previous_tendency'] = self.df[f'{col}_delta_previous'].apply(tendency_func)
        self.df[f'{col}_two_previous_tendency'] = self.df[f'{col}_delta_two_previous'].apply(tendency_func)
        self.df[f'{col}_current_tendency'] = self.df[f'{col}_delta_one_hour'].apply(tendency_func)

    def get_features(self) -> DataFrame:
        """Create all features for the outage model.

        :return: DataFrame: dataframe with all the features calculated.
        """
        self.df.sort_values(
            by=['episode_fips_id', 'meteorological_current_datetime_val'],
            inplace=True
        )
        self.get_diff_features()
        for col in self.work_cols:
            for ix in RANGE:
                self.get_feature_previous_n_hours(col, ix)
            self.get_delta_featues(col)
            self.get_tendency_features(col)

        return self.df
