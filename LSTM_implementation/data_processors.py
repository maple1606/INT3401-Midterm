import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class DataGrouper:
    """
    This class supports grouping data before 
    """
    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters

    def cluster_df(self, df: pd.DataFrame):
        model = KMeans(n_clusters=self.n_clusters)
        rc_df = df[['row', 'col']].groupby(
            ['row', 'col']).count().reset_index()
        model.fit(rc_df)
        df['cluster_no'] = model.predict(df[['row', 'col']])

    def get_grouped_data(self, df: pd.DataFrame):
        self.cluster_df(df)
        data_clusters = []
        for i in range(self.n_clusters):
            coord_df_map = dict()
            cluster_df = df[df['cluster_no'] == i]
            row_col_combination = cluster_df[['row', 'col']].groupby(
                ['row', 'col']).count().reset_index()
            for j in range(len(row_col_combination)):
                row, col = row_col_combination.iloc[j]['row'], row_col_combination.iloc[j]['col']
                row_col_df = cluster_df[(
                    cluster_df['row'] == row & cluster_df['col'] == col)]
                if not row_col_df.empty:
                    row_col_df = row_col_df.sort_values(by=['date'])
                    row_col_df = row_col_df.drop(
                        columns=['cluster_no', 'row', 'col'])
                    row_col_df = row_col_df.reset_index(drop=True)
                    coord_df_map[tuple(row, col)] = row_col_df
            data_clusters.append(coord_df_map)
        return data_clusters


class TimeSeriesProcessor:
    def __init__(self, time_step: int = 1, horizon: int = 1):
        self.time_step = time_step
        self.horizon = horizon

    def find_continuous(self, df: pd.DataFrame):
        continuous_index = []
        start = 0
        for i in range(1, len(df)):
            if (df.iloc[i]['date'] - df.iloc[i - 1]['date']) == pd.Timedelta(hours=1):
                continue
            else:
                continuous_index.append(tuple([start, i]))
                start = i
        continuous_index.append(tuple([start, len(df)]))
        # Drop date for training (Maybe use a more flexible implementation in training method?)
        continuous_df = [df.iloc[start:end].drop(
            columns=['date']) for start, end in continuous_index]
        print(f"find_continuous: len(continuous_df): {len(continuous_df)}")
        return continuous_df

    def create_sequence_data(self, df: pd.DataFrame,
                             target_col: str = 'aws'):
        if (self.time_step < 1 or self.horizon < 1):
            raise ValueError(
                "time_step and forward step must be greater than 0")

        if (self.time_step + (self.horizon - 1) >= len(df) - 1):
            raise ValueError(
                f"time_step ({self.time_step}) must be less than the length of the DataFrame({len(df)})")
        X = []
        y = []
        for i in range(len(df) - (self.time_step + (self.horizon - 1))):
            X.append(df.iloc[i:i + self.time_step])
            y.append(df.iloc[i + self.time_step + (self.horizon - 1)][target_col])

        X = np.array(X)
        y = np.array(y)
        print(f"create_sequence_data: X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def get_data(self, df: pd.DataFrame):
        final_X = []
        final_y = []
        continuous_dataframes = self.find_continuous(df)
        for dataframe in continuous_dataframes:
            try:
                X, y = self.create_sequence_data(
                    dataframe, self.time_step, self.horizon)
                for i in range(len(X)):
                    final_X.append(X[i])
                    final_y.append(y[i])
            except ValueError:
                pass

        final_X = np.array(final_X)
        final_y = np.array(final_y)
        print(f"get_data: final_X shape: {
            final_X.shape}, final_y shape: {final_y.shape}")
        return final_X, final_y
