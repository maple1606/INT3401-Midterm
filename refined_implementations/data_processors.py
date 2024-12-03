import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import skfuzzy as fuzz
import pandas.api.types as ptypes
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


class DataGrouper:
    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters

    def knn_cluster_df(self, df: pd.DataFrame):
        model = KMeans(n_clusters=self.n_clusters)
        rc_df = df[['row', 'col']].groupby(
            ['row', 'col']).count().reset_index()
        model.fit(rc_df)
        df['cluster_no'] = model.predict(df[['row', 'col']])

    def fcm_cluster_df(self, df: pd.DataFrame):
        data_points = df[['row', 'col']].values.T
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data_points, self.n_clusters, 2, error=0.005, maxiter=1000, init=None)
        cluster_no = np.argmax(u, axis=0)
        df['cluster_no'] = cluster_no

    def get_grouped_data(self, df: pd.DataFrame):
        """
        Returns: A LIST of dictionaries (with key of row number and col number for time-series processing), where each dictionary contains the dataframes of each cluster. E.g. [cluster1: dict, cluster2: dict, ...]
        """
        data_clusters = []
        for i in range(self.n_clusters):
            coord_df_map = dict()
            cluster_df = df[df['cluster_no'] == i]
            row_col_combination = cluster_df[['row', 'col']].groupby(
                ['row', 'col']).count().reset_index()
            for j in range(len(row_col_combination)):
                row, col = row_col_combination.iloc[j]['row'], row_col_combination.iloc[j]['col']
                row_col_df = cluster_df[(
                    cluster_df['row'] == row) & (cluster_df['col'] == col)]
                if not row_col_df.empty:
                    # SORT BY DATE
                    row_col_df = row_col_df.sort_values(by=['date'])
                    row_col_df = row_col_df.drop(
                        columns=['cluster_no', 'row', 'col'])
                    row_col_df = row_col_df.reset_index(drop=True)
                    coord_df_map[tuple([row, col])] = row_col_df
            data_clusters.append(coord_df_map)
        #     print(f"get_grouped_data: len(coord_df_map): {len(coord_df_map)}")
        # print(f"get_grouped_data: len(data_clusters): {len(data_clusters)}")
        return data_clusters

    def get_grouped_data_no_cluster(self, df: pd.DataFrame):
        """
        Returns: A dictionary of dataframes, key = coord, value = df
        """
        coord_df_map = dict()
        row_col_combination = row_col_combination = df[['row', 'col']].groupby(
            ['row', 'col']).count().reset_index()
        for i in range(len(row_col_combination)):
            row, col = row_col_combination.iloc[i]['row'], row_col_combination.iloc[i]['col']
            row_col_df = df[(df['row'] == row) & (df['col'] == col)]
            if not row_col_df.empty:
                # SORT BY DATE
                row_col_df = row_col_df.sort_values(by=['date'])
                row_col_df = row_col_df.drop(
                    columns=['row', 'col'])
                row_col_df = row_col_df.reset_index(drop=True)
                coord_df_map[tuple([row, col])] = row_col_df
        return coord_df_map


class TimeSeriesProcessor:
    """
    2 main function:
    Lagged data - for shifting the target column (for traditional ML which can't handle time-series)
    Sequence data - for LSTM, GRU, etc.
    """

    def __init__(self, time_step: int = 1, future_step: int = 1):
        self.time_step = time_step
        self.future_step = future_step

    def find_continuous(self, df: pd.DataFrame):
        """Return a list of continuous dataframes"""
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
        continuous_dfs = [df.iloc[start:end].drop(
            columns=['date']) for start, end in continuous_index]
        # print(f"find_continuous: len(continuous_dfs): {len(continuous_dfs)}")
        return continuous_dfs

    def create_sequence_data(self, df: pd.DataFrame, target_col: str = 'aws'):
        if (self.time_step < 1 or self.future_step < 1):
            raise ValueError(
                "time_step and forward step must be greater than 0")

        if (self.time_step + (self.future_step - 1) >= len(df)):
            raise ValueError(
                f"time_step + future_step - 1 = {self.time_step + self.future_step - 1} must be less than the length of the DataFrame({len(df)})")
        X = []
        y = []
        for i in range(len(df) - (self.time_step + (self.future_step - 1))):
            X.append(df.iloc[i:i + self.time_step])
            y.append(df.iloc[i + self.time_step +
                     (self.future_step - 1)][target_col])

        X = np.array(X)
        y = np.array(y)
        # print(f"create_sequence_data: X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def get_sequence_data(self, df: pd.DataFrame):
        final_X = []
        final_y = []
        continuous_dataframes = self.find_continuous(df)
        for dataframe in continuous_dataframes:
            try:
                X, y = self.create_sequence_data(
                    dataframe)
                for i in range(len(X)):
                    final_X.append(X[i])
                    final_y.append(y[i])
            except ValueError as e:
                # print(e)
                continue

        final_X = np.array(final_X)
        final_y = np.array(final_y)
        # print(f"get_data: final_X shape: {
        #       final_X.shape}, final_y shape: {final_y.shape}")
        return final_X, final_y

    def get_sequence_data_from_coord_df_map(self, df_dict):
        X = []
        y = []
        for key in df_dict.keys():
            df = df_dict[key]
            # print(f"df.shape: {df.shape}")
            X_temp, y_temp = self.get_sequence_data(df)
            # Watch out for empty data
            if not (X_temp.size > 0 or y_temp.size > 0):
                continue
            X.append(X_temp)
            y.append(y_temp)
            # print(f"X: {X_temp.shape}, y: {y_temp.shape}")
        # Concatenate all the data
        X = np.concatenate(X)
        y = np.concatenate(y)
        if X.size == 0 or y.size == 0:
            raise ValueError("X or y is empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        return X, y

    def create_lagged_data(self, df: pd.DataFrame, target_col: str = 'aws'):
        if (self.time_step < 1 or self.future_step < 1):
            raise ValueError(
                "time_step and forward step must be greater than 0")
        if (self.time_step + (self.future_step - 1) >= len(df) - 1):
            raise ValueError(
                f"time_step + future_step - 1 = {self.time_step + self.future_step - 1} must be less than the length of the DataFrame({len(df)})")
        df_lagged = df.copy().reset_index(drop=True)
        for lag in range(1, self.time_step):
            df_lagged[f'{target_col}_lag_{lag}'] = df_lagged[target_col].shift(lag)
        # print(f"create_lagged_data-df.shape:{df.shape}")
        # print(f"create_lagged-data-df_lagged.shape{df_lagged.shape}")
        df_lagged = df_lagged.iloc[self.time_step:]
        X = []
        y = []
        for i in range(len(df_lagged) - self.future_step):
            X.append(df_lagged.iloc[i])
            y.append(df_lagged.iloc[i + self.future_step][target_col])

        X = np.array(X)
        y = np.array(y)
        # del df_lagged  # release memory
        # gc.collect()
        # print(f"create_lagged_data: X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def get_lagged_data(self, df: pd.DataFrame):
        final_X = []
        final_y = []
        continuous_dataframes = self.find_continuous(df)
        for dataframe in continuous_dataframes:
            try:
                X, y = self.create_lagged_data(
                    dataframe)
                for i in range(len(X)):
                    final_X.append(X[i])
                    final_y.append(y[i])
            except ValueError as e:
                # print(e)
                continue

        final_X = np.array(final_X)
        final_y = np.array(final_y)
        # print(f"get_data: final_X shape: {
        #       final_X.shape}, final_y shape: {final_y.shape}")
        return final_X, final_y

    def get_lagged_data_from_coord_df_map(self, df_dict):
        """
        Returns X and y for a cluster.
        """
        X = []
        y = []
        for key in df_dict.keys():
            df = df_dict[key]
            # print(f"df.shape: {df.shape}")
            X_temp, y_temp = self.get_lagged_data(df)
            # Watch out for empty data
            if not (X_temp.size > 0 and y_temp.size > 0):
                continue
            X.append(X_temp)
            y.append(y_temp)
            # print(f"X: {X_temp.shape}, y: {y_temp.shape}")
        # print(len(X))
        # print(len(y))
        # Concatenate all the data
        X = np.concatenate(X)
        y = np.concatenate(y)
        if X.size == 0 or y.size == 0:
            raise ValueError("X or y is empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        return X, y


class GetData:
    @staticmethod
    def get_non_cluster_lagged_data(train_df: pd.DataFrame,
                                    test_df: pd.DataFrame,
                                    time_step: int = 1,
                                    future_step: int = 1,
                                    target_col: str = 'aws'):
        data_grouper = DataGrouper()
        time_series_processor = TimeSeriesProcessor(
            time_step=time_step, future_step=future_step)
        train_coord_df_map = data_grouper.get_grouped_data_no_cluster(train_df)
        test_coord_df_map = data_grouper.get_grouped_data_no_cluster(test_df)
        X_train, y_train = time_series_processor.get_lagged_data_from_coord_df_map(
            train_coord_df_map)
        X_test, y_test = time_series_processor.get_lagged_data_from_coord_df_map(
            test_coord_df_map)
        return X_train, y_train, X_test, y_test

    def get_non_cluster_sequence_data(train_df: pd.DataFrame,
                                      test_df: pd.DataFrame,
                                      time_step: int = 1,
                                      future_step: int = 1,
                                      target_col: str = 'aws'):
        data_grouper = DataGrouper()
        time_series_processor = TimeSeriesProcessor(
            time_step=time_step, future_step=future_step)
        train_coord_df_map = data_grouper.get_grouped_data_no_cluster(
            train_df)
        test_coord_df_map = data_grouper.get_grouped_data_no_cluster(
            test_df)
        X_train, y_train = time_series_processor.get_sequence_data_from_coord_df_map(
            train_coord_df_map)
        X_test, y_test = time_series_processor.get_sequence_data_from_coord_df_map(
            test_coord_df_map)
        return X_train, y_train, X_test, y_test


class OutlierProcessor:
    pass


class ImbalanceProcessor:
    """If X is 3D, takes the last dimension as the feature dimension"""

    def __init__(self, model):
        """
        Args:
                model: Imbalanced-learning model, imported from imblearn.over_sampling
                or imblearn.under_sampling, or imblearn.combine
        """
        self.model = model

    def process_imbalance(self, X, y):
        """
        Even if X is 3D, the last dimension is taken as the feature dimension
        Returns:
                X_resampled: Resampled X (2D either if X is 2D or 3D)
                y_resampled: Resampled y
        """
        if (len(X.shape) == 2):
            X_resampled, y_resampled = self.model.fit_resample(X, y)
            return X_resampled, y_resampled
        if (len(X.shape) == 3):
            X_resampled, y_resampled = self.model.fit_resample(X[:, -1, :], y)
            return X_resampled, y_resampled
        else:
            raise ValueError(f"X must be 2D or 3D, got {len(X.shape)}D")


class DataScaler:
    def __init__(self, target_col: str = 'aws'):
        """
        Args:
                scaler: Scaler object, imported from sklearn.preprocessing
        """
        self.target_col = target_col

    def scale_data(self, train_df, test_df, exclude_cols=['date', 'row', 'col', 'cluster_no']):
        # For every column in the data frame, apply each column's a scaler
        scalers = dict()  # column name: scaler
        columns_name = train_df.columns.tolist()
        for col in columns_name:
            if (not ptypes.is_numeric_dtype(train_df[col])) or (col in exclude_cols):
                continue
            else:
                print(col)
                scaler = MinMaxScaler()
                train_df.loc[:, col] = scaler.fit_transform(train_df[[col]])
                test_df.loc[:, col] = scaler.transform(test_df[[col]])
                scalers[col] = scaler
        self.target_scaler = scalers[self.target_col]
