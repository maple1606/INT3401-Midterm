import os
from dotenv import load_dotenv

load_dotenv()


import sys

IMPL_PATH = os.getenv("IMPL_PATH")
sys.path.append(IMPL_PATH)
DATA_PATH = IMPL_PATH + "/data/"
print(DATA_PATH)


# ### PARAMS



# ### Training configs:
# 1. XGB classifier, Random Forest Classifier
# 2. Random Forest regression, MLP regression


# ### Load data from pickle files (made from 1_group_data.ipynb)

import pickle
import pandas as pd
import numpy as np

train_4_clusters = pickle.load(open(DATA_PATH + "train_4_clusters.pkl", "rb"))
test_4_clusters = pickle.load(open(DATA_PATH + "test_4_clusters.pkl", "rb"))
train_10_clusters = pickle.load(open(DATA_PATH + "train_10_clusters.pkl", "rb"))
test_10_clusters = pickle.load(open(DATA_PATH + "test_10_clusters.pkl", "rb"))


# More insights
print(type(train_4_clusters))
print(type(train_4_clusters[0]))
print(type(train_4_clusters[0][list(train_4_clusters[0].keys())[0]]))
train_4_clusters[0][list(train_4_clusters[0].keys())[0]].head()


# ### For each month, for each cluster, convert the dictionary into corresponding X and y
def train_with_configs(TIME_STEP, FUTURE_STEP, N_CLUSTERS):
    import data_processors as dp
    ts_processor = dp.TimeSeriesProcessor(time_step=TIME_STEP, future_step=FUTURE_STEP)
    train_4_clustered_list = []
    test_4_clustered_list = []
    train_10_clustered_list = []
    test_10_clustered_list = []

    import data_processors as dp
    ts_processor = dp.TimeSeriesProcessor(time_step=TIME_STEP, future_step=FUTURE_STEP)
    train_4_clustered_list = []
    test_4_clustered_list = []
    train_10_clustered_list = []
    test_10_clustered_list = []

    for cluster in range(N_CLUSTERS):
        train_4_clustered_list.append(ts_processor.get_sequence_data_for_cluster(train_4_clusters[cluster]))
        test_4_clustered_list.append(ts_processor.get_sequence_data_for_cluster(test_4_clusters[cluster]))
        train_10_clustered_list.append(ts_processor.get_sequence_data_for_cluster(train_10_clusters[cluster]))
        test_10_clustered_list.append(ts_processor.get_sequence_data_for_cluster(test_10_clusters[cluster]))

    # test for april training data
    print(len(train_4_clustered_list)) # 4 Cluster
    train_4_cluster_0 = train_4_clustered_list[0] # lots of train data
    print(type(train_4_cluster_0))
    X, y = train_4_cluster_0
    print(X.shape)
    print(y.shape)

    # Count number of nan values
    print(np.sum(np.isnan(X)))
    print(np.sum(np.isnan(y)))

    # Count number of 0 values
    print(np.sum(X == 0))
    print(np.sum(y == 0))


    # ### Normalize X features

    from imblearn.under_sampling import RandomUnderSampler
    for i in range(N_CLUSTERS):
        for dataset_list in [train_4_clustered_list, train_10_clustered_list]:
            X_train, y_train = dataset_list[i]
            n_samples, time_steps, n_features = X_train.shape
            X_train_flat = X_train.reshape((n_samples, time_steps * n_features))
            y_train_binary = (y_train > 0).astype(int).ravel()  # Ensure y_train_binary is 1D
            rus = RandomUnderSampler()
            X_resampled_flat, y_resampled_binary = rus.fit_resample(X_train_flat, y_train_binary)
            resampled_indices = rus.sample_indices_
            y_resampled = y_train[resampled_indices]
            X_resampled = X_resampled_flat.reshape((-1, time_steps, n_features))
            if (len(y_resampled.shape) == 1):
                raise ValueError("y_resampled shape is not 1D, current shape: ", y_resampled.shape)
            dataset_list[i] = (X_resampled, y_resampled)

    
    from sklearn.preprocessing import MinMaxScaler
    for i in range(N_CLUSTERS):
        X_train_4, y_train_4 = train_4_clustered_list[i]
        X_test_4, y_test_4 = test_4_clustered_list[i]
        X_train_10, y_train_10 = train_10_clustered_list[i]
        X_test_10, y_test_10 = test_10_clustered_list[i]
        scaler = MinMaxScaler()
        data_scaler = dp.DataScaler(scaler)
        X_train_4, X_test_4 = data_scaler.scale_data(X_train_4, X_test_4)
        X_train_10, X_test_10 = data_scaler.scale_data(X_train_10, X_test_10)
        train_4_clustered_list[i] = (X_train_4, y_train_4)
        train_10_clustered_list[i] = (X_train_10, y_train_10)

    import matplotlib.pyplot as plt
    def draw_scatter_plot(y_true, y_pred, title, model_name, month, cluster, optional=""):
        plt.scatter(y_true, y_pred)
        plt.title(title)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.savefig(RESULT_PATH + f"{model_name}_{month}-ts{TIME_STEP}_fs{FUTURE_STEP}_cluster_{cluster}{optional}.png")
        plt.show()


    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping


    def create_lstm_model(time_step, no_features):
        """one hidden layer with 100 nodes and is trained with a learning rate of 9 × 10−5"""
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(time_step, no_features)),
            tf.keras.layers.LSTM(256, return_sequences=True, activation='relu'),
            tf.keras.layers.Dense(1, activation='relu')
        ])
        return model


    april_models_regression = []
    october_models_regression = []

    for cluster in range(N_CLUSTERS):
        X_4, y_4 = train_4_clustered_list[cluster]
        X_10, y_10 = train_10_clustered_list[cluster]
        model_4 = create_lstm_model(TIME_STEP, X_4.shape[2])
        model_10 = create_lstm_model(TIME_STEP, X_10.shape[2])
        model_4.compile(optimizer=Adam(learning_rate=9e-5), loss='mse')
        model_10.compile(optimizer=Adam(learning_rate=9e-5), loss='mse')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_4.fit(X_4, y_4, epochs=100, validation_split=0.2, callbacks=[early_stopping])
        model_10.fit(X_10, y_10, epochs=100, validation_split=0.2, callbacks=[early_stopping])
        april_models_regression.append(model_4)
        october_models_regression.append(model_10)


    RESULT_PATH = IMPL_PATH + f"/results/LSTM/ts{TIME_STEP}_fs{FUTURE_STEP}/"
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    def draw_scatter_plot(y_true, y_pred, title, model_name, month, cluster, optional=""):
        plt.scatter(y_true, y_pred)
        plt.title(title)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.savefig(RESULT_PATH + f"{model_name}_{month}-ts{TIME_STEP}_fs{FUTURE_STEP}_cluster_{cluster}{optional}.png")
        plt.close()

    from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error



    april_predictions_scores_reg = dict()
    october_predictions_scores_reg = dict()

    for cluster in range(N_CLUSTERS):
        X_test_4, y_test_4 = test_4_clustered_list[cluster]
        X_test_10, y_test_10 = test_10_clustered_list[cluster]
        model_4 = april_models_regression[cluster]
        model_10 = october_models_regression[cluster]
        y_pred_4 = model_4.predict(X_test_4)
        y_pred_10 = model_10.predict(X_test_10)
        april_predictions_scores_reg[cluster] = {
            "rmse": root_mean_squared_error(y_test_4, y_pred_4),
            "r2": r2_score(y_test_4, y_pred_4),
            "mae": mean_absolute_error(y_test_4, y_pred_4)
        }
        october_predictions_scores_reg[cluster] = {
            "rmse": root_mean_squared_error(y_test_10, y_pred_10),
            "r2": r2_score(y_test_10, y_pred_10),
            "mae": mean_absolute_error(y_test_10, y_pred_10)
        }
        draw_scatter_plot(y_test_4, y_pred_4, f"LSTM Regressor cluster {cluster}", "LSTM", "April", cluster, "_regression")
        draw_scatter_plot(y_test_10, y_pred_10, f"LSTM Regressor cluster {cluster}", "LSTM", "October", cluster, "_regression")


    # Save the results
    april_predictions_scores_reg_df = pd.DataFrame(april_predictions_scores_reg, index=["rmse", "r2", "mae"])
    october_predictions_scores_reg_df = pd.DataFrame(october_predictions_scores_reg, index=["rmse", "r2", "mae"])

    april_predictions_scores_reg_df.to_csv(RESULT_PATH + f"april_predictions_scores_reg_df_ts{TIME_STEP}_fs{FUTURE_STEP}.csv")
    october_predictions_scores_reg_df.to_csv(RESULT_PATH + f"october_predictions_scores_reg_df_ts{TIME_STEP}_fs{FUTURE_STEP}.csv")

def main():
    for time_step in [3, 6, 12]:
        for future_step in [1, 3, 6]:
            train_with_configs(time_step, future_step, 4)

if __name__ == "__main__":
    main()