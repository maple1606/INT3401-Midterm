
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from dotenv import load_dotenv


load_dotenv()


import sys

IMPL_PATH = os.getenv("IMPL_PATH")
sys.path.append(IMPL_PATH)
DATA_PATH = IMPL_PATH + "/data/"
print(DATA_PATH)

import data_processors as dp
from imblearn.combine import SMOTEENN

def train_with_config(TIME_STEP, FUTURE_STEP, N_CLUSTERS):
 
# ### Training configs:
# 1. XGB classifier, Random Forest Classifier
# 2. Random Forest regression, MLP regression

# ### Load data from pickle files (made from 1_group_data.ipynb)
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
    ts_processor = dp.TimeSeriesProcessor(time_step=TIME_STEP, future_step=FUTURE_STEP)
    train_4_clustered_list = []
    test_4_clustered_list = []
    train_10_clustered_list = []
    test_10_clustered_list = []

    for cluster in range(N_CLUSTERS):
        train_4_clustered_list.append(ts_processor.get_lagged_data_for_cluster(train_4_clusters[cluster]))
        test_4_clustered_list.append(ts_processor.get_lagged_data_for_cluster(test_4_clusters[cluster]))
        train_10_clustered_list.append(ts_processor.get_lagged_data_for_cluster(train_10_clusters[cluster]))
        test_10_clustered_list.append(ts_processor.get_lagged_data_for_cluster(test_10_clusters[cluster]))


    # test for april training data
    print(len(train_4_clustered_list)) # 4 Cluster
    train_4_cluster_0 = train_4_clustered_list[0] # lots of train data
    X, y = train_4_cluster_0
    print(X.shape)
    print(y.shape)

    # Count number of nan values
    print(np.sum(np.isnan(X)))
    print(np.sum(np.isnan(y)))

    # Count number of 0 values
    print(np.sum(X == 0))
    print(np.sum(y == 0))

    
    # ### Classification
    # #### Encode y for training and test sets
def main():
    time_step = [3, 6, 12]
    future_step = [1, 3, 6]

    for ts in time_step:
        for fs in future_step:
            train_with_config(ts, fs, 4)

if __name__ == "__main__":
    main()