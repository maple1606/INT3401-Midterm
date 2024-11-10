
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


    train_4_clustered_list_classification = []
    test_4_clustered_list_classification = []
    train_10_clustered_list_classification = []
    test_10_clustered_list_classification = []

    for cluster in range(N_CLUSTERS):
        X_4_train, y_4_train = train_4_clustered_list[cluster]
        X_4_test, y_4_test = test_4_clustered_list[cluster]
        X_10_train, y_10_train = train_10_clustered_list[cluster]
        X_10_test, y_10_test = test_10_clustered_list[cluster]

        y_4_train = np.where(y_4_train > 0, 1, 0)
        y_4_test = np.where(y_4_test > 0, 1, 0)
        y_10_train = np.where(y_10_train > 0, 1, 0)
        y_10_test = np.where(y_10_test > 0, 1, 0)

        train_4_clustered_list_classification.append([X_4_train, y_4_train])
        test_4_clustered_list_classification.append([X_4_test, y_4_test])
        train_10_clustered_list_classification.append([X_10_train, y_10_train])
        test_10_clustered_list_classification.append([X_10_test, y_10_test])

    
    # #### Oversampling for training sets




    train_4_clustered_list_resampled_smoteenn = []
    train_10_clustered_list_resampled_smoteenn = []

    for cluster in range(N_CLUSTERS):
        X, y = train_4_clustered_list_classification[cluster]
        model = SMOTEENN()
        imbl_processor = dp.ImbalanceProcessor(model)
        X_resampled, y_resampled = imbl_processor.process_imbalance(X, y)
        train_4_clustered_list_resampled_smoteenn.append([X_resampled, y_resampled])
        print(X_resampled.shape)
        print(y_resampled.shape)

        X, y = train_10_clustered_list_classification[cluster]
        model = SMOTEENN()
        imbl_processor = dp.ImbalanceProcessor(model)
        X_resampled, y_resampled = imbl_processor.process_imbalance(X, y)
        train_10_clustered_list_resampled_smoteenn.append([X_resampled, y_resampled])
        print(X_resampled.shape)
        print(y_resampled.shape)

    
    # #### Create classification models and evaluate




    april_models_classification = [] # 4 items for 4 clusters, each item contains 2 models
    october_models_classification = [] # 4 items for 4 clusters, each item contains 2 models

    for cluster in range(N_CLUSTERS):
        X_4, y_4 = train_4_clustered_list_resampled_smoteenn[cluster]
        class_weight_4 = {0: y_4[y_4 == 0].shape[0], 1: y_4[y_4 == 1].shape[0]}
        X_10, y_10 = train_10_clustered_list_resampled_smoteenn[cluster]
        class_weight_10 = {0: y_10[y_10 == 0].shape[0], 1: y_10[y_10 == 1].shape[0]}
        rfc_4 = RandomForestClassifier(n_estimators=250, max_depth=10, class_weight=class_weight_4)
        rfc_10 = RandomForestClassifier(n_estimators=250, max_depth=10, class_weight=class_weight_10)
        xgb_4 = XGBClassifier(n_estimators=250, max_depth=10, scale_pos_weight=class_weight_4[0]/class_weight_4[1])
        xgb_10 = XGBClassifier(n_estimators=250, max_depth=10, scale_pos_weight=class_weight_10[0]/class_weight_10[1])
        rfc_4.fit(X_4, y_4)
        rfc_10.fit(X_10, y_10)
        xgb_4.fit(X_4, y_4)
        xgb_10.fit(X_10, y_10)
        april_models_classification.append([rfc_4, xgb_4])
        october_models_classification.append([rfc_10, xgb_10])


    # Test the models
    april_predictions_scores = dict()
    october_predictions_scores = dict()
    for cluster in range(N_CLUSTERS):
        X_4, y_4 = test_4_clustered_list_classification[cluster]
        X_10, y_10 = test_10_clustered_list_classification[cluster]
        rfc_4, xgb_4 = april_models_classification[cluster]
        rfc_10, xgb_10 = october_models_classification[cluster]
        rfc_4_pred = rfc_4.predict(X_4)
        rfc_10_pred = rfc_10.predict(X_10)
        xgb_4_pred = xgb_4.predict(X_4)
        xgb_10_pred = xgb_10.predict(X_10)
        april_predictions_scores[f"rfc_{cluster}"] = [f1_score(y_4, rfc_4_pred), accuracy_score(y_4, rfc_4_pred), precision_score(y_4, rfc_4_pred), recall_score(y_4, rfc_4_pred)]
        april_predictions_scores[f"xgb_{cluster}"] = [f1_score(y_4, xgb_4_pred), accuracy_score(y_4, xgb_4_pred), precision_score(y_4, xgb_4_pred), recall_score(y_4, xgb_4_pred)]
        october_predictions_scores[f"rfc_{cluster}"] = [f1_score(y_10, rfc_10_pred), accuracy_score(y_10, rfc_10_pred), precision_score(y_10, rfc_10_pred), recall_score(y_10, rfc_10_pred)]
        october_predictions_scores[f"xgb_{cluster}"] = [f1_score(y_10, xgb_10_pred), accuracy_score(y_10, xgb_10_pred), precision_score(y_10, xgb_10_pred), recall_score(y_10, xgb_10_pred)]
        


    print(april_predictions_scores)
    print(october_predictions_scores)

    # Convert to pandas dataframe
    april_predictions_scores_df = pd.DataFrame(april_predictions_scores, index=["f1", "accuracy", "precision", "recall"])
    october_predictions_scores_df = pd.DataFrame(october_predictions_scores, index=["f1", "accuracy", "precision", "recall"])

    print(april_predictions_scores_df.shape)
    print(october_predictions_scores_df.shape)


    april_predictions_scores_df.head()


    october_predictions_scores_df.head()


    RESULT_PATH = IMPL_PATH + f"/results/traditional_ml/ts{TIME_STEP}_fs{FUTURE_STEP}/"
    # Create the directory if not exists
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    april_predictions_scores_df.to_csv(RESULT_PATH + f"traditional_ml-ts{TIME_STEP}_fs{FUTURE_STEP}_april_predictions_scores_df.csv")
    october_predictions_scores_df.to_csv(RESULT_PATH + f"traditional_ml-ts{TIME_STEP}_fs{FUTURE_STEP}_october_predictions_scores_df.csv")

    
    # ### Regression model on overall data


    import matplotlib.pyplot as plt

    def draw_scatter_plot(y_true, y_pred, title, model_name, month, cluster, optional=""):
        plt.scatter(y_true, y_pred)
        plt.title(title)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.savefig(RESULT_PATH + f"{model_name}_{month}-ts{TIME_STEP}_fs{FUTURE_STEP}_cluster_{cluster}{optional}.png")
        plt.close()




    april_models_reg = [] # 4 models for 4 clusters
    october_models_reg = [] # 4 models for 4 clusters

    for cluster in range(N_CLUSTERS):
        X_4, y_4 = train_4_clustered_list[cluster]
        X_10, y_10 = train_10_clustered_list[cluster]
        rfr_4 = RandomForestRegressor(n_estimators=250, max_depth=10)
        rfr_10 = RandomForestRegressor(n_estimators=250, max_depth=10)
        xgbr_4 = XGBRegressor(n_estimators=250, max_depth=10)
        xgbr_10 = XGBRegressor(n_estimators=250, max_depth=10)
        rfr_4.fit(X_4, y_4)
        rfr_10.fit(X_10, y_10)
        xgbr_4.fit(X_4, y_4)
        xgbr_10.fit(X_10, y_10)
        april_models_reg.append([rfr_4, xgbr_4])
        october_models_reg.append([rfr_10, xgbr_10])


    # Test the models
    april_predictions_scores_reg = dict()
    october_predictions_scores_reg = dict()

    for cluster in range(N_CLUSTERS):
        X_4, y_4 = test_4_clustered_list[cluster]
        X_10, y_10 = test_10_clustered_list[cluster]
        rfr_4, xgbr_4 = april_models_reg[cluster]
        rfr_10, xgbr_10 = october_models_reg[cluster]
        rfr_4_pred = rfr_4.predict(X_4)
        rfr_10_pred = rfr_10.predict(X_10)
        xgbr_4_pred = xgbr_4.predict(X_4)
        xgbr_10_pred = xgbr_10.predict(X_10)
        draw_scatter_plot(y_4, rfr_4_pred, f"Random Forest Regressor Cluster {cluster} April", "rfr", "april", cluster)
        draw_scatter_plot(y_4, xgbr_4_pred, f"XGBoost Regressor Cluster {cluster} April", "xgbr", "april", cluster)
        draw_scatter_plot(y_10, rfr_10_pred, f"Random Forest Regressor Cluster {cluster} October", "rfr", "october", cluster)
        draw_scatter_plot(y_10, xgbr_10_pred, f"XGBoost Regressor Cluster {cluster} October", "xgbr", "october", cluster)
        april_predictions_scores_reg[f"rfr_{cluster}"] = [root_mean_squared_error(y_4, rfr_4_pred), r2_score(y_4, rfr_4_pred), mean_absolute_error(y_4, rfr_4_pred)]
        april_predictions_scores_reg[f"xgbr_{cluster}"] = [root_mean_squared_error(y_4, xgbr_4_pred), r2_score(y_4, xgbr_4_pred), mean_absolute_error(y_4, xgbr_4_pred)]
        october_predictions_scores_reg[f"rfr_{cluster}"] = [root_mean_squared_error(y_10, rfr_10_pred), r2_score(y_10, rfr_10_pred), mean_absolute_error(y_10, rfr_10_pred)]
        october_predictions_scores_reg[f"xgbr_{cluster}"] = [root_mean_squared_error(y_10, xgbr_10_pred), r2_score(y_10, xgbr_10_pred), mean_absolute_error(y_10, xgbr_10_pred)]


    # Convert to pandas dataframe
    april_predictions_scores_reg_df = pd.DataFrame(april_predictions_scores_reg, index=["rmse", "r2", "mae"])
    october_predictions_scores_reg_df = pd.DataFrame(october_predictions_scores_reg, index=["rmse", "r2", "mae"])


    april_predictions_scores_reg_df.head()


    october_predictions_scores_reg_df.head()


    # Save the results
    april_predictions_scores_reg_df.to_csv(RESULT_PATH + f"traditional_ml-ts{TIME_STEP}_fs{FUTURE_STEP}_april_predictions_scores_reg_df.csv")
    october_predictions_scores_reg_df.to_csv(RESULT_PATH + f"traditional_ml-ts{TIME_STEP}_fs{FUTURE_STEP}_october_predictions_scores_reg_df.csv")

    
    # ### Combine rain only regression and rain classification


    # Select only rainy Y values
    train_4_clustered_list_rain_only = []
    train_10_clustered_list_rain_only = []

    for cluster in range(N_CLUSTERS):
        X_4, y_4 = train_4_clustered_list[cluster]
        X_10, y_10 = train_10_clustered_list[cluster]

        y_4_rain = y_4[y_4 > 0]
        X_4_rain = X_4[y_4 > 0]
        y_10_rain = y_10[y_10 > 0]
        X_10_rain = X_10[y_10 > 0]

        train_4_clustered_list_rain_only.append([X_4_rain, y_4_rain])
        train_10_clustered_list_rain_only.append([X_10_rain, y_10_rain])


    # Train the model with rain only data
    april_models_rain_only = [] # 4 models for 4 clusters
    october_models_rain_only = [] # 4 models for 4 clusters

    for cluster in range(N_CLUSTERS):
        X_4, y_4 = train_4_clustered_list_rain_only[cluster]
        X_10, y_10 = train_10_clustered_list_rain_only[cluster]
        rfr_4 = RandomForestRegressor(n_estimators=250, max_depth=10)
        rfr_10 = RandomForestRegressor(n_estimators=250, max_depth=10)
        xgbr_4 = XGBRegressor(n_estimators=250, max_depth=10)
        xgbr_10 = XGBRegressor(n_estimators=250, max_depth=10)
        rfr_4.fit(X_4, y_4)
        rfr_10.fit(X_10, y_10)
        xgbr_4.fit(X_4, y_4)
        xgbr_10.fit(X_10, y_10)
        april_models_rain_only.append([rfr_4, xgbr_4])
        october_models_rain_only.append([rfr_10, xgbr_10])


    # Random Forest Classifier + Regressor
    april_prediction_scores_combined_rfc = dict()
    october_prediction_scores_combined_rfc = dict()
    for cluster in range(N_CLUSTERS):
        X_4, y_4 = test_4_clustered_list[cluster]
        X_10, y_10 = test_10_clustered_list[cluster]
        rfc_4 = april_models_classification[cluster][0]
        rfr_4, xgbr_4 = april_models_rain_only[cluster]
        rfc_10 = october_models_classification[cluster][0]
        rfr_10, xgbr_10 = october_models_rain_only[cluster]
        rfc_4_pred = rfc_4.predict(X_4)
        rfc_10_pred = rfc_10.predict(X_10)
        rfr_4_pred = rfr_4.predict(X_4)
        rfr_10_pred = rfr_10.predict(X_10)
        xgbr_4_pred = xgbr_4.predict(X_4)
        xgbr_10_pred = xgbr_10.predict(X_10)
        # Merge the predictions where rfc is 0 when rfc is 0, otherwise use rfr and xgbr
        rfc_rfr_4_pred = np.where(rfc_4_pred == 0, rfc_4_pred, rfr_4_pred)
        rfc_xgbr_4_pred = np.where(rfc_4_pred == 0, rfc_4_pred, xgbr_4_pred)
        rfc_rfr_10_pred = np.where(rfc_10_pred == 0, rfc_10_pred, rfr_10_pred)
        rfc_xgbr_10_pred = np.where(rfc_10_pred == 0, rfc_10_pred, xgbr_10_pred)
        draw_scatter_plot(y_4, rfc_rfr_4_pred, f"Random Forest Classifier + Random Forest Regressor Cluster {cluster} April", "rfc_rfr", "april", cluster, "combined")
        draw_scatter_plot(y_4, rfc_xgbr_4_pred, f"Random Forest Classifier + XGBoost Regressor Cluster {cluster} April", "rfc_xgbr", "april", cluster, "combined")
        draw_scatter_plot(y_10, rfc_rfr_10_pred, f"Random Forest Classifier + Random Forest Regressor Cluster {cluster} October", "rfc_rfr", "october", cluster, "combined")
        draw_scatter_plot(y_10, rfc_xgbr_10_pred, f"Random Forest Classifier + XGBoost Regressor Cluster {cluster} October", "rfc_xgbr", "october", cluster, "combined")
        april_prediction_scores_combined_rfc[f"rfc_rfr_{cluster}"] = [root_mean_squared_error(y_4, rfc_rfr_4_pred), r2_score(y_4, rfc_rfr_4_pred), mean_absolute_error(y_4, rfc_rfr_4_pred)]
        april_prediction_scores_combined_rfc[f"rfc_xgbr_{cluster}"] = [root_mean_squared_error(y_4, rfc_xgbr_4_pred), r2_score(y_4, rfc_xgbr_4_pred), mean_absolute_error(y_4, rfc_xgbr_4_pred)]
        october_prediction_scores_combined_rfc[f"rfc_rfr_{cluster}"] = [root_mean_squared_error(y_10, rfc_rfr_10_pred), r2_score(y_10, rfc_rfr_10_pred), mean_absolute_error(y_10, rfc_rfr_10_pred)]
        october_prediction_scores_combined_rfc[f"rfc_xgbr_{cluster}"] = [root_mean_squared_error(y_10, rfc_xgbr_10_pred), r2_score(y_10, rfc_xgbr_10_pred), mean_absolute_error(y_10, rfc_xgbr_10_pred)]

    # Convert to pandas dataframe
    april_prediction_scores_combined_rfc_df = pd.DataFrame(april_prediction_scores_combined_rfc, index=["rmse", "r2", "mae"])
    october_prediction_scores_combined_rfc_df = pd.DataFrame(october_prediction_scores_combined_rfc, index=["rmse", "r2", "mae"])

    # Save the results
    april_prediction_scores_combined_rfc_df.to_csv(RESULT_PATH + f"traditional_ml-ts{TIME_STEP}_fs{FUTURE_STEP}_april_prediction_scores_combined_rfc_df.csv")
    october_prediction_scores_combined_rfc_df.to_csv(RESULT_PATH + f"traditional_ml-ts{TIME_STEP}_fs{FUTURE_STEP}_october_prediction_scores_combined_rfc_df.csv")

    # XGBoost Classifier + Regressor
    april_prediction_scores_combined_xgb = dict()
    october_prediction_scores_combined_xgb = dict()

    for cluster in range(N_CLUSTERS):
        X_4, y_4 = test_4_clustered_list[cluster]
        X_10, y_10 = test_10_clustered_list[cluster]
        xgb_4 = april_models_classification[cluster][1]
        rfr_4, xgbr_4 = april_models_rain_only[cluster]
        xgb_10 = october_models_classification[cluster][1]
        rfr_10, xgbr_10 = october_models_rain_only[cluster]
        xgb_4_pred = xgb_4.predict(X_4)
        xgb_10_pred = xgb_10.predict(X_10)
        rfr_4_pred = rfr_4.predict(X_4)
        rfr_10_pred = rfr_10.predict(X_10)
        xgbr_4_pred = xgbr_4.predict(X_4)
        xgbr_10_pred = xgbr_10.predict(X_10)
        # Merge the predictions where xgb is 0 when xgb is 0, otherwise use rfr and xgbr
        xgb_rfr_4_pred = np.where(xgb_4_pred == 0, xgb_4_pred, rfr_4_pred)
        xgb_xgbr_4_pred = np.where(xgb_4_pred == 0, xgb_4_pred, xgbr_4_pred)
        xgb_rfr_10_pred = np.where(xgb_10_pred == 0, xgb_10_pred, rfr_10_pred)
        xgb_xgbr_10_pred = np.where(xgb_10_pred == 0, xgb_10_pred, xgbr_10_pred)
        draw_scatter_plot(y_4, xgb_rfr_4_pred, f"XGBoost Classifier + Random Forest Regressor Cluster {cluster} April", "xgb_rfr", "april", cluster, "combined")
        draw_scatter_plot(y_4, xgb_xgbr_4_pred, f"XGBoost Classifier + XGBoost Regressor Cluster {cluster} April", "xgb_xgbr", "april", cluster, "combined")
        draw_scatter_plot(y_10, xgb_rfr_10_pred, f"XGBoost Classifier + Random Forest Regressor Cluster {cluster} October", "xgb_rfr", "october", cluster, "combined")
        draw_scatter_plot(y_10, xgb_xgbr_10_pred, f"XGBoost Classifier + XGBoost Regressor Cluster {cluster} October", "xgb_xgbr", "october", cluster, "combined")
        april_prediction_scores_combined_xgb[f"xgb_rfr_{cluster}"] = [root_mean_squared_error(y_4, xgb_rfr_4_pred), r2_score(y_4, xgb_rfr_4_pred), mean_absolute_error(y_4, xgb_rfr_4_pred)]
        april_prediction_scores_combined_xgb[f"xgb_xgbr_{cluster}"] = [root_mean_squared_error(y_4, xgb_xgbr_4_pred), r2_score(y_4, xgb_xgbr_4_pred), mean_absolute_error(y_4, xgb_xgbr_4_pred)]
        october_prediction_scores_combined_xgb[f"xgb_rfr_{cluster}"] = [root_mean_squared_error(y_10, xgb_rfr_10_pred), r2_score(y_10, xgb_rfr_10_pred), mean_absolute_error(y_10, xgb_rfr_10_pred)]
        october_prediction_scores_combined_xgb[f"xgb_xgbr_{cluster}"] = [root_mean_squared_error(y_10, xgb_xgbr_10_pred), r2_score(y_10, xgb_xgbr_10_pred), mean_absolute_error(y_10, xgb_xgbr_10_pred)]

    # Convert to pandas dataframe
    april_prediction_scores_combined_xgb_df = pd.DataFrame(april_prediction_scores_combined_xgb, index=["rmse", "r2", "mae"])
    october_prediction_scores_combined_xgb_df = pd.DataFrame(october_prediction_scores_combined_xgb, index=["rmse", "r2", "mae"])

    # Save the results
    april_prediction_scores_combined_xgb_df.to_csv(RESULT_PATH + f"traditional_ml-ts{TIME_STEP}_fs{FUTURE_STEP}_april_prediction_scores_combined_xgb_df.csv")
    october_prediction_scores_combined_xgb_df.to_csv(RESULT_PATH + f"traditional_ml-ts{TIME_STEP}_fs{FUTURE_STEP}_october_prediction_scores_combined_xgb_df.csv")


def main():
    time_step = [3, 6, 12]
    future_step = [1, 3, 6]

    for ts in time_step:
        for fs in future_step:
            train_with_config(ts, fs, 4)

if __name__ == "__main__":
    main()