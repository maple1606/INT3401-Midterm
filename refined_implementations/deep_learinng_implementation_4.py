# %% [markdown]
# ### Load data

# %%
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
DATA_PATH=os.getenv("DATA_PATH")

month = 4

# %%
train_data = pd.read_csv(f"{DATA_PATH}/{month}/train_data_{month}.csv")
train_data['date']
train_data

# %%
test_data = pd.read_csv(f"{DATA_PATH}/{month}/test_data_{month}.csv")
test_data

# %%
import missingno as msno
msno.matrix(train_data)
msno.matrix(test_data)

# %%
def format_date_column(df, date_column='date'):
    """
    This function checks if the date column is in datetime format.
    If not, it converts the column to datetime format.

    Parameters:
    df (pd.DataFrame): The dataframe containing the date column.
    date_column (str): The name of the date column. Default is 'date'.

    Returns:
    pd.DataFrame: The dataframe with the formatted date column.
    """
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(
            df[date_column],  format='%Y-%m-%d %H:%M:%S')
    return df


train_data = format_date_column(train_data)
test_data = format_date_column(test_data)
important_columns = ['date', 'row', 'col', 'aws', 'imerge',
                     'tclw', 'r850', 'r250', 'v850', 'r500', 'u250']
train_data = train_data[important_columns]
test_data = test_data[important_columns]

# %%
msno.matrix(train_data)
msno.matrix(test_data)

# %%
train_data = train_data.dropna()
test_data = test_data.dropna()
print(train_data.isna().sum())
print(test_data.isna().sum())

# %%
from data_processors import DataScaler
scaler = DataScaler()
scaler.scale_data(train_data, test_data)

# %% [markdown]
# ### Create Sequence data

# %%
TIME_STEPS=6
from data_processors import GetData

X_train, y_train, X_test, y_test = GetData.get_non_cluster_sequence_data(train_df=train_data, test_df=test_data, time_step=TIME_STEPS)

# %%
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# %% [markdown]
# ### Stacking models

# %%
# 5 models each with different number of layers and units
UNITS1 = [64, 64]
UNITS2 = [32, 32]
UNITS3 = [64, 128, 64]
UNITS4 = [32, 64, 32]
UNITS5 = [128, 128, 256, 256]

DROPOUT1 = [0.2, 0.2]
DROPOUT2 = [0.1, 0.1]
DROPOUT3 = [0.2, 0.3, 0.2]
DROPOUT4 = [0.1, 0.2, 0.1]
DROPOUT5 = [0.2, 0.3, 0.4, 0.5]

EPOCHS = 250
KERNEL_REGULARIZER = None
PATIENCE = 15

UNITS = [UNITS1, UNITS2, UNITS3, UNITS4, UNITS5]
DROPOUTS = [DROPOUT1, DROPOUT2, DROPOUT3, DROPOUT4, DROPOUT5]
VALIDATION_SPLIT = 0.1

# %%
from recurrent_regressor import LSTMRegressor, GRURegressor

lstm_regressors = []
gru_regressors = []

for i in range(5):
    lstm_regressors.append(LSTMRegressor(units=UNITS[i], dropouts=DROPOUTS[i], kernel_regularizer=KERNEL_REGULARIZER, patience=PATIENCE, validation_split=VALIDATION_SPLIT, epochs=EPOCHS))
    gru_regressors.append(GRURegressor(units=UNITS[i], dropouts=DROPOUTS[i], kernel_regularizer=KERNEL_REGULARIZER, patience=PATIENCE, validation_split=VALIDATION_SPLIT, epochs=EPOCHS))

# %%
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

# Init base estimators
base_estimators = []

for i in range(len(lstm_regressors)):
    base_estimators.append((f"LSTM_{i+1}", lstm_regressors[i]))
    base_estimators.append((f"GRU_{i+1}", gru_regressors[i]))
    
# Init final estimator
final_estimator = LinearRegression(n_jobs=-1)

# Pre-fit base estimators
for name, estimator in base_estimators:
	estimator.fit(X_train, y_train)

# Init stacking regressor
stacking_regressor = StackingRegressor(estimators=base_estimators, final_estimator=final_estimator, cv="prefit", n_jobs=-1)

# %%
# Fit stacking regressor
stacking_regressor.fit(X_train, y_train)

# %%
y_pred = stacking_regressor.predict(X_test)

# %%
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

plt.figure(figsize=(10, 5))
plt.title(f"RFR - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
plt.scatter(y_test, y_pred, marker='o', alpha=0.5)
plt.xlabel("True values")
plt.ylabel("Predicted values")
# Save results image
plt.savefig(f"/home/hoangbaoan1901/Documents/INT3401-Midterm/refined_implementations/results/stacking_models_no_cluster_{month}.png")


