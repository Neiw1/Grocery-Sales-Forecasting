import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

cwd = os.getcwd()
data_folder = os.path.join(cwd, "sales_forecasting_data")

df = pd.read_csv(os.path.join(data_folder, "clean_train.csv"))

print(df)