import pandas as pd
import numpy as np
import lightgbm as lgb

# Load test data
df_test = pd.read_csv('sales_forecasting_data/engineered_data.csv')  # Replace with your actual test path

# Categorical columns to label encode (must match training)
cat_cols = ['onpromotion', 'family', 'city', 'state', 'type_x']

# Apply label encoding (same as in training)
# You must use the **same mappings** used during training.
# For a quick solution (may not be ideal), re-fit LabelEncoders here assuming same categories:
from sklearn.preprocessing import LabelEncoder

for col in cat_cols:
    le = LabelEncoder()
    df_test[col] = le.fit_transform(df_test[col].astype(str))

# Feature columns used during training
feature_cols = [
    'store_nbr', 'item_nbr', 'onpromotion', 'family', 'class', 'perishable',
    'city', 'state', 'type_x', 'cluster', 'transactions', 'dcoilwtico',
    'month', 'is_weekend'
]

# Prepare feature matrix
X_test = df_test[feature_cols].fillna(0)

# Load the trained LightGBM model
model = lgb.Booster(model_file='model.txt')

# Predict
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# Undo log1p if needed (to get predictions in original scale)
y_pred_original = np.expm1(y_pred)

# Save predictions
df_test['predicted_unit_sales'] = y_pred_original.clip(0)
df_test[['id', 'predicted_unit_sales']].to_csv('predictions.csv', index=False)

print("✅ Predictions saved to predictions.csv")
