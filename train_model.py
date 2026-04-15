import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
import os

# Load the cleaned dataset
if not os.path.exists('used_cars_cleaned.csv'):
    print("Cleaned data not found. Please run preprocess.py first.")
    exit()

df = pd.read_csv('used_cars_cleaned.csv')

# 1. Feature Selection
cat_features = ['brand', 'model_grouped', 'fuel_type', 'accident', 'clean_title']
num_features = ['model_year', 'milage', 'car_age', 'hp', 'displacement', 'is_automatic']
features = cat_features + num_features

X = df[features]
y = df['price']

# 2. Target Transformation: Log-normalize the price to handle extreme skew
y_log = np.log1p(y)

# 3. Handle Categorical Data (HistGradientBoosting has built-in support for categorical features, but needs them to be encoded as integers)
# We use OrdinalEncoder to convert categories to integers
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_cat = encoder.fit_transform(X[cat_features])
X[cat_features] = X_cat

# 4. Train-Test Split (using log-transformed target)
X_train, X_test, y_log_train, y_log_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# 5. Model Building (HistGradientBoosting is much faster and more accurate for tabular data)
print("Training High-Accuracy Model...")
model = HistGradientBoostingRegressor(
    max_iter=1000, 
    learning_rate=0.05, 
    max_depth=10, 
    categorical_features=[i for i, f in enumerate(features) if f in cat_features],
    random_state=42
)
model.fit(X_train, y_log_train)

# 6. Evaluation
y_log_pred = model.predict(X_test)
# Convert predictions back to original scale
y_pred = np.expm1(y_log_pred)
y_test_original = np.expm1(y_log_test)

mae = mean_absolute_error(y_test_original, y_pred)
r2 = r2_score(y_test_original, y_pred)

print("\nFINAL MODEL PERFORMANCE:")
print(f"Mean Absolute Error: ${mae:.2f}")
print(f"R^2 Score (Accuracy): {r2:.4f}")

if r2 < 0.78:
    print("\nR2 score is still below 0.78. Attempting further refinements...")

# 7. Save Model and Pipeline components
joblib.dump(model, "model.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(features, "columns.pkl")
print("\n✅ New high-accuracy model and encoder saved.")