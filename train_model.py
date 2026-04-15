import pandas as pd
import joblib
import re
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("used_cars.csv")

# Clean column names
df.columns = df.columns.str.lower().str.strip()

print("Available columns:", df.columns)

# Data Cleaning function for numeric values in strings
def clean_numeric(x):
    if isinstance(x, str):
        # Remove currency symbols, units, and non-numeric characters except decimal point
        x = re.sub(r'[^\d.]', '', x)
        return float(x) if x else 0.0
    return float(x) if x else 0.0

# Apply cleaning to 'price' and 'milage' (actual name in CSV)
df['price'] = df['price'].apply(clean_numeric)
df['milage'] = df['milage'].apply(clean_numeric)

# Map columns to internal names
column_map = {
    'model_year': 'year',
    'milage': 'mileage',
    'price': 'selling_price',
    'fuel_type': 'fuel',
    'transmission': 'transmission'
}

# Rename columns
df = df.rename(columns=column_map)

print("Renamed columns:", df.columns)

# Check required columns
required_cols = ['year', 'mileage', 'fuel', 'transmission', 'selling_price']
for col in required_cols:
    if col not in df.columns:
        print(f"❌ Missing column: {col}")

# Select features and target
df = df[required_cols].dropna()

# Prepare features (X) and target (y)
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# One-hot encoding for categorical variables
X = pd.get_dummies(X)

# Save the feature columns to ensure consistent input during prediction
joblib.dump(X.columns, "columns.pkl")

# Train the model
print("Training model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the trained model
joblib.dump(model, "model.pkl")

print("✅ Model trained successfully. Files created: model.pkl, columns.pkl")