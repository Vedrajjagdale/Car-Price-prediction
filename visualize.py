import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix

# Set style
sns.set_theme(style="whitegrid")

def generate_visualizations(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # [Existing plots code...]
    
    # 1. Price Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], kde=True, color='skyblue')
    plt.title('Distribution of Car Prices')
    plt.xlabel('Price ($)')
    plt.savefig(os.path.join(output_dir, 'price_dist.png'))
    plt.close()
    
    # 2. Milage vs Price
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='milage', y='price', data=df, alpha=0.5, color='coral')
    plt.title('Price vs Milage')
    plt.xlabel('Milage (mi)')
    plt.ylabel('Price ($)')
    plt.savefig(os.path.join(output_dir, 'price_vs_milage.png'))
    plt.close()
    
    # 3. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # [NEW: Actual vs Predicted Plot]
    if os.path.exists('model.pkl'):
        model = joblib.load('model.pkl')
        cols = joblib.load('columns.pkl')
        
        # Prepare a small sample for visualization if needed, but let's use the full cleaned data
        features = ['model_year', 'milage', 'brand', 'fuel_type', 'accident', 'car_age']
        X = pd.get_dummies(df[features], drop_first=True)
        # Handle missing columns in the dummy set compared to the trained model
        X = X.reindex(columns=cols, fill_value=0)
        
        y_actual = df['price']
        y_pred = model.predict(X)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_actual, y_pred, alpha=0.3, color='green')
        plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
        plt.title('Actual vs Predicted Prices')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'))
        plt.close()

        # [NEW: Confusion Matrix (Binned Price Tiers)]
        # Since this is regression, we create a pseudo-confusion matrix by binning prices into "Budget", "Mid", "Luxury"
        bins = [0, 20000, 50000, np.inf]
        labels = ['Budget', 'Mid', 'Luxury']
        y_actual_cat = pd.cut(y_actual, bins=bins, labels=labels)
        y_pred_cat = pd.cut(y_pred, bins=bins, labels=labels)
        
        cm = confusion_matrix(y_actual_cat, y_pred_cat, labels=labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Price Tier Confusion Matrix')
        plt.xlabel('Predicted Tier')
        plt.ylabel('Actual Tier')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()

    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    generate_visualizations('used_cars_cleaned.csv', 'visualizations')
