import pandas as pd
import numpy as np
import re

def clean_dataset(df):
    # 1. Clean Price and Milage
    df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
    df['milage'] = df['milage'].str.replace(' mi.', '').str.replace(',', '').astype(float)
    
    # 2. Handle Missing Values
    df['accident'] = df['accident'].fillna('None reported')
    df['clean_title'] = df['clean_title'].fillna('Unknown')
    
    # 3. Feature Engineering: Car Age
    current_year = 2024
    df['car_age'] = current_year - df['model_year']
    
    # 4. Extract Horsepower (HP) and Displacement (L) from engine
    def extract_hp(engine_str):
        if not isinstance(engine_str, str): return np.nan
        match = re.search(r'(\d+\.?\d*)HP', engine_str)
        return float(match.group(1)) if match else np.nan

    def extract_displacement(engine_str):
        if not isinstance(engine_str, str): return np.nan
        match = re.search(r'(\d+\.?\d*)L', engine_str)
        return float(match.group(1)) if match else np.nan

    df['hp'] = df['engine'].apply(extract_hp)
    df['displacement'] = df['engine'].apply(extract_displacement)
    
    # Fill missing HP and L with median
    df['hp'] = df['hp'].fillna(df['hp'].median())
    df['displacement'] = df['displacement'].fillna(df['displacement'].median())
    
    # 5. Simplify Transmission (Automatic vs Manual)
    df['is_automatic'] = df['transmission'].str.contains('A/T|Automatic|CVT', case=False, na=False).astype(int)
    
    # 6. Group rare models into 'Other'
    model_counts = df['model'].value_counts()
    rare_models = model_counts[model_counts < 5].index
    df['model_grouped'] = df['model'].replace(rare_models, 'Other')
    
    # 7. Remove Extreme Outliers (Price > $200k) for better general model
    # Note: Keep enough data but remove things that are clearly "one-offs"
    df = df[df['price'] < 250000].reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    df = pd.read_csv('used_cars.csv')
    df_cleaned = clean_dataset(df)
    df_cleaned.to_csv('used_cars_cleaned.csv', index=False)
    print("Dataset cleaned and saved to used_cars_cleaned.csv")
    print(f"Final dataset size: {len(df_cleaned)}")
    print(df_cleaned[['hp', 'displacement', 'is_automatic']].head())
