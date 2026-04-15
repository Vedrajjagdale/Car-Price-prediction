import streamlit as st
import pandas as pd
import joblib

# Set page config for a more premium look
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Load model and columns
try:
    model = joblib.load("model.pkl")
    columns = joblib.load("columns.pkl")
except FileNotFoundError:
    st.error("Model files not found. Please run train_model.py first.")
    st.stop()

# Custom CSS for a more premium feel
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 15px;
        background: white;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        text-align: center;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# App Title
st.title("🚗 Luxury Car Price Prediction")
st.markdown("Predict the market value of used cars using our advanced machine learning model.")

# Create columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Vehicle Details")
    year = st.number_input("Manufacturing Year", min_value=1990, max_value=2025, value=2018)
    mileage = st.number_input("Total Running of car", min_value=0, value=30000, step=1000)

with col2:
    st.subheader("Configuration")
    # Extract unique values from columns for selectboxes (heuristic mapping)
    # Since columns are one-hot encoded like 'fuel_Gasoline', 'transmission_Automatic'
    fuels = sorted([c.replace('fuel_', '') for c in columns if c.startswith('fuel_')])
    transmissions = sorted([c.replace('transmission_', '') for c in columns if c.startswith('transmission_')])
    
    # Fallbacks if columns parsing is complex
    if not fuels: fuels = ["Gasoline", "Hybrid", "Diesel", "E85 Flex Fuel"]
    if not transmissions: transmissions = ["Automatic", "Manual", "6-Speed A/T", "M/T"]

    fuel = st.selectbox("Fuel Type", fuels)
    transmission = st.selectbox("Transmission", transmissions)

# Prediction Logic
if st.button("Calculate Predictive Value"):
    # Create input dataframe
    input_data = pd.DataFrame({
        'year': [year],
        'mileage': [mileage],
        'fuel': [fuel],
        'transmission': [transmission]
    })
    
    # One-hot encode the input
    input_data = pd.get_dummies(input_data)
    
    # Align columns with training data
    input_data = input_data.reindex(columns=columns, fill_value=0)
    
    # Prediction
    prediction = model.predict(input_data)[0]
    
    # Display Result
    st.markdown(f"""
        <div class="prediction-card">
            <h3>Estimated Market Value</h3>
            <h1 style="color: #28a745;">$ {int(prediction):,}</h1>
            <p style="color: #6c757d;">Based on current market trends and vehicle configuration.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #6c757d;'>Powered by Random Forest Regression | Car Price Prediction App</p>", unsafe_allow_html=True)