#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Set page config with solar theme
st.set_page_config(
    page_title="Solar Power Generation Predictor",
    page_icon="☀️",
    layout="wide"
)

# Load the saved model, scaler, and feature names
@st.cache_resource
def load_model():
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('features.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

# Embedded CSS styling
st.markdown(
    """
    <style>
    /* Main app styling */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1509391366360-2e959784a276?ixlib=rb-4.0.3");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    /* Main content container */
    .main {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
        margin: 1rem;
    }
    
    /* Header styling */
    .header {
        color: #FFA500;
        text-shadow: 1px 1px 2px #000;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    /* Input boxes */
    .input-box {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Result boxes */
    .result-box {
        background-color: rgba(255, 215, 0, 0.3);
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        border-left: 5px solid #FFA500;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #FFA500;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #FF8C00;
        transform: scale(1.05);
    }
    
    /* Text input styling */
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 5px;
    }
    
    /* Sidebar styling */
    .st-emotion-cache-6qob1r {
        background-color: rgba(255, 255, 255, 0.85) !important;
    }
    
    /* Status indicators */
    .low-status {
        background-color: #FF6B6B;
        color: white;
    }
    
    .medium-status {
        background-color: #FFD166;
        color: black;
    }
    
    .high-status {
        background-color: #06D6A0;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to make predictions
def predict_power(input_data):
    # Convert input data to DataFrame with correct feature order
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Create a dummy row with all features + target (for scaling)
    dummy_row = input_df.copy()
    dummy_row['power_generated'] = 0  # Add target column with dummy value
    
    # Scale the input data (excluding the target)
    scaled_data = scaler.transform(dummy_row)
    input_scaled = scaled_data[:, :-1]  # Exclude the last column (target)
    
    # Make prediction (returns log-transformed value)
    log_prediction = model.predict(input_scaled)
    
    # Reverse the log transformation
    prediction = np.exp(log_prediction)[0]
    
    return prediction

# Main app
with st.container():
    st.markdown("<h1 class='header'>☀️ Solar Power Generation Predictor</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: rgba(255, 255, 255, 0.7); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
    Predict solar power generation based on weather and environmental conditions.<br>
    Enter the parameters below and click the predict button.
    </div>
    """, unsafe_allow_html=True)

    # Create input sections
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='input-box'>", unsafe_allow_html=True)
        st.subheader("🌤️ Solar & Weather Parameters")
        distance_to_solar_noon = st.text_input("Distance to Solar Noon (degrees)", value="0.0", key="solar_noon")
        temperature = st.text_input("Temperature (°C)", value="25.0", key="temp")
        wind_speed = st.text_input("Wind Speed (km/h)", value="10.0", key="wind_speed")
        sky_cover = st.text_input("Sky Cover (oktas, 0-8)", value="2.0", key="sky_cover")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='input-box'>", unsafe_allow_html=True)
        st.subheader("🌬️ Atmospheric Conditions")
        wind_direction = st.text_input("Wind Direction (degrees, 0-360)", value="180.0", key="wind_dir")
        visibility = st.text_input("Visibility (km)", value="10.0", key="visibility")
        humidity = st.text_input("Humidity (%)", value="50.0", key="humidity")
        average_pressure = st.text_input("Average Pressure (hPa)", value="1013.0", key="pressure")
        st.markdown("</div>", unsafe_allow_html=True)

    # Prediction button
    if st.button('🔮 Predict Power Generation', use_container_width=True, key="predict_btn"):
        try:
            # Convert inputs to float
            input_data = {
                'distance_to_solar_noon': float(distance_to_solar_noon),
                'temperature': float(temperature),
                'wind_direction': float(wind_direction),
                'wind_speed': float(wind_speed),
                'sky_cover': float(sky_cover),
                'visibility': float(visibility),
                'humidity': float(humidity),
                'average_pressure': float(average_pressure)
            }

            # Make prediction
            prediction = predict_power(input_data)
            
            # Display results with nice styling
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.success(f"Predicted Power Generation: {prediction:.2f} MW")
            
            energy_joules = prediction * 1_000_000 * 3600
            st.info(f"Estimated Energy for 1 hour: {energy_joules:,.0f} J")
            
            # Visual representation
            st.subheader("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Power Output", f"{prediction:.2f} MW", 
                         help="Estimated power generation in megawatts")
                
            with col2:
                efficiency = (prediction / 10) * 100  # Assuming 10 MW is max capacity
                st.metric("System Efficiency", f"{efficiency:.1f}%", 
                         help="Percentage of maximum possible output")
                
            with col3:
                if prediction < 2:
                    status = "Low ☁️"
                    status_class = "low-status"
                elif prediction < 5:
                    status = "Moderate ⛅"
                    status_class = "medium-status"
                else:
                    status = "High ☀️"
                    status_class = "high-status"
                st.markdown(f"""
                <div class="{status_class}" style="padding: 1rem; border-radius: 10px; text-align: center;">
                <h3>Generation Status</h3>
                <h2>{status}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Add solar panel visualization
            st.markdown("### 🌞 System Performance")
            solar_capacity = min(prediction / 10, 1.0)  # Cap at 100%
            st.progress(solar_capacity)
            st.markdown(f"Solar panels operating at {solar_capacity*100:.1f}% of capacity")
            st.markdown("</div>", unsafe_allow_html=True)
            
        except ValueError:
            st.error("Please enter valid numbers for all parameters")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Sidebar with additional info
with st.sidebar:
    st.markdown("<div style='background-color: rgba(255, 255, 255, 0.8); padding: 1rem; border-radius: 10px;'>", unsafe_allow_html=True)
    st.header("ℹ️ About This App")
    st.markdown("""
    This predictive model uses machine learning to estimate solar power generation 
    based on environmental conditions.
    
    **Model Features:**
    - Solar position
    - Weather conditions
    - Atmospheric measurements
    """)
    
    st.header("📝 Instructions")
    st.markdown("""
    1. Enter values for all parameters
    2. Click the predict button
    3. View the power generation estimate
    """)
    
    st.header("⚙️ Model Details")
    st.write("Algorithm: Random Forest Regressor")
    st.write("Trained on historical solar generation data")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; background-color: rgba(255, 255, 255, 0.7); border-radius: 10px;">
    <p>©  Solar Power Analytics | Sustainable Energy Solutions</p>
    <p>☀️ Harnessing the power of the sun ☀️</p>
</div>
""", unsafe_allow_html=True)

