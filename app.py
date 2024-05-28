import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pickle  # for loading models
import matplotlib.pyplot as plt

# Load models
@st.cache(allow_output_mutation=True)
def load_models():
    with open('XGBoost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    return xgb_model

xgb_model = load_models()

# App title
st.title('Energy Usage Forecasting')

# Sidebar for user input
st.sidebar.header('Specify Forecast Details')
years = st.sidebar.number_input('Years into the future:', min_value=0, max_value=20, value=5)
months = st.sidebar.number_input('Additional months:', min_value=0, max_value=11, value=2)

# Functions to preprocess and add features
def generate_future_dates(years, months):
    end_date = datetime.now() + timedelta(days=(years * 365) + (months * 30))
    future_dates = pd.date_range(start=datetime.now(), end=end_date, freq='H')
    future_df = pd.DataFrame(future_dates, columns=['Datetime'])
    future_df.set_index('Datetime', inplace=True)
    future_df = add_features(future_df)  # Apply your existing feature engineering function
    return future_df

# Predict function
def make_predictions(model, features):
    predictions = model.predict(features)
    return features.index, predictions

# Displaying the forecast
if st.sidebar.button('Show Forecast'):
    features = generate_future_dates(years, months)
    dates, predictions = make_predictions(model, features)
    plt.figure(figsize=(10, 5))
    plt.plot(dates, predictions, label='Forecasted Energy Usage')
    plt.xlabel('Date')
    plt.ylabel('Energy Usage')
    plt.title('Energy Usage Forecast')
    plt.legend()
    st.pyplot(plt)
