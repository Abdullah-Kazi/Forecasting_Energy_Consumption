import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pickle
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor

st.title('Energy Forecasting Demo')
st.write("""
         Welcome to our interactive demo showcasing our advanced forecasting capabilities using machine learning.
         This tool demonstrates how we leverage data to provide accurate energy usage predictions, 
         helping businesses and consumers optimize their energy management.
         """)

# Load trained models
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_models():
    models = {}
    try:
        model_names = ['XGBoost_model', 'Prophet_model', 'LightGBM_model']
        for model_name in model_names:
            with open(f'{model_name}.pkl', 'rb') as file:
                models[model_name] = pickle.load(file)
    except Exception as e:
        st.error(f"Failed to load model due to: {e}")
    return models

models = load_models()


# Sidebar for model selection and forecasting details
st.sidebar.header('Forecast Settings')
model_names = list(models.keys())
selected_model_name = st.sidebar.selectbox('Choose a Forecasting Model:', model_names)

# Date range picker for forecast
today = datetime.today().date()
tomorrow = today + timedelta(days=1)
start_date = st.sidebar.date_input('Start date', tomorrow)
end_date = st.sidebar.date_input('End date', tomorrow + timedelta(days=30))
if start_date > end_date:
    st.sidebar.error('Error: End date must fall after start date.')



def generate_dates(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    return dates

def prepare_features(dates):
    df = pd.DataFrame(dates, columns=['Datetime'])
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    df = add_features(df)
    return df

# Button to generate forecast
if st.sidebar.button('Generate Forecast'):
    dates = generate_dates(start_date, end_date)
    features = prepare_features(dates)
    model = models[selected_model_name]
    _, predictions = make_predictions(model, features)

    # Display the forecast results
    st.subheader('Forecast Results')
    plt.figure(figsize=(10, 5))
    plt.plot(dates, predictions, label='Predicted Energy Usage')
    plt.xlabel('Date')
    plt.ylabel('Energy Usage (kWh)')
    plt.title(f'Energy Usage Forecast from {start_date} to {end_date}')
    plt.legend()
    st.pyplot(plt)

    # Display data in a table
    results_df = pd.DataFrame({'Date': dates, 'Predicted Usage': predictions})
    st.write(results_df)

