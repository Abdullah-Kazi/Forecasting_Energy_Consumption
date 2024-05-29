import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pickle
import matplotlib.pyplot as plt

st.title('Energy Forecasting Demo')
st.write("""
         Welcome to our interactive demo showcasing our advanced forecasting capabilities using machine learning.
         This tool demonstrates how we leverage data to provide accurate energy usage predictions,
         helping businesses and consumers optimize their energy management.
         """)

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_models():
    models = {}
    try:
        model_names = ['XGBoost_model', 'Linear Regression_model', 'LightGBM_model', 'Decision Tree_model', 'Random Forest_model', 'Prophet_model']
        for model_name in model_names:
            with open(f'{model_name}.pkl', 'rb') as file:
                models[model_name] = pickle.load(file)
    except Exception as e:
        st.error(f"Failed to load model due to: {e}")
    return models

models = load_models()

def generate_future_dates(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    df = pd.DataFrame(dates, columns=['Datetime'])
    df.set_index('Datetime', inplace=True)
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

st.sidebar.header('Forecast Future Energy Usage')
selected_model_name = st.sidebar.selectbox('Choose a Forecasting Model:', list(models.keys()))

today = datetime.today().date()
start_date = st.sidebar.date_input('Start Date', today + timedelta(days=1), min_value=today)
end_date = st.sidebar.date_input('End Date', today + timedelta(days=30), min_value=today)

if st.sidebar.button('Generate Forecast'):
    future_dates = generate_future_dates(start_date, end_date)
    model = models[selected_model_name]
    features = future_dates.drop(columns=[], errors='ignore')  # Adjust based on the model's expected input features
    dates, predictions = make_predictions(model, features)

    if predictions is not None:
        forecast_df = pd.DataFrame({
            'Date': pd.to_datetime(dates),
            'Predicted Energy Usage': predictions
        })
        forecast_df.set_index('Date', inplace=True)

        st.subheader('Forecast Results for Future Energy Usage')
        plt.figure(figsize=(10, 5))
        plt.plot(forecast_df.index, forecast_df['Predicted Energy Usage'], label='Predicted Energy Usage (kWh)')
        plt.xlabel('Date')
        plt.ylabel('Energy Usage (kWh)')
        plt.title(f'Predicted Energy Usage from {start_date} to {end_date}')
        plt.legend()
        st.pyplot(plt)
    else:
        st.error("Failed to generate predictions. Please check the model and input features.")


