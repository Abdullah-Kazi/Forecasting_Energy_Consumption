import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb

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

st.sidebar.header('Upload Your Data')
st.sidebar.write("Please upload a CSV file with columns: Datetime, Energy Consumption")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

def preprocess_data(df):
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data = preprocess_data(data)
    energy_column = data.columns[0]  # Dynamically identifying the energy column


def generate_dates(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    return dates


def make_predictions(model, features):
    try:
        predictions = model.predict(features)
        return features.index, predictions
    except Exception as e:
        st.error(f"Error in making predictions: {str(e)}")
        return None, None
             
feature_names = ['hour', 'dayofweek', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']  # Used for consistent feature handling

def aggregate_predictions(df, freq):
    if freq == 'Daily':
        df = df.resample('D').sum()
    elif freq == 'Weekly':
        df = df.resample('W').sum()
    elif freq == 'Monthly':
        df = df.resample('M').sum()
    else:
        df = df.resample('H').sum()
    return df

def calculate_costs(df, cost_per_kwh):
    df['Cost'] = df['Predicted Usage'] * cost_per_kwh
    return df

if uploaded_file is not None:
    st.sidebar.header('Forecast Settings')
    model_names = list(models.keys())
    selected_model_name = st.sidebar.selectbox('Choose a Forecasting Model:', model_names)

    today = datetime.today().date()
    tomorrow = today + timedelta(days=1)
    start_date = st.sidebar.date_input('Start date', tomorrow)
    end_date = st.sidebar.date_input('End date', tomorrow + timedelta(days=30))
    if start_date > end_date:
        st.sidebar.error('Error: End date must fall after start date.')

    aggregation = st.sidebar.selectbox(
        'Choose Aggregation Level:',
        ['Hourly', 'Daily', 'Weekly', 'Monthly'],
        index=1  # Default to 'Daily'
    )

    cost_per_kwh = st.sidebar.number_input('Cost per kWh in $', value=0.10, min_value=0.01, max_value=1.00, step=0.01)

    if st.sidebar.button('Generate Forecast'):
        dates = generate_dates(start_date, end_date)
        features = prepare_features(dates)
        model = models[selected_model_name]
        dates, predictions = make_predictions(model, feature_names)

        if dates is not None and predictions is not None:
            forecast_df = pd.DataFrame({
                'Date': pd.to_datetime(dates),
                'Predicted Usage': predictions
            })
            forecast_df.set_index('Date', inplace=True)
            aggregated_df = aggregate_predictions(forecast_df, aggregation)
            cost_df = calculate_costs(aggregated_df.copy(), cost_per_kwh)

            # Plot for Energy Usage
            st.subheader('Forecast Results for Energy Usage')
            plt.figure(figsize=(10, 5))
            plt.plot(aggregated_df.index, aggregated_df['Predicted Usage'], label='Energy Usage (kWh)')
            plt.xlabel('Date')
            plt.ylabel('Energy Usage (kWh)')
            plt.title(f'Energy Usage from {start_date} to {end_date} - Aggregated {aggregation}')
            plt.legend()
            st.pyplot(plt)

            # Plot for Cost
            st.subheader('Forecast Results with Cost Analysis')
            plt.figure(figsize=(10, 5))
            plt.plot(cost_df.index, cost_df['Cost'], label='Forecasted Cost')
            plt.xlabel('Date')
            plt.ylabel('Cost ($)')
            plt.title(f'Cost from {start_date} to {end_date} - Aggregated {aggregation}')
            plt.legend()
            st.pyplot(plt)
            st.write(cost_df)
        else:
            st.error("Failed to generate predictions. Please check the model and input features.")

    if st.sidebar.button('Predict Uploaded Data'):
        try:
            prepared_data = add_features(data)
            predictions = make_predictions(models[selected_model_name], prepared_data.drop(columns=[energy_column]))
            if predictions[0] is not None:
                prepared_data['Predictions'] = predictions[1]
                st.write("Predictions on Uploaded Data:")
                st.write(prepared_data)
        except Exception as e:
            st.error(f"Failed to predict on uploaded data: {str(e)}")


