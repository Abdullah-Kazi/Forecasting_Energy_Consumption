import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Title and Introduction
st.title('Energy Forecasting Dashboard')
st.write("""
         Welcome to our advanced forecasting tool for energy management. This interactive dashboard leverages machine learning
         to predict future energy usage, helping businesses optimize their energy expenses by planning ahead effectively.
         """)

# Load Machine Learning Models
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

# Data Preprocessing Function
def preprocess_data(df):
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)  # Set 'Datetime' as the index if you want to use it directly from the index
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

# Function to Generate Future Date Features
def generate_future_dates(last_date, end_date):
    dates = pd.date_range(start=last_date + timedelta(hours=1), end=end_date, freq='H')
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

def make_predictions(model, features):
    try:
        predictions = model.predict(features)
        return features.index, predictions
    except Exception as e:
        st.error(f"Error in making predictions: {str(e)}")
        return None, None

# Sidebar Elements
st.sidebar.header('Upload Your Data')
use_test_data = st.sidebar.checkbox('Use Example Test Data')

if use_test_data:
    data = pd.read_csv('test.csv')  # Adjust path as needed
else:
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        st.info('Please upload a CSV file or select the option to use test data.')
        st.stop()

data = preprocess_data(data)

# Forecast Settings
st.sidebar.header('Forecast Settings')
model_names = list(models.keys())
selected_model_name = st.sidebar.selectbox('Choose a Forecasting Model:', model_names)

last_date = data.index.max()
end_date = st.sidebar.date_input('End Date for Prediction', last_date + timedelta(days=7), min_value=last_date + timedelta(days=1))

energy_rate = st.sidebar.number_input("Enter your energy rate (cost per kWh):", value=0.1)  # Default to 0.1 or appropriate rate
aggregate_by = st.sidebar.selectbox('Aggregate Predictions By:', ['Hourly', 'Daily', 'Weekly', 'Monthly', 'Yearly'])

if st.sidebar.button('Generate Future Forecast'):
    future_features = generate_future_dates(last_date, end_date)
    model = models[selected_model_name]
    _, predictions = make_predictions(model, future_features)

    if predictions is not None:
        forecast_df = pd.DataFrame({
            'Predicted Energy Usage': predictions
        }, index=future_features.index)
             st.write(forecast_df)

        # Monetary Calculations
        forecast_df['Cost'] = forecast_df['Predicted Energy Usage'] * energy_rate
        total_cost = forecast_df['Cost'].sum()

        # Data Aggregation
        if aggregate_by == 'Daily':
            forecast_df = forecast_df.resample('D').sum()
        elif aggregate_by == 'Weekly':
            forecast_df = forecast_df.resample('W').sum()
        elif aggregate_by == 'Monthly':
            forecast_df = forecast_df.resample('M').sum()
        elif aggregate_by == 'Yearly':
            forecast_df = forecast_df.resample('Y').sum()

        # Display Data and Plot
        st.write(f"Total Estimated Cost: ${total_cost:.2f}")
        st.write(forecast_df)

        # Plotting results
        st.subheader('Forecast Results for Future Energy Usage')
        plt.figure(figsize=(10, 5))
        plt.plot(forecast_df.index, forecast_df['Predicted Energy Usage'], label='Predicted Energy Usage (kWh)')
        plt.xlabel('Date')
        plt.ylabel('Energy Usage (kWh)')
        plt.title('Future Energy Usage Forecast')
        plt.legend()
        st.pyplot(plt)
    else:
        st.error("Failed to generate predictions. Please check the model and input features.")

