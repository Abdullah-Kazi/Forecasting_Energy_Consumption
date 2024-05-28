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
         
# Sidebar option for selecting the aggregation frequency
aggregation = st.sidebar.selectbox(
    'Choose Aggregation Level:',
    ['Hourly', 'Daily', 'Weekly', 'Monthly'],
    index=1  # Default to 'Daily'
)

def make_predictions(model, features):
    try:
        predictions = model.predict(features)
        return features.index, predictions
    except Exception as e:
        st.error(f"Error in making predictions: {str(e)}")
        return None, None

def add_features(df):
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df
         
def generate_dates(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    return dates

def prepare_features(dates):
    df = pd.DataFrame(dates, columns=['Datetime'])
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    df = add_features(df)
    return df

def aggregate_predictions(df, freq):
    if freq == 'Daily':
        df = df.resample('D').sum()  # Summing up the predictions for each day
    elif freq == 'Weekly':
        df = df.resample('W').sum()  # Summing up the predictions for each week
    elif freq == 'Monthly':
        df = df.resample('M').sum()  # Summing up the predictions for each month
    else:
        df = df.resample('H').sum()  # Keeping hourly data as it is
    return df

# Button to generate and display forecast
if st.sidebar.button('Generate Forecast'):
    dates = generate_dates(start_date, end_date)
    features = prepare_features(dates)
    model = models[selected_model_name]
    dates, predictions = make_predictions(model, features)

    if dates is not None and predictions is not None:
        # Create a DataFrame for plotting and manipulation
        forecast_df = pd.DataFrame({
            'Date': pd.to_datetime(dates),
            'Predicted Usage': predictions
        })
        forecast_df.set_index('Date', inplace=True)
        
        # Aggregate the data based on selected frequency
        aggregated_df = aggregate_predictions(forecast_df, aggregation)

        # Display the forecast results
        st.subheader('Forecast Results')
        plt.figure(figsize=(10, 5))
        plt.plot(aggregated_df.index, aggregated_df['Predicted Usage'], label='Aggregated Energy Usage')
        plt.xlabel('Date')
        plt.ylabel('Energy Usage (kWh)')
        plt.title(f'Energy Usage Forecast from {start_date} to {end_date} - Aggregated {aggregation}')
        plt.legend()
        st.pyplot(plt)

        # Display data in a table
        aggregated_df.reset_index(inplace=True)  # Resetting index for better table display
        st.write(aggregated_df)
    else:
        st.error("Failed to generate predictions. Please check the model and input features.")


