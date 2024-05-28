import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pickle
import matplotlib.pyplot as plt

# Load and cache models
@st.cache(allow_output_mutation=True)
def load_models():
    models = {}
    try:
        for model_name in ['XGBoost_model', 'Prophet_model', 'LightGBM_model']:
            with open(f'{model_name}.pkl', 'rb') as file:
                models[model_name] = pickle.load(file)
    except Exception as e:
        st.error(f"Failed to load model due to: {e}")
    return models

# Generate future dates
def generate_future_dates(years, months):
    end_date = datetime.now() + timedelta(days=(years * 365) + (months * 30))
    future_dates = pd.date_range(start=datetime.now(), end=end_date, freq='H')
    future_df = pd.DataFrame(future_dates, columns=['Datetime'])
    future_df.set_index('Datetime', inplace=True)
    return future_df

# Add features to dates
def add_features(df):
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

# Function to make predictions
def make_predictions(model, features):
    try:
        predictions = model.predict(features)
        return features.index, predictions
    except Exception as e:
        st.error(f"Error in making predictions: {str(e)}")
        return None, None

# Aggregation function
def aggregate_predictions(dates, predictions, freq):
    df = pd.DataFrame({'Date': dates, 'Predicted Energy Usage': predictions})
    df.set_index('Date', inplace=True)
    if freq == 'Daily':
        resampled = df.resample('D').mean()
    elif freq == 'Weekly':
        resampled = df.resample('W').mean()
    elif freq == 'Monthly':
        resampled = df.resample('M').mean()
    elif freq == 'Yearly':
        resampled = df.resample('A').mean()
    return resampled

# Streamlit UI
st.title('Energy Usage Forecasting')
st.sidebar.header('Specify Forecast Details')
models = load_models()
model_names = list(models.keys())
selected_model_name = st.sidebar.selectbox('Choose a model:', model_names)
years = st.sidebar.number_input('Years into the future:', min_value=0, max_value=20, value=5)
months = st.sidebar.number_input('Additional months:', min_value=0, max_value=11, value=2)
frequency = st.sidebar.selectbox('Aggregate Frequency:', ['Daily', 'Weekly', 'Monthly', 'Yearly'])

# Show forecast button
if st.sidebar.button('Show Forecast'):
    model = models[selected_model_name]
    future_df = generate_future_dates(years, months)
    future_df = add_features(future_df)
    dates, predictions = make_predictions(model, future_df)
    if dates is not None and predictions is not None:
        aggregated_data = aggregate_predictions(dates, predictions, frequency)
        plt.figure(figsize=(10, 5))
        plt.plot(aggregated_data.index, aggregated_data['Predicted Energy Usage'], label='Aggregated Energy Usage')
        plt.xlabel('Date')
        plt.ylabel('Energy Usage')
        plt.title(f'Future Energy Usage Forecast - {frequency}')
        plt.legend()
        st.pyplot(plt)
        st.write(f"Forecasted Energy Usage ({frequency}):")
        st.dataframe(aggregated_data)
    else:
        st.error("Failed to generate predictions. Please check the model and input features.")

