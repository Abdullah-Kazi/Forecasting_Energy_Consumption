import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pickle
import matplotlib.pyplot as plt

# Function to load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    with open('XGBoost_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function to generate future dates based on user input
def generate_future_dates(years, months):
    end_date = datetime.now() + timedelta(days=(years * 365) + (months * 30))
    future_dates = pd.date_range(start=datetime.now(), end=end_date, freq='H')
    future_df = pd.DataFrame(future_dates, columns=['Datetime'])
    future_df.set_index('Datetime', inplace=True)
    return future_df

# Function to add necessary features to the future dates DataFrame
def add_features(df):
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

# Function to make predictions using the model and features DataFrame
def make_predictions(model, features):
    try:
        if set(model.get_booster().feature_names) != set(features.columns):
            raise ValueError("Feature mismatch between the model and input data")
        predictions = model.predict(features)
        return features.index, predictions
    except Exception as e:
        st.error(f"Error in making predictions: {str(e)}")
        return None, None

# Streamlit UI
st.title('Energy Usage Forecasting')

# Sidebar for user input
st.sidebar.header('Specify Forecast Details')
years = st.sidebar.number_input('Years into the future:', min_value=0, max_value=20, value=5)
months = st.sidebar.number_input('Additional months:', min_value=0, max_value=11, value=2)

# Load the model
model = load_model()

# Button to show forecast
if st.sidebar.button('Show Forecast'):
    # Generate future dates and add features
    future_df = generate_future_dates(years, months)
    future_df = add_features(future_df)
    
    # Make predictions
    dates, predictions = make_predictions(model, future_df)
    
    # Plot the predictions
    if dates is not None and predictions is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(dates, predictions, label='Forecasted Energy Usage')
        plt.xlabel('Date')
        plt.ylabel('Energy Usage')
        plt.title('Future Energy Usage Forecast')
        plt.legend()
        st.pyplot(plt)
    else:
        st.error("Failed to generate predictions. Please check the model and input features.")
