import streamlit as st
import pandas as pd
from datetime import datetime
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

def make_predictions(model, features):
    try:
        predictions = model.predict(features)
        return features.index, predictions
    except Exception as e:
        st.error(f"Error in making predictions: {str(e)}")
        return None, None

def calculate_costs(df, cost_per_kwh):
    df['Cost'] = df['Energy Consumption'] * cost_per_kwh
    return df

st.sidebar.header('Upload Your Data')
st.sidebar.write("Please upload a CSV file with columns: Datetime, Energy Consumption")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data = preprocess_data(data)

    st.sidebar.header('Prediction Settings')
    selected_model_name = st.sidebar.selectbox('Choose a Forecasting Model:', list(models.keys()))
    cost_per_kwh = st.sidebar.number_input('Cost per kWh in $', value=0.10, min_value=0.01, max_value=1.00, step=0.01)

    min_date, max_date = data.index.min(), data.index.max()
    start_date = st.sidebar.date_input('Start Date', min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input('End Date', min_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.sidebar.error('Error: End date must be after start date.')

    if st.sidebar.button('Predict Uploaded Data'):
        filtered_data = data[start_date:end_date]
        if not filtered_data.empty:
            model = models[selected_model_name]
            features = filtered_data.drop(columns=['Energy Consumption'])
            dates, predictions = make_predictions(model, features)
            
            if predictions is not None:
                filtered_data['Predicted Usage'] = predictions
                cost_df = calculate_costs(filtered_data, cost_per_kwh)
                
                st.write("Predictions on Uploaded Data:")
                st.write(filtered_data[['Energy Consumption', 'Predicted Usage', 'Cost']])

                # Plotting results
                st.subheader('Forecast Results for Energy Usage')
                plt.figure(figsize=(10, 5))
                plt.plot(filtered_data.index, filtered_data['Energy Consumption'], label='Actual Energy Usage (kWh)')
                plt.plot(filtered_data.index, filtered_data['Predicted Usage'], label='Predicted Usage (kWh)', linestyle='--')
                plt.xlabel('Date')
                plt.ylabel('Energy Usage (kWh)')
                plt.title('Comparison of Actual and Predicted Energy Usage')
                plt.legend()
                st.pyplot(plt)

                # Cost plot
                st.subheader('Forecast Results with Cost Analysis')
                plt.figure(figsize=(10, 5))
                plt.plot(cost_df.index, cost_df['Cost'], label='Forecasted Cost', color='red')
                plt.xlabel('Date')
                plt.ylabel('Cost ($)')
                plt.title('Cost Analysis of Predicted Energy Usage')
                plt.legend()
                st.pyplot(plt)
            else:
                st.error("Failed to generate predictions. Please check the model and input features.")
        else:
            st.error("No data available for the selected date range. Please choose a different range.")





