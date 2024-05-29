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

st.sidebar.header('Upload Your Data')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data = preprocess_data(data)
    st.sidebar.header('Forecast Settings')
    model_names = list(models.keys())
    selected_model_name = st.sidebar.selectbox('Choose a Forecasting Model:', model_names)

    last_date = data.index.max()
    end_date = st.sidebar.date_input('End Date for Prediction', last_date + timedelta(days=7), min_value=last_date + timedelta(days=1))

    if st.sidebar.button('Generate Future Forecast'):
        future_features = generate_future_dates(last_date, end_date)
        model = models[selected_model_name]
        _, predictions = make_predictions(model, future_features)

        if predictions is not None:
            forecast_df = pd.DataFrame({
                'Predicted Energy Usage': predictions
            }, index=future_features.index)
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
else:
    st.info('Please upload a CSV file to begin the forecasting process.')


