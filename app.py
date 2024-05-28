import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pickle
import matplotlib.pyplot as plt

# Function to load the trained models
@st.cache(allow_output_mutation=True)

def load_models():
    models = {}
    try:
        for model_name in ['XGBoost_model', 'Prophet_model', 'LightGBM_model']:
            with open(f'{model_name}.pkl', 'rb') as file:
                models[model_name] = pickle.load(file)
        return models, None  # Return models and 'None' for error
    except Exception as e:
        return None, str(e)  # Return None for models and the error message

# Generate future dates based on user input
def generate_future_dates(years, months):
    end_date = datetime.now() + timedelta(days=(years * 365) + (months * 30))
    future_dates = pd.date_range(start=datetime.now(), end=end_date, freq='H')
    future_df = pd.DataFrame(future_dates, columns=['Datetime'])
    future_df.set_index('Datetime', inplace=True)
    return future_df

# Add necessary features to the future dates DataFrame
def add_features(df):
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

# Make predictions using the selected model and features DataFrame
def make_predictions(model, features):
    try:
        predictions = model.predict(features)
        return features.index, predictions
    except Exception as e:
        st.error(f"Error in making predictions: {str(e)}")
        return None, None

# Streamlit UI setup
st.title('Energy Usage Forecasting')

# Sidebar for user input
st.sidebar.header('Specify Forecast Details')
models = load_models()
model_names = list(models.keys())
selected_model_name = st.sidebar.selectbox('Choose a model:', model_names)
years = st.sidebar.number_input('Years into the future:', min_value=0, max_value=20, value=5)
months = st.sidebar.number_input('Additional months:', min_value=0, max_value=11, value=2)

# Button to show forecast
if st.sidebar.button('Show Forecast'):
    model = models[selected_model_name]
    future_df = generate_future_dates(years, months)
    future_df = add_features(future_df)
    dates, predictions = make_predictions(model, future_df)

    if dates is not None and predictions is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(dates, predictions, label='Forecasted Energy Usage')
        plt.xlabel('Date')
        plt.ylabel('Energy Usage')
        plt.title('Future Energy Usage Forecast')
        plt.legend()
        st.pyplot(plt)

        # Display predictions in a DataFrame
        prediction_df = pd.DataFrame({
            'Date': dates,
            'Predicted Energy Usage': predictions
        })
        st.write("Forecasted Energy Usage:")
        st.dataframe(prediction_df)
    else:
        st.error("Failed to generate predictions. Please check the model and input features.")

