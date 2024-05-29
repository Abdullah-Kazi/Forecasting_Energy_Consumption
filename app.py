import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pickle
import matplotlib.pyplot as plt

st.title('Energy Management Forecasting Tool')
st.write("""
         Welcome to our advanced forecasting tool for energy management. 
         This application provides future energy consumption predictions to help businesses optimize their energy strategies and reduce costs.
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
    date_range = pd.date_range(start=last_date + timedelta(hours=1), end=end_date, freq='H')
    future_df = pd.DataFrame(index=date_range, columns=['hour', 'dayofweek', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear'])
    future_df['hour'] = future_df.index.hour
    future_df['dayofweek'] = future_df.index.dayofweek
    future_df['month'] = future_df.index.month
    future_df['year'] = future_df.index.year
    future_df['dayofyear'] = future_df.index.dayofyear
    future_df['dayofmonth'] = future_df.index.day
    future_df['weekofyear'] = future_df.index.isocalendar().week
    return future_df

def make_predictions(model, future_features):
    try:
        predictions = model.predict(future_features)
        logging.info('Forecast completed successfully.')
        return future_features.index, predictions
    except Exception as e:
        logging.error(f'Failed to make forecast due to: {e}')
        return None, None

def aggregate_data(df, aggregation_level):
    aggregation_dict = {
        'Hourly': 'H',
        'Daily': 'D',
        'Weekly': 'W',
        'Monthly': 'M',
        'Yearly': 'Y'
    }
    return df.resample(aggregation_dict[aggregation_level]).sum()


def calculate_costs(df, cost_per_kwh):
    df['Cost'] = df['Predicted Energy Usage'] * cost_per_kwh
    return df

st.sidebar.header('Upload Your Data')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data = preprocess_data(data)

    st.sidebar.header('Forecast Settings')
    model_names = list(models.keys())
    selected_model_name = st.sidebar.selectbox('Choose a Forecasting Model:', model_names)
    aggregation = st.sidebar.selectbox(
        'Choose Aggregation Level:',
        ['Hourly', 'Daily', 'Weekly', 'Monthly', 'Yearly'],
        index=0  # Default to 'Hourly'
    )
    cost_per_kwh = st.sidebar.number_input('Cost per kWh in $', value=0.10, min_value=0.01, max_value=1.00, step=0.01)

    last_date = data.index.max()
    end_date = st.sidebar.date_input('End Date for Prediction', last_date + timedelta(days=7), min_value=last_date + timedelta(days=1))

    if st.sidebar.button('Generate Forecast'):
        future_features = generate_future_dates(last_date, end_date)
        all_features = pd.concat([data, future_features], sort=False).fillna(0)  # Combining existing and future data
        model = models[selected_model_name]
        _, predictions = make_predictions(model, future_features)

        if predictions is not None:
            all_features['Predicted Energy Usage'] = predictions
            aggregated_df = aggregate_data(all_features[['Predicted Energy Usage']], aggregation)
            cost_df = calculate_costs(aggregated_df.copy(), cost_per_kwh)

            st.write("Forecasted Energy Usage and Costs:")
            st.dataframe(cost_df.style.format({'Predicted Energy Usage': "{:.2f}", 'Cost': "${:.2f}"}))

            # Plot for Energy Usage
            st.subheader('Forecast Results for Energy Usage')
            plt.figure(figsize=(10, 5))
            plt.plot(aggregated_df.index, aggregated_df['Predicted Energy Usage'], label='Energy Usage (kWh)')
            plt.xlabel('Date')
            plt.ylabel('Energy Usage (kWh)')
            plt.title(f'Energy Usage from {data.index.min()} to {end_date} - Aggregated {aggregation}')
            plt.legend()
            st.pyplot(plt)

            # Plot for Cost
            st.subheader('Forecast Results with Cost Analysis')
            plt.figure(figsize=(10, 5))
            plt.plot(cost_df.index, cost_df['Cost'], label='Forecasted Cost')
            plt.xlabel('Date')
            plt.ylabel('Cost ($)')
            plt.title(f'Cost from {data.index.min()} to {end_date} - Aggregated {aggregation}')
            plt.legend()
            st.pyplot(plt)
        else:
            st.error("Failed to generate predictions. Please check the model and input features.")
else:
    st.info('Please upload a CSV file to begin the forecasting process.')

