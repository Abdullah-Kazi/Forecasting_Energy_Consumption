# Energy Forecasting Dashboard

Welcome to the Energy Forecasting Dashboard! This advanced forecasting tool leverages machine learning to predict future energy usage, helping businesses optimize their energy expenses by planning ahead effectively.

## Features

- Interactive dashboard for visualizing and analyzing energy usage forecasts
- Multiple machine learning models available for generating predictions
- Customizable forecast settings, including end date and energy rate
- Aggregation of predictions by hourly, daily, weekly, monthly, or yearly intervals
- Monetary calculations to estimate the cost of future energy consumption
- User-friendly interface for uploading historical energy data or using example test data

## Installation

To run the Energy Forecasting Dashboard locally, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/energy-forecasting-dashboard.git
   ```

2. Navigate to the project directory:
   ```
   cd energy-forecasting-dashboard
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

5. Open your web browser and visit `http://localhost:8501` to access the dashboard.

## Usage

1. Upload your historical energy usage data as a CSV file or select the option to use example test data.
2. Choose a forecasting model from the dropdown menu in the sidebar.
3. Specify the end date for the prediction using the date picker.
4. Enter your energy rate (cost per kWh) in the sidebar.
5. Select the desired aggregation level for the predictions (hourly, daily, weekly, monthly, or yearly).
6. Click the "Generate Future Forecast" button to generate predictions.
7. View the forecasted energy usage, estimated costs, and visualizations in the main panel.

## Models

The Energy Forecasting Dashboard supports the following machine learning models:

- XGBoost
- Linear Regression
- LightGBM
- Decision Tree
- Random Forest
- Prophet

The trained models are stored as pickle files in the repository. You can replace or add new models by updating the corresponding pickle files.

## Dependencies

The project relies on the following dependencies:

- Python 3.7+
- Streamlit
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost
- LightGBM
- Prophet

You can install the dependencies by running `pip install -r requirements.txt`.

## Contributing

Contributions to the Energy Forecasting Dashboard are welcome! If you find any bugs, have suggestions for improvements, or would like to add new features, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions or inquiries, please contact [kaziabdullah61@gmail.com](mailto:kaziabdullah61@gmail.com).

Happy forecasting!
