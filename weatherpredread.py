import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import matplotlib.pyplot as plt
import pickle

# Load the data
df = pd.read_csv('/Users/moizibrar/Downloads/weatherAUS.csv')

# Filter data for Melbourne
melb = df[df['Location'] == 'Melbourne'].copy()
melb['Date'] = pd.to_datetime(melb['Date'])

# Prepare data for SARIMA
data = melb[['Date', 'Temp3pm']].dropna().copy()
data.columns = ['ds', 'y']  # Ensure columns are named 'ds' and 'y'
data.set_index('ds', inplace=True)

# Ensure the data index has a daily frequency
data = data.asfreq('D')

# Load the model parameters using pickle
try:
    with open('/Users/moizibrar/Downloads/final/Weather_Predictions.pkl', 'rb') as f:
        results = pickle.load(f)
    print("Model parameters loaded successfully.")
except Exception as e:
    print(f"Error loading model parameters: {e}")

# Predict using the SARIMA model
try:
    # Predict for the next 365 days
    forecast = results.get_forecast(steps=365)
    forecast_df = forecast.summary_frame()

    # Create a date range for the forecast
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=365, freq='D')
    forecast_df.index = future_dates  # Ensure the forecast dates are properly indexed

    # Plot the forecast
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['y'], label='Actual')
    plt.plot(future_dates, forecast_df['mean'], label='Forecast', linestyle='--')
    plt.fill_between(future_dates, forecast_df['mean_ci_lower'], forecast_df['mean_ci_upper'], color='k', alpha=0.1)
    plt.title('Forecast for Temperature at 3pm in Melbourne')
    plt.xlabel('Date')
    plt.ylabel('Temp3pm')
    plt.legend()
    plt.show()

    # Example of making prediction for a specific date
    specific_date = '2017-05-05'
    specific_date_pd = pd.to_datetime(specific_date)

    try:
        # Ensure specific_date is beyond the training data range
        if specific_date_pd > data.index[-1]:
            steps = (specific_date_pd - data.index[-1]).days
            specific_date_data = results.get_forecast(steps=steps).summary_frame()
            specific_date_data.index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
            prediction = specific_date_data.loc[specific_date, "mean"]
            print(f'Prediction for specific date ({specific_date}): {prediction:.2f}')
        else:
            specific_date_data = results.get_prediction(start=specific_date_pd, end=specific_date_pd).summary_frame()
            prediction = specific_date_data["mean"].values[0]
            print(f'Prediction for specific date ({specific_date}): {prediction:.2f}')
    except ValueError:
        print(f"No prediction available for specific date {specific_date}")

except Exception as e:
    print(f"Error predicting with SARIMA model: {e}")