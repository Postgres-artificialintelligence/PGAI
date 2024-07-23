import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine
import pickle
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Database connection parameters
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')

# Create the database connection
engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
cursor = conn.cursor()

# Function to load model from PostgreSQL
def load_model_from_db(model_name):
    query = "SELECT model_data FROM google_model_storage WHERE model_name = %s;"
    cursor.execute(query, (model_name,))
    result = cursor.fetchone()
    
    if result is None:
        raise ValueError(f"Model with name '{model_name}' not found in database.")
    
    model_data = result[0]
    model = pickle.loads(model_data)
    return model

# Load the model from the database
model_name = 'sarimax_google_stock_model'
model = load_model_from_db(model_name)

# Load data from the PostgreSQL table
query = "SELECT date, close FROM google_stock"
df = pd.read_sql(query, engine)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Ensure 'close' is numeric
df['close'] = pd.to_numeric(df['close'], errors='coerce')
df = df.dropna(subset=['close'])
df = df[np.isfinite(df['close'])]

# Function to get forecast for a specific date
def get_forecast_for_date(model, start_date, target_date):
    # Calculate the number of days between the start date and the target date
    days_diff = (target_date - start_date).days
    
    if days_diff <= 0:
        raise ValueError("Target date must be after the start date.")
    
    # Forecasting with the loaded model
    forecast = model.get_forecast(steps=days_diff)
    forecast_mean = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()
    
    # Get the forecast value for the target date
    forecast_value = forecast_mean.iloc[-1]
    conf_int = forecast_conf_int.iloc[-1]
    
    return forecast_value, conf_int

# Example usage
# Set the start date (last date in the training data)
start_date = df.index[-1]

# Set the target date for which you want the forecast
target_date = pd.Timestamp('2024-07-16')

# Get the forecast for the target date
forecast_value, conf_int = get_forecast_for_date(model, start_date, target_date)
print(f"Forecast value for {target_date}: {forecast_value}")
print(f"95% Confidence interval: {conf_int}")

# Close the database connection
cursor.close()
conn.close()

# Plotting historical data and forecasts from the loaded model
forecast_dates = pd.date_range(start=start_date + pd.DateOffset(days=1), periods=(target_date - start_date).days, freq='D')
forecast = model.get_forecast(steps=(target_date - start_date).days)
forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

plt.figure(figsize=(14,7))
plt.plot(df.index, df['close'], label='Actual')
plt.plot(forecast_dates, forecast_mean.values, color='red', label='Forecast')
plt.fill_between(forecast_dates, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink', alpha=0.5, label='Confidence Intervals')

plt.title('Google Stock Closing Prices - Forecast')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
