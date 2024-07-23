import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()

# Database connection parameters
db_params = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

# Function to load the model and scaler from PostgreSQL
def load_model_and_scaler_from_postgres(model_name):
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    select_query = sql.SQL("""
        SELECT model_data, scaler_data
        FROM tesla_model_storage
        WHERE model_name = %s;
    """)
    cursor.execute(select_query, (model_name,))
    model_data, scaler_data = cursor.fetchone()

    model = pickle.loads(model_data)
    scaler = pickle.loads(scaler_data)

    cursor.close()
    conn.close()
    
    return model, scaler

# Function to fetch historical data from PostgreSQL
def fetch_historical_data():
    conn = psycopg2.connect(**db_params)
    query = "SELECT * FROM tesla_stock ORDER BY date ASC"
    historical_data = pd.read_sql(query, conn, parse_dates=['date'])
    conn.close()
    historical_data.set_index('date', inplace=True)
    return historical_data

# Function to predict close prices for specific dates
def predict_for_dates(dates, historical_data, model, scaler):
    # Create a DataFrame for the input dates
    date_df = pd.DataFrame({'date': pd.to_datetime(dates)})
    date_df.set_index('date', inplace=True)

    # Feature engineering on historical data
    historical_data['Year'] = historical_data.index.year
    historical_data['Month'] = historical_data.index.month
    historical_data['Day'] = historical_data.index.day
    historical_data['DayOfWeek'] = historical_data.index.dayofweek

    # Adding rolling averages and other features
    historical_data['MA10'] = historical_data['close'].rolling(window=10).mean()
    historical_data['MA50'] = historical_data['close'].rolling(window=50).mean()
    historical_data['MA200'] = historical_data['close'].rolling(window=200).mean()
    historical_data['Volatility'] = historical_data['close'].rolling(window=10).std()

    # Predicting for future dates
    future_data = pd.DataFrame(index=date_df.index)
    future_data['Year'] = future_data.index.year
    future_data['Month'] = future_data.index.month
    future_data['Day'] = future_data.index.day
    future_data['DayOfWeek'] = future_data.index.dayofweek

    # Assume the future rolling averages and volatility are same as the last available historical data
    last_row = historical_data.iloc[-1]
    future_data['MA10'] = last_row['MA10']
    future_data['MA50'] = last_row['MA50']
    future_data['MA200'] = last_row['MA200']
    future_data['Volatility'] = last_row['Volatility']

    # Define the features
    X_dates = future_data[['Year', 'Month', 'Day', 'DayOfWeek', 'MA10', 'MA50', 'MA200', 'Volatility']]

    # Standardize the features
    X_dates_scaled = scaler.transform(X_dates)

    # Predict the close prices
    predictions = model.predict(X_dates_scaled)
    
    # Add predictions to DataFrame
    future_data['Predicted_Close'] = predictions
    
    return future_data

# Load the model and scaler
model, scaler = load_model_and_scaler_from_postgres('tesla_gradient_boosting_model')

# Fetch historical data from PostgreSQL
historical_data = fetch_historical_data()

# Example usage
dates = ['2024-06-19', '2024-06-20', '2024-06-21']
predictions_df = predict_for_dates(dates, historical_data, model, scaler)

# Print the predictions
print(predictions_df[['Predicted_Close']])

# Plot the predictions
plt.figure(figsize=(10, 6))
plt.plot(predictions_df.index, predictions_df['Predicted_Close'], label='Predicted Close Prices', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Predicted Close Price')
plt.title('Predicted Close Prices for Specific Dates')
plt.legend()
plt.show()
