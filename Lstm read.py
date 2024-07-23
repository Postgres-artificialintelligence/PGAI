import numpy as np
import pandas as pd
import psycopg2
import pickle
from tensorflow.keras.models import load_model
import io
import os
from dotenv import load_dotenv
import tempfile

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

# Load the model and scaler from the PostgreSQL database
def load_model_from_db(model_name):
    # Connect to PostgreSQL database
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    
    # Retrieve model and scaler from the database
    cursor.execute("SELECT model_data, scaler_data FROM apple_model_storage WHERE model_name = %s", (model_name,))
    result = cursor.fetchone()
    
    model_bytes = result[0]
    scaler_bytes = result[1]
    
    cursor.close()
    conn.close()
    
    # Deserialize the model from a temporary file
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as temp_model_file:
        temp_model_file.write(model_bytes)
        temp_model_file.seek(0)
        model = load_model(temp_model_file.name)
    
    # Deserialize the scaler
    scaler = pickle.load(io.BytesIO(scaler_bytes))
    
    return model, scaler

# Function to read the trained model and predict future prices
def predict_future_prices(model, scaler, close_data, future_date, look_back=60):
    # Normalize the data
    scaled_data = scaler.transform(close_data)

    # Start with the last look_back days from the scaled data
    last_look_back_data = scaled_data[-look_back:]

    predictions = []
    
    # Calculate the number of future days to predict
    last_date = pd.to_datetime(data.index[-1])
    future_date = pd.to_datetime(future_date)
    future_days = (future_date - last_date).days
    
    if future_days <= 0:
        raise ValueError("Future date must be later than the last date in the dataset.")

    for _ in range(future_days):
        # Prepare the input data
        input_data = np.reshape(last_look_back_data, (1, look_back, 1))
        
        # Make the prediction
        prediction = model.predict(input_data)
        
        # Append the prediction to the predictions list
        predictions.append(prediction[0, 0])
        
        # Update the input data with the new prediction
        last_look_back_data = np.append(last_look_back_data[1:], prediction, axis=0)

    # Invert the normalization of the predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    return predictions[-1]

# Load the dataset from the PostgreSQL database
def load_stock_data():
    # Connect to PostgreSQL database
    conn = psycopg2.connect(**db_params)
    query = "SELECT * FROM apple_stock"
    data = pd.read_sql(query, conn)
    conn.close()
    return data

# Example usage
model_name = 'apple_lstm_model'
future_date = '2024-02-23'  # Change this to the specific future date you want to predict

# Load the dataset from the database
data = load_stock_data()

# Preprocessing the data
data['Date'] = pd.to_datetime(data['date'])  # Assuming the date column is named 'date'
data.set_index('Date', inplace=True)

# Select only the 'Close' column
close_data = data['close'].values  # Assuming the close price column is named 'close'
close_data = close_data.reshape(-1, 1)

# Load the model and scaler from the database
model, scaler = load_model_from_db(model_name)

# Predict the price for the specific future date
predicted_price = predict_future_prices(model, scaler, close_data, future_date)

print(f"Predicted Close Price on {future_date}: {predicted_price}")
