import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
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

# Function to connect to PostgreSQL
def connect_to_postgres():
    return psycopg2.connect(**db_params)

# Function to drop and create a new table
def recreate_model_storage_table():
    conn = connect_to_postgres()
    cursor = conn.cursor()
    
    drop_table_query = "DROP TABLE IF EXISTS tesla_model_storage;"
    cursor.execute(drop_table_query)
    
    create_table_query = """
    CREATE TABLE tesla_model_storage (
        id SERIAL PRIMARY KEY,
        model_name TEXT NOT NULL UNIQUE,
        model_data BYTEA,
        scaler_data BYTEA
    );
    """
    cursor.execute(create_table_query)
    
    conn.commit()
    cursor.close()
    conn.close()

# Function to save the model and scaler to PostgreSQL
def save_model_and_scaler_to_postgres(model_name, model, scaler):
    conn = connect_to_postgres()
    cursor = conn.cursor()
    
    model_data = pickle.dumps(model)
    scaler_data = pickle.dumps(scaler)
    
    insert_query = sql.SQL("""
    INSERT INTO tesla_model_storage (model_name, model_data, scaler_data)
    VALUES (%s, %s, %s)
    ON CONFLICT (model_name) 
    DO UPDATE SET 
        model_data = EXCLUDED.model_data,
        scaler_data = EXCLUDED.scaler_data;
    """)
    
    cursor.execute(insert_query, (model_name, model_data, scaler_data))
    conn.commit()
    cursor.close()
    conn.close()

# Function to load the model and scaler from PostgreSQL
def load_model_and_scaler_from_postgres(model_name):
    conn = connect_to_postgres()
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
    conn = connect_to_postgres()
    query = "SELECT * FROM tesla_stock"
    historical_data = pd.read_sql(query, conn, parse_dates=['date'])
    conn.close()
    historical_data.set_index('date', inplace=True)
    return historical_data

# Function to perform feature engineering
def perform_feature_engineering(historical_data):
    historical_data['Year'] = historical_data.index.year
    historical_data['Month'] = historical_data.index.month
    historical_data['Day'] = historical_data.index.day
    historical_data['DayOfWeek'] = historical_data.index.dayofweek

    # Adding rolling averages and other features
    historical_data['MA10'] = historical_data['close'].rolling(window=10).mean()
    historical_data['MA50'] = historical_data['close'].rolling(window=50).mean()
    historical_data['MA200'] = historical_data['close'].rolling(window=200).mean()
    historical_data['Volatility'] = historical_data['close'].rolling(window=10).std()
    
    return historical_data

# Function to predict close prices for specific dates
def predict_for_dates(dates, historical_data, model, scaler):
    # Create a DataFrame for the input dates
    date_df = pd.DataFrame({'date': pd.to_datetime(dates)})
    date_df.set_index('date', inplace=True)

    # Perform feature engineering
    historical_data = perform_feature_engineering(historical_data)

    # Predicting for future dates
    future_data = pd.DataFrame(index=date_df.index)
    future_data['Year'] = future_data.index.year
    future_data['Month'] = future_data.index.month
    future_data['Day'] = future_data.index.day
    future_data['DayOfWeek'] = future_data.index.dayofweek

    # Assume the future rolling averages and volatility are the same as the last available historical data
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

# Example usage
if __name__ == "__main__":
    # Recreate the tesla_model_storage table
    recreate_model_storage_table()
    
    # Load historical data
    historical_data = fetch_historical_data()
    
    # Perform feature engineering on historical data
    historical_data = perform_feature_engineering(historical_data)
    
    # Train a new model (example using Gradient Boosting Regressor with modified parameters)
    features = ['Year', 'Month', 'Day', 'DayOfWeek', 'MA10', 'MA50', 'MA200', 'Volatility']
    X = historical_data[features].dropna()
    y = historical_data['close'].loc[X.index]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.5, max_depth=10)
    model.fit(X_scaled, y)
    
    # Save the new model and scaler to PostgreSQL
    save_model_and_scaler_to_postgres('tesla_gradient_boosting_model', model, scaler)
    
    # Load the model and scaler
    model, scaler = load_model_and_scaler_from_postgres('tesla_gradient_boosting_model')
    
    # Predict for specific dates
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
