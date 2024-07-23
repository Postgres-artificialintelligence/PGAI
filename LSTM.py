import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import psycopg2
import pickle
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')

# Function to get data from PostgreSQL
def get_data_from_db():
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )
    query = "SELECT date, close FROM apple_stock"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Load the dataset from PostgreSQL
data = get_data_from_db()

# Preprocessing the data
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Select only the 'Close' column
close_data = data['close'].values
close_data = close_data.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_data)

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create a dataset with a look_back period
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60
X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Create the LSTM model
model = Sequential()
model.add(Input(shape=(look_back, 1)))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Define the checkpoint and early stopping callbacks
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', verbose=2, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1, validation_data=(X_test, y_test),
                    callbacks=[checkpoint, early_stopping])

# Load the best model
best_model = load_model('best_model.keras')

# Save the model and scaler into PostgreSQL
def save_model_to_postgres(model, scaler, model_name):
    # Save model to a file first
    model.save('temp_model.h5')
    
    # Read the model file into a bytes object
    with open('temp_model.h5', 'rb') as model_file:
        model_byte_data = model_file.read()
    
    # Serialize the scaler
    scaler_byte_data = pickle.dumps(scaler)
    
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )
    cur = conn.cursor()
    
    # Drop the table if it exists and create a new table
    cur.execute("""
        DROP TABLE IF EXISTS apple_model_storage;
        CREATE TABLE apple_model_storage (
            id SERIAL PRIMARY KEY,
            model_name TEXT UNIQUE NOT NULL,
            model_data BYTEA,
            scaler_data BYTEA
        );
    """)
    
    # Insert serialized model and scaler into the database
    cur.execute("""
        INSERT INTO apple_model_storage (model_name, model_data, scaler_data)
        VALUES (%s, %s, %s)
        ON CONFLICT (model_name) 
        DO UPDATE SET model_data = EXCLUDED.model_data, scaler_data = EXCLUDED.scaler_data;
    """, (model_name, psycopg2.Binary(model_byte_data), psycopg2.Binary(scaler_byte_data)))
    
    conn.commit()
    cur.close()
    conn.close()

# Save the best model and scaler
save_model_to_postgres(best_model, scaler, 'apple_lstm_model')

# Function to load the model and scaler from PostgreSQL
def load_model_from_postgres(model_name):
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )
    cur = conn.cursor()
    
    # Fetch serialized model and scaler from the database
    cur.execute("SELECT model_data, scaler_data FROM apple_model_storage WHERE model_name = %s", (model_name,))
    result = cur.fetchone()
    if result is None:
        cur.close()
        conn.close()
        raise ValueError(f"Model or scaler not found for model name: {model_name}")
    
    model_data, scaler_data = result
    
    # Deserialize model and scaler
    with open('temp_model.h5', 'wb') as model_file:
        model_file.write(model_data)
    model = load_model('temp_model.h5')
    scaler = pickle.loads(scaler_data)
    
    cur.close()
    conn.close()
    
    return model, scaler

# Load the model and scaler
loaded_model, loaded_scaler = load_model_from_postgres('apple_lstm_model')

# Make predictions with the loaded model
train_predict = loaded_model.predict(X_train)
test_predict = loaded_model.predict(X_test)

# Invert predictions
train_predict = loaded_scaler.inverse_transform(train_predict)
y_train = loaded_scaler.inverse_transform([y_train])
test_predict = loaded_scaler.inverse_transform(test_predict)
y_test = loaded_scaler.inverse_transform([y_test])

# Function to make predictions for a specific date
def predict_for_date(date_str, look_back=60):
    date = pd.to_datetime(date_str)
    if date not in data.index:
        raise ValueError("The date provided is not in the dataset.")
    
    # Find the index of the date
    date_idx = data.index.get_loc(date)
    
    if date_idx < look_back:
        raise ValueError("Not enough data to make a prediction for this date.")
    
    # Prepare the input data
    input_data = scaled_data[date_idx-look_back:date_idx].reshape(1, look_back, 1)
    
    # Make the prediction
    prediction = loaded_model.predict(input_data)
    prediction = loaded_scaler.inverse_transform(prediction)
    
    return prediction[0][0]

# Example usage: Predict the closing price for a specific date
prediction_date = '2024-07-07'  # Replace with your desired date
try:
    predicted_price = predict_for_date(prediction_date)
    print(f"The predicted closing price for {prediction_date} is: {predicted_price:.2f}")
except ValueError as e:
    print(e)

# Plot baseline and predictions
plt.figure(figsize=(12, 6))
plt.plot(data.index, close_data, label='Actual Close Price')
plt.plot(data.index[look_back:len(train_predict)+look_back], train_predict, label='Train Prediction')
plt.plot(data.index[len(train_predict)+(look_back*2)+1:len(close_data)-1], test_predict, label='Test Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
