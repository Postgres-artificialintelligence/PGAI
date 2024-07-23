import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pickle
import h5py
from statsmodels.tsa.stattools import adfuller
from datetime import date, timedelta
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import psycopg2
from psycopg2 import sql

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

# Load data from the PostgreSQL table
query = "SELECT date, close FROM google_stock"
df = pd.read_sql(query, engine)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Ensure 'close' is numeric
df['close'] = pd.to_numeric(df['close'], errors='coerce')
df = df.dropna(subset=['close'])
df = df[np.isfinite(df['close'])]

# Check stationarity function
def check_stationarity(series):
    result = adfuller(series)
    print('ADF Statistics: %f' % result[0])
    print('P-Value: %f' % result[1])
    for key, value in result[4].items():
        print(f'Critical Value {key}: {value}')
    if result[1] <= 0.05:
        print("Reject the Null Hypothesis: Data is stationary")
    else:
        print("Fail to reject the Null Hypothesis: Data is not stationary")

check_stationarity(df['close'])

# Model parameters
p = 2
d = 1
q = 2
P = 2
D = 1
Q = 2
s = 12

# Fit the SARIMAX model
model = sm.tsa.statespace.SARIMAX(df['close'], order=(p, d, q), seasonal_order=(P, D, Q, s))
fit_model = model.fit()
print(fit_model.summary())

# Save the model using pickle
model_data = pickle.dumps(fit_model)

# Store the model in the PostgreSQL table
model_name = 'sarimax_google_stock_model'

# Create table if not exists
create_table_query = """
 DROP TABLE IF EXISTS google_model_storage;
        CREATE TABLE google_model_storage (
            id SERIAL PRIMARY KEY,
            model_name TEXT UNIQUE NOT NULL,
            model_data BYTEA,
            scaler_data BYTEA
        );
"""
cursor.execute(create_table_query)

# Insert or update the model
insert_query = """
INSERT INTO google_model_storage (model_name, model_data)
VALUES (%s, %s)
ON CONFLICT (model_name)
DO UPDATE SET model_data = EXCLUDED.model_data;
"""
cursor.execute(insert_query, (model_name, psycopg2.Binary(model_data)))

# Commit and close connection
conn.commit()
cursor.close()
conn.close()

print("Model saved successfully.")
