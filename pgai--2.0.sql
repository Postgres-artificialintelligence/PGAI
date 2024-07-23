CREATE OR REPLACE FUNCTION get_prediction_for_date(prediction_date DATE)
RETURNS TABLE(
    date DATE,
    pseudo_column DOUBLE PRECISION
) AS $$
import pandas as pd
import pickle
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

# Load the data
df = pd.read_csv('/Users/moizibrar/Downloads/weatherAUS.csv')

# Filter data for Melbourne
melb = df[df['Location'] == 'Melbourne'].copy()
melb['Date'] = pd.to_datetime(melb['Date'])

# Prepare data for SARIMA
data = melb[['Date', 'Temp3pm']].dropna().copy()
data.columns = ['ds', 'y']
data.set_index('ds', inplace=True)
data = data.asfreq('D').fillna(method='ffill')

# Load the model parameters using pickle
try:
    with open('/Users/moizibrar/Downloads/final/Weather_Predictions.pkl', 'rb') as f:
        results = pickle.load(f)
except Exception as e:
    plpy.error(f"Error loading model parameters: {e}")

# Predict using the SARIMA model
try:
    specific_date_pd = pd.to_datetime(prediction_date)
    
    if specific_date_pd > data.index[-1]:
        steps = (specific_date_pd - data.index[-1]).days
        specific_date_data = results.get_forecast(steps=steps).summary_frame()
        specific_date_data.index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
        prediction = specific_date_data.loc[specific_date_pd, "mean"]
    else:
        specific_date_data = results.get_prediction(start=specific_date_pd, end=specific_date_pd).summary_frame()
        prediction = specific_date_data["mean"].values[0]

    result = [(specific_date_pd.date(), prediction)]
    return result

except Exception as e:
    plpy.error(f"Error predicting with SARIMA model: {e}")

$$ LANGUAGE plpython3u;






