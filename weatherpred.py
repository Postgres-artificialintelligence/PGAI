import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import pickle

# Load the data
df = pd.read_csv('/Users/moizibrar/Downloads/weatherAUS.csv')

# Filter data for Melbourne
melb = df[df['Location'] == 'Melbourne'].copy()
melb['Date'] = pd.to_datetime(melb['Date'])

# Plot initial data
plt.figure(figsize=(10, 5))
plt.plot(melb['Date'], melb['Temp3pm'])
plt.title('Temperature at 3pm in Melbourne')
plt.xlabel('Date')
plt.ylabel('Temp3pm')
plt.show()

# Filter by year and plot
melb = melb[melb['Date'].dt.year <= 2015].copy()

plt.figure(figsize=(10, 5))
plt.plot(melb['Date'], melb['Temp3pm'])
plt.title('Temperature at 3pm in Melbourne (Up to 2015)')
plt.xlabel('Date')
plt.ylabel('Temp3pm')
plt.show()

# Prepare data for SARIMA
data = melb[['Date', 'Temp3pm']].dropna().copy()
data.columns = ['ds', 'y']  # Ensure columns are named 'ds' and 'y'
data.set_index('ds', inplace=True)

# Ensure the data index has a daily frequency
data = data.asfreq('D')

# Define SARIMA model parameters
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)  # yearly seasonality (12 months)

# Fit the SARIMA model
model = SARIMAX(data['y'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
results = model.fit()

# Save the model parameters using pickle
try:
    with open('/Users/moizibrar/Downloads/final/Weather_Predictions.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Model parameters saved successfully.")
except Exception as e:
    print(f"Error saving model parameters: {e}")