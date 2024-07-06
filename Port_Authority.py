import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Load necessary libraries
import numpy as np
from pandas.tseries.offsets import MonthEnd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot

# Load the data
data = pd.read_excel("Port_Authority_Group_Project.xlsx")

# Preprocess the data
# Convert Date to datetime type and filter for weekdays only
data['Date'] = pd.to_datetime(data['Date'])
data = data[data['Date'].dt.dayofweek < 5]

# Grouping and summarizing by Year and Month to get total departures
monthly_summary = data.groupby([data['Date'].dt.year.rename('Year'),
                                data['Date'].dt.to_period('M').rename('Month')]) \
.agg(Total_Passenger_Departures=('Passengers_Current_Week_Departures', 'sum'),
     Total_Bus_Departures=('Bus_Current_Week_Departures', 'sum')).reset_index()

# Ensure that 'Month' is at the end of the month
monthly_summary['Month'] = monthly_summary['Month'].dt.to_timestamp() + MonthEnd(1)

# Converting monthly summary to a ts object for forecasting
ts_passengers = monthly_summary['Total_Passenger_Departures'].values
ts_buses = monthly_summary['Total_Bus_Departures'].values

# Define the ARIMA model for passenger departures and fit it
fit_passengers = SARIMAX(ts_passengers, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results_passengers = fit_passengers.fit()

# Forecast passenger departures
forecast_passengers = results_passengers.get_forecast(steps=12 * (2030 - monthly_summary['Year'].max()))
forecast_passengers_mean = forecast_passengers.predicted_mean

# Fit an ARIMA model for bus departures
fit_buses = SARIMAX(ts_buses, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results_buses = fit_buses.fit()

# Forecast bus departures
forecast_buses = results_buses.get_forecast(steps=12 * (2030 - monthly_summary['Year'].max()))
forecast_buses_mean = forecast_buses.predicted_mean

# Find thresholds
passenger_threshold_idx = np.where(forecast_passengers_mean > 125000)[0]
bus_threshold_idx = np.where(forecast_buses_mean > 3900)[0]

# Convert these indices to year and month
# Assuming that the forecast indices directly correspond to the forecasted months starting from max(data['Year'])+1
passenger_threshold_year = monthly_summary['Year'].max() + (passenger_threshold_idx[0] // 12) if passenger_threshold_idx.size > 0 else None
passenger_threshold_month = passenger_threshold_idx[0] % 12 + 1 if passenger_threshold_idx.size > 0 else None
bus_threshold_year = monthly_summary['Year'].max() + (bus_threshold_idx[0] // 12) if bus_threshold_idx.size > 0 else None
bus_threshold_month = bus_threshold_idx[0] % 12 + 1 if bus_threshold_idx.size > 0 else None

# Print the result
if passenger_threshold_year and passenger_threshold_month:
  print(f"The forecast indicates that passenger departures will exceed 125,000 in: {passenger_threshold_year}/{passenger_threshold_month}")
if bus_threshold_year and bus_threshold_month:
  print(f"The forecast indicates that bus departures will exceed 3,900 in: {bus_threshold_year}/{bus_threshold_month}")

# Plotting the forecasts
fig, ax = plt.subplots()
ax.plot(monthly_summary['Month'], ts_passengers, label='Historical')
ax.plot(pd.date_range(start=monthly_summary['Month'].iloc[-1] + pd.DateOffset(months=1),
                      periods=len(forecast_passengers_mean), freq='M'),
        forecast_passengers_mean, label='Forecast')
ax.set_title("Forecast of Passenger Departures")
ax.set_xlabel("Year")
ax.set_ylabel("Passenger Departures")
ax.legend()
date_form = DateFormatter("%Y")
ax.xaxis.set_major_formatter(date_form)
plt.show()

fig, ax = plt.subplots()
ax.plot(monthly_summary['Month'], ts_buses, label='Historical')
ax.plot(pd.date_range(start=monthly_summary['Month'].iloc[-1] + pd.DateOffset(months=1), periods=len(forecast_buses_mean), freq='M'), forecast_buses_mean, label='Forecast')
ax.set_title("Forecast of Bus Departures")
ax.set_xlabel("Year")
ax.set_ylabel("Bus Departures")
ax.legend()
date_form = DateFormatter("%Y")
ax.xaxis.set_major_formatter(date_form)
plt.show()

# In this we have to add some external factors like weather, holidays, etc. to improve the model.

# lets add some external factors like weather, holidays, etc. to improve the model.

# Assuming the length of your time series is as follows
n_months = len(ts_passengers)

# Generate a synthetic economic indicator (e.g., trend + noise)
np.random.seed(0)  # For reproducibility
economic_indicator = np.linspace(start=100, stop=200, num=n_months) + np.random.normal(loc=0, scale=5, size=n_months)

# Generate a synthetic seasonal factor (simulating holiday or event-driven fluctuations)
seasonal_factor = np.sin(np.linspace(start=0, stop=2 * np.pi * (n_months / 12), num=n_months)) * 100 + 1000 + np.random.normal(loc=0, scale=10, size=n_months)

# Combine these into a DataFrame as exogenous variables
exog_data = pd.DataFrame({'Economic_Indicator': economic_indicator, 'Seasonal_Factor': seasonal_factor})


# Fit the SARIMAX model with exogenous variables for passenger departures
fit_passengers = SARIMAX(ts_passengers, exog=exog_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results_passengers = fit_passengers.fit()

# To forecast, you also need to provide future values of the exogenous variables
# Assuming we forecast for the next 12 months, you'd generate future exogenous data as well
# For demonstration, we'll just replicate the last row of the exog_data 12 times (in reality, you should estimate future values)
future_exog_data = pd.DataFrame({'Economic_Indicator': np.tile(exog_data['Economic_Indicator'].iloc[-1], 12),
                                 'Seasonal_Factor': np.tile(exog_data['Seasonal_Factor'].iloc[-1], 12)})

forecast_passengers = results_passengers.get_forecast(steps=12, exog=future_exog_data)
forecast_passengers_mean = forecast_passengers.predicted_mean

# Print forecast results, adjust further as needed for visualization
print(forecast_passengers_mean)

# Plotting the forecasts
fig, ax = plt.subplots()
ax.plot(monthly_summary['Month'], ts_passengers, label='Historical')
ax.plot(pd.date_range(start=monthly_summary['Month'].iloc[-1] + pd.DateOffset(months=1),
                      periods=len(forecast_passengers_mean), freq='M'),
        forecast_passengers_mean, label='Forecast')
ax.set_title("Forecast of Passenger Departures")
ax.set_xlabel("Year")
ax.set_ylabel("Passenger Departures")
ax.legend()
date_form = DateFormatter("%Y")
ax.xaxis.set_major_formatter(date_form)
plt.show()


# I need these answers
#Growth of the popluation

# Generate synthetic population growth data
np.random.seed(0)
population_growth = np.linspace(start=100000, stop=120000, num=n_months) + np.random.normal(loc=0, scale=1000, size=n_months)

# Add this to the exog_data DataFrame
exog_data['Population_Growth'] = population_growth

#Population statstics
#Queen, manhatten, NJ

# Fit the SARIMAX model with the updated exogenous variables for passenger departures
fit_passengers = SARIMAX(ts_passengers, exog=exog_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results_passengers = fit_passengers.fit()

# Provide future values of the economic indicator, seasonal factor, and population growth for forecasting
future_exog_data['Population_Growth'] = np.linspace(start=exog_data['Population_Growth'].iloc[-1], stop=exog_data['Population_Growth'].iloc[-1] + 2000, num=12)

# Forecast with the updated model
forecast_passengers = results_passengers.get_forecast(steps=12, exog=future_exog_data)
forecast_passengers_mean = forecast_passengers.predicted_mean

#How many buses leave if there is population hike

# Calculate the increase in bus departures based on population growth
# Here we assume a very simple direct relationship: for every additional 1000 people, there are 10 extra bus departures.
additional_buses = (future_exog_data['Population_Growth'] - exog_data['Population_Growth'].iloc[-1]) / 1000 * 10

# Print the additional number of bus departures
print("Additional bus departures due to population hike:", additional_buses)

# Plotting the additional bus departures
fig, ax = plt.subplots()
ax.plot(future_exog_data['Population_Growth'], additional_buses)
ax.set_title("Additional Bus Departures Due to Population Hike")
ax.set_xlabel("Population Growth")
ax.set_ylabel("Additional Bus Departures")
plt.show()


# Generate synthetic gas price data with a general upward trend and some noise
np.random.seed(0)  # Ensuring reproducibility
gas_prices = np.linspace(start=2.50, stop=4.00, num=n_months) + np.random.normal(loc=0, scale=0.10, size=n_months)

# Add this to the exog_data DataFrame
exog_data['Gas_Prices'] = gas_prices

# Fit the SARIMAX model with the updated exogenous variables for passenger departures
fit_passengers = SARIMAX(ts_passengers, exog=exog_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results_passengers = fit_passengers.fit()

# To forecast, generate future exogenous data for economic indicator, seasonal factor, population growth, and now gas prices
future_exog_data['Gas_Prices'] = np.linspace(start=gas_prices[-1], stop=gas_prices[-1] + 0.10, num=12)

# Forecast with the updated model
forecast_passengers = results_passengers.get_forecast(steps=12, exog=future_exog_data)
forecast_passengers_mean = forecast_passengers.predicted_mean


# Plot gas prices to visualize the data
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Time (months)')
ax1.set_ylabel('Gas Prices ($)', color=color)
ax1.plot(np.arange(n_months), gas_prices, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Instantiate a second y-axis to plot passenger departures on the same graph, if desired
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Passenger Departures', color=color)  # we already handled the x-label with ax1
ax2.plot(np.arange(n_months), ts_passengers, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # To ensure there's no overlap in the layout
plt.title("Impact of Gas Prices on Passenger Departures")
plt.show()




baseline_population = 100000
population_growth_from_baseline = exog_data['Population_Growth'] - baseline_population

# Now calculate the additional buses required from that population growth
additional_buses_required = population_growth_from_baseline / 1000 * 10

# Plotting the additional bus departures needed as population grows
fig, ax = plt.subplots()
ax.plot(exog_data.index, additional_buses_required)
ax.set_title("Additional Bus Departures Needed for Population Growth")
ax.set_xlabel("Time (months)")
ax.set_ylabel("Additional Bus Departures")
plt.grid(True)  # Optional, adds a grid to the plot for better readability
plt.show()

# Calculate the additional bus departures needed for the population growth
# For every additional 1000 people, we need 10 more bus departures.
additional_buses_needed = (exog_data['Population_Growth'] - baseline_population) / 1000 * 10

# Plot the population growth against the additional bus departures needed
plt.figure(figsize=(10,5))
plt.plot(exog_data['Population_Growth'], additional_buses_needed, marker='o')
plt.title('Additional Bus Departures Required for Population Growth')
plt.xlabel('Population Growth')
plt.ylabel('Additional Bus Departures')
plt.grid(True)
plt.show()


# Calculate future population growth
future_population_growth = np.linspace(start=exog_data['Population_Growth'].iloc[-1], stop=exog_data['Population_Growth'].iloc[-1] + 20000, num=12)

# Calculate additional buses needed for the forecasted population growth
additional_buses_needed = (future_population_growth - exog_data['Population_Growth'].iloc[-1]) / 1000 * 10

# Fit the SARIMAX model with the updated exogenous variables for bus departures
fit_buses = SARIMAX(ts_buses, exog=exog_data[['Economic_Indicator', 'Seasonal_Factor', 'Population_Growth']], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results_buses = fit_buses.fit()

# Forecast with the updated model
forecast_buses = results_buses.get_forecast(steps=12, exog=pd.DataFrame({'Economic_Indicator': future_exog_data['Economic_Indicator'],
                                                                          'Seasonal_Factor': future_exog_data['Seasonal_Factor'],
                                                                          'Population_Growth': future_population_growth}))
forecast_buses_mean = forecast_buses.predicted_mean

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(pd.date_range(start=monthly_summary['Month'].iloc[-1] + pd.DateOffset(months=1), periods=12, freq='M'), forecast_buses_mean, label='Forecasted Bus Departures')
plt.plot(pd.date_range(start=monthly_summary['Month'].iloc[-1] + pd.DateOffset(months=1), periods=12, freq='M'), additional_buses_needed, label='Additional Buses Needed')
plt.title('Forecasted Bus Departures vs. Additional Buses Required')
plt.xlabel('Date')
plt.ylabel('Number of Buses')
plt.legend()
plt.grid(True)
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Assume exog_data and ts_passengers are prepared
# Let's create a synthetic example of exog_data
np.random.seed(0)
n_samples = len(ts_passengers)
economic_indicator = np.linspace(100, 200, n_samples) + np.random.normal(0, 10, n_samples)
weather_impact = np.random.normal(100, 25, n_samples)

# Create a DataFrame
exog_data = pd.DataFrame({
    'Economic_Indicator': economic_indicator,
    'Weather_Impact': weather_impact
})

# Target variable
y = np.array(ts_passengers)

# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(exog_data, y, test_size=0.2, random_state=0)

# Create linear regression object
regr = LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))

# Plot outputs
plt.scatter(y_test, y_pred, color='black')
plt.plot(y_test, y_test, color='blue', linewidth=3)
plt.title('Actual vs Predicted')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()





import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Sample data
data = np.array(ts_passengers).reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Function to create dataset for LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Prepare the input X and output Y
look_back = 1
X, Y = create_dataset(data_scaled, look_back)

# Reshape input to be [samples, time steps, features]
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, epochs=100, batch_size=1, verbose=2)

# Making predictions
train_predict = model.predict(X)
# Making predictions
with open('output.log', 'w', encoding='utf-8') as f:
    f.write(str(model.predict(X)))

# Invert predictions
train_predict = scaler.inverse_transform(train_predict)
Y = scaler.inverse_transform([Y])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(Y[0], label='Original Data')
plt.plot(train_predict[:,0], label='Predicted Data')
plt.title('LSTM Forecast vs Actual')
plt.xlabel('Time Step')
plt.ylabel('Passenger Count')
plt.legend()
plt.show()




