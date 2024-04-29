import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Load data from Excel sheetDesktop\
excel_file = r'path-of-your-data-set'
data = pd.read_excel(excel_file, names=['Year', 'GNP', 'iGNPi']) #column names of my dataset


def least_squares_regression(data):
    # Filter data for the period 2000 to 2014
    filtered_data = data[(data['Year'] >= 2000) & (data['Year'] <= 2014)]
    # Calculate necessary sums
    n = len(filtered_data)
    Sxx = (n ** 2 * (n + 1) * (2 * n + 1) / 6) - ((n * (n + 1) / 2) ** 2)
    Sxy = n * sum(filtered_data['iGNPi']) - (n * (n + 1) / 2) * sum(filtered_data['GNP'])
    b = Sxy / Sxx
    a = filtered_data['GNP'].mean() - (b * (n + 1) / 2)
    return a, b

alpha = float(input("Enter the value of alpha : "))
beta = float(input("Enter the value of beta : "))
S0, G0 = least_squares_regression(data)

# Forecasting
forecast_years = range(2015, 2023)
forecast_values = []

for year in forecast_years:
    S = alpha * data['GNP'].values[year-2000] + (1 - alpha) * (S0 + G0)
    G = beta * (S - S0) + (1 - beta) * G0
    F = S + G
    forecast_values.append(F)
    S0, G0 = S, G

actual_values = data['GNP'].values[-8:]
print("Actual:" , actual_values)
print("Forecasted:" , forecast_values)

n = 8  # from 2015 to 2022
MAD = (1 / n) * sum((abs(forecast_values - actual_values)))
MAPE = (1 / n) * sum(abs((actual_values - forecast_values) / (actual_values))) * 100
MSE = (1 / n) * sum((forecast_values - actual_values) ** 2)


print("MAD:", MAD)
print("MSE:", MSE)
print("MAPE:", MAPE)

years = data['Year'].values[-8:]  # Assuming you have the corresponding years for the actual data

plt.plot(years, actual_values, label='Actual GNP')
plt.plot(range(2015, 2023), forecast_values, label=f'Forecasted GNP (alpha={alpha})', linestyle='dashed')

plt.xlabel('Year')
plt.ylabel('GNP')
plt.title('Actual vs Forecasted GNP')
plt.legend()
plt.show()
