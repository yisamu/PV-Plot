


#box-plot
#Calculate the averag month of outliers

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime
from scipy.interpolate import interp1d
import os
import matplotlib as mpl
import csv
from scipy.fft import fft

# Customize the font size for Matplotlib
mpl.rcParams.update({'font.size': 6})

# Load data from a CSV file into a Pandas DataFrame
data = pd.read_csv('C:/Users/yandcuo/Desktop/Dr/project/process/data/SunEnergy1_DataSet/Data/CONFIDENTIAL RANCHLAND SOLAR DATA - 5 min data.csv', header=[0, 1])

# Function to convert date and time format
def convert_date_time(date_string):
    date_object = datetime.strptime(date_string, "%d-%m月-%y %H:%M:%S")  # Convert date string to datetime object
    formatted_date = date_object.strftime("%m/%d/%Y %H:%M")              # Format datetime object as desired
    return formatted_date

# Convert columns in the DataFrame
column_name = data.columns[0]  # Get the name of the first column
data[column_name] = data[column_name].apply(convert_date_time)  # Apply the date and time conversion function

# Handle multi-level column labels
a = data.columns.get_level_values(0).to_series()  # Extract the first level of column labels
b = a.mask(a.str.startswith('Unnamed')).ffill().fillna('')  # Fill 'Unnamed' labels with previous values
data.columns = [b, data.columns.get_level_values(1)]  # Set new column labels

# Handle missing values by forward filling
data = data.fillna(method="ffill")

# Convert 'Date' and 'Timestamp' columns to datetime objects and set 'Date' as the index
data['Date', 'Timestamp'] = pd.to_datetime(data['Date', 'Timestamp'])
data.set_index(data['Date', 'Timestamp'], inplace=True)
data.drop(('Date', 'Timestamp'), inplace=True, axis=1)  # Drop the 'Date' and 'Timestamp' columns

# Replace 'Bad' and '#DIV/0!' entries with Pandas' missing value representation (pd.NA)
data.replace(['Bad', '#DIV/0!'], pd.NA, inplace=True)


####pay attention Jan##############
# If choose Jan , Set starttime = '2022-01-01 00:05'
#Need change the start_date = datetime.strptime(starttime, '%Y-%m-%d %H:%M') ，otherwise delete %H:%M'

# Define the target variable (inverter name) and time period
name = 'Inv 11'

start_date_str = '2022-02-01'
end_date_str = '2022-02-28'

# Convert start and end dates to datetime objects
start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
end_date = datetime.strptime(end_date_str, '%Y-%m-%d')


year = start_date.year
month = start_date.month

# Generate a date range for each month
# date_ranges = pd.date_range(start=start_date, end=end_date, freq='M')

# Set Seaborn plotting style
sns.set_style("darkgrid")



month_directory = f'C:/Users/yandcuo/Desktop/Dr/project/process/data/SunEnergy1_DataSet/2023-10/Health indicator/{name}'
os.makedirs(month_directory, exist_ok=True)


# Extract temperature data for the specified month
temperature_data = data.loc[start_date_str:end_date_str, name][['PM1 Temp', 'PM2 Temp', 'PM3 Temp', 'PM4 Temp']]
temperature_data = temperature_data.apply(pd.to_numeric, errors='coerce')

# Plotting boxplot for temperature data
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=temperature_data, palette='Set2', ax=ax)

# Set title and labels for the plot
ax.set_title(f'{name} - Temperature Distribution\n{start_date_str} to {end_date_str}', fontsize=12)
ax.set_xlabel('Sensor', fontsize=12)
ax.set_ylabel('Temperature (°C)', fontsize=12)

# Set font sizes for tick labels
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)

# Rotate x-axis tick labels for better visibility
ax.tick_params(axis='x', labelrotation=45)

# Ensure a tight layout for better visualization
plt.tight_layout()

# Save the boxplot to a file (SVG format)
boxplot_file_name = f'{name}-Temperature-Boxplot-{start_date_str}-{end_date_str}.svg'
plt.savefig(os.path.join(month_directory, boxplot_file_name))
# plt.show()


# Find outliers above the upper bound (Q3 + 1.5 * IQR)
Q1 = temperature_data.quantile(0.25)
Q3 = temperature_data.quantile(0.75)
IQR = Q3 - Q1

upper_bound = Q3 + 1.5 * IQR

# Extract outliers above the upper bound
outliers = temperature_data[temperature_data > upper_bound]

# Find corresponding dates and temperatures for outliers
outlier_dates = []
outlier_temperatures = []

for sensor in outliers.columns:
    sensor_outliers = outliers[sensor].dropna()
    for date, temperature in zip(sensor_outliers.index, sensor_outliers.values):
        outlier_dates.append(date)
        outlier_temperatures.append(temperature)

# Plot the original temperature data
fig, ax = plt.subplots(figsize=(15, 6))
colors = ['blue', 'red', 'green', 'purple']
ax.plot(temperature_data.index, temperature_data['PM1 Temp'], label='PM1 Temp',color='blue')
ax.plot(temperature_data.index, temperature_data['PM2 Temp'], label='PM2 Temp',color='red')
ax.plot(temperature_data.index, temperature_data['PM3 Temp'], label='PM3 Temp',color='green')
ax.plot(temperature_data.index, temperature_data['PM4 Temp'], label='PM4 Temp',color='purple')

# Highlight outliers on the plot
ax.scatter(outlier_dates, outlier_temperatures, color='red', label='Outliers')

# Set labels and title
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Temperature (°C)', fontsize=12)
ax.set_title(f'{name} - Temperature Data with Outliers\n{start_date_str} to {end_date_str}', fontsize=12)
# Set font sizes for tick labels
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
# Set legend
ax.legend()

# Rotate x-axis tick labels for better visibility
ax.tick_params(axis='x', labelrotation=45)

# Ensure a tight layout for better visualization
plt.tight_layout()

plt.show()
# # Create an empty dictionary to store temperatures corresponding to each outlier date
# outlier_temperatures_dict = {}
#
# # Store outlier dates as keys and corresponding temperatures as values in the dictionary
# for date, temperature in zip(outlier_dates, outlier_temperatures):
#     date_key = date.date()  # Keep only the date part, remove the specific time
#     if date_key not in outlier_temperatures_dict:
#         outlier_temperatures_dict[date_key] = []
#     outlier_temperatures_dict[date_key].append(temperature)
#
#
#
# # Calculate the number of days with outliers
# total_outlier_days = len(outlier_temperatures_dict)
#
# # Save outlier dates, temperatures, and the total number of days to a CSV file
# outlier_dates_csv_file_name = f'{name}-Outlier-Dates-Temperatures-{start_date_str}-{end_date_str}.csv'
# with open(os.path.join(month_directory, outlier_dates_csv_file_name), 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerow(['Dates', 'Temperatures'])  # Write column names to the CSV file
#     for date, temperatures in outlier_temperatures_dict.items():
#         csvwriter.writerow([date, ', '.join(map(str, temperatures))])  # Write each outlier date and corresponding temperatures
#     csvwriter.writerow(['Total Days with Outliers', total_outlier_days])  # Write the total number of outlier days
#
# # Output the total number of days with outliers
# print(f'Total days with outliers: {total_outlier_days}')

# Create dictionaries to store outlier dates for each sensor
# Create dictionaries to store outlier dates for each sensor
pm1_outliers = {}
pm2_outliers = {}
pm3_outliers = {}
pm4_outliers = {}

# Extract outliers above the upper bound and store dates for each sensor
for sensor in outliers.columns:
    sensor_outliers = outliers[sensor].dropna()
    for date, temperature in zip(sensor_outliers.index, sensor_outliers.values):
        if sensor == 'PM1 Temp' and temperature > upper_bound['PM1 Temp']:
            pm1_outliers[date.date()] = pm1_outliers.get(date.date(), 0) + 1
        elif sensor == 'PM2 Temp' and temperature > upper_bound['PM2 Temp']:
            pm2_outliers[date.date()] = pm2_outliers.get(date.date(), 0) + 1
        elif sensor == 'PM3 Temp' and temperature > upper_bound['PM3 Temp']:
            pm3_outliers[date.date()] = pm3_outliers.get(date.date(), 0) + 1
        elif sensor == 'PM4 Temp' and temperature > upper_bound['PM4 Temp']:
            pm4_outliers[date.date()] = pm4_outliers.get(date.date(), 0) + 1

# Create dictionaries to store sensor names and their corresponding outlier days and dates
sensor_outliers_dict = {
    'PM1 Temp': {'Outlier Days': len(pm1_outliers), 'Dates': pm1_outliers},
    'PM2 Temp': {'Outlier Days': len(pm2_outliers), 'Dates': pm2_outliers},
    'PM3 Temp': {'Outlier Days': len(pm3_outliers), 'Dates': pm3_outliers},
    'PM4 Temp': {'Outlier Days': len(pm4_outliers), 'Dates': pm4_outliers}
}

# Create a dictionary to store sensor names and their corresponding outlier days and formatted dates
formatted_dates_dict = {
    'PM1 Temp': {
        'Outlier Days': len(pm1_outliers),
        'Dates': [date.strftime('%Y-%m-%d') for date in pm1_outliers.keys()]
    },
    'PM2 Temp': {
        'Outlier Days': len(pm2_outliers),
        'Dates': [date.strftime('%Y-%m-%d') for date in pm2_outliers.keys()]
    },
    'PM3 Temp': {
        'Outlier Days': len(pm3_outliers),
        'Dates': [date.strftime('%Y-%m-%d') for date in pm3_outliers.keys()]
    },
    'PM4 Temp': {
        'Outlier Days': len(pm4_outliers),
        'Dates': [date.strftime('%Y-%m-%d') for date in pm4_outliers.keys()]
    }
}

# Create a DataFrame from the formatted dictionary
result_df = pd.DataFrame(formatted_dates_dict).T

# Save the DataFrame to a CSV file
result_csv_file_name = f'Sensor-Outlier-Days-{month}.csv'
result_df.to_csv(os.path.join(month_directory, result_csv_file_name))

# Print the number of days with outliers for each sensor
print(f'Total days with PM1 outliers: {len(pm1_outliers)} days')
print(f'Total days with PM2 outliers: {len(pm2_outliers)} days')
print(f'Total days with PM3 outliers: {len(pm3_outliers)} days')
print(f'Total days with PM4 outliers: {len(pm4_outliers)} days')



sns.boxplot(data=temperature_data, palette='Set2')
plt.show()
