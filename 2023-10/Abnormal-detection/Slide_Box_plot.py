


#slide-box-plot
#mark every day  outliers

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
name = 'Inv 2'

start_date_str = '2022-01-01 00:05'
end_date_str = '2022-01-31'

# Convert start and end dates to datetime objects
start_date = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M')
end_date = datetime.strptime(end_date_str, '%Y-%m-%d')


year = start_date.year
month = start_date.month

# Generate a date range for each month
# date_ranges = pd.date_range(start=start_date, end=end_date, freq='M')


# Extract temperature data for the specified month
temperature_data = data.loc[start_date_str:end_date_str, name][['PM1 Temp', 'PM2 Temp', 'PM3 Temp', 'PM4 Temp']]
temperature_data = temperature_data.apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values
temperature_data = temperature_data.dropna()

# Extract unique dates from the index and convert it to a Pandas Series to use the unique() method
unique_dates = pd.Series(temperature_data.index.date).unique()


# Set Seaborn plotting style
sns.set_style("darkgrid")

# Create a directory to save the plots if it doesn't exist
month_directory = f'C:/Users/yandcuo/Desktop/Dr/project/process/data/SunEnergy1_DataSet/2023-10/Health indicator/slide/{name}'
os.makedirs(month_directory, exist_ok=True)



# Plotting boxplot for temperature data for each day
fig, axs = plt.subplots(4, 1, figsize=(15, 6), sharex=True)

# Group temperature data by day and plot daily boxplots for each sensor
for i, sensor in enumerate(temperature_data.columns):
    # Group data by day and extract daily temperatures for the sensor
    daily_temperature_data = temperature_data[sensor].groupby(temperature_data.index.date)
    daily_temperatures = [daily_data.dropna().values for _, daily_data in daily_temperature_data]

    # Plot daily boxplot
    sns.boxplot(data=daily_temperatures, ax=axs[i], palette='Set2')

    # Set title and labels
    axs[i].set_title(f'{sensor} -Distribution', fontsize=12)
    axs[i].set_ylabel('Temperature (°C)', fontsize=10)
    axs[i].tick_params(axis='y', labelsize=8)

# Set x-axis labels as dates
date_labels = [date.strftime('%m/%d') for date in unique_dates]
plt.xticks(range(len(date_labels)), date_labels, rotation=45, fontsize=10)
plt.xlabel('Date', fontsize=12)

# Adjust layout to prevent overlapping subplots
plt.tight_layout()

# Save the figure as SVG
monthly_boxplot_file_name = f'{name}-{month}-Boxplot.svg'
plt.savefig(os.path.join(month_directory, monthly_boxplot_file_name))
# plt.show()


# print abnormal detection


# Create a grid layout with 4 subplots
fig, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

# Iterate through temperature data for each sensor
for i, (sensor, sensor_temperatures) in enumerate(temperature_data.items()):
    # Store dates and temperatures of outliers
    outlier_dates = []
    outlier_temperatures = []

    # Iterate through temperature data for each day
    for date, daily_temperatures in sensor_temperatures.groupby(sensor_temperatures.index.date):
        # Calculate Q1, Q3, and upper bound for the day
        Q1 = daily_temperatures.quantile(0.25)
        Q3 = daily_temperatures.quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR

        # Detect and output outliers above the upper bound
        outliers = daily_temperatures[daily_temperatures > upper_bound]
        if not outliers.empty:
            print(f'Outliers for {sensor} on {date}')

        # Plot temperature curve (in blue)
        axs[i].plot(daily_temperatures.index, daily_temperatures, color='blue', label=f'{sensor} ')

        # Plot outliers above the upper bound (in red)
        axs[i].scatter(outliers.index, outliers.values, color='red', label=f'{sensor} Outliers', zorder=5)

        # Store dates and temperatures of outliers
        outlier_dates.extend([date] * len(outliers))
        outlier_temperatures.extend(outliers.values)

        # Add legend only on the first day
        if date == sensor_temperatures.index.date[0]:
            axs[i].legend()

    # Set subplot title and y-axis label
    axs[i].set_title(f'{sensor} Temperature Data with Outliers')
    axs[i].set_ylabel('Temperature (°C)')

# Set x-axis label as 'Date' and rotate x-axis labels by 45 degrees
plt.xlabel('Date')
# plt.xticks(rotation=45)

# Adjust layout for better visualization
plt.tight_layout()

# Save the plot as an SVG file
mark_name = f'{name}-{month}-Temp.svg'
plt.savefig(os.path.join(month_directory, mark_name))
plt.show()
