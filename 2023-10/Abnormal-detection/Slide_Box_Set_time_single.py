# #slide-box-plot
# #mark every day from 09:00-17:00  outliers
# #chose any PM to plt
#
# import pandas as pd
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mtick
# from datetime import datetime
# from scipy.interpolate import interp1d
# import os
# import matplotlib as mpl
# import csv
# from scipy.fft import fft
#
# # Customize the font size for Matplotlib
# mpl.rcParams.update({'font.size': 6})
#
# # Load data from a CSV file into a Pandas DataFrame
# data = pd.read_csv('C:/Users/yandcuo/Desktop/Dr/project/process/data/SunEnergy1_DataSet/Data/CONFIDENTIAL RANCHLAND SOLAR DATA - 5 min data.csv', header=[0, 1])
#
# # Function to convert date and time format
# def convert_date_time(date_string):
#     date_object = datetime.strptime(date_string, "%d-%m月-%y %H:%M:%S")  # Convert date string to datetime object
#     formatted_date = date_object.strftime("%m/%d/%Y %H:%M")              # Format datetime object as desired
#     return formatted_date
#
# # Convert columns in the DataFrame
# column_name = data.columns[0]  # Get the name of the first column
# data[column_name] = data[column_name].apply(convert_date_time)  # Apply the date and time conversion function
#
# # Handle multi-level column labels
# a = data.columns.get_level_values(0).to_series()  # Extract the first level of column labels
# b = a.mask(a.str.startswith('Unnamed')).ffill().fillna('')  # Fill 'Unnamed' labels with previous values
# data.columns = [b, data.columns.get_level_values(1)]  # Set new column labels
#
# # Handle missing values by forward filling
# data = data.fillna(method="ffill")
#
# # Convert 'Date' and 'Timestamp' columns to datetime objects and set 'Date' as the index
# data['Date', 'Timestamp'] = pd.to_datetime(data['Date', 'Timestamp'])
# data.set_index(data['Date', 'Timestamp'], inplace=True)
# data.drop(('Date', 'Timestamp'), inplace=True, axis=1)  # Drop the 'Date' and 'Timestamp' columns
#
# # Replace 'Bad' and '#DIV/0!' entries with Pandas' missing value representation (pd.NA)
# data.replace(['Bad', '#DIV/0!'], pd.NA, inplace=True)
#
#
# ###########pay attention Jan##############
# # If choose Jan , Set starttime = '2022-01-01 00:05'
# #Need change the start_date = datetime.strptime(starttime, '%Y-%m-%d %H:%M') ，otherwise delete %H:%M'
#
# # Define the target variable (inverter name) and time period
# name = 'Inv 2'
#
# start_date_str = '2022-12-01'
# end_date_str = '2022-12-31'
#
# # Convert start and end dates to datetime objects
# start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
# end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
#
#
# year = start_date.year
# month = start_date.month
#
#
#
# # Extract temperature data for the specified month
# temperature_data = data.loc[start_date_str:end_date_str, name][['PM1 Temp', 'PM2 Temp', 'PM3 Temp', 'PM4 Temp']]
# temperature_data = temperature_data.apply(pd.to_numeric, errors='coerce')
#
# # Drop rows with missing values
# temperature_data = temperature_data.dropna()
#
# # Extract unique dates from the index and convert it to a Pandas Series to use the unique() method
# unique_dates = pd.Series(temperature_data.index.date).unique()
#
# # Set Seaborn plotting style
# sns.set_style("darkgrid")
#
# # Filter data for the specified time range (09:00 to 17:00)
# start_time = '09:00'
# end_time = '17:00'
#
# # Convert start and end times to datetime objects
# start_datetime = datetime.strptime(start_time, '%H:%M').time()
# end_datetime = datetime.strptime(end_time, '%H:%M').time()
#
#
# # Create a directory to save the plots if it doesn't exist
# hourly_directory = f'C:/Users/yandcuo/Desktop/Dr/project/process/data/SunEnergy1_DataSet/2023-10/Health indicator/hourly/{name}'
# os.makedirs(hourly_directory, exist_ok=True)
#
# # Plotting hourly boxplot for the specified sensor's temperature data
# plt.figure(figsize=(12, 3))
#
# # Choose the specific sensor you want to plot (e.g., 'PM1 Temp')
# sensor_to_plot = 'PM1 Temp'
#
# # Filter temperature data for the specified sensor and time range
# filtered_temperature_data = temperature_data[sensor_to_plot].between_time(start_time, end_time)
#
# # Group filtered temperature data by day and extract hourly temperatures for the sensor
# hourly_temperature_data = filtered_temperature_data.groupby(filtered_temperature_data.index.date)
# hourly_temperatures = [hourly_data.dropna().values for _, hourly_data in hourly_temperature_data]
#
# # Plot hourly boxplot
# sns.boxplot(data=hourly_temperatures, palette='Set2')
#
# # Set title and labels
# plt.title(f'{sensor_to_plot} - Distribution (09:00 to 17:00)', fontsize=12)
# plt.ylabel('Temperature (°C)', fontsize=12)
# plt.tick_params(axis='y', labelsize=10)
# plt.xlabel('Date', fontsize=12)
#
#
# # Set x-axis labels as dates
# date_labels = [date.strftime('%m/%d') for date in unique_dates]
# plt.xticks(range(len(date_labels)), date_labels, rotation=45, fontsize=10)
# # Adjust layout to prevent overlapping subplots
# plt.tight_layout()
# # Save the figure as SVG
# hourly_boxplot_file_name = f'{sensor_to_plot}-{month}-Boxplot.svg'
# plt.savefig(os.path.join(hourly_directory, hourly_boxplot_file_name))
# # plt.show()
#
#
# #########print abnormal detection################
#
# from matplotlib.lines import Line2D
# # Create a dictionary to store the total number of outlier days for each sensor
# outlier_days_dict = {}
#
# # Create a grid layout with 1 subplot
# fig, axs = plt.subplots(1, 1, figsize=(12, 3))
#
# # Get temperature data for the specific sensor
# sensor_temperatures = temperature_data[sensor_to_plot]
#
# # Lists to store dates and outlier temperatures
# outlier_dates = []
# outlier_temperatures = []
# outlier_days = 0  # Variable to calculate total outlier days
#
# # Iterate through the temperature data for the sensor
# for date, daily_temperatures in sensor_temperatures.groupby(sensor_temperatures.index.date):
#     # Plot temperature curve (in blue)
#     axs.plot(daily_temperatures.index, daily_temperatures, color='blue', label=f'{sensor_to_plot} ')
#
#     # Filter data for the specified time range (09:00 to 17:00)
#     daily_temperatures_range = daily_temperatures.between_time('09:00', '17:00')
#
#     # Calculate Q1, Q3, and upper bound for the day
#     Q1 = daily_temperatures_range.quantile(0.25)
#     Q3 = daily_temperatures_range.quantile(0.75)
#     IQR = Q3 - Q1
#     upper_bound = Q3 + 1.5 * IQR
#
#     # Detect and output outliers above the upper bound
#     outliers = daily_temperatures_range[daily_temperatures_range > upper_bound]
#     if not outliers.empty:
#         outlier_days += 1
#         print(f'Outliers for {sensor_to_plot} on {date}')
#
#         # Plot outliers above the upper bound (in red)
#         axs.scatter(outliers.index, outliers.values, color='red', label=f'{sensor_to_plot} Outliers', zorder=5)
#
#     # Store dates and temperatures of outliers
#     outlier_dates.extend([date] * len(outliers))
#     outlier_temperatures.extend(outliers.values)
#
# # Store total outlier days for the sensor
# outlier_days_dict[sensor_to_plot] = outlier_days
#
# # Add legend for the plot
# legend_handles = [
#     Line2D([0], [0], color='blue', label=f'{sensor_to_plot}'),
#     Line2D([0], [0], marker='o', color='red', markerfacecolor='red', markersize=5,
#            linestyle='None', label=f'Outliers')
# ]
# axs.legend(handles=legend_handles)
#
# # Set subplot title and y-axis label
# axs.set_title(f'{sensor_to_plot} Data with Outliers', fontsize=12)
# axs.set_ylabel('Temperature (°C)', fontsize=12)
# axs.tick_params(axis='y', labelsize=9)
#
# # Set x-axis label as 'Date' and rotate x-axis labels by 45 degrees
# plt.xlabel('Date', fontsize=11)
# plt.xticks( fontsize=9)
#
# # Adjust layout for better visualization
# plt.tight_layout()
#
# # Save the plot as an SVG file
# mark_name = f'{sensor_to_plot}-{month}-Outliers.svg'
# plt.savefig(os.path.join(hourly_directory, mark_name))
# plt.show()
#
# # Print the total number of outlier days for the sensor
# print('*************************************')
# print(f"Outlier Days for {sensor_to_plot}: {outlier_days_dict[sensor_to_plot]} days")




# add average
#slide-box-plot
#mark every day from 09:00-17:00  outliers
#chose any PM to plt

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


###########pay attention Jan##############
# If choose Jan , Set starttime = '2022-01-01 00:05'
#Need change the start_date = datetime.strptime(starttime, '%Y-%m-%d %H:%M') ，otherwise delete %H:%M'

# Define the target variable (inverter name) and time period
name = 'Inv 10'

start_date_str = '2022-12-01'
end_date_str = '2022-12-31'

# Convert start and end dates to datetime objects
start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
end_date = datetime.strptime(end_date_str, '%Y-%m-%d')


year = start_date.year
month = start_date.month



# Extract temperature data for the specified month
temperature_data = data.loc[start_date_str:end_date_str, name][['PM1 Temp', 'PM2 Temp', 'PM3 Temp', 'PM4 Temp']]
temperature_data = temperature_data.apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values
temperature_data = temperature_data.dropna()

# Extract unique dates from the index and convert it to a Pandas Series to use the unique() method
unique_dates = pd.Series(temperature_data.index.date).unique()

# Set Seaborn plotting style
sns.set_style("darkgrid")

# Filter data for the specified time range (09:00 to 17:00)
start_time = '08:00'
end_time = '17:00'

# Convert start and end times to datetime objects
start_datetime = datetime.strptime(start_time, '%H:%M').time()
end_datetime = datetime.strptime(end_time, '%H:%M').time()


# Create a directory to save the plots if it doesn't exist
hourly_directory = f'C:/Users/yandcuo/Desktop/Dr/project/process/data/SunEnergy1_DataSet/2023-10/Health indicator/hourly/{name}'
os.makedirs(hourly_directory, exist_ok=True)

# Plotting hourly boxplot for the specified sensor's temperature data
plt.figure(figsize=(12, 3))

# Choose the specific sensor you want to plot (e.g., 'PM1 Temp')
sensor_to_plot = 'PM4 Temp'

# Filter temperature data for the specified sensor and time range
filtered_temperature_data = temperature_data[sensor_to_plot].between_time(start_time, end_time)

# Group filtered temperature data by day and extract hourly temperatures for the sensor
hourly_temperature_data = filtered_temperature_data.groupby(filtered_temperature_data.index.date)
hourly_temperatures = [hourly_data.dropna().values for _, hourly_data in hourly_temperature_data]

# Plot hourly boxplot
sns.boxplot(data=hourly_temperatures, palette='Set2')

# Set title and labels
plt.title(f'{sensor_to_plot} - Distribution (08:00 to 17:00)', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.tick_params(axis='y', labelsize=10)
plt.xlabel('Date', fontsize=12)


# Set x-axis labels as dates
date_labels = [date.strftime('%Y-%m-%d') for date in unique_dates]
# plt.xticks(range(len(date_labels)), date_labels, rotation=45, fontsize=10)

# Set x-axis labels as dates with a step to display only some labels
step = max(len(date_labels) // 5, 1)  # Show every 10th label, adjust as needed
plt.xticks(range(0, len(date_labels), step), [date_labels[i] for i in range(0, len(date_labels), step)],  fontsize=10)


# Adjust layout to prevent overlapping subplots
plt.tight_layout()
# Save the figure as SVG
hourly_boxplot_file_name = f'{sensor_to_plot}-{month}-Boxplot.svg'
plt.savefig(os.path.join(hourly_directory, hourly_boxplot_file_name))
# plt.show()


#########print abnormal detection################

from matplotlib.lines import Line2D
# Create a dictionary to store the total number of outlier days for each sensor
outlier_days_dict = {}

# Create a grid layout with 1 subplot
fig, axs = plt.subplots(1, 1, figsize=(12, 3))

# Get temperature data for the specific sensor
sensor_temperatures = temperature_data[sensor_to_plot]

# Lists to store dates and outlier temperatures
outlier_dates = []
outlier_temperatures = []
outlier_days = 0  # Variable to calculate total outlier days
extreme_days= 0
# Iterate through the temperature data for the sensor
for date, daily_temperatures in sensor_temperatures.groupby(sensor_temperatures.index.date):
    # Calculate daily average temperature
    daily_avg_temperature = daily_temperatures.between_time('08:00', '17:00').mean()

    # Plot temperature curve (in blue)
    axs.plot(daily_temperatures.index, daily_temperatures, color='blue', label=f'{sensor_to_plot} ')

    # Filter data for the specified time range (09:00 to 17:00)
    daily_temperatures_range = daily_temperatures.between_time('08:00', '17:00')

    # Calculate Q1, Q3, and upper bound for the day
    Q1 = daily_temperatures_range.quantile(0.25)
    Q3 = daily_temperatures_range.quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR

    # Detect and output outliers above the upper bound
    outliers = daily_temperatures_range[daily_temperatures_range > upper_bound]
    if not outliers.empty:
        outlier_days += 1
        # print(f'Outliers for {sensor_to_plot} on {date}')

        # Check if outliers are 10 degrees above the daily average temperature
        extreme_outliers = outliers[outliers > daily_avg_temperature + 10]

        if not extreme_outliers.empty:
            # print(f'Extreme Outliers for {sensor_to_plot} on {date}')
            extreme_days +=1
            # Plot extreme outliers above the upper bound (in orange)
            axs.scatter(extreme_outliers.index, extreme_outliers.values, color='red',
                        label=f'{sensor_to_plot} Extreme Outliers',
                        zorder=5)
            # Store dates and temperatures of extreme outliers
            outlier_dates.extend([date] * len(extreme_outliers))
            outlier_temperatures.extend(extreme_outliers.values)



# Store total outlier days for the sensor
outlier_days_dict[sensor_to_plot] = outlier_days

# Add legend for the plot
legend_handles = [
    Line2D([0], [0], color='blue', label=f'{sensor_to_plot}'),
    Line2D([0], [0], marker='o', color='red', markerfacecolor='red', markersize=5,
           linestyle='None', label=f'Outliers')
]
axs.legend(handles=legend_handles)

# Set subplot title and y-axis label
axs.set_title(f'{sensor_to_plot} Outliers (08:00 to 17:00)', fontsize=12)
axs.set_ylabel('Temperature (°C)', fontsize=12)
axs.tick_params(axis='y', labelsize=10)

# Set x-axis label as 'Date' and rotate x-axis labels by 45 degrees
plt.xlabel('Date', fontsize=11)
plt.xticks( fontsize=10)

# Adjust layout for better visualization
plt.tight_layout()

# Save the plot as an SVG file
mark_name = f'{sensor_to_plot}-{month}-Outliers.svg'
plt.savefig(os.path.join(hourly_directory, mark_name))


# Print the total number of outlier days for the sensor
print('*************************************')
print(f"Outlier Days for {sensor_to_plot}: {extreme_days} days")
plt.show()


