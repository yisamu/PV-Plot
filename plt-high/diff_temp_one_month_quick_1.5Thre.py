# plot the temp distribution diagram and the DC AC DC VTG  every day
# plot temp for each month any period of time
# Get the number of days above 55 degrees
# set the 1.5 threshold

# plot the temp distribution diagram and the DC AC DC VTG  every day
# plot temp for each month any period of time
# Get the number of days above 55 degrees


#caculate one year auto swutch

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

# Define the target variable (inverter name) and time period
name = 'Inv 15'
start_date_str = '2022-01-01'
end_date_str = '2022-12-31'

# Convert start and end dates to datetime objects
start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

# Generate a date range for each month
date_ranges = pd.date_range(start=start_date, end=end_date, freq='M')

# Set Seaborn plotting style
sns.set_style("darkgrid")

for current_month in date_ranges:
    # Extract year and month
    year = current_month.year
    month = current_month.month

    # Define the start and end dates for the current month
    start_date = current_month.replace(day=1)
    end_date = current_month.replace(day=pd.Timestamp(year, month, 1).days_in_month)

    # Convert start and end dates to strings
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Create a directory for the current month if it doesn't exist
    month_directory = f'C:/Users/yandcuo/Desktop/Dr/project/process/data/SunEnergy1_DataSet/effi_temp_every_month/effi_temp_no_consider/diff_temp_one_month_extract/z_method/Inv15/{year}-{month:02d}'
    os.makedirs(month_directory, exist_ok=True)

    # Create a Matplotlib figure and axis for plotting
    fig, ax = plt.subplots(figsize=(15, 6))

    # Define legend labels and line colors for temperature sensors
    legend_labels = ['PM1 Temp', 'PM2 Temp', 'PM3 Temp', 'PM4 Temp']
    colors = ['blue', 'red', 'green', 'purple']

    # Iterate through each day in the current month
    for day in pd.date_range(start=start_date_str, end=end_date_str, freq='D'):
        day_str = day.strftime('%Y-%m-%d')

        # Extract temperature data for the current day
        temp_data = data[name][['PM1 Temp', 'PM2 Temp', 'PM3 Temp', 'PM4 Temp']].loc[day_str]
        temp_data = temp_data.apply(pd.to_numeric, errors='coerce')  # Convert temperature data to numeric format

        # Plot temperature data for each sensor
        for i in range(4):
            ax.plot(temp_data.index, temp_data.iloc[:, i], color=colors[i])

    # Set labels for the title, x-axis and y-axis
    ax.set_title(f'{name}-Temperature Distribution-{year}-{month:02d}', fontsize=12)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    # Increase the font size of x-axis tick labels
    ax.tick_params(axis='x', labelsize=10)
    # Increase the font size of y-axis tick labels
    ax.tick_params(axis='y', labelsize=10)
    # Add a legend to the plot
    ax.legend(legend_labels, fontsize=8, loc='upper right')

    # Ensure a tight layout for better visualization
    plt.tight_layout()

    # Define the file name for saving the plot (SVG format)
    file_name = f'{name}-AD-DC-Temp {year}-{month:02d}.svg'

    # Save the plot to a file
    plt.savefig(os.path.join(month_directory, file_name))
# #################     Judge the temp       #################  average*1.5
#     # Initialize a dictionary to store average temperatures for each sensor
#     average_temperatures = {}
#
#     # Iterate through each day in the current month
#     for day in pd.date_range(start=start_date_str, end=end_date_str, freq='D'):
#         day_str = day.strftime('%Y-%m-%d')
#
#         # Extract temperature data for the current day
#         temp_data = data[name][['PM1 Temp', 'PM2 Temp', 'PM3 Temp', 'PM4 Temp']].loc[day_str]
#         temp_data = temp_data.apply(pd.to_numeric, errors='coerce')  # Convert temperature data to numeric format
#         # Calculate the average temperature for each sensor and store it in the dictionary
#         for pm in ['PM1 Temp', 'PM2 Temp', 'PM3 Temp', 'PM4 Temp']:
#             if pm not in average_temperatures:
#                 average_temperatures[pm] = []
#             average_temp = temp_data[pm].mean()
#             average_temperatures[pm].append(average_temp)
#
#     # Calculate the threshold temperature as 1.5 times the average temperature
#     threshold_temp = {sensor: 1.5 * np.mean(avg_temp) for sensor, avg_temp in average_temperatures.items()}
#
#     # Initialize a variable to count the number of days with temperatures above the threshold
#     days_above_threshold = {sensor: 0 for sensor in threshold_temp}
#
#     # Initialize a dictionary to store dates with temperatures above the threshold for each sensor
#     dates_above_threshold = {sensor: [] for sensor in threshold_temp}
#
#     # Iterate through each day in the current month
#     for day in pd.date_range(start=start_date_str, end=end_date_str, freq='D'):
#         day_str = day.strftime('%Y-%m-%d')
#
#         # Extract temperature data for the current day
#         temp_data = data[name][['PM1 Temp', 'PM2 Temp', 'PM3 Temp', 'PM4 Temp']].loc[day_str]
#         temp_data = temp_data.apply(pd.to_numeric, errors='coerce')  # Convert temperature data to numeric format
#
#         # Check if any temperature reading exceeds the threshold for each sensor
#         for sensor in threshold_temp:
#             if temp_data[sensor].max() > threshold_temp[sensor]:
#                 days_above_threshold[sensor] += 1
#                 dates_above_threshold[sensor].append(day_str)  # Add the date to the list
#
#     # Output the total number of days with temperatures above the threshold for each sensor
#     for sensor in threshold_temp:
#         print(f'The total number of days above {threshold_temp[sensor]:.2f}°C for {sensor} in {year}-{month:02d} is: {days_above_threshold[sensor]} days')
#
#     # Output the dates when temperatures exceeded the threshold for each sensor
#     for sensor in threshold_temp:
#         if len(dates_above_threshold[sensor]) > 0:
#             print(f'Dates with temperatures above {threshold_temp[sensor]:.2f}°C for {sensor} in {year}-{month:02d}:')
#             for date in dates_above_threshold[sensor]:
#                 print(date)
#         else:
#             print(f'No dates with temperatures above {threshold_temp[sensor]:.2f}°C found for {sensor} in {year}-{month:02d}')
#
#     # Define the file name for saving the print outputs in CSV format for the current month
#     output_file_name = os.path.join(
#         month_directory,
#         f'{name}-High-Temp {year}-{month:02d}.csv'
#     )
#
#     # Define the data to be saved in the CSV file for the current month
#     output_data = [
#         ['Sensor', 'Threshold Temperature (°C)', 'Total Days Above Threshold', 'Dates Above Threshold'],
#     ]
#
#     # Add data for each sensor to the output data
#     for sensor in threshold_temp:
#         output_data.append([
#             sensor,
#             f'{threshold_temp[sensor]:.2f} °C',
#             days_above_threshold[sensor],
#             ', '.join(dates_above_threshold[sensor])
#         ])
#
#     # Save the data to a CSV file for the current month
#     with open(output_file_name, 'w', newline='') as csv_file:
#         csv_writer = csv.writer(csv_file)
#         csv_writer.writerows(output_data)
#
#     print(f'Print outputs saved to {output_file_name}')
#
# plt.show()

    #################     Judge the temp       ################# z_method
    # Initialize a dictionary to store average temperatures for each sensor
    average_temperatures = {}

    # Iterate through each day in the current month
    for day in pd.date_range(start=start_date_str, end=end_date_str, freq='D'):
        day_str = day.strftime('%Y-%m-%d')

        # Extract temperature data for the current day
        temp_data = data[name][['PM1 Temp', 'PM2 Temp', 'PM3 Temp', 'PM4 Temp']].loc[day_str]
        temp_data = temp_data.apply(pd.to_numeric, errors='coerce')  # Convert temperature data to numeric format

        # Calculate the average temperature for each sensor and store it in the dictionary
        for pm in ['PM1 Temp', 'PM2 Temp', 'PM3 Temp', 'PM4 Temp']:
            if pm not in average_temperatures:
                average_temperatures[pm] = []
            average_temp = temp_data[pm].mean()
            average_temperatures[pm].append(average_temp)

    # Calculate the threshold temperature as 1.5 times the average temperature
    threshold_temp = {sensor: 1.5 * np.mean(avg_temp) for sensor, avg_temp in average_temperatures.items()}

    # Initialize a variable to count the number of days with temperatures above the threshold
    days_above_threshold = {sensor: 0 for sensor in threshold_temp}

    # Initialize a dictionary to store dates with temperatures above the threshold for each sensor
    dates_above_threshold = {sensor: [] for sensor in threshold_temp}

    # Iterate through each day in the current month
    for day in pd.date_range(start=start_date_str, end=end_date_str, freq='D'):
        day_str = day.strftime('%Y-%m-%d')

        # Extract temperature data for the current day
        temp_data = data[name][['PM1 Temp', 'PM2 Temp', 'PM3 Temp', 'PM4 Temp']].loc[day_str]
        temp_data = temp_data.apply(pd.to_numeric, errors='coerce')  # Convert temperature data to numeric format

        # Check if any temperature reading exceeds the threshold for each sensor
        for sensor in threshold_temp:
            if temp_data[sensor].max() > threshold_temp[sensor]:
                days_above_threshold[sensor] += 1
                dates_above_threshold[sensor].append(day_str)  # Add the date to the list

    # Output the total number of days with temperatures above the threshold for each sensor
    for sensor in threshold_temp:
        print(f'The total number of days above {threshold_temp[sensor]:.2f}°C for {sensor} in {year}-{month:02d} is: {days_above_threshold[sensor]} days')

    # Output the dates when temperatures exceeded the threshold for each sensor
    for sensor in threshold_temp:
        if len(dates_above_threshold[sensor]) > 0:
            print(f'Dates with temperatures above {threshold_temp[sensor]:.2f}°C for {sensor} in {year}-{month:02d}:')
            for date in dates_above_threshold[sensor]:
                print(date)
        else:
            print(f'No dates with temperatures above {threshold_temp[sensor]:.2f}°C found for {sensor} in {year}-{month:02d}')

    # Define the file name for saving the print outputs in CSV format for the current month
    output_file_name = os.path.join(        month_directory,        f'{name}-High-Temp {year}-{month:02d}.csv'    )

    # Define the data to be saved in the CSV file for the current month
    output_data = ['Sensor', 'Threshold Temperature (°C)', 'Total Days Above Threshold', 'Dates Above Threshold']

    # Add data for each sensor to the output data
    for sensor in threshold_temp:
        output_data.append([
            sensor,
            f'{threshold_temp[sensor]:.2f} °C',
            days_above_threshold[sensor],
            ', '.join(dates_above_threshold[sensor])
        ])

    # Save the data to a CSV file for the current month
    with open(output_file_name, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(output_data)

    print(f'Print outputs saved to {output_file_name}')

plt.show()






