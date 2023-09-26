# plot the temp distribution diagram and the DC AC DC VTG  every day
# plot temp for each month any period of time
# Get the number of days above 55 degrees
# set the 3sigma



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
name = 'Inv 11'
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
    month_directory = f'C:/Users/yandcuo/Desktop/Dr/project/process/data/SunEnergy1_DataSet/effi_temp_every_month/effi_temp_no_consider/diff_temp_one_month_extract/3 Sigma/{name}/{year}-{month:02d}'
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

        # Calculate mean and standard deviation for each sensor
        sensor_means = temp_data.mean()
        sensor_std = temp_data.std()

        # Calculate upper and lower limits for each sensor (3 times standard deviation)
        upper_limits = sensor_means + 3 * sensor_std
        lower_limits = sensor_means - 3 * sensor_std

        # Plot temperature data for each sensor
        for i in range(4):
            sensor_name = legend_labels[i]
            ax.plot(temp_data.index, temp_data.iloc[:, i], color=colors[i], label=sensor_name)

            # Identify points that exceed the upper or lower limit using 3σ rule
            exceed_upper_limit = temp_data.iloc[:, i] > upper_limits[i]
            # exceed_lower_limit = temp_data.iloc[:, i] < lower_limits[i]

            # Mark points that exceed the upper or lower limit
            ax.scatter(
                temp_data.index[exceed_upper_limit], temp_data.iloc[:, i][exceed_upper_limit],
                color=colors[i], marker='x', s=40)

            # ax.scatter(
            #     temp_data.index[exceed_lower_limit], temp_data.iloc[:, i][exceed_lower_limit],
            #     color=colors[i], marker='x', s=20, label=f'{sensor_name} < -3σ'
            # )

    # Set labels for the title, x-axis and y-axis
    ax.set_title(f'{name}-Temperature Distribution-{year}-{month:02d}', fontsize=12)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    # Increase the font size of x-axis tick labels
    ax.tick_params(axis='x', labelsize=10)
    # Increase the font size of y-axis tick labels
    ax.tick_params(axis='y', labelsize=10)
    # Add a legend to the plot
    # ax.legend(labels=legend_labels, fontsize=8, loc='upper right')

    # Ensure a tight layout for better visualization
    plt.tight_layout()

    # Define the file name for saving the plot (SVG format)
    file_name = f'{name}-AD-DC-Temp {year}-{month:02d}.svg'

    # Save the plot to a file
    plt.savefig(os.path.join(month_directory, file_name))




# Initialize dictionaries to store information about exceeded thresholds for each month
exceeded_threshold_info_by_month = {sensor: [] for sensor in ['PM1 Temp', 'PM2 Temp', 'PM3 Temp', 'PM4 Temp']}
total_days_above_threshold_by_month = {sensor: 0 for sensor in ['PM1 Temp', 'PM2 Temp', 'PM3 Temp', 'PM4 Temp']}

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
    month_directory = f'C:/Users/yandcuo/Desktop/Dr/project/process/data/SunEnergy1_DataSet/effi_temp_every_month/effi_temp_no_consider/diff_temp_one_month_extract/3 Sigma/{name}/{year}-{month:02d}'
    os.makedirs(month_directory, exist_ok=True)

    # Initialize dictionaries to store information about exceeded thresholds for the current month
    exceeded_threshold_info = {sensor: set() for sensor in ['PM1 Temp', 'PM2 Temp', 'PM3 Temp', 'PM4 Temp']}
    total_days_above_threshold = {sensor: 0 for sensor in ['PM1 Temp', 'PM2 Temp', 'PM3 Temp', 'PM4 Temp']}

    # Create a Matplotlib figure and axis for plotting for the current month
    # fig, ax = plt.subplots(figsize=(15, 6))

    # Iterate through each day in the current month
    for day in pd.date_range(start=start_date_str, end=end_date_str, freq='D'):
        day_str = day.strftime('%Y-%m-%d')

        # Extract temperature data for the current day
        temp_data = data[name][['PM1 Temp', 'PM2 Temp', 'PM3 Temp', 'PM4 Temp']].loc[day_str]
        temp_data = temp_data.apply(pd.to_numeric, errors='coerce')  # Convert temperature data to numeric format

        # Calculate the mean and standard deviation for each sensor's temperature data
        mean_temps = temp_data.mean()
        std_dev_temps = temp_data.std()

        # Calculate the threshold temperature as 3 times the standard deviation
        threshold_temp = mean_temps + 3 * std_dev_temps

        # Check if any temperature reading exceeds the threshold for each sensor
        for sensor in ['PM1 Temp', 'PM2 Temp', 'PM3 Temp', 'PM4 Temp']:
            exceed_threshold = temp_data[sensor] > threshold_temp[sensor]

            # If any temperature reading exceeds the threshold, update the dictionaries
            if exceed_threshold.any():
                exceed_dates = temp_data.index[exceed_threshold].strftime('%Y-%m-%d').tolist()
                for date in exceed_dates:
                    exceeded_threshold_info[sensor].add(date)

    # Output the total number of days with temperatures above the threshold for each sensor for the current month
    for sensor, dates_set in exceeded_threshold_info.items():
        total_days_above_threshold[sensor] = len(dates_set)
        print(f'The total number of days above {threshold_temp[sensor]:.2f}°C for {sensor} in {year}-{month:02d} is: {len(dates_set)} days')

        # Output the dates when temperatures exceeded the threshold for the current sensor for the current month
        print(f'Dates with temperatures above {threshold_temp[sensor]:.2f}°C for {sensor} in {year}-{month:02d}:')
        for date in sorted(dates_set):
            print(date)


    # Save the print outputs to a CSV file for the current month
    output_file_name = os.path.join(
        month_directory,
        f'{name}-High-Temp {year}-{month:02d}.csv'
    )

    # Define the data to be saved in the CSV file for the current month
    output_data = [['Sensor', 'Threshold Temperature (°C)', 'Total Days Above Threshold', 'Dates Above Threshold']]

    # Add data for each sensor to the output data for the current month
    for sensor in total_days_above_threshold:
        total_days = total_days_above_threshold[sensor]
        exceed_dates = exceeded_threshold_info[sensor]
        output_data.append([
            sensor,
            f'{threshold_temp[sensor]:.2f} °C',
            total_days,
            ', '.join(exceed_dates)
        ])

    # Save the data to a CSV file for the current month
    with open(output_file_name, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(output_data)

    print(f'Print outputs saved to {output_file_name}')

# Close the Matplotlib plot to release resources
plt.show()
