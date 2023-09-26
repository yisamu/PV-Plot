#plt 15 inv high temp

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

data = pd.read_csv( 'Inv1-15-high-temp.csv')

# Get unique labels for all inverters
inverters = data['Inverter'].unique()

# define the number of rows and columns for subplots
num_rows = 5
num_cols = 3

# Create a new Figure object and set the layout for subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
fig.subplots_adjust(hspace=0.5)

# Iterate through all inverters
for i, inverter in enumerate(inverters):
    row = i // num_cols  # Calculate the current row for the inverter  exact division
    col = i % num_cols   # Calculate the current column for the inverter
    # Print the values of row and col for each iteration
    # print(f"i = {i}, row = {row}, col = {col}")
    # Get data for the x-axis (months) and y-axis (Number of Alerts)
    months = data.columns[2:]  # Month data starts from the 3rd column

    alerts = data.loc[data['Inverter'] == inverter, data.columns[2:]].astype(int).values[0]
    # Print the names of inverter for each iteration
    # print(f"i = {i}, inverter = {inverter}")
    # Plot the subplot

    axes[row, col].plot(months, alerts, marker='o', linestyle='-', color='b')
    axes[row, col].set_title(inverter, fontsize=11)
    axes[row, col].set_xlabel('Month', fontsize=11)
    axes[row, col].set_ylabel('Number of High temp', fontsize=11)

# Remove unused subplots
for i in range(len(inverters), num_rows * num_cols):
    row = i // num_cols
    col = i % num_cols
    fig.delaxes(axes[row, col])


plt.tight_layout()
file_path = r'C:\Users\yandcuo\Desktop\Dr\project\process\data\SunEnergy1_DataSet\effi_temp_every_month\effi_temp_no_consider\plt_inv_high_temp'
file_name = f' 2022-High-Temp.svg'
plt.savefig(os.path.join(file_path, file_name))
plt.show()

