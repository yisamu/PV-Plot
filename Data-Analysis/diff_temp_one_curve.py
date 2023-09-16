# plot one day  one temp four temp


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime
from scipy.interpolate import interp1d
import os
import matplotlib as mpl


mpl.rcParams.update({'font.size': 6})

data = pd.read_csv('C:/Users/yandcuo/Desktop/Dr/project/process/data/SunEnergy1_DataSet/Data/CONFIDENTIAL RANCHLAND SOLAR DATA - 5 min data.csv', header=[0, 1])



#  convert time style
def convert_date_time(date_string):
    date_object = datetime.strptime(date_string, "%d-%m月-%y %H:%M:%S")
    formatted_date = date_object.strftime("%m/%d/%Y %H:%M")
    return formatted_date

# convert columns
column_name = data.columns[0]  # 'Your_Column_Name'
data[column_name] = data[column_name].apply(convert_date_time)


a = data.columns.get_level_values(0).to_series() # get columns label
b = a.mask(a.str.startswith('Unnamed')).ffill().fillna('') #fill unnamed label
data.columns = [b, data.columns.get_level_values(1)]
data = data.fillna(method="ffill")

data['Date','Timestamp'] = pd.to_datetime(data['Date','Timestamp']) # convert column to datetime object
data.set_index(data['Date','Timestamp'], inplace=True) # set column 'date' to index
data.drop(('Date','Timestamp'), inplace=True, axis=1)  # delete 'Date','Timestamp'

data.replace(['Bad', '#DIV/0!'], pd.NA, inplace=True)

name = 'Inv 2'
starttime = '2022-12-25'

start_date = datetime.strptime(starttime, '%Y-%m-%d')
month_name = start_date.strftime('%m')
day_name = start_date.strftime('%d')
file_path=r'C:\Users\yandcuo\Desktop\Dr\project\process\data\SunEnergy1_DataSet\effi_temp_every_month\effi_temp_no_consider\Single-temp-ac-dc'

day_str=starttime
sns.set_style("darkgrid")
fig, axes = plt.subplots(4, 1, figsize=(12, 6))

# DC Power
dc_data = data[name]['DC W'].loc[day_str]
dc_data = pd.to_numeric(dc_data, errors='coerce')
axes[0].plot(dc_data.index, dc_data.values, color='green')
axes[0].set_title(f'DC Power for {day_str}', fontsize=12)
axes[0].set_xlabel('Time', fontsize=10)
axes[0].set_ylabel('DC Power (W)', fontsize=10)
# axes[0].grid(True)

# AC Power
ac_data = data[name]['AC W'].loc[day_str]
ac_data = pd.to_numeric(ac_data, errors='coerce')
axes[1].plot(ac_data.index, ac_data.values, color='red')
axes[1].set_title(f'AC Power for {day_str}', fontsize=12)
axes[1].set_xlabel('Time', fontsize=10)
axes[1].set_ylabel('AC Power (W)', fontsize=10)
# axes[1].grid(True)

# DC Voltge
ac_data = data[name]['DC VTG'].loc[day_str]
ac_data = pd.to_numeric(ac_data, errors='coerce')
axes[2].plot(ac_data.index, ac_data.values, color='red')
axes[2].set_title(f'DC VTG for {day_str}', fontsize=12)
axes[2].set_xlabel('Time', fontsize=10)
axes[2].set_ylabel('DC VTG (V)', fontsize=10)
# axes[1].grid(True)

# PM1 Temperature
# caculate PM1

temp_data_pm1 = data[name]['PM1 Temp'].loc[day_str]
# temp_data_pm1 = pd.to_numeric(temp_data_pm1, errors='coerce')
# axes[3].plot(temp_data_pm1.index, temp_data_pm1.values, color='blue')
# axes[3].set_title(f'Temperature for {day_str}', fontsize=12)
# axes[3].set_xlabel('Time', fontsize=10)
# axes[3].set_ylabel('Temperature (°C)', fontsize=10)

# plt PM1 PM2 PM3 PM4 with time
# Temperature
temp_data = data[name][['PM1 Temp', 'PM2 Temp', 'PM3 Temp', 'PM4 Temp']].loc[day_str]
temp_data = temp_data.apply(pd.to_numeric, errors='coerce')

colors = ['blue', 'red', 'green', 'purple']
labels = ['PM1 Temp', 'PM2 Temp', 'PM3 Temp', 'PM4 Temp']

for i in range(4):
    axes[3].plot(temp_data.index, temp_data.iloc[:, i], color=colors[i], label=labels[i])

axes[3].set_title(f'Temperature for {day_str}', fontsize=12)
axes[3].set_xlabel('Time', fontsize=10)
axes[3].set_ylabel('Temperature (°C)', fontsize=10)
axes[3].legend(fontsize=8)



plt.tight_layout()
file_name1 = f'AD-DC-Temp 2022-{month_name}-{day_name}.svg'
plt.savefig(os.path.join(file_path, file_name1))

# plt PM1 histogram
plt.figure(figsize=(12,5))
sns.histplot(data=temp_data_pm1,color='blue', edgecolor='black', kde=True, label='PM1 Temp')
plt.title(f'Temp for {day_str}', fontsize=10)
plt.xlabel('Temperature (°C)', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.legend(fontsize=10)
file_name2 = f'Temp-hist 2022-{month_name}-{day_name}.svg'
plt.savefig(os.path.join(file_path, file_name2))



# plt PM1 PM2 PM3 PM4  histogram
temp_data_list = [data[name]['PM1 Temp'], data[name]['PM2 Temp'], data[name]['PM3 Temp'], data[name]['PM4 Temp']]
colors = ['blue', 'red', 'green', 'purple']
fig, ax = plt.subplots(figsize=(12, 5))
for i in range(4):
    temp_data = temp_data_list[i].loc[day_str]
    # print(temp_data_list[1].loc[day_str])
    temp_data = pd.to_numeric(temp_data, errors='coerce')

    # Plot histogram for each temperature
    sns.histplot(temp_data.dropna(), bins=30, color=colors[i], edgecolor='black', kde=True,  label=labels[i])
    ax.set_title(f'Temperature Histograms for {day_str}', fontsize=12)
ax.set_xlabel('Temperature (°C)', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.legend(fontsize=8)


plt.tight_layout()
file_name3= f'Temp-4 2022-{month_name}-{day_name}.svg'
plt.savefig(os.path.join(file_path, file_name3))

plt.show()