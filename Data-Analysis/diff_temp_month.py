
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
starttime = '2022-12-01'
endtime = '2022-12-31'


start_date = datetime.strptime(starttime, '%Y-%m-%d')
month_name = start_date.strftime('%m')
file_name = f'temperature_distribution_{month_name}_{start_date.year}.svg'
file_path = r'C:\Users\yandcuo\Desktop\Dr\project\process\data\SunEnergy1_DataSet\effi_temp_every_month\one month\effici_temp_every_month'


def calculate_efficiency(ac_w, dc_w):
    efficiency = (ac_w / dc_w) * 100 if not dc_w.empty and (dc_w != 0).any() else None
    return efficiency

# file_path = r'C:\Users\yandcuo\Desktop\Dr\project\process\data\SunEnergy1_DataSet\July_Dic_e_t\one month\efficiency'


# date_range = pd.date_range(starttime, endtime, freq='D')
#
#
# fig, axes = plt.subplots(4, 8, figsize=(16, 7))
# axes = axes.ravel()




date_range = pd.date_range(starttime, endtime, freq='D')
fig, axes = plt.subplots(4, 8, figsize=(18, 9))
axes = axes.ravel()

for i, day in enumerate(date_range):
    day_str = day.strftime('%Y-%m-%d')

    temp_data_pm1 = data[name]['PM1 Temp'].loc[day_str]
    temp_data_pm2 = data[name]['PM2 Temp'].loc[day_str]
    temp_data_pm3 = data[name]['PM3 Temp'].loc[day_str]
    temp_data_pm4 = data[name]['PM4 Temp'].loc[day_str]

    # Convert to numeric and handle non-numeric values
    temp_data_pm1 = pd.to_numeric(temp_data_pm1, errors='coerce')
    temp_data_pm2 = pd.to_numeric(temp_data_pm2, errors='coerce')
    temp_data_pm3 = pd.to_numeric(temp_data_pm3, errors='coerce')
    temp_data_pm4 = pd.to_numeric(temp_data_pm4, errors='coerce')

    # Check if there are any valid temperature data points before plotting
    if not temp_data_pm1.dropna().empty:
        sns.histplot(data=temp_data_pm1, ax=axes[i], color='blue', edgecolor='black', kde=True, label='PM1 Temp')
        sns.histplot(data=temp_data_pm2, ax=axes[i], color='red', edgecolor='black', kde=True, label='PM2 Temp')
        sns.histplot(data=temp_data_pm3, ax=axes[i], color='green', edgecolor='black', kde=True, label='PM3 Temp')
        sns.histplot(data=temp_data_pm4, ax=axes[i], color='purple', edgecolor='black', kde=True, label='PM4 Temp')

        axes[i].set_title(f'Temp for {day_str}', fontsize=6)
        axes[i].set_xlabel('Temperature (°C)', fontsize=6)
        axes[i].set_ylabel('Frequency', fontsize=6)
        axes[i].legend(fontsize=6)
    else:
        # Handle cases where there is no valid temperature data
        axes[i].axis('off')




for i in range(len(date_range), len(axes)):
    fig.delaxes(axes[i])


plt.tight_layout()
plt.savefig(os.path.join(file_path, file_name))
plt.show()
