
# caculate  effcicy every month

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

def convert_date_time(date_string):
    date_object = datetime.strptime(date_string, "%d-%m月-%y %H:%M:%S")
    formatted_date = date_object.strftime("%m/%d/%Y %H:%M")
    return formatted_date

column_name = data.columns[0]
data[column_name] = data[column_name].apply(convert_date_time)

a = data.columns.get_level_values(0).to_series()
b = a.mask(a.str.startswith('Unnamed')).ffill().fillna('')
data.columns = [b, data.columns.get_level_values(1)]
data = data.fillna(method="ffill")

data['Date', 'Timestamp'] = pd.to_datetime(data['Date', 'Timestamp'])
data.set_index(data['Date', 'Timestamp'], inplace=True)
data.drop(('Date', 'Timestamp'), inplace=True, axis=1)
data.replace(['Bad', '#DIV/0!'], pd.NA, inplace=True)

starttime = '2022-07-01'
endtime = '2022-07-31'

start_date = datetime.strptime(starttime, '%Y-%m-%d')
month_name = start_date.strftime('%m')
file_name1 = f'efficiency_{month_name}.svg'
file_name2 = f'average_efficiency_{month_name}.svg'
file_path = r'C:\Users\yandcuo\Desktop\Dr\project\process\data\SunEnergy1_DataSet\effi_temp_every_month\one month\effici_temp_every_month'


def calculate_efficiency(ac_w, dc_w):
    ac_w = pd.to_numeric(ac_w, errors='coerce')
    dc_w = pd.to_numeric(dc_w, errors='coerce')
    if ac_w.isna().any() or dc_w.isna().any() or (dc_w == 0).all():
        efficiency = None
    else:
        efficiency = (ac_w / dc_w) * 100
    return efficiency



date_range = pd.date_range(starttime, endtime, freq='D')

fig, axes = plt.subplots(4, 8, figsize=(18, 8))
axes = axes.ravel()

daily_mean_efficiency_values = []

for i, day in enumerate(date_range):
    day_str = day.strftime('%Y-%m-%d')

    ac_w_data = data['Inv 2']['AC W'].loc[day_str]
    dc_w_data = data['Inv 2']['DC W'].loc[day_str]

    non_zero_mask = dc_w_data != 0

    interp_dc_w_data = dc_w_data.copy()
    interp_dc_w_data[~non_zero_mask] = np.nan
    interp_dc_w_data.interpolate(method='linear', inplace=True)

    efficiency_data = calculate_efficiency(ac_w_data, interp_dc_w_data)

    efficiency_df = pd.DataFrame({'Efficiency (%)': efficiency_data}, index=ac_w_data.index)

    efficiency_df['Efficiency (%)'] = efficiency_df['Efficiency (%)'].apply(
        lambda x: x if (x is not None) and (0 <= x <= 100) else np.nan)
    efficiency_df.dropna(subset=['Efficiency (%)'], inplace=True)  # 删除包含NaN值的行

    mean_efficiency = efficiency_df['Efficiency (%)'].mean()
    daily_mean_efficiency_values.append(mean_efficiency)

    hist_color = 'blue'
    title_color = 'black'

    axes[i].hist(efficiency_df['Efficiency (%)'], bins=20, color=hist_color, alpha=0.7)
    axes[i].set_title(f'Efficiency for {day_str}', color=title_color)
    axes[i].set_xlabel('Efficiency (%)')
    axes[i].set_ylabel('Frequency')

for i in range(len(date_range), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig(os.path.join(file_path, file_name1))
plt.show()

plt.figure(figsize=(14, 7))
x_labels = [day.strftime('%Y-%m-%d') for day in date_range]
plt.plot(x_labels, daily_mean_efficiency_values, marker='o', linestyle='-', color='blue')
plt.title(f'Daily Average Efficiency for {month_name}')
plt.xlabel('Date')
plt.ylabel('Average Efficiency (%)')
plt.xticks(rotation=45, fontsize=7)
plt.grid(True)

for i, label in enumerate(x_labels):
    plt.text(i, daily_mean_efficiency_values[i], label, ha='center', va='bottom', fontsize=7)

plt.grid(False)
plt.savefig(os.path.join(file_path, file_name2))
plt.show()
