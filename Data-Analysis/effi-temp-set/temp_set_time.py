
##efficiency caculation 9:00  15:00


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime
from scipy.interpolate import interp1d
import os

data = pd.read_csv('CONFIDENTIAL RANCHLAND SOLAR DATA - 5 min data.csv', header=[0,1],) #header  inx row 1 2


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

# 使用fillna方法将所有NaN值替换为特定值，例如0
data.fillna(0, inplace=True)

name = 'Inv 2'
starttime = '2022-12-01'
endtime = '2022-12-31'

timepoint_start = '9:00'
timepoint_end = '15:00'

file_name = f'temperature_distribution_nov1_2022.svg'


file_path = r'C:\Users\yandcuo\Desktop\Dr\project\process\data\SunEnergy1_DataSet\July_Dic_e_t\one month'


date_range = pd.date_range(starttime, endtime, freq='D')


fig, axes = plt.subplots(4, 8, figsize=(20, 10))
axes = axes.ravel()

for i, day in enumerate(date_range):
    day_str = day.strftime('%Y-%m-%d')
    day_data = data.loc[day_str]
    time_range_data=day_data.between_time(timepoint_start,timepoint_end)

    #  temp and convert to float
    temp_data_pm1 = time_range_data[name]['PM1 Temp'].loc[day_str].astype(float)
    temp_data_pm2 = time_range_data[name]['PM2 Temp'].loc[day_str].astype(float)
    temp_data_pm3 = time_range_data[name]['PM3 Temp'].loc[day_str].astype(float)
    temp_data_pm4 = time_range_data[name]['PM4 Temp'].loc[day_str].astype(float)

    # 使用Seaborn绘制直方图和KDE曲线
    sns.histplot(data=temp_data_pm1, ax=axes[i], color='blue', edgecolor='black', kde=True, label='PM1 Temp')
    sns.histplot(data=temp_data_pm2, ax=axes[i], color='red', edgecolor='black', kde=True, label='PM2 Temp')
    sns.histplot(data=temp_data_pm3, ax=axes[i], color='green', edgecolor='black', kde=True, label='PM3 Temp')
    sns.histplot(data=temp_data_pm4, ax=axes[i], color='purple', edgecolor='black', kde=True, label='PM4 Temp')

    axes[i].set_title(f'Temp for {day_str}', fontsize=6)
    axes[i].set_xlabel('Temperature (°C)', fontsize=6)
    axes[i].set_ylabel('Frequency', fontsize=6)
    axes[i].legend(fontsize=6)

    # 可以根据需要自定义坐标轴刻度等

# 删除多余的子图
for i in range(len(date_range), len(axes)):
    fig.delaxes(axes[i])

# 调整子图之间的间距
plt.tight_layout()

# 保存图像或显示图像
# plt.savefig(os.path.join(file_path, file_name))
plt.show()