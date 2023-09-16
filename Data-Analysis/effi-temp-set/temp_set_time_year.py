# one year temp

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

data = pd.read_csv('CONFIDENTIAL RANCHLAND SOLAR DATA - 5 min data.csv', header=[0, 1])


#  convert time style
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
data.fillna(0, inplace=True)

name = 'Inv 2'
starttime = '2022-01-01'
endtime = '2022-12-31'

timepoint_start = '9:00'
timepoint_end = '15:00'

file_path = r'C:\Users\yandcuo\Desktop\Dr\project\process\data\SunEnergy1_DataSet\July_Dic_e_t\one month\temp 9-15'


start_date = datetime.strptime(starttime, "%Y-%m-%d")
end_date = datetime.strptime(endtime, "%Y-%m-%d")


current_month = start_date
while current_month <= end_date:
    # 获取当前月份的起始日期和结束日期
    current_month_start = current_month.replace(day=1)
    next_month_start = (current_month + pd.DateOffset(months=1)).replace(day=1) - pd.DateOffset(days=1)

    # 获取该月的天数
    month_days = (next_month_start - current_month_start).days + 1


    num_rows = 4
    num_cols = 8


    fig_width = 18
    fig_height = 9
    colors = ['blue', 'red', 'green', 'purple']

    if month_days <= num_rows * num_cols:
        num_cols = month_days // num_rows + 1 if month_days % num_rows != 0 else month_days // num_rows

    # 创建子图，每个子图代表一个月
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    axes = axes.ravel()

    for i in range(month_days):
        # 计算当前日期
        current_day = current_month_start + pd.DateOffset(days=i)
        current_day_str = current_day.strftime('%Y-%m-%d')

        # 获取当前日期的数据
        day_data = data.loc[current_day_str]
        time_range_data = day_data.between_time(timepoint_start, timepoint_end)

        for j, temp_type in enumerate(['PM1 Temp', 'PM2 Temp', 'PM3 Temp', 'PM4 Temp']):
            temp_data = time_range_data[name][temp_type].astype(float)
            sns.histplot(data=temp_data, ax=axes[i], color=colors[j], edgecolor='black', kde=True, label=temp_type)

        axes[i].set_title(f'Temp for {current_day_str}', fontsize=7)
        axes[i].set_xlabel('Temperature (°C)', fontsize=6)
        axes[i].set_ylabel('Frequency', fontsize=6)
        axes[i].legend(fontsize=6)
    # 删除多余的子图
    for j in range(month_days, num_rows * num_cols):
        fig.delaxes(axes[j])

    # 保存每个月的图像
    month_name = current_month.strftime('%B %Y')
    file_name = f'temperature_distribution_{month_name}.svg'
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, file_name))
    plt.show()
    plt.close()
    # 切换到下一个月份
    current_month = next_month_start + pd.DateOffset(days=1)
