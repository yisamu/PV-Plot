
# caculate every month  9:00  15:00  ok but some point miss

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

data = pd.read_csv('CONFIDENTIAL RANCHLAND SOLAR DATA - 5 min data.csv', header=[0, 1])

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

name = 'Inv 2'
starttime = '2022-01-01'
endtime = '2022-12-31'

timepoint_start = '9:00'
timepoint_end = '15:00'

#
# file_name1 = 'efficiency_09.svg'
# file_name2 = 'average_efficiency_09.svg'

file_path = r'C:\Users\yandcuo\Desktop\Dr\project\process\data\SunEnergy1_DataSet\July_Dic_e_t\one month\efficiency 9-15'

start_date = datetime.strptime(starttime, "%Y-%m-%d")
end_date = datetime.strptime(endtime, "%Y-%m-%d")

def calculate_efficiency(ac_w, dc_w):
    ac_w = pd.to_numeric(ac_w, errors='coerce')
    dc_w = pd.to_numeric(dc_w, errors='coerce')
    if ac_w.isna().any() or dc_w.isna().any() or (dc_w == 0).all():
        efficiency = None
    else:
        efficiency = (ac_w / dc_w) * 100
    return efficiency



monthly_mean_efficiency_values = []
current_month = start_date
while current_month <= end_date:

    current_month_start = current_month.replace(day=1)
    next_month_start = (current_month + pd.DateOffset(months=1)).replace(day=1) - pd.DateOffset(days=1)

    month_days = (next_month_start - current_month_start).days + 1


    num_rows = 4
    num_cols = 8
    fig_width = 18
    fig_height = 9

    if month_days <= num_rows * num_cols:
        num_cols = month_days // num_rows + 1 if month_days % num_rows != 0 else month_days // num_rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    axes = axes.ravel()
    daily_mean_efficiency_values = []





    for i in range(month_days):
        # 计算当前日期
        current_day = current_month_start + pd.DateOffset(days=i)
        current_day_str = current_day.strftime('%Y-%m-%d')

        day_data = data.loc[current_day_str]
        time_range_data = day_data.between_time(timepoint_start, timepoint_end)



        ac_w_data = time_range_data['Inv 2']['AC W']
        dc_w_data = time_range_data['Inv 2']['DC W']

        non_zero_mask = dc_w_data != 0

        interp_dc_w_data = dc_w_data.copy()
        interp_dc_w_data[~non_zero_mask] = np.nan
        interp_dc_w_data.interpolate(method='linear', inplace=True)

        efficiency_data = calculate_efficiency(ac_w_data, interp_dc_w_data)

        efficiency_df = pd.DataFrame({'Efficiency (%)': efficiency_data}, index=ac_w_data.index)

        efficiency_df['Efficiency (%)'] = efficiency_df['Efficiency (%)'].apply(
            lambda x: x if (x is not None) and (0 <= x <= 100) else np.nan)
        efficiency_df.dropna(subset=['Efficiency (%)'], inplace=True)  # 删除包含NaN值的行

        mean_efficiency = np.nanmean(efficiency_df['Efficiency (%)'])
        daily_mean_efficiency_values.append(mean_efficiency)



        hist_color = 'blue'
        title_color = 'black'

        axes[i].hist(efficiency_df['Efficiency (%)'], bins=20, color=hist_color, alpha=0.7)
        axes[i].set_title(f'Efficiency for {current_day_str}', color=title_color)
        axes[i].set_xlabel('Efficiency (%)')
        axes[i].set_ylabel('Frequency')

    for j in range(len(time_range_data), len(axes)):
        fig.delaxes(axes[j])



    month_name = current_month.strftime('%B %Y')
    file_name = f'efficiency_{month_name}.svg'
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, file_name))
    # plt.show()
    plt.close()
    # 切换到下一个月份
    # current_month = next_month_start + pd.DateOffset(days=1)

#average effi

    plt.figure(figsize=(14, 7))
    x_labels = [str(j + 1) for j in range(len(daily_mean_efficiency_values))]
    plt.plot(x_labels, daily_mean_efficiency_values, marker='o', linestyle='-', color='blue')
    plt.title(f'Monthly Average Efficiency for {current_month.strftime("%B %Y")}')
    plt.xlabel('Day')
    plt.ylabel('Average Efficiency (%)')
    plt.xticks(rotation=45, fontsize=7)
    plt.grid(True)

    for j, label in enumerate(x_labels):
        plt.text(j, daily_mean_efficiency_values[j], label, ha='center', va='bottom', fontsize=7)

    plt.grid(False)
    plt.tight_layout()
    file_name = f'monthly_efficiency_and_average_{current_month.strftime("%B %Y")}.svg'
    plt.savefig(os.path.join(file_path, file_name))
    # plt.show()
    plt.close()
    # 将每个月的平均效率值添加到列表中
    monthly_mean_efficiency_values.append(daily_mean_efficiency_values)

    # 切换到下一个月份
    current_month = next_month_start + pd.DateOffset(days=1)












        # plt.figure(figsize=(14, 7))
        # x_labels = [day.strftime('%Y-%m-%d') for day in time_range_data]
        # plt.plot(x_labels, daily_mean_efficiency_values, marker='o', linestyle='-', color='blue')
        # plt.title('Daily Average Efficiency for December 2022')
        # plt.xlabel('Date')
        # plt.ylabel('Average Efficiency (%)')
        # plt.xticks(rotation=45, fontsize=7)
        # plt.grid(True)
        #
        # for i, label in enumerate(x_labels):
        #     plt.text(i, daily_mean_efficiency_values[i], label, ha='center', va='bottom', fontsize=7)
        #
        # plt.grid(False)
        # # plt.savefig(os.path.join(file_path, file_name2))
        # plt.show()
