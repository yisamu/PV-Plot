
##5 min data
# hilight fault


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime
import os

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
starttime = '2022-12-25 03:00'
endtime = '2022-12-25 21:00'
time_min='2022-12-25 08:45'
time_max='2022-12-25 12:30'

file_path=r'C:\Users\yandcuo\Desktop\Dr\project\process\data\SunEnergy1_DataSet\process-data\IGBT-2\Inverter 2 -12-28-10-55'


sns.set_style("darkgrid")
plt.figure(figsize=(8,3))
sns.lineplot( data = data[name]['DC W'].loc[starttime:endtime])
plt.axvspan(time_min, time_max, color='red', alpha=0.2)  # Highlight the desired region


plt.tight_layout()

# plt.show()

plt.figure(figsize=(8,3))
sns.lineplot( data = data[name]['DC VTG'].loc[starttime:endtime])
plt.axvspan(time_min, time_max, color='red', alpha=0.2)  # Highlight the desired region

plt.tight_layout()

plt.figure(figsize=(8,3))
sns.lineplot( data = data[name]['DC AMP'].loc[starttime:endtime])
plt.axvspan(time_min, time_max, color='red', alpha=0.2)  # Highlight the desired region

plt.tight_layout()

plt.figure(figsize=(8,3))
sns.lineplot( data = data[name]['AC W'].loc[starttime:endtime])
plt.axvspan(time_min, time_max, color='red', alpha=0.2)  # Highlight the desired region

plt.tight_layout()

plt.figure(figsize=(8,3))
sns.lineplot( data = data[name]['AC VTG'].loc[starttime:endtime])
plt.axvspan(time_min, time_max, color='red', alpha=0.2)  # Highlight the desired region

plt.tight_layout()

plt.figure(figsize=(8,3))
sns.lineplot( data = data[name]['AC AMP'].loc[starttime:endtime])
plt.axvspan(time_min, time_max, color='red', alpha=0.2)  # Highlight the desired region

plt.tight_layout()

plt.figure(figsize=(8,3))
sns.lineplot( data = data[name]['VAR'].loc[starttime:endtime])
plt.axvspan(time_min, time_max, color='red', alpha=0.2)  # Highlight the desired region
plt.tight_layout()

plt.show()

# chose data






#
# plt.figure(figsize=(12,3))
# sns.lineplot( data = data[name]['PM1 Temp'].loc[starttime:endtime])
# plt.axvspan(time_min, time_max, color='red', alpha=0.2)  # Highlight the desired region
#
#
# plt.tight_layout()
#
# plt.figure(figsize=(12,3))
# sns.lineplot( data = data[name]['PM2 Temp'].loc[starttime:endtime])
# plt.axvspan(time_min, time_max, color='red', alpha=0.2)  # Highlight the desired region
#
#
# plt.tight_layout()
# plt.figure(figsize=(12,3))
# sns.lineplot( data = data[name]['PM3 Temp'].loc[starttime:endtime])
# plt.axvspan(time_min, time_max, color='red', alpha=0.2)  # Highlight the desired region
#
#
# plt.tight_layout()
# plt.figure(figsize=(12,3))
# sns.lineplot( data = data[name]['PM4 Temp'].loc[starttime:endtime])
# plt.axvspan(time_min, time_max, color='red', alpha=0.2)  # Highlight the desired region
# plt.tight_layout()
###############################################################



#####################histplot#################################
plt.figure(figsize=(4,3))
sns.histplot(data = data[name]['PM1 Temp'].loc[starttime:endtime])
plt.tight_layout()

plt.figure(figsize=(4,3))
sns.histplot(data = data[name]['PM2 Temp'].loc[starttime:endtime])
plt.tight_layout()

plt.figure(figsize=(4,3))
sns.histplot(data = data[name]['PM3 Temp'].loc[starttime:endtime])
plt.tight_layout()


plt.figure(figsize=(4,3 ))
sns.histplot(data = data[name]['PM4 Temp'].loc[starttime:endtime])
plt.tight_layout()

plt.show()


# Define the function to save graphics as SVG
def save_plot_as_svg(data, name, starttime, endtime, ylabel):
    sns.set_style("darkgrid")
    plt.figure(figsize=(12, 3))

    # Use the ylabel parameter to select the correct data column
    data_column = data[name][ylabel].loc[starttime:endtime]

    sns.lineplot(data=data_column)
    plt.axvspan(time_min, time_max, color='red', alpha=0.2)  # Highlight the desired region
    plt.tight_layout()

    plt.ylabel(ylabel)

    # Use ylabel to save SVG
    svg_filename = f"{ylabel.replace(' ', '_')}.svg"
    svg_filepath = os.path.join(file_path, svg_filename)

    # Define svg style
    plt.savefig(svg_filepath, format='svg')

    plt.close()


# Update the function calls to use the correct ylabel
save_plot_as_svg(data, name, starttime, endtime, 'DC W')
save_plot_as_svg(data, name, starttime, endtime, 'DC VTG')
save_plot_as_svg(data, name, starttime, endtime, 'DC AMP')
save_plot_as_svg(data, name, starttime, endtime, 'AC W')
save_plot_as_svg(data, name, starttime, endtime, 'AC VTG')
save_plot_as_svg(data, name, starttime, endtime, 'AC AMP')
save_plot_as_svg(data, name, starttime, endtime, 'VAR')

# save_plot_as_svg(data, name, starttime, endtime, 'PM1 Temp')
# save_plot_as_svg(data, name, starttime, endtime, 'PM2 Temp')
# save_plot_as_svg(data, name, starttime, endtime, 'PM3 Temp')
# save_plot_as_svg(data, name, starttime, endtime, 'PM4 Temp')


# Save the histogram as an SVG file
def save_histogram_as_svg(data, name, starttime, endtime, column):
    plt.figure(figsize=(4, 3))

    # Use the ylabel parameter to select the correct data column
    data_column = data[name][column].loc[starttime:endtime]

    sns.histplot(data=data_column)
    plt.tight_layout()
    plt.ylabel('Frequency')

    svg_filename = f"{column.replace(' ', '_')}.svg"
    svg_filepath = os.path.join(file_path, svg_filename)
    plt.savefig(svg_filepath, format='svg')
    plt.close()


# Update the function calls to use the correct column
save_histogram_as_svg(data, name, starttime, endtime, 'PM1 Temp')
save_histogram_as_svg(data, name, starttime, endtime, 'PM2 Temp')
save_histogram_as_svg(data, name, starttime, endtime, 'PM3 Temp')
save_histogram_as_svg(data, name, starttime, endtime, 'PM4 Temp')


sns.set_style("darkgrid")
palette = ["#f4a582", "#053061", "#4393c3", "#b2182b"]
plt.figure(figsize=(10,6))
sns.histplot(data = data[name]['PM3 Temp'].loc[starttime:endtime], color=palette[0], edgecolor=palette[0],  kde=True, alpha = 0.5, label="PM3 Temp")
sns.histplot(data = data[name]['PM2 Temp'].loc[starttime:endtime], color=palette[1], edgecolor=palette[1], kde=True, alpha = 0.5, label="PM2 Temp")
sns.histplot(data = data[name]['PM1 Temp'].loc[starttime:endtime], color=palette[2], edgecolor=palette[2], kde=True, alpha = 0.5, label="PM1 Temp")
sns.histplot(data = data[name]['PM4 Temp'].loc[starttime:endtime], color=palette[3], edgecolor=palette[3], kde=True, alpha = 0.5, label="PM4 Temp")
# sns.histplot(data=df, x="sepal_length", color="skyblue", label="Sepal Length", kde=True)
# sns.histplot(data=df, x="sepal_width", color="red", label="Sepal Width", kde=True)
plt.xlabel('PM Temp')
plt.legend()
# plt.show()

svg_filename = name + ' Temp fusion.svg'
svg_filepath = os.path.join(file_path, svg_filename)
plt.savefig(svg_filepath, format='svg')

