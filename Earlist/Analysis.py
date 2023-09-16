# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 20:51:11 2023

@author: limingl
@upgrade：Yi Luo
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

#%% loading data
data = pd.read_csv('CONFIDENTIAL RANCHLAND SOLAR DATA.csv', header=[0,1])
a = data.columns.get_level_values(0).to_series()
b = a.mask(a.str.startswith('Unnamed')).ffill().fillna('')
data.columns = [b, data.columns.get_level_values(1)]
data = data.fillna(method="ffill")

data['Date','Timestamp'] = pd.to_datetime(data['Date','Timestamp']) # convert column to datetime object
data.set_index(data['Date','Timestamp'], inplace=True) # set column 'date' to index
data.drop(('Date','Timestamp'), inplace=True, axis=1)  # delete 'Date','Timestamp'
#%%



sns.set_style("darkgrid")  # ste graph style
plt.figure(figsize=(12,3))
plt.plot(data['Inv 19', 'DC W']) # DC W is need
plt.ylabel('DC W')
plt.figure(figsize=(12,3))
plt.plot(data['Inv 19', 'AC W'])
plt.ylabel('AC W')
plt.figure(figsize=(12,3))
plt.plot(data['Inv 19', 'DC AMP'])
plt.ylabel('DC AMP')
plt.figure(figsize=(12,3))
plt.plot(data['Inv 19', 'DC VTG'])
plt.ylabel('DC VTG')
plt.figure(figsize=(12,3))
plt.plot(data['Inv 19', 'AC VTG'])
plt.ylabel('AC VTG')
plt.figure(figsize=(12,3))
plt.plot(data['Inv 19', 'AC AMP'])
plt.ylabel('AC AMP')
plt.figure(figsize=(12,3))
plt.plot(data['Inv 19', 'DC W'] / data['Inv 14', 'DC VTG'])
plt.ylabel('DC W/DC VTG')



view = (data['Inv 2', 'DC W'] / data['Inv 2', 'DC VTG'])

#%% 
view2 = data['Inv 1', 'DC W']*data['Inv 1', 'DC AMP'] - data['Inv 1', 'DC W']

#%% Peak active power for 40 inverters
peak_AC_power = []
for i in range(40):
    if i==3:
        continue
    name = 'Inv ' + str(i + 1)
    peak_AC_power.append(data[name, 'AC W'].max()/1000)
peak_AC_power = pd.DataFrame({'Inverter Peak AC Power / kW' : peak_AC_power})
plt.figure(figsize=(8,5))
sns.set_style("darkgrid")
sns.histplot(data=peak_AC_power, x="Inverter Peak AC Power / kW", bins=12)
#%% inverter_efficiency calculation
inverter_efficiency = []
timepoint_start = '9:00'
timepoint_end = '14:59'
for i in range(40):
    if i==3:
        inverter_efficiency.append([])
        continue
    name = 'Inv ' + str(i + 1)
    inverter_efficiency_temp = np.array(data[name, 'AC W'].between_time(timepoint_start, timepoint_end) / data[name, 'DC W'].between_time(timepoint_start, timepoint_end)).reshape(-1,12)
    temp_record = []
    for k in range(365):
        temp_record.append( inverter_efficiency_temp[k,:][inverter_efficiency_temp[k,:]<1].mean()) 
    inverter_efficiency.append(np.array(temp_record))
IE = pd.DataFrame(inverter_efficiency)


# inverter_efficiency = []
# timepoint_start = '9:00'
# timepoint_end = '15:00'
# for i in range(40):
#     if i==3:
#         continue     
#     name = 'Inv ' + str(i + 1)
#     temp = data[name, 'AC W'].between_time(timepoint_start, timepoint_end) / data[name, 'DC W'].between_time(timepoint_start, timepoint_end).reshape(-1,13)
#     inverter_efficiency.append(np.array(temp.mean(axis=1) ))
# IE = pd.DataFrame(inverter_efficiency)

#%%EI
sns.set_style("darkgrid")
plt.figure(figsize=(6,4))
for i in range(40):
    if i==3:
        continue
    name = 'Inv ' + str(i + 1)    
    ax = sns.ecdfplot(data = inverter_efficiency[i])
plt.xlim(-0.1, 1.02)
plt.xlabel('Efficiency (AC/DC %)')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

#%% 
ax = sns.histplot(data = np.array(IE).flatten(), bins=50)
# plt.xlim(-0.1, 1.02)
plt.xlabel('Efficiency (AC/DC %)')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))


plt.figure(figsize=(6,4))
ax = sns.lineplot(data = inverter_efficiency[0])
plt.xlabel('Data points')
plt.ylabel('Efficiency (AC/DC %)')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
#%%  temperature curve with up-low intervals
Temp_AMB = np.array(data['MET Device Averages Across Site', 'Amb Temp']).reshape(-1,48)
ave_temp = pd.DataFrame({'Daily Temp':Temp_AMB.mean(axis=1), 'Days':[i for i in range(365)]})
sns.set_style("darkgrid")
plt.figure(figsize=(10,5))
ax = sns.lineplot( data = ave_temp, x =ave_temp['Days'], y = ave_temp['Daily Temp'])
ax.fill_between(ave_temp['Days'], Temp_AMB.max(axis=1), Temp_AMB.min(axis=1), color='b', alpha=.1)
plt.tight_layout()
plt.show()
#%%  POA curve with up-low intervals
Temp_AMB = np.array(data['MET Device Averages Across Site', 'POA W/m2']).reshape(-1,48)
ave_temp = pd.DataFrame({'Daily POA':Temp_AMB.mean(axis=1), 'Days':[i for i in range(365)]})
sns.set_style("darkgrid")
plt.figure(figsize=(8,5))
ax = sns.lineplot( data = ave_temp, x =ave_temp['Days'], y = ave_temp['Daily POA'])
ax.fill_between(ave_temp['Days'], Temp_AMB.max(axis=1), Temp_AMB.min(axis=1), color='b', alpha=.1)
plt.tight_layout()
plt.show()
#%%  POA curve with up-low intervals
Temp_AMB = np.array(data['MET Device Averages Across Site', 'GHI']).reshape(-1,48)
ave_temp = pd.DataFrame({'Daily GHI':Temp_AMB.mean(axis=1), 'Days':[i for i in range(365)]})
sns.set_style("darkgrid")
plt.figure(figsize=(8,5))
ax = sns.lineplot( data = ave_temp, x =ave_temp['Days'], y = ave_temp['Daily GHI'])
ax.fill_between(ave_temp['Days'], Temp_AMB.max(axis=1), Temp_AMB.min(axis=1), color='b', alpha=.1)
plt.tight_layout()
plt.show()
#%%  temperature curve with up-low intervals
Temp_Cell = np.array(data['MET Device Averages Across Site', 'Cell temp']).reshape(-1,48)
ave_temp = pd.DataFrame({'Cell Temp':Temp_Cell.mean(axis=1), 'Days':[i for i in range(365)]})
sns.set_style("darkgrid")
plt.figure(figsize=(10,5))
ax = sns.lineplot( data = ave_temp, x =ave_temp['Days'], y = ave_temp['Cell Temp'] )
ax.fill_between(ave_temp['Days'], Temp_Cell.max(axis=1), Temp_Cell.min(axis=1), color='b', alpha=.1)
plt.tight_layout()
plt.show()
#%%  Wind velocity curve with up-low intervals
Temp_Cell = np.array(data['MET Device Averages Across Site', 'Wind Vel']).reshape(-1,48)
ave_temp = pd.DataFrame({'Wind Vel':Temp_Cell.mean(axis=1), 'Days':[i for i in range(365)]})
sns.set_style("darkgrid")
plt.figure(figsize=(10,5))
ax = sns.lineplot( data = ave_temp, x =ave_temp['Days'], y = ave_temp['Wind Vel'] )
ax.fill_between(ave_temp['Days'], Temp_Cell.max(axis=1), Temp_Cell.min(axis=1), color='b', alpha=.1)
plt.tight_layout()
plt.show()
#%% operational features

name = 'Inv 1'
starttime = '2022-6-1'
endtime = '2022-6-10'

sns.set_style("darkgrid")
plt.figure(figsize=(12,3))
sns.lineplot( data = data[name]['DC W'].loc[starttime:endtime])
plt.tight_layout()

plt.figure(figsize=(12,3))
sns.lineplot( data = data[name]['DC VTG'].loc[starttime:endtime])
plt.tight_layout()

plt.figure(figsize=(12,3))
sns.lineplot( data = data[name]['DC AMP'].loc[starttime:endtime])
plt.tight_layout()

plt.figure(figsize=(12,3))
sns.lineplot( data = data[name]['AC W'].loc[starttime:endtime])
plt.tight_layout()

plt.figure(figsize=(12,3))
sns.lineplot( data = data[name]['AC VTG'].loc[starttime:endtime])
plt.tight_layout()

plt.figure(figsize=(12,3))
sns.lineplot( data = data[name]['AC AMP'].loc[starttime:endtime])
plt.tight_layout()

plt.figure(figsize=(12,3))
sns.lineplot( data = data[name]['VAR'].loc[starttime:endtime])
plt.tight_layout()


plt.figure(figsize=(4,3))
sns.histplot(data = data[name]['PM3 Temp'].loc[starttime:endtime])
plt.tight_layout()

plt.figure(figsize=(4,3))
sns.histplot(data = data[name]['PM2 Temp'].loc[starttime:endtime])
plt.tight_layout()

plt.figure(figsize=(4,3))
sns.histplot(data = data[name]['PM1 Temp'].loc[starttime:endtime])
plt.tight_layout()

plt.figure(figsize=(4,3 ))
sns.histplot(data = data[name]['PM4 Temp'].loc[starttime:endtime])
plt.tight_layout()


#%%
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
plt.show()

#%% temperature comparsion
name = 'Inv 1'
Temp_Cell = np.array(data[name, 'CL1 AMB']).reshape(-1,48)
ave_temp = pd.DataFrame({'Temp.':Temp_Cell.mean(axis=1), 'Days':[i for i in range(365)]})
sns.set_style("darkgrid")
plt.figure(figsize=(10,5))
ax = sns.lineplot( data = ave_temp, x =ave_temp['Days'], y = ave_temp['Temp.'], label='CL1 AMB Temp.' )
ax.fill_between(ave_temp['Days'], Temp_Cell.max(axis=1), Temp_Cell.min(axis=1), color='b', alpha=.1)

Temp_AMB = np.array(data['MET Device Averages Across Site', 'Amb Temp']).reshape(-1,48)
ave_temp = pd.DataFrame({'Daily Temp':Temp_AMB.mean(axis=1), 'Days':[i for i in range(365)]})
ax = sns.lineplot( data = ave_temp, x =ave_temp['Days'], y = ave_temp['Daily Temp'], label='AMB Temp.')

plt.tight_layout()
plt.show()
#%% temperature comparsion
name = 'Inv 1'
Temp_Cell = np.array(data[name, 'PM1 Temp']).reshape(-1,48)
ave_temp = pd.DataFrame({'Temp.':Temp_Cell.mean(axis=1), 'Days':[i for i in range(365)]})
sns.set_style("darkgrid")
plt.figure(figsize=(10,5))
ax = sns.lineplot( data = ave_temp, x =ave_temp['Days'], y = ave_temp['Temp.'], label='Heatsink1 Temp.' )
ax.fill_between(ave_temp['Days'], Temp_Cell.max(axis=1), Temp_Cell.min(axis=1), color='b', alpha=.1)

Temp_AMB = np.array(data['MET Device Averages Across Site', 'Amb Temp']).reshape(-1,48)
ave_temp = pd.DataFrame({'Daily Temp':Temp_AMB.mean(axis=1), 'Days':[i for i in range(365)]})
ax = sns.lineplot( data = ave_temp, x =ave_temp['Days'], y = ave_temp['Daily Temp'], label='AMB Temp.')

plt.tight_layout()
plt.show()
