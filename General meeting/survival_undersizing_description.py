# set the low and high undersizing rate

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter #Cox Proportional Hazard model
from scipy.stats import spearmanr
import os
from sklearn.preprocessing import StandardScaler
# Read CSV file
data = pd.read_csv('Dependence Test.csv')

# Map labels to events (1: normal, i.e., truncated events, and 0: failure, i.e., non-truncated events)
data['Event'] = data['Label'].apply(lambda x: 1 if x == 'normal' else 0)

# Create a Standard Scaler
scaler = StandardScaler()

# Standardize the 'Undersizing Rate' column
data['Undersizing Rate'] = scaler.fit_transform(data['Undersizing Rate'].values.reshape(-1, 1))

# Map Undersizing Rate to color
data['color'] = data['Undersizing Rate'].apply(lambda x: 'blue' if x < 0 else 'red')

#
# ################ Survival Model #########################
# # Create Kaplan-Meier estimator object
# kmf = KaplanMeierFitter()
#
# # Compute survival curve
# kmf.fit(durations=data['Working Time'], event_observed=data['Event'], label='KM Estimate')
#
# # Plot survival curve with color mapping
# kmf.plot(ci_show=False, color=data['color'])
#
# # Other plot code remains the same
# # Get censored events' time points
# censored_times = data[data['Event'] == 1]['Working Time']
# censored_probabilities = kmf.survival_function_.loc[censored_times].values
#
# # Median survival time
# median_survival_time = kmf.median_survival_time_
#
# # Survival probability at specific time (e.g., at t=1500)
# specific_time_survival_prob = kmf.predict([1500])
#
# # Calculate area under the survival curve (AUC) using trapezoidal rule
# # Calculate area under the survival curve (AUC) using trapezoidal rule
# survival_probabilities = kmf.survival_function_.iloc[:,0]
# area_under_curve = (survival_probabilities.diff().fillna(0) * survival_probabilities.index.to_series()).sum()
#
# print(f"Median Survival Time: {median_survival_time}")
# print(f"Survival Probability at t=1500: {specific_time_survival_prob}")
# print(f"Area under the Survival Curve (AUC) using trapezoidal rule: {area_under_curve}")
#
# # Show the survival curve plot
# plt.title('Censor Data Analysis')
# plt.xlabel('Working Time')
# plt.ylabel('Survival Probability')
# plt.legend()
# plt.show()

# Divide the data into three groups based on quantiles
# quantiles = data['Undersizing Rate'].quantile([0.33, 0.66])
# labels = ['low', 'mid', 'high']
#
# # Restore the quantiles to original scale
# quantiles_original_scale = scaler.inverse_transform(quantiles.values.reshape(-1, 1))
#
# # Print the restored quantiles
# print("Original Scale Quantiles:")
# print(quantiles_original_scale)

# # Create groups based on original scale quantiles
# data['group'] = pd.cut(data['Undersizing Rate'], bins=[min(data['Undersizing Rate']), quantiles[0.33], quantiles[0.66], max(data['Undersizing Rate'])], labels=labels)
#
# # Plot survival curves for the three groups
# for g in ['low', 'mid', 'high']:
#     kmf = KaplanMeierFitter()
#     kmf.fit(data[data['group']==g]['Working Time'], data[data['group']==g]['Event'], label=g)
#     ax = kmf.plot(ci_show=False)
#
# plt.title('Censor Data Analysis')
# plt.xlabel('Working Time')
# plt.ylabel('Survival Probability')
# plt.legend()
#
# # Get the current directory where the script is located
# current_directory = os.getcwd()
# # Specify the file name and save the plot in the current directory
# file_name = "Survival_Curves_sperate.pdf"
# plt.savefig(os.path.join(current_directory, file_name))
# plt.show()

# ######################获取Undersizing Rate的中位数
median_undersizing_rate = data['Undersizing Rate'].quantile(0.5)

# 将标准化后的中位数还原为原始数据
median_undersizing_rate_original_scale = scaler.inverse_transform([[median_undersizing_rate]])

# 输出还原后的中位数
print("Original Scale Median Undersizing Rate:")
print(median_undersizing_rate_original_scale[0][0])



# 将Undersizing Rate分为前50%和后50%两组
data['group'] = pd.cut(data['Undersizing Rate'], bins=[min(data['Undersizing Rate']), median_undersizing_rate, max(data['Undersizing Rate'])], labels=['Low Undersizing Rate', 'High Undersizing Rate'])

# 绘制生存曲线
for g in ['Low Undersizing Rate', 'High Undersizing Rate']:
    kmf = KaplanMeierFitter()
    kmf.fit(data[data['group']==g]['Working Time'], data[data['group']==g]['Event'], label=g)
    ax = kmf.plot(ci_show=True)

plt.title('Censor Data Analysis')
plt.xlabel('Working Time')
plt.ylabel('Survival Probability')
plt.legend()

# 获取当前脚本所在的目录
current_directory = os.getcwd()
# 指定文件名并将图保存在当前目录
file_name = "Survival_Curves_separated.pdf"
plt.savefig(os.path.join(current_directory, file_name))
plt.show()


