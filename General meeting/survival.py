# 2023/10/15
#Kaplan-Meier
#Spearman correlation coefficient
#Cox Proportional Hazard Model
# research working time with undersizing rate coef

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

# 创建一个标准化器
scaler = StandardScaler()

# 对'Undersizing Rate'列进行标准化
data['Undersizing Rate'] = scaler.fit_transform(data['Undersizing Rate'].values.reshape(-1, 1))



################ Survival Model #########################
# Create Kaplan-Meier estimator object
kmf = KaplanMeierFitter()

# Compute survival curve and confidence interval
kmf.fit(durations=data['Working Time'], event_observed=data['Event'])
kmf.plot()

# Get censored events' time points
censored_times = data[data['Event'] == 1]['Working Time']
censored_probabilities = kmf.survival_function_.loc[censored_times]['KM_estimate']

# Median survival time
median_survival_time = kmf.median_survival_time_

# Survival probability at specific time (e.g., at t=1500)
specific_time_survival_prob = kmf.predict([1500])

# Calculate area under the survival curve (AUC) using trapezoidal rule
survival_probabilities = kmf.survival_function_['KM_estimate']
area_under_curve = (survival_probabilities.diff().fillna(0) * survival_probabilities.index.to_series()).sum()

print(f"Median Survival Time: {median_survival_time}")
print(f"Survival Probability at t=1500: {specific_time_survival_prob}")
print(f"Area under the Survival Curve (AUC) using trapezoidal rule: {area_under_curve}")

# Show the survival curve plot
plt.title('Censor Data Analysis')
plt.xlabel('Working Time')
plt.ylabel('Survival Probability')
plt.legend()

# Get the current directory where the script is located
current_directory = os.getcwd()
# Specify the file name and save the plot in the current directory
file_name = "Survival_Curves.pdf"
plt.savefig(os.path.join(current_directory, file_name))
plt.show()

################ Cox Proportional Hazard Model #########################

# Calculate Spearman correlation coefficient
corr, _ = spearmanr(data['Undersizing Rate'], data['Working Time'])
print('Spearman Correlation: ', corr)

plt.scatter(data[data['Label']=='normal']['Working Time'], data[data['Label']=='normal']['Undersizing Rate'], c='b', label='normal')
plt.scatter(data[data['Label']=='failure']['Working Time'], data[data['Label']=='failure']['Undersizing Rate'], c='r', label='failure')

plt.xlabel('Working Time')
plt.ylabel('Undersizing Rate')
plt.legend()
file_name = "scatter.svg"
plt.savefig(os.path.join(current_directory, file_name))
plt.show()
plt.show()

# Split data into truncated events and non-truncated events
truncated_data = data[data['Event'] == 1]
non_truncated_data = data[data['Event'] == 0]

# Create Cox Proportional Hazard Regression model objects with L2 regularization
cph_truncated = CoxPHFitter(penalizer=0.1)
cph_non_truncated = CoxPHFitter(penalizer=0.1)

# Fit Cox Proportional Hazard Regression models separately for truncated and non-truncated events considering "Working Time" and "Undersizing Rate"
cph_truncated.fit(truncated_data, duration_col='Working Time', event_col='Event', formula="Q('Undersizing Rate')")
cph_non_truncated.fit(non_truncated_data, duration_col='Working Time', event_col='Event', formula="Q('Undersizing Rate')")

# Print regression coefficients
print("Truncated Data Model Summary:")
print(cph_truncated.summary)

print("\nNon-Truncated Data Model Summary:")
print(cph_non_truncated.summary)

############# Create Kaplan-Meier Survival Function Objects ##################
kmf_truncated = KaplanMeierFitter()
kmf_non_truncated = KaplanMeierFitter()

# Calculate Kaplan-Meier survival curves
kmf_truncated.fit(truncated_data['Working Time'], event_observed=truncated_data['Event'], label='Truncated')
kmf_non_truncated.fit(non_truncated_data['Working Time'], event_observed=non_truncated_data['Event'], label='Non-Truncated')

# Plot Kaplan-Meier survival curves
plt.figure(figsize=(10, 6))
kmf_truncated.plot()
kmf_non_truncated.plot()

plt.title('Kaplan-Meier Survival Curves')
plt.xlabel('Working Time')
plt.ylabel('Survival Probability')
plt.legend(loc='best')

# Write model summary information to files
with open('truncated_data_output.txt', 'w') as f:
    f.write(cph_truncated.summary.to_string())

with open('non_truncated_data_output.txt', 'w') as f:
    f.write(cph_non_truncated.summary.to_string())
