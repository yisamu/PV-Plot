# caculat cox hazard coef,hr,p
# for every factors.
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter #Cox Proportional Hazard model
from scipy.stats import spearmanr
import os
from sklearn.preprocessing import StandardScaler

# Read data
data = pd.read_csv('Feature_Results_Analysis.csv')

# Map labels to events (1: normal, i.e., truncated events, and 0: failure, i.e., non-truncated events)
data['Event'] = data['Label'].apply(lambda x: 1 if x == 'Normal' else 0)

# Create a standard scaler
scaler = StandardScaler()

# List to store results for each factor
results_list = []

# Loop through columns from the 5th to 17th
for column in data.columns[4:17]:
    # Standardize the current column
    data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1))

    # Create Cox Proportional Hazard model object
    cph = CoxPHFitter()

    # Fit the model considering 'Working Time' and the current column, and truncation events
    cph.fit(data, duration_col='working_time', event_col='Event', formula=f"Q('{column}')")

    # Get the coefficient for the current column
    coef = cph.summary['coef'][f'Q(\'{column}\')']

    # Get the Hazard Ratio for the current column
    hr = cph.hazard_ratios_[f'Q(\'{column}\')']

    # Get the p-value for the current column
    p_value = cph.summary['p'][f'Q(\'{column}\')']

    # Print results for the current column, including column name prefix
    print(f"Column: {column}")
    print(f"Coefficient: {coef}")
    print(f"Hazard Ratio: {hr}")
    print(f"P-Value: {p_value}")
    print("--------------------------------------")

    # Append results to the results list
    results_list.append([column, coef, hr, p_value])

# Create a DataFrame for the results
results_df = pd.DataFrame(results_list, columns=['Column', 'Coefficient', 'Hazard Ratio', 'P-Value'])

# Save results to a CSV file including column names
results_df.to_csv('cox_result.csv', index=False)

print("Results saved as 'cox_result.csv'.")