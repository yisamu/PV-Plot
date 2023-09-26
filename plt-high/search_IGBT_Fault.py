# Extract the igbt fault outage
# plt distribution




import pandas as pd
import matplotlib.pyplot as plt
import os

# Set the path to the Excel file to be read
file_path = 'CONFIDENTIAL 2022 Ranchland Outage Report.xlsx'

# Set the path to the directory where you want to save the output
save_path = r'C:\Users\yandcuo\Desktop\Dr\project\process\data\SunEnergy1_DataSet\effi_temp_every_month\effi_temp_no_consider\search-igbt-fault'

# Create an empty DataFrame to store the results
result_df = pd.DataFrame()

# Iterate through all the worksheets in the Excel file
for sheet_name in pd.ExcelFile(file_path).sheet_names:
    # Read the Excel worksheet without column labels
    # print(f"Processing worksheet: {sheet_name}")  # Print the sheet_name

    data = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    # print("Sample of data:")
    # print(data.head())  # Print a sample of data
    # print(data)
    # Check if each row contains "IGBT Fault" and save the matching rows
    igbt_fault_rows = []

    for index, row in data.iterrows():
        if "igbt fault" in str(row.values).lower(): # not consider capital
            # Add the month to the beginning of the data before saving
            row_with_month = [sheet_name] + row.tolist()

            igbt_fault_rows.append(row_with_month)
            # print(row_with_month)
    # Create a DataFrame containing rows with "IGBT Fault"
    igbt_fault_df = pd.DataFrame(igbt_fault_rows)

    # Append rows containing "IGBT Fault" to the result DataFrame
    result_df = pd.concat([result_df, igbt_fault_df], ignore_index=True)

# Save the results to a new Excel file
result_df.to_excel(f'{save_path}\\igbt_fault_data_with_month.xlsx', index=False, header=False)  #\\ separate directories and files.




# Extract failure months
fault_months = result_df[0]

# Count the number of occurrences in each month
month_counts = fault_months.value_counts()

# Define the order of months as per your data
month_order = list(fault_months.unique())

# Sort the month_counts Series by the custom order
month_counts = month_counts[month_order]

# print(month_counts)
# plt a histogram
plt.figure(figsize=(10, 6))
month_counts.plot(kind='bar', color='blue', edgecolor='k')
plt.xlabel('Month', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Distribution of IGBT Faults by Month', fontsize=11)
plt.xticks(rotation=45)

plt.tight_layout()

file_name = f' IGBT-Fault-distribution.svg'
plt.savefig(os.path.join(save_path, file_name))
plt.show()