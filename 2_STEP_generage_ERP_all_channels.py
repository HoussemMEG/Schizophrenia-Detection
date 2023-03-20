import os
import pandas as pd


# reading subjects information and data
column_list = pd.read_csv(r"./data/columnLabels.csv").columns
print("Column list\n", list(column_list))

channels = column_list[4:-6]
print("Channels\n", list(channels))

demographic = pd.read_csv(r"./data/demographic.csv")
diagnosis_dict = dict(zip(demographic.subject, demographic[" group"]))  # 1 SZ 0 CTL
print("Diagnosis dict\n", diagnosis_dict)
print("Categories: <0 healthy control>, <1 schizophrenia>")
print(demographic[' group'].value_counts(), '\n')

df_ERP = pd.read_csv(r'./data/ERPdata.csv')
time = df_ERP['time_ms']
print(df_ERP)


# Walk through all the data to read it and parse it and put it in only one csv file
global_df = pd.DataFrame()
for root, dirs, filenames in os.walk(r'./data'):
    if dirs:
        continue

    for filename in filenames:
        print(filename)
        df = pd.read_csv(os.path.join(root, filename), header=None, names=column_list)
        df = df.sort_values(by=['condition', 'trial', 'sample'])
        df = df.reset_index(drop=True)
        df = df.groupby(by=['condition', 'sample']).apply('mean')
        df.reset_index(drop=False, inplace=True)
        df.drop('trial', axis=1, inplace=True)
        df = df.astype({'subject': int, 'condition': int, 'sample': int})
        df = df[['subject', 'condition', *channels]]
        df['time_ms'] = time
        global_df = pd.concat([global_df, df], ignore_index=True)

global_df.sort_values(by=['subject', 'condition'], inplace=True)
print(global_df)

# Save the read information into all_channels_ERP
global_df.to_csv(os.path.join(r'./all_channels_ERP.csv'))
