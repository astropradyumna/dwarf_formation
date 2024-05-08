'''
This code is to merge all the csv files for FoFs0, 1,  and 2
'''

import pandas as pd



outpath  = '/rhome/psadh003/bigdata/tng50/output_files/'

'''
First set of files
'''
# File paths for the three CSV files
csv_file_1 = "fof0_merged_evolved_everything.csv"
csv_file_2 = "fof1_merged_evolved_everything.csv"
csv_file_3 = "fof2_merged_evolved_everything.csv"

# Read the CSV files into DataFrames
df1 = pd.read_csv(outpath + csv_file_1, low_memory=False, delimiter = ',')
df2 = pd.read_csv(outpath + csv_file_2, low_memory=False, delimiter = ',')
df3 = pd.read_csv(outpath + csv_file_3, low_memory=False, delimiter = ',')

df1['fof'] = 0
df2['fof'] = 1
df3['fof'] = 2

# print(df1.head())


# Concatenate the DataFrames vertically (row-wise)
merged_df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_csv_file = "fof210_merged_evolved_everything.csv"
merged_df.to_csv( outpath + merged_csv_file, index=False)

'''
Second set of files
'''
# File paths for the three CSV files
csv_file_1 = "fof0_merged_evolved_wmbp_everything.csv"
csv_file_2 = "fof1_merged_evolved_wmbp_everything.csv"
csv_file_3 = "fof2_merged_evolved_wmbp_everything.csv"

# Read the CSV files into DataFrames
df1 = pd.read_csv(outpath + csv_file_1, low_memory=False, delimiter = ',')
df2 = pd.read_csv(outpath + csv_file_2, low_memory=False, delimiter = ',')
df3 = pd.read_csv(outpath + csv_file_3, low_memory=False, delimiter = ',')

df1['fof'] = 0
df2['fof'] = 1
df3['fof'] = 2

# print(df1.head())


# Concatenate the DataFrames vertically (row-wise)
merged_df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_csv_file = "fof210_merged_evolved_wmbp_everything.csv"
merged_df.to_csv( outpath + merged_csv_file, index=False)



'''
Third set of files
'''
# File paths for the three CSV files
csv_file_1 = "fof0_surviving_evolved_everything.csv"
csv_file_2 = "fof1_surviving_evolved_everything.csv"
csv_file_3 = "fof2_surviving_evolved_everything.csv"

# Read the CSV files into DataFrames
df1 = pd.read_csv(outpath + csv_file_1, low_memory=False, delimiter = ',')
df2 = pd.read_csv(outpath + csv_file_2, low_memory=False, delimiter = ',')
df3 = pd.read_csv(outpath + csv_file_3, low_memory=False, delimiter = ',')

df1['fof'] = 0
df2['fof'] = 1
df3['fof'] = 2

# print(df1.head())


# Concatenate the DataFrames vertically (row-wise)
merged_df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_csv_file = "fof210_surviving_evolved_everything.csv"
merged_df.to_csv( outpath + merged_csv_file, index=False)