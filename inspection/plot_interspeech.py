import os, sys
sys.path.append('./')
import glob
import pandas as pd
from hrtfs.cipic_db import CipicDatabase 

# Use glob to find all files ending with '.csv' in the current directory
csv_files = glob.glob('*.csv')
out_files = glob.glob('*.out')

# Print the list of files
print(csv_files)
#print(out_files)


# Load the CSV file into a DataFrame
sub_3_chan_0_full_dataset = None
for csv_file in csv_files:
    if ("collated_results" in csv_file):
        sub_3_chan_0_full_dataset = pd.read_csv(csv_file)

print(sub_3_chan_0_full_dataset.head())
print(sub_3_chan_0_full_dataset.tail())
