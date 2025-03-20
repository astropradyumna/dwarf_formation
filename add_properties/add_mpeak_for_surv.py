# This code is to run through all the list of surviving subhalos and append the position from the MBP at infall for the subhalos
#This is the assumption that we are using the combined file of all the three FoF groups

import sys
import ast
import os
import re
sys.path.append(os.path.abspath('..'))

import numpy as np
import pandas as pd
from errani_plus_tng_subhalo import Subhalo
from tqdm import tqdm
import illustris_python as il
from joblib import Parallel, delayed #This is to parallelize the code
# import sys
# import ast



'''
file import
'''
filepath = '/rhome/psadh003/bigdata/tng50/tng_files/'
outpath  = '/rhome/psadh003/bigdata/tng50/output_files/'
baseUrl = 'https://www.tng-project.org/api/TNG50-1/'
headers = {"api-key":"894f4df036abe0cb9d83561e4b1efcf1"}
basePath = '/rhome/psadh003/bigdata/L35n2160TNG_fixed/output'
fof_path = '/bigdata/saleslab/psadh003/tng50/fof_partdata/'






fof_no = int(sys.argv[1]) #We are taking FoF number as an argument
fof_str = 'fof' + str(fof_no)

ages_df = pd.read_csv(filepath + 'ages_tng.csv', comment = '#')

all_snaps = np.array(ages_df['snapshot'])
all_redshifts = np.array(ages_df['redshift'])
all_ages = np.array(ages_df['age(Gyr)'])

this_fof = il.groupcat.loadSingle(basePath, 99, haloID = fof_no)
central_sfid_99 = this_fof['GroupFirstSub']

def convert_to_float(value):
    try:
        if isinstance(value, float) or isinstance(value, int):
            return value
        blah = ast.literal_eval(value)
        if isinstance(blah, list):
            if len(blah) == 1:
                blah2 = float(blah[0])   
            elif len(blah) == 3:
                blah2 = np.array([float(blah[0]), float(blah[1]), float(blah[2])])
        else:
            blah2 = float(blah)        
        return blah2
    except Exception as e:
        try:
            value_fixed = re.sub(r'(?<=\d)\s+(?=-?\d)', ', ', value)  # Handle spaces between numbers
            value_fixed = re.sub(r'(?<=\d)\s+(?=\d)', ', ', value_fixed)
            value_fixed = re.sub(r'(?<=\.)\s+(?=-\d)', ', ', value_fixed)
            value_fixed = re.sub(r'(?<=\.)\s+(?=\d)', ', ', value_fixed)
            # Now use ast.literal_eval
            # print(value_fixed)
            result = ast.literal_eval(value_fixed)
            return result
        except Exception as e:
            print(value)
            return value

df = pd.read_csv(outpath + fof_str + '_surviving_evolved_everything.csv', delimiter = ',', low_memory=False)  
df = df.applymap(convert_to_float)

df['mpeak_ar'] = ''

# Following is the list of snapshot and subfind ID arrays
snap_if_ar = df['snap_if_ar'].values
sfid_if_ar = df['sfid_if_ar'].values

h = 0.6774
def get_mpeak(ix):
    snap = snap_if_ar[ix]
    sfid = sfid_if_ar[ix]
    sub = il.sublink.loadTree(basePath, snap, sfid, fields = ['Group_M_Crit200', 'SnapNum', 'SubfindID', 'GroupFirstSub'], onlyMPB = True)
    # Mpeak is defined as the mass of the FoF group last time when it was a central
    # We are going to find the last time when it was a central by going through the tree and checking if SubfindID is equal to GroupFirstSub
    # If it is, we are going to return the mass of the group
    for i in range(len(sub['SnapNum'])):
        # print(sub['SnapNum'][i])
        # assert 
        if sub['SubfindID'][i] == sub['GroupFirstSub'][i]:
            return sub['Group_M_Crit200'][i] * 1e10/h
    return None # If it was never a central, we are going to return None

def get_vpeak(ix):
    snap = snap_if_ar[ix]
    sfid = sfid_if_ar[ix]
    sub = il.sublink.loadTree(basePath, snap, sfid, fields = ['SubhaloVmax', 'SnapNum', 'SubfindID', 'GroupFirstSub'], onlyMPB = True)
    subh_vpeak = sub['SubhaloVmax']
    return max(subh_vpeak) # We are going to return the maximum value of Vmax
    # Mpeak is defined as the mass of the FoF group last time when it was a central
    # We are going to find the last time when it was a central by going through the tree and checking if SubfindID is equal to GroupFirstSub
    # If it is, we are going to return the mass of the group
    # for i in range(len(sub['SnapNum'])):
    #     # print(sub['SnapNum'][i])
    #     # assert 
    #     if sub['SubfindID'][i] == sub['GroupFirstSub'][i]:
    #         return sub['Group_Vmax'][i]
    # return None # If it was never a central, we are going to return None

results = Parallel(n_jobs=5, pre_dispatch='1.5*n_jobs')(delayed(get_vpeak)(ix) for ix in tqdm(range(int(len(snap_if_ar)))))

df['vpeak_ar'] = results

df.to_csv(outpath + fof_str + '_surviving_evolved_everything.csv', index = False)