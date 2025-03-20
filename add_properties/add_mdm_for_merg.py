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


def convert_to_float(value):
    try:
        if isinstance(value, float) or isinstance(value, int):
            return value

        if value == '[inf, inf, inf]':
            return np.array([np.inf, np.inf, np.inf])

        if value == '[inf]':
            return np.inf

        if value == '[-inf]':
            return -np.inf

        

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

fof_no = int(sys.argv[1]) #We are taking FoF number as an argument
fof_str = 'fof' + str(fof_no)

ages_df = pd.read_csv(filepath + 'ages_tng.csv', comment = '#')

all_snaps = np.array(ages_df['snapshot'])
all_redshifts = np.array(ages_df['redshift'])
all_ages = np.array(ages_df['age(Gyr)'])

this_fof = il.groupcat.loadSingle(basePath, 99, haloID = fof_no)
central_sfid_99 = this_fof['GroupFirstSub']


df = pd.read_csv(outpath + fof_str +'_merged_evolved_wmbp_everything.csv', delimiter = ',', low_memory=False)  
df = df.applymap(convert_to_float)

snap_if_ar = df['snap_if_ar'].values
sfid_if_ar = df['sfid_if_ar'].values


df2 = pd.read_csv(filepath + fof_str + '_sh_merged_after_z3_tng50_1_everything.csv') #This is the ones before evolution since we need the last snapshot as well.
msh_sfid = df2['SubfindID']
msh_sfid = np.array([s.strip('[]') for s in msh_sfid], dtype = int) #snap ID at infall
msh_snap = np.array(df2['SnapNum'], dtype = int) #Snap at infall
msh_ift = all_ages[msh_snap]


msh_sfid1 = df2['inf1_subid']
msh_sfid1 = np.array([s.strip('[]') for s in msh_sfid1], dtype = int)
msh_snap1 = df2['inf1_snap']
msh_tinf1 = all_ages[msh_snap1] 


msh_merger_snap = np.array(df2['MergerSnapNum'], dtype = int) #SnapNum at the last snapshot of survival
msh_merger_sfid = np.array(df2['MergerSubfindID'], dtype = int) #this is the subfind ID at the last snapshot of survival
msh_mt = all_ages[msh_merger_snap] #The time of merger

df['mdm_ar'] = ''
h = 0.6774
def get_mpeak(ix):
    snap = snap_if_ar[ix]
    sfid = sfid_if_ar[ix]
    sub = il.sublink.loadTree(basePath, snap, sfid, fields = ['Group_M_Crit200', 'SnapNum', 'SubfindID', 'GroupFirstSub'], onlyMPB = True)
    # Mpeak is defined as the mass of the FoF group last time when it was a central
    print(sub['SnapNum'])
    for i in range(len(sub['SnapNum'])):
        if sub['SubfindID'][i] == sub['GroupFirstSub'][i]:
            return sub['Group_M_Crit200'][i]*1e10/h
    return None

mpeak_ar = Parallel(n_jobs=5, pre_dispatch='1.5*n_jobs')(delayed(get_mpeak)(ix) for ix in tqdm(range(int(len(snap_if_ar)))))

df['mpeak_ar'] = mpeak_ar

df.to_csv(outpath + fof_str + '_merged_evolved_wmbp_everything.csv', index = False)