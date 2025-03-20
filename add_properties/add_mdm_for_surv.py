'''
This code is to get the dark matter mass of the subhalos at z = 0.
'''


import sys
import ast
import os
import re
sys.path.append(os.path.abspath('..'))

import numpy as np
import pandas as pd
# from errani_plus_tng_subhalo import Subhalo
from tng_subhalo_and_halo import TNG_Subhalo
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


df = pd.read_csv(outpath + fof_str + '_surviving_evolved_everything.csv', delimiter = ',', low_memory=False)  

df['mdm_f_ar'] = '' 

snap_if_ar = df['snap_if_ar'].values
sfid_if_ar = df['sfid_if_ar'].values

mdm_ar = np.zeros(0) #These are the dark matter masses of the subhalos at z = 0

def get_mdm_tot(ix):
    subh = TNG_Subhalo(sfid = int(sfid_if_ar[ix]), snap = int(snap_if_ar[ix]), last_snap = 99)
    mdm = subh.get_mdm(where = int(99))
    return mdm

results = Parallel(n_jobs=5, pre_dispatch='1.5*n_jobs')(delayed(get_mdm_tot)(ix) for ix in tqdm(range(int(len(snap_if_ar)))))

df['mdm_f_ar'] = np.array(results)

df.to_csv(outpath + fof_str + '_surviving_evolved_everything.csv', index = False)