# This code is to run through all the list of surviving subhalos and append the position from the MBP at infall for the subhalos
#This is the assumption that we are using the combined file of all the three FoF groups

import sys
import ast
import os
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

this_fof = il.groupcat.loadSingle(basePath, 99, haloID = fof_no)
central_sfid_99 = this_fof['GroupFirstSub']

df = pd.read_csv(outpath + fof_str +'_merged_evolved_wmbp_everything.csv', delimiter = ',', low_memory=False)  
# df = df.applymap(convert_to_float)
snap_if_ar = df['snap_if_ar'].values
sfid_if_ar = df['sfid_if_ar'].values


ages_df = pd.read_csv(filepath + 'ages_tng.csv', comment = '#')

all_snaps = np.array(ages_df['snapshot'])
all_redshifts = np.array(ages_df['redshift'])
all_ages = np.array(ages_df['age(Gyr)'])


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

#Starting two empty columns, in an effort to have everything in one file
df['pos_if_ar'] = '' 
df['vel_if_ar'] = '' 
# df['dist_f_ar'] = '' 

# Following is the list of snapshot and subfind ID arrays
snap_if_ar = df['snap_if_ar'].values
sfid_if_ar = df['sfid_if_ar'].values

pos_ar = np.zeros(0) #These are the positions of the MBP at z = 0
vel_ar = np.zeros(0) #These are the velocities of the MBP at z = 0
popix_ar = np.zeros(0) #list of indices for which we cannot find the MBP in FoF at z = 0


def get_positions(ix):
    last_snap = msh_merger_snap[np.where((msh_sfid == sfid_if_ar[ix]) & (msh_snap == snap_if_ar[ix]))[0]]
    subh = Subhalo(sfid = sfid_if_ar[ix], snap = snap_if_ar[ix], last_snap = int(last_snap), central_sfid_99 = central_sfid_99)
    # cen = Subhalo(sfid = central_sfid_99, snap = 99, last_snap = 99, central_sfid_99 = central_sfid_99) #this is the central subhalo at z = 0
    pos = subh.get_position_wrt_center(where = int(snap_if_ar[ix]))
    pos = pos.reshape(1, -1)
    # vel_f_ar = np.array(subh.tree['SubhaloVel'][0, :]) #This is the peculiar velocity of the group
    # avg_vel = np.
    vel_f_ar = subh.get_velocity_wrt_center(where = int(snap_if_ar[ix]))
    # vel_f_ar = vel_f_ar.tolist()
    vel_f_ar = vel_f_ar.reshape(1, -1)

    return pos, vel_f_ar

results = Parallel(n_jobs=5, pre_dispatch='1.5*n_jobs')(delayed(get_positions)(ix) for ix in tqdm(range(int(len(snap_if_ar)))))
#The "results" will have positions and velocities of the MBP at z = 0 for the subhalos that survived

for ix in range(len(results)):
    # if ix > 10:
    #     break
    if results[ix] is None:
        popix_ar = np.append(popix_ar, ix)
        
    else:
        # print( np.array(results[ix].reshape(1, -1)[0]))
        # print( results[ix].reshape(1, -1)[0] )
        posavg, velavg = results[ix]
        df['pos_if_ar'][ix] = posavg.reshape(1, -1)[0].tolist()
        df['vel_if_ar'][ix] = velavg.reshape(1, -1)[0].tolist()
        # df['dist_f_ar'][ix] = np.linalg.norm(posavg.reshape(1, -1))
        if len(pos_ar) == 0:
            pos_ar = posavg.reshape(1, -1)
            vel_ar = velavg.reshape(1, -1)
        else:
            pos_ar = np.append(pos_ar, posavg.reshape(1, -1), axis = 0)
            vel_ar = np.append(vel_ar, velavg.reshape(1, -1), axis = 0)


print(f'Number of subhalos being lost are {len(popix_ar)} out of {len(snap_if_ar)}')
df.to_csv(outpath + fof_str + '_merged_evolved_wmbp_everything.csv', index = False) 
