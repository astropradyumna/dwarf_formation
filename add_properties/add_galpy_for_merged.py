import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
import illustris_python as il
import ast
from tqdm import tqdm
import sys
import os
from astropy import units as u
from errani_plus_tng_subhalo import Subhalo


filepath = '/rhome/psadh003/bigdata/tng50/tng_files/'
outpath  = '/rhome/psadh003/bigdata/tng50/output_files/'
baseUrl = 'https://www.tng-project.org/api/TNG50-1/'
headers = {"api-key":"894f4df036abe0cb9d83561e4b1efcf1"}
basePath = '/rhome/psadh003/bigdata/L35n2160TNG_fixed/output'
filepath = '/rhome/psadh003/bigdata/tng50/tng_files/'

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
    except (ValueError, SyntaxError):
        # print(f"Error converting {value}") #Looks like only inf values are not being converted, which is good
        return value 

fof_no = int(sys.argv[1])
fof_str = 'fof' + str(fof_no)

this_fof = il.groupcat.loadSingle(basePath, 99, haloID = fof_no)
central_sfid_99 = this_fof['GroupFirstSub']


df = pd.read_csv(outpath + fof_str +'_merged_evolved_wmbp_everything.csv', delimiter = ',', low_memory=False)  
df = df.applymap(convert_to_float)
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
df['gp_pos_f_ar'] = ''
df['gp_dist_f_ar'] = ''
df['gp_vel_f_ar'] = ''


#Here is what I would like to do --  for each subhalos in the list, we are going to look at the last snapshot from the df2 and then find position from galpy by orbit integration
def get_gp_position(ix):
    #Let us look where snap_if_ar[ix] and sfid_if_ar[ix] are in  msh_sfid and msh_snap and then get the last snapshot from df2
    last_snap = msh_merger_snap[np.where((msh_sfid == sfid_if_ar[ix]) & (msh_snap == snap_if_ar[ix]))[0]]
    subh = Subhalo(sfid = sfid_if_ar[ix], snap = snap_if_ar[ix], last_snap = int(last_snap), central_sfid_99 = central_sfid_99) #FIXME: Please update the last_snap variable after you are done. This wil onl hold for ix = 0
    return subh.get_gp_position(merged = True) # We will have to make a new function in the Subhalo class to get the galpy position

results = Parallel(n_jobs=32, pre_dispatch='1.5*n_jobs')(delayed(get_gp_position)(ix) for ix in tqdm(range(len(snap_if_ar))))

popix_ar = np.zeros(0)
pos_ar = np.zeros(0)
vel_ar = np.zeros(0)

for ix in range(len(results)):
    # if ix > 10:
    #     break
    if results[ix] is None:
        popix_ar = np.append(popix_ar, ix)
        
    else:
        # print( np.array(results[ix].reshape(1, -1)[0]))
        # print( results[ix].reshape(1, -1)[0] )
        posavg, velavg = results[ix]
        posavg = np.array(posavg)
        velavg = np.array(velavg)
        df['gp_pos_f_ar'][ix] = posavg.reshape(1, -1)[0].tolist()
        df['gp_vel_f_ar'][ix] = velavg.reshape(1, -1)[0].tolist()
        df['gp_dist_f_ar'][ix] = np.linalg.norm(posavg.reshape(1, -1))
        if len(pos_ar) == 0:
            pos_ar = posavg.reshape(1, -1)
            vel_ar = velavg.reshape(1, -1)
        else:
            pos_ar = np.append(pos_ar, posavg.reshape(1, -1), axis = 0)
            vel_ar = np.append(vel_ar, velavg.reshape(1, -1), axis = 0)


print(f'Number of subhalos being lost are {len(popix_ar)} out of {len(snap_if_ar)}')
df.to_csv(outpath + fof_str + '_merged_evolved_wmbp_everything.csv', index = False) 
                                                         
