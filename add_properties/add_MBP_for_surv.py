# This code is to run through all the list of surviving subhalos and append the position from the MBP at infall for the subhalos
#This is the assumption that we are using the combined file of all the three FoF groups

import numpy as np
import pandas as pd
from errani_plus_tng_subhalo import Subhalo
from tqdm import tqdm
import illustris_python as il
from joblib import Parallel, delayed #This is to parallelize the code
import sys
import ast



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



# FoF data input =======================================
fof_no = int(sys.argv[1]) #We are taking FoF number as an argument
fof_str = 'fof' + str(fof_no)

this_fof = il.groupcat.loadSingle(basePath, 99, haloID = fof_no)
central_sfid_99 = this_fof['GroupFirstSub']

this_fof_path = fof_path + fof_str + '_partdata/'

star_ids = np.load(this_fof_path+'star_ids.npy')
star_pos = np.load(this_fof_path+'star_pos.npy') #in kpc, wrt center of the fof halo
star_vel = np.load(this_fof_path+'star_vel.npy') #in km/s wrt 1000 central stellar particles of the FoF

dm_ids = np.load(this_fof_path+'dm_ids.npy')
dm_pos = np.load(this_fof_path+'dm_pos.npy')
dm_vel = np.load(this_fof_path+'dm_vel.npy')
# =======================================


df = pd.read_csv(outpath + fof_str + '_surviving_evolved_everything.csv', delimiter = ',', low_memory=False)  
# df = df.applymap(convert_to_float)

#Starting two empty columns, in an effort to have everything in one file
df['mbp_pos_f_ar'] = '' #This is MBP positions
df['mbp_vel_f_ar'] = '' #This is MBP velocities
df['mbp_dist_f_ar'] = '' 

# Following is the list of snapshot and subfind ID arrays
snap_if_ar = df['snap_if_ar'].values
sfid_if_ar = df['sfid_if_ar'].values


pos_ar = np.zeros(0) #These are the positions of the MBP at z = 0
vel_ar = np.zeros(0) #These are the velocities of the MBP at z = 0

popix_ar = np.zeros(0) #list of indices for which we cannot find the MBP in FoF at z = 0


def get_positions(ix):
    '''
    This function is for parallelizing the process of finding the positions of the subhalos
    '''
    subh = Subhalo(sfid = sfid_if_ar[ix], snap = snap_if_ar[ix], last_snap = 99, central_sfid_99 = central_sfid_99)
    mbpid =  subh.get_mbpid(where = int(snap_if_ar[ix]))[0] #This will be the MBP ID of the subhalo at the infall snapshot

    pos = [None]
    vel = [None]
    pos2 = [None]
    vel2 = [None]
    index = np.where(np.isin(star_ids, mbpid))[0]
    # print(index)
    if len(index) == 1: 
        pos = star_pos[index][0]
        vel = star_vel[index][0]
    if len(index) == 0:
        index = np.where(np.isin(dm_ids, mbpid))[0]
        if len(index) == 1: 
            pos = dm_pos[index][0]
            vel = dm_vel[index][0]

    # index2 = np.where(np.isin(star_ids, mbpidp_ar[ix]))[0]
    # # print(index2)
    # if len(index2) == 1: 
    #     pos2 = star_pos[index2][0]
    #     vel2 = star_vel[index2][0]
    # if len(index2) == 0:
    #     index2 = np.where(np.isin(dm_ids, mbpidp_ar[ix]))[0]
    #     if len(index2) == 1: 
    #         pos2 = dm_pos[index2][0]
    #         vel2 = dm_vel[index2][0]
    
    # print(pos, pos2)
    # posavg = pos + pos2 #This is the average position of the subhalo

    # In the case of having a position for MBP ID of the merger snapshot and the previous snapshot, 
        # the position would be the average of both positions, else, it is only one of these positions. 
        # It should either be a stellar particle or a DM particle
    posavg = []
    velavg = []
    if len(pos) == 3 and len(pos2) == 3:
        posavg = np.array(pos + pos2)/2.
        velavg = np.array(vel + vel2)/2.
    elif len(pos) ==3 and len(pos2) == 1:
        posavg = np.array(pos)
        velavg = np.array(vel)
    elif len(pos2) == 3 and len(pos) == 1:
        posavg = np.array(pos2)
        velavg = np.array(vel2)
    elif len(pos2) == 1 and len(pos) == 1:
        return None

    if len(posavg) == 3:
        return posavg, velavg
        # if len(pos_ar) == 0:
        #     pos_ar = posavg.reshape(1, -1)
        # else:
        #     return posavg
            # pos_ar = np.append(pos_ar, posavg.reshape(1, -1), axis = 0)
    else:
        # popix_ar = np.append(popix_ar, ix) #FIXME: #12 Some of the particles are not in the FoF0 particle file
        return None


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
        df['mbp_pos_f_ar'][ix] = posavg.reshape(1, -1)[0].tolist()
        df['mbp_vel_f_ar'][ix] = velavg.reshape(1, -1)[0].tolist()
        df['mbp_dist_f_ar'][ix] = np.linalg.norm(posavg.reshape(1, -1))
        if len(pos_ar) == 0:
            pos_ar = posavg.reshape(1, -1)
            vel_ar = velavg.reshape(1, -1)
        else:
            pos_ar = np.append(pos_ar, posavg.reshape(1, -1), axis = 0)
            vel_ar = np.append(vel_ar, velavg.reshape(1, -1), axis = 0)


print(f'Number of subhalos being lost are {len(popix_ar)} out of {len(snap_if_ar)}')
df.to_csv(outpath + fof_str + '_surviving_evolved_everything.csv', index = False) 
