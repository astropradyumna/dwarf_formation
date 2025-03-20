'''
This is to add the crossing time for satellites which survive in the simulation
'''

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

        if value == '[inf, inf, inf]' or value == '[inf inf inf]':
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
        

df = pd.read_csv(outpath + fof_str + '_surviving_evolved_everything.csv', delimiter = ',', low_memory=False)  
df = df.applymap(convert_to_float)

# We are now going to add an empty column for crossing time, snapshot and the sfid at crossing time
df['t_cross'] = ''
df['snap_cross'] = ''
df['sfid_cross'] = ''
df['pos_cross'] = ''
df['pot_cross'] = ''
df['vel_cross'] = ''

snap_if_ar = df['snap_if_ar'].values
sfid_if_ar = df['sfid_if_ar'].values

h = 0.6774


central_fields = ['GroupFirstSub', 'SubhaloGrNr', 'SnapNum', 'GroupNsubs', 'SubhaloPos', 'Group_R_Crit200', 'Group_M_Crit200', 'SubhaloVel', 'GroupMass']
central_tree = il.sublink.loadTree(basePath, 99, central_sfid_99, fields = central_fields, onlyMPB = True)
central_snaps = central_tree['SnapNum']
central_redshift = all_redshifts[central_snaps]
central_x =  central_tree['SubhaloPos'][:, 0]/(1 + central_redshift)/h
central_y =  central_tree['SubhaloPos'][:, 1]/(1 + central_redshift)/h
central_z =  central_tree['SubhaloPos'][:, 2]/(1 + central_redshift)/h
central_r200 = central_tree['Group_R_Crit200']/(1 + central_redshift)/h #This is the virial radius of the group
# ages_rvir = all_ages[central_snaps] #Ages corresponding to the virial radii
central_grnr = central_tree['SubhaloGrNr']
central_gr_m200 = central_tree['Group_M_Crit200']*1e10/h #This is the M200 of the central group
central_gr_m = central_tree['GroupMass']*1e10/h #This is the total mass of the central group
central_vx = central_tree['SubhaloVel'][:, 0] #km/s
central_vy = central_tree['SubhaloVel'][:, 1]
central_vz = central_tree['SubhaloVel'][:, 2]

central_v0 = np.sqrt(4.3e-6 * central_gr_m200 / central_r200) #this is the isothermal speed of the FoF halo for all snapshots

def get_potential_snapshot(r, snap):
    G = 4.30092e-6 #kpc/Msun * (km/s)^2
    Mvir = central_gr_m200[snap == central_snaps]
    rvir = central_r200[snap == central_snaps]
    # print(Mvir, rvir)
    r3v = 3 * rvir
    M = central_gr_m[snap == central_snaps]
    vc = np.sqrt(G*Mvir / rvir)
    pot = vc**2 * np.log(r/r3v) - G*M/r3v

    # print()
    return pot


def get_crossing_time(ix):
    # Let us now get the sfid of the subhalo at z = 0
    temp_tree = il.sublink.loadTree(basePath, snap_if_ar[ix], sfid_if_ar[ix], fields = ['SnapNum', 'SubfindID'], onlyMDB = True)
    sfid_at_99 = temp_tree['SubfindID'][temp_tree['SnapNum'] == 99]
    # let us now have the full tree
    subh_tree = il.sublink.loadTree(basePath, 99, sfid_at_99[0], fields = ['SnapNum', 'SubfindID', 'SubhaloPos', 'SubhaloVel'], onlyMPB = True)
    subh_snaps = subh_tree['SnapNum']
    snap_len = len(subh_snaps) #This is the length of the snapshot array
    snap_cross = -1
    for ix in range(snap_len):
        '''
        This loop is to go through all the snaps in order to obtain the snap where infall happened
        '''
        
        snap_ix = subh_snaps[snap_len - ix - 1]
        if snap_ix <= 10:
            continue

        subh_pos = subh_tree['SubhaloPos'][snap_ix == subh_snaps][0, :]/(1 + all_redshifts[snap_ix])/h
        cen_pos_this_snap = np.array([central_x[snap_ix == central_snaps], central_y[snap_ix == central_snaps], central_z[snap_ix == central_snaps]])
        subh_dist = np.linalg.norm(subh_pos - cen_pos_this_snap[:,0]) #This is the distance of the subhalo from the center of the FoF group at this snap
        if subh_dist < central_r200[snap_ix == central_snaps]:
            snap_cross = snap_ix
            pos_cross = subh_pos - cen_pos_this_snap[:,0]
            pot_cross = get_potential_snapshot(subh_dist, snap_cross)
            vel_cross = subh_tree['SubhaloVel'][snap_ix == subh_snaps][0, :]
            avg_vel = np.array([central_vx[snap_ix == central_snaps], central_vy[snap_ix == central_snaps], central_vz[snap_ix == central_snaps]])
            # if fof_no == 0: # These values have been taken from add_pos_vel_for_surv.py
            #     avg_vel =  np.array([-32.431934, -31.424938, -37.76695 ] )
            # elif fof_no == 1:
            #     avg_vel =  np.array([124.3889,   99.56441,  -32.406525] )
            # elif fof_no == 2:
            #     avg_vel =  np.array( [-110.63492,    28.92303,  5.6502633]  )
            vel_cross = vel_cross - avg_vel[:,0]
            break
    tcross = all_ages[snap_cross]
    sfid_cross = subh_tree['SubfindID'][snap_cross == subh_tree['SnapNum']]
    if snap_cross != -1:
        return tcross, snap_cross, sfid_cross[0], pos_cross, pot_cross[0], vel_cross
    else:
        return None

results = Parallel(n_jobs=5, pre_dispatch='1.5*n_jobs')(delayed(get_crossing_time)(ix) for ix in tqdm(range(int(len(snap_if_ar)))))


for ix in range(len(results)):
    if results[ix] is None:
        continue
    tcross, snap_cross, sfid_cross, pos_cross, pot_cross, vel_cross = results[ix]
    df['t_cross'][ix] = tcross
    df['snap_cross'][ix] = snap_cross
    df['sfid_cross'][ix] = sfid_cross
    df['pos_cross'][ix] = pos_cross.reshape(1, -1)[0].tolist()
    df['pot_cross'][ix] = pot_cross
    df['vel_cross'][ix] = vel_cross.reshape(1, -1)[0].tolist()
    
df.to_csv(outpath + fof_str + '_surviving_evolved_everything.csv', index = False) 
