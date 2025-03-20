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
#Starting two empty columns, in an effort to have everything in one
#Starting two empty columns, in an effort to have everything in one file
df['pot_if_ar'] = '' 
df['pot_f_ar'] = '' 
# df['dist_f_ar'] = '' 

# Following is the list of snapshot and subfind ID arrays
snap_if_ar = df['snap_if_ar'].values
sfid_if_ar = df['sfid_if_ar'].values
pspos_f_ar = np.stack(np.array(df['pos_f_ar'].values))
psvel_f_ar = np.stack(np.array(df['vel_f_ar'].values))
psdist_f_ar = np.array(df['dist_f_ar'].values)

pspos_if_ar = np.stack(np.array(df['pos_if_ar'].values))
psvel_if_ar = np.stack(np.array(df['vel_if_ar'].values))
psdist_if_ar = np.linalg.norm(pspos_if_ar, axis = 1)


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


def get_potential(ix):
    snap = snap_if_ar[ix]
    pot_f = get_potential_snapshot(psdist_f_ar[ix], 99)
    pot_if = get_potential_snapshot(psdist_if_ar[ix], snap)
    return pot_if, pot_f


results = Parallel(n_jobs=5, pre_dispatch='1.5*n_jobs')(delayed(get_potential)(ix) for ix in tqdm(range(int(len(snap_if_ar)))))
#The "results" will have positions and velocities of the MBP at z = 0 for the subhalos that survived

for ix in range(len(results)):
    # if ix > 10:
    #     break
    if results[ix] is None:
        popix_ar = np.append(popix_ar, ix)
        
    else:
        # print( np.array(results[ix].reshape(1, -1)[0]))
        # print( results[ix].reshape(1, -1)[0] )
        pot_if, pot_f = results[ix]
        df['pot_if_ar'][ix] = pot_if
        df['pot_f_ar'][ix] = pot_f

df.to_csv(outpath + fof_str + '_surviving_evolved_everything.csv', index = False)
                                                        