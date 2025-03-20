'''
This program is for finding the energy distribution of FOF halos
'''


import numpy as np
import matplotlib.pyplot as plt 
import os 
# os.environ["USE_LZMA"] = "0"
import pandas as pd
# from errani_plus_tng_subhalo import Subhalo
from tqdm import tqdm
# import galpy
import IPython
import illustris_python as il
from matplotlib.backends.backend_pdf import PdfPages
import sys
import ast
from joblib import Parallel, delayed #This is to parallelize the code

# from subhalo_profiles import ExponentialProfile, NFWProfile
# import warnings
# from colossus.cosmology import cosmology
# from colossus.halo import concentration

# cosmology.setCosmology('planck18')

# Suppress the lzma module warning
# warnings.filterwarnings("ignore", category=UserWarning, module="pandas.compat")
# warnings.simplefilter(action='ignore', category=FutureWarning)


# This is currently being used for finding the position of the MBP 
basePath = '/rhome/psadh003/bigdata/L35n2160TNG_fixed/output'
fof_path = '/bigdata/saleslab/psadh003/tng50/fof_partdata/'
outpath  = '/rhome/psadh003/bigdata/tng50/output_files/'

h = 0.6774
mass_dm = 3.07367708626464e-05 * 1e10/h #This is for TNG50-1


# fof_no = int(sys.argv[1])
# fof_no = 288
# fof_no = 981
fof_no = 1092
fof_str = 'fof' + str(fof_no)

this_fof = il.groupcat.loadSingle(basePath, 99, haloID = fof_no)
central_sfid_99 = this_fof['GroupFirstSub']
rvir_this_fof = this_fof['Group_R_Crit200']/h #in kpc

this_fof_path = fof_path + fof_str + '_partdata/'
filepath = '/rhome/psadh003/bigdata/tng50/tng_files/'



star_ids = np.load(this_fof_path+'star_ids.npy')
star_pos = np.load(this_fof_path+'star_pos.npy') #in kpc, wrt center of the fof halo
star_vel = np.load(this_fof_path+'star_vel.npy') #in km/s wrt 1000 central stellar particles of the FoF

dm_ids = np.load(this_fof_path+'dm_ids.npy')
dm_pos = np.load(this_fof_path+'dm_pos.npy')
dm_vel = np.load(this_fof_path+'dm_vel.npy')
dm_masses = np.ones(len(dm_pos[:, 0]))*mass_dm

xar = dm_pos[:, 0]
yar = dm_pos[:, 1]
zar = dm_pos[:, 2]

dist_ar = np.sqrt(xar**2 + yar**2 + zar**2)

vxar = dm_vel[:, 0]
vyar = dm_vel[:, 1]
vzar = dm_vel[:, 2]

# cut_ixs = (dist_ar < 2*rvir_this_fof) #Planning to only look at particles within 2*rvir_this_fof
cut_ixs = (dist_ar < np.inf) #Planning to only look at all particles 
xar = xar[cut_ixs]
yar = yar[cut_ixs]
zar = zar[cut_ixs]
vxar = vxar[cut_ixs]
vyar = vyar[cut_ixs]
vzar = vzar[cut_ixs]


def get_potential_energy(x, y, z):
    '''
    This returns the potential energy of the star at the given position in (km/s)^2
    '''
    G1 = 4.30092e-6 #kpc/Msun * (km/s)^2
    # G1 = 1 #This is for this program only

    # pe_dm = mass_dm * np.sum(1/np.sqrt((x - dm_xcoord)**2 + (y - dm_ycoord)**2 + (z - dm_zcoord)**2 ))
    pe_dm = 0 #It should be noted that this is for stars
    for ix in range(len(xar)):
        if xar[ix] == x and yar[ix] == y and zar[ix] == z:
            continue
        pe_dm = pe_dm + mass_dm/np.sqrt((x - xar[ix])**2 + (y - yar[ix])**2 + (z - zar[ix])**2)
    pe = -G1 * pe_dm
    return pe #this should be in (km/s)^2

def get_total_energy(ix):
    G1 = 4.30092e-6
    ke = 0.5 * (vxar[ix]**2 + vyar[ix]**2 + vzar[ix]**2)  #This is in (km/s)^2
    pe = get_potential_energy(xar[ix], yar[ix], zar[ix])
    # te = pe + ke
    return ke, pe


results = Parallel(n_jobs=32, pre_dispatch='1.5*n_jobs')(delayed(get_total_energy)(ix) for ix in tqdm(range(len(xar)))) #CHANGE: This is only for testing!
ke_ar = np.zeros(0)
pe_ar = np.zeros(0)
for ix in range(len(results)):
    ke_ar = np.append(ke_ar, results[ix][0])
    pe_ar = np.append(pe_ar, results[ix][1])

np.save(filepath + 'energy_files/inside_ke_'+fof_str+'.npy', ke_ar)
np.save(filepath + 'energy_files/inside_pe_'+fof_str+'.npy', pe_ar)
