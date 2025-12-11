# This is to repeat the same analysis as that of main.py but now for the complete sample of suhalos that has been input in tng50-dark-analysis directory

'''
This is a second attempt to evolve subhaolos from TNG50hydro 
'''
import numpy as np
import matplotlib.pyplot as plt 
import os 
import sys
os.environ["USE_LZMA"] = "0"
import pandas as pd
sys.path.append(os.path.abspath('/bigdata/saleslab/psadh003/tng50-dark-analysis'))
from errani_plus_tng_subhalo import Subhalo
from tqdm import tqdm
import galpy
import IPython
import illustris_python as il
from matplotlib.backends.backend_pdf import PdfPages
from subhalo_profiles import ExponentialProfile, NFWProfile
import warnings
from populating_stars import *
from joblib import Parallel, delayed #This is to parallelize the code
from hydro_path_for_files import * #This is to get the paths for the files
from constants import * #This is to get the constants


warnings.simplefilter(action='ignore', category=FutureWarning)

suffix = '_Vpeak_newall_isothermal' # this run is for the new power law model

ages_df = pd.read_csv('ages_tng.csv', comment = '#')

all_snaps = np.array(ages_df['snapshot'])
all_redshifts = np.array(ages_df['redshift'])
all_ages = np.array(ages_df['age(Gyr)'])


fof_no = int(sys.argv[1]) # This is the FOF number that comes in as an argument, analysis is done for this fof
fof_str = 'fof' + str(fof_no)

this_fof = il.groupcat.loadSingle(basePath, 99, haloID = fof_no)
central_sfid_99 = this_fof['GroupFirstSub']

# Load the central tree for descendant tracking (needed for Type-2 additional details)
central_mpb = il.sublink.loadTree(basePath, 99, central_sfid_99, fields = ['SubhaloID', 'SnapNum', 'SubfindID'], onlyMPB = True)
central_sfids = central_mpb['SubfindID']
central_snaps = central_mpb['SnapNum']

# Load the full central tree for Type-2 additional details
cft = il.sublink.loadTree(basePath, 99, central_sfid_99, fields = ['SubhaloID', 'DescendantID', 'SnapNum', 'SubfindID', 'SubhaloIDMostbound'])
cft_shid = cft['SubhaloID']
cft_desc = cft['DescendantID']
cft_snap = cft['SnapNum']
cft_sfid = cft['SubfindID']
cft_mbpid = cft['SubhaloIDMostbound']

np.random.seed(42)

# Let us now import the Type-1 subhalos 
t1_df = pd.read_csv(filepath + 'hydrofofno_' + str(fof_no) + '.csv', comment = '#')
t1_snap_if_ar = np.array(t1_df['snap_if_ar'], dtype=int) # This is the infall snap of the subhalo
t1_sfid_if_ar = np.array(t1_df['sfid_if_ar'], dtype=int) # This is the infall sfid of the subhalo


def save_surviving_subhalos(ix):
    vmx_f_ar, rmx_f_ar, mmx_f_ar, mstar_f_ar, rh_f_ar, vd_f_ar = -1 * np.ones(6, dtype = int) # These are the final values of the subhalo
    mstar_f_pl_ar, rh_f_pl_ar, vd_f_pl_ar = -1 * np.ones(3, dtype = int) # These are the final values of the subhalo for the power law model
    mstar_f_co_ar, rh_f_co_ar, vd_f_co_ar = -1 * np.ones(3, dtype = int) # These are the final values of the subhalo for the cutoff model

    subh = Subhalo(snap = t1_snap_if_ar[ix], sfid = t1_sfid_if_ar[ix], last_snap = 99, central_sfid_99 = central_sfid_99)
    
    if np.array(subh.mstar).size * np.array(subh.mstar_pl).size * np.array(subh.mstar_co).size  == 0 :
        return None
    try:
        t = subh.get_orbit(merged = False, when_te = 'last') #after this, the subhalo has rperi, rapo and torb
    except Exception as e:
        print(e) 
        return 

    
    rh_max_ar = subh.Rh  # this is the 2d half light radius
    vd_max_ar = subh.vd  # los vd
    mstar_max_ar = subh.mstar

    rh_max_pl_ar = subh.Rh_pl
    rh_max_co_ar = subh.Rh_co

    mstar_max_co_ar = subh.mstar_co
    mstar_max_pl_ar = subh.mstar_pl

    vmx_if_ar = subh.vmx0
    rmx_if_ar = subh.rmx0
    mmx_if_ar = subh.mmx0
    vpeak_ar = subh.vpeak

    mstar_if_ar = subh.get_mstar(where = int(subh.snap), how = 'total')
    vmax_if_ar = subh.get_vmax(where = int(subh.snap))


    sfid_if_ar = subh.sfid
    snap_if_ar = subh.snap
    rperi_ar = subh.rperi
    rapo_ar = subh.rapo
    torb_ar = subh.torb
    tinf_ar = all_ages[int(subh.snap)]

    

    if subh.resolved == True:
        vmx_f_ar, rmx_f_ar, mmx_f_ar, vd_f_ar, rh_f_ar, mstar_f_ar = subh.get_model_values(float(tinf_ar), t)  # FIXME: Some orbits are not unbound as galpy reports
    else: #If unresolved, we calculate the power law and cutoff model stellar masses. Note that mmx, vmx, rmx would still be the same
        vmx_f_ar, rmx_f_ar, mmx_f_ar, vd_f_pl_ar, rh_f_pl_ar, mstar_f_pl_ar = subh.get_model_values(float(tinf_ar), t, porc = 'p')
        vmx_f_ar, rmx_f_ar, mmx_f_ar, vd_f_co_ar, rh_f_co_ar, mstar_f_co_ar = subh.get_model_values(float(tinf_ar), t, porc = 'c')

    # # FIXME: mstarf would be from tng. change accordingly
    # vmx_f_ar = vmxf
    # rmx_f_ar = rmxf
    # mmx_f_ar = mmxf
    # mstar_f_ar = mstarf
    # rh_f_ar = rhf
    # vd_f_ar = vdf

    with warnings.catch_warnings(record=True) as w:
        vmxf_tng, rmxf_tng, mmxf_tng = subh.get_mx_values(where=int(99))
        if len(w) > 0:
            vmxf_tng, rmxf_tng, mmxf_tng = subh.get_rot_curve(where=int(99))
    # Following are from TNG
    vmx_f_ar_tng = vmxf_tng
    rmx_f_ar_tng = rmxf_tng
    mmx_f_ar_tng = mmxf_tng
    mstar_f_ar_tng = subh.get_mstar(where=99, how='total')
    try:
        rh_f_ar_tng = subh.get_rh(where=99) * 3. / 4
        vd_f_ar_tng = subh.get_vd(where=99)
    except ValueError:
        rh_f_ar_tng = 0
        vd_f_ar_tng = 0

    # if subh.snap < 25: #These would be the cases where the infall was before z = 3 and we only consider these if they survive with mstar > 5e6 Msun at z = 0.
    #     mstar_at_0 = subh.get_mstar(where = 99, how = 'total')
    #     if mstar_at_0 > 5e6:
    #         vmx_f_ar, rmx_f_ar, mmx_f_ar = vmx_f_ar_tng, rmx_f_ar_tng, mmx_f_ar_tng
    #         mstar_f_ar, rh_f_ar, vd_f_ar = mstar_f_ar_tng, rh_f_ar_tng, vd_f_ar_tng
    #         # vmx_f_ar, rmx_f_ar, mmx_f_ar, vd_f_ar, rh_f_ar, mstar_f_ar
    #     else:
    #         return None

    # if len(pos_f_ar) == 0:
    # pos_f_ar = subh.tree['SubhaloPos'][0, :] / h #Maybe we should take pos at 0 which is snap 99?
    # pos_f_ar = pos_f_ar.reshape(1, -1)

    pos_f_ar = subh.get_position_wrt_center(where = 99)
    pos_f_ar = pos_f_ar.reshape(1, -1) #This is the array after beign reshaped

    vel_f_ar = np.array(subh.tree['SubhaloVel'][0, :]) #This is the peculiar velocity of the group
    if fof_no == 0:
        avg_vel =  np.array([-32.431934, -31.424938, -37.76695 ] )
    elif fof_no == 1:
        avg_vel =  np.array([124.3889,   99.56441,  -32.406525] )
    elif fof_no == 2:
        avg_vel =  np.array( [-110.63492,    28.92303,  5.6502633]  )
    else:
        # print('Average velocity not defined for this FoF group')
        raise ValueError('Average velocity not defined for this FoF group')
    vel_f_ar = vel_f_ar - avg_vel
    # vel_f_ar = vel_f_ar.tolist()
    vel_f_ar = vel_f_ar.reshape(1, -1)
    # else:
    #     this_pos = np.array(subh.tree['SubhaloPos'][-1, :] / h)
    #     pos_f_ar = np.append(pos_f_ar, this_pos.reshape(1, -1), axis=0)

    dist_f_ar = subh.get_dist_from_cen(where=99)
    return vmx_if_ar, rmx_if_ar, mmx_if_ar, vmx_f_ar, rmx_f_ar, mmx_f_ar,sfid_if_ar, snap_if_ar, mstar_max_ar, rh_max_ar, vd_max_ar, mstar_f_ar, rh_f_ar, vd_f_ar,  rperi_ar, rapo_ar, torb_ar, tinf_ar, vmx_f_ar_tng, rmx_f_ar_tng, mmx_f_ar_tng, mstar_f_ar_tng, rh_f_ar_tng, vd_f_ar_tng, pos_f_ar.tolist()[0], dist_f_ar, vd_f_pl_ar, rh_f_pl_ar, mstar_f_pl_ar, vd_f_co_ar, rh_f_co_ar, mstar_f_co_ar, rh_max_pl_ar, rh_max_co_ar, mstar_max_pl_ar, mstar_max_co_ar, vel_f_ar.tolist()[0], mstar_if_ar, vmax_if_ar, vpeak_ar






results = Parallel(n_jobs=32, pre_dispatch='1.5*n_jobs')(delayed(save_surviving_subhalos)(ix) for ix in tqdm(range(len(t1_sfid_if_ar))))
results = [value for value in results if value is not None] #Getting rid of all the None entries


column_names = ['vmx_if_ar', 'rmx_if_ar', 'mmx_if_ar',
    'vmx_f_ar', 'rmx_f_ar', 'mmx_f_ar',
    'sfid_if_ar', 'snap_if_ar',
    'mstar_max_ar', 'rh_max_ar', 'vd_max_ar',
    'mstar_f_ar', 'rh_f_ar', 'vd_f_ar',
    'rperi_ar', 'rapo_ar', 'torb_ar', 'tinf_ar',
    'vmx_f_ar_tng', 'rmx_f_ar_tng', 'mmx_f_ar_tng',
    'mstar_f_ar_tng', 'rh_f_ar_tng', 'vd_f_ar_tng',
    'pos_f_ar', 'dist_f_ar', 'vd_f_pl_ar', 'rh_f_pl_ar', 'mstar_f_pl_ar', 
    'vd_f_co_ar', 'rh_f_co_ar', 'mstar_f_co_ar', 
    'rh_max_pl_ar', 'rh_max_co_ar', 'mstar_max_pl_ar', 'mstar_max_co_ar', 'vel_f_ar', 'mstar_if_ar', 'vmax_if_ar', 'vpeak_ar']

# Create an empty DataFrame with the specified column names
df = pd.DataFrame(columns=column_names)
for ix in range(len(results)):
    df.loc[len(df)] = results[ix]

df['fof'] = fof_no

# Add hof_flag for Type-1 subhalos
t1_vpeak_ar = df['vpeak_ar'].values

def type1_additional_details(ix):
    '''
    This function will return the additional details for the Type-1 subhalos. 

    Returns:
    hof_flag
    '''
    # Set seed for reproducibility in parallel processing
    # Using base seed + index ensures each item gets consistent random numbers
    np.random.seed(42 + ix)
    def hof(lvpeak, x0 = 1.29, x1 = 0.05):
        return 0.5 * (1 + np.tanh((lvpeak - x0) / x1)) 
    lvpeak = np.log10(t1_vpeak_ar[ix])
    this_hof = hof(lvpeak)
    hof_flag = np.random.rand() < this_hof
    return hof_flag

results_hof = Parallel(n_jobs=32, pre_dispatch='1.5*n_jobs')(delayed(type1_additional_details)(ix) for ix in tqdm(range(len(df))))
df['hof_flag'] = [value for value in results_hof]

df.to_csv(filepath + fof_str + '_surviving_evolved_everything' + suffix + '.csv', index = False)


# To save on memory, let us delete all the variables that we do not need

del df, results, results_hof, t1_df, t1_sfid_if_ar, t1_snap_if_ar, t1_vpeak_ar

# Let us now import the Type-2 subhalos

df0 = pd.read_csv(filepath + 'hydrofofno_' + str(fof_no) + '_t2.csv', comment = '#')
t2_snap_if_ar = np.array(df0['snap_if_ar'], dtype=int) # This is the infall snap of the subhalo
t2_sfid_if_ar = np.array(df0['sfid_if_ar'], dtype=int) # This is the infall sfid of the subhalo
t2_snap_merger_ar = np.array(df0['snap_merger_ar'], dtype=int) # This is the merger snap of the subhalo

def save_merged_subhalos(ix):
    '''
    This is a function to save return all the parameters of interest for the merged subhalos
    '''
    vmx_f_ar, rmx_f_ar, mmx_f_ar, mstar_f_ar, rh_f_ar, vd_f_ar = -1 * np.ones(6, dtype = int)
    mstar_f_pl_ar, rh_f_pl_ar, vd_f_pl_ar = -1 * np.ones(3, dtype = int)
    mstar_f_co_ar, rh_f_co_ar, vd_f_co_ar = -1 * np.ones(3, dtype = int)

    subh  = Subhalo(snap = int(t2_snap_if_ar[ix]), sfid = int(t2_sfid_if_ar[ix]), last_snap = int(t2_snap_merger_ar[ix]), central_sfid_99=central_sfid_99) #these are at infall
    if np.array(subh.mstar).size * np.array(subh.mstar_pl).size * np.array(subh.mstar_co).size  == 0: #this would be the mass cutoff at infall for the subhalos
        return None
    # t = subh.get_orbit(merged = True, when_te = 'last')
    try:
        t = subh.get_orbit(merged = True, when_te = 'last') #after this, the subhalo has rperi, rapo and torb
    except Exception as e:
        print(e)
        # ctr = ctr + 1
        # skipped_ixs = np.append(skipped_ixs, ix)
        return None

    rh_max_ar = subh.Rh  # this is the 2d half-light radius
    vd_max_ar = subh.vd  # los vd

    vmx_if_ar = subh.vmx0
    rmx_if_ar = subh.rmx0
    mmx_if_ar = subh.mmx0
    vpeak_ar = subh.vpeak

    sfid_if_ar = t2_sfid_if_ar[ix]
    snap_if_ar = t2_snap_if_ar[ix]
    rperi_ar = subh.rperi
    rapo_ar = subh.rapo
    torb_ar = subh.torb
    tinf_ar = all_ages[t2_snap_if_ar[ix]]

    mstar_max_ar = subh.mstar

    mstar_if_ar = subh.get_mstar(where = int(subh.snap), how = 'total')
    vmax_if_ar = subh.get_vmax(where = int(subh.snap))

    rh_max_pl_ar = subh.Rh_pl
    rh_max_co_ar = subh.Rh_co

    mstar_max_co_ar = subh.mstar_co
    mstar_max_pl_ar = subh.mstar_pl


    # if subh.torb == np.inf:
    #     vmxf, rmxf, mmxf, vdf, rhf, mstarf = subh.vmx0, subh.rmx0, subh.mmx0, vd_max_ar[-1], rh_max_ar[-1], subh.mstar
    # else:
    if subh.resolved == True:
        vmx_f_ar, rmx_f_ar, mmx_f_ar, vd_f_ar, rh_f_ar, mstar_f_ar = subh.get_model_values(float(tinf_ar), t)  # FIXME: Some orbits are not unbound as galpy reports
    else: #If unresolved, we calculate the power law and cutoff model stellar masses. Note that mmx, vmx, rmx would still be the same
        vmx_f_ar, rmx_f_ar, mmx_f_ar, vd_f_pl_ar, rh_f_pl_ar, mstar_f_pl_ar = subh.get_model_values(float(tinf_ar), t, porc = 'p')
        vmx_f_ar, rmx_f_ar, mmx_f_ar, vd_f_co_ar, rh_f_co_ar, mstar_f_co_ar = subh.get_model_values(float(tinf_ar), t, porc = 'c')



    mbpid_ar = subh.get_mbpid(where = subh.last_snap) #Get the MBP ID at infall
    mbpidp =  np.array(subh.get_mbpid(where = subh.last_snap-1))
    # print(mbpid_ar[-1], mbpidp)
    # print(len(mbpidp), mbpidp.shape)
    if len(mbpidp) != 0:
        mbpidp_ar = mbpidp
    else:
        mbpidp_ar = -1

    return vmx_if_ar, rmx_if_ar, mmx_if_ar, vmx_f_ar, rmx_f_ar, mmx_f_ar, sfid_if_ar, snap_if_ar, mstar_max_ar, rh_max_ar, vd_max_ar, mstar_f_ar, rh_f_ar, vd_f_ar, rperi_ar, rapo_ar, torb_ar, tinf_ar, mbpid_ar, mbpidp_ar, vd_f_pl_ar, rh_f_pl_ar, mstar_f_pl_ar, vd_f_co_ar, rh_f_co_ar, mstar_f_co_ar, rh_max_pl_ar, rh_max_co_ar, mstar_max_pl_ar, mstar_max_co_ar, mstar_if_ar, vmax_if_ar, vpeak_ar


results = Parallel(n_jobs=32, pre_dispatch='1.5*n_jobs')(delayed(save_merged_subhalos)(ix) for ix in tqdm(range(len(t2_snap_if_ar))))
results = [value for value in results if value is not None] #Getting rid of all the None entries


column_names = ['vmx_if_ar', 'rmx_if_ar', 'mmx_if_ar',
    'vmx_f_ar', 'rmx_f_ar', 'mmx_f_ar',
    'sfid_if_ar', 'snap_if_ar',
    'mstar_max_ar', 'rh_max_ar', 'vd_max_ar',
    'mstar_f_ar', 'rh_f_ar', 'vd_f_ar',
    'rperi_ar', 'rapo_ar', 'torb_ar', 'tinf_ar',
    'mbpid_ar', 'mbpidp_ar',  'vd_f_pl_ar', 'rh_f_pl_ar', 
    'mstar_f_pl_ar', 'vd_f_co_ar', 'rh_f_co_ar', 'mstar_f_co_ar', 
    'rh_max_pl_ar', 'rh_max_co_ar', 'mstar_max_pl_ar', 'mstar_max_co_ar', 'mstar_if_ar', 'vmax_if_ar', 'vpeak_ar']


df = pd.DataFrame(columns=column_names)
for ix in range(len(results)):
    df.loc[len(df)] = results[ix]

df['fof'] = fof_no

# Add additional details for Type-2 subhalos (merger_snap, merger_sfid, posx, posy, posz, desc_flag, desc_flag2, mbpID, hof_flag)
# Use the parent catalog that was already loaded (df0)
par_snap_if_ar = np.array(df0['snap_if_ar'].values, dtype = int)
par_sfid_if_ar = np.array(df0['sfid_if_ar'].values, dtype = int)
par_snap_merger_ar = np.array(df0['snap_merger_ar'].values, dtype = int)
par_sfid_merger_ar = np.array(df0['sfid_merger_ar'].values, dtype = int)
par_posx_ar = np.array(df0['posx_ar'].values)
par_posy_ar = np.array(df0['posy_ar'].values)
par_posz_ar = np.array(df0['posz_ar'].values)

# Get arrays from evolved results dataframe
t2_snap_if_evolved_ar = df['snap_if_ar'].values
t2_sfid_if_evolved_ar = df['sfid_if_ar'].values
t2_vpeak_evolved_ar = df['vpeak_ar'].values

def type2_additional_details(ix):
    '''
    This function will return the additional details for the Type-2 subhalos. 

    Returns:
    merger_snap, merger_sfid, posx, posy, posz, desc_flag, desc_flag2, mbpID, hof_flag
    '''
    # Set seed for reproducibility in parallel processing
    # Using base seed + index ensures each item gets consistent random numbers
    np.random.seed(42 + ix)
    # For each subhalo in the evolved catalog, lets get the subhalo from the parent catalog
    snap_if = t2_snap_if_evolved_ar[ix]
    sfid_if = t2_sfid_if_evolved_ar[ix]

    # look for these in the par_snap_if_ar and par_sfid_if_ar
    ix2 = np.where((par_snap_if_ar == snap_if) & (par_sfid_if_ar == sfid_if))[0][0]
    merger_snap = par_snap_merger_ar[ix2]
    merger_sfid = par_sfid_merger_ar[ix2]

    subh_tree = il.sublink.loadTree(basePath, merger_snap, merger_sfid, fields = ['SubhaloID', 'DescendantID', 'SnapNum', 'SubfindID', 'SubhaloIDMostbound'], onlyMDB = True)
    subh_sfid = subh_tree['SubfindID']
    subh_snap = subh_tree['SnapNum']
    subh_desc = subh_tree['DescendantID']
    subh_shid = subh_tree['SubhaloID']
    if len(subh_snap[subh_snap == merger_snap+1]) > 1 :
        desc_flag2 = 0
    else:
        desc_flag2 = 1

    ix_merged = np.where((cft_snap == merger_snap) & (cft_sfid == merger_sfid))[0][0]
    ix_desc = np.where(cft_shid == cft_desc[ix_merged])[0][0] # This is the index of the descendant in the cft
    desc_sfid = cft_sfid[ix_desc]
    desc_snap = cft_snap[ix_desc]
    if desc_sfid == central_sfids[central_snaps == desc_snap]:
        desc_flag = 1
    else: 
        desc_flag = 0

    # hof_flag calculation
    def hof(lvpeak, x0 = 1.29, x1 = 0.05):
        return 0.5 * (1 + np.tanh((lvpeak - x0) / x1))
    lvpeak = np.log10(t2_vpeak_evolved_ar[ix])
    this_hof = hof(lvpeak)
    hof_flag = np.random.rand() < this_hof

    return merger_snap, merger_sfid, par_posx_ar[ix2], par_posy_ar[ix2], par_posz_ar[ix2], desc_flag, desc_flag2, cft_mbpid[ix_merged] , hof_flag

results_add = Parallel(n_jobs=32, pre_dispatch='1.5*n_jobs')(delayed(type2_additional_details)(ix) for ix in tqdm(range(len(df))))

df['merger_snap'] = [value[0] for value in results_add]
df['merger_sfid'] = [value[1] for value in results_add]
df['posx'] = [value[2] for value in results_add]
df['posy'] = [value[3] for value in results_add]
df['posz'] = [value[4] for value in results_add]
df['desc_flag'] = [value[5] for value in results_add]
df['desc_flag2'] = [value[6] for value in results_add]
df['mbpID'] = [value[7] for value in results_add]
df['hof_flag'] = [value[8] for value in results_add]

df.to_csv(filepath + fof_str + '_merged_evolved_everything' + suffix + '.csv', index = False)





