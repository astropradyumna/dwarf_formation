'''
This program is currently being used to ind the position of the MBP for the input subhalo file of merged subhalos
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


fof_no = int(sys.argv[1])
fof_str = 'fof' + str(fof_no)

this_fof = il.groupcat.loadSingle(basePath, 99, haloID = fof_no)
central_sfid_99 = this_fof['GroupFirstSub']

this_fof_path = fof_path + fof_str + '_partdata/'


def convert_to_float(value):
    try:
        blah = ast.literal_eval(value)
        if isinstance(blah, list):
            blah2 = float(blah[0])     
        else:
            blah2 = float(blah)        
        return blah2
    except (ValueError, SyntaxError):
        return value 



# df = pd.read_csv(outpath + 'merged_evolved_fof0_everything.csv', delimiter = ',')
df = pd.read_csv(outpath + fof_str +'_merged_evolved_everything.csv', delimiter = ',')  
df = df.applymap(convert_to_float)

mbpid_ar = np.array(df['mbpid_ar'])
mbpidp_ar = np.array(df['mbpidp_ar']) #MBP ID of one snapshot before 

star_ids = np.load(this_fof_path+'star_ids.npy')
star_pos = np.load(this_fof_path+'star_pos.npy')

dm_ids = np.load(this_fof_path+'dm_ids.npy')
dm_pos = np.load(this_fof_path+'dm_pos.npy')

pos_ar = np.zeros(0)

popix_ar = np.zeros(0)

# print(star_pos[0])
# print(dm_pos[0])
# print(star_ids[0])
# print(dm_ids[0])
# IPython.embed()
# print(len(star_ids), type(star_ids))
# print(len(mbpid_ar), type(mbpid_ar))
# print(star_ids)
# print(mbpid_ar)
# print(mbpid_ar[0] in dm_ids)

def get_positions(ix):
    '''
    This function is for parallelizing the process of finding the positions of the subhalos
    '''
    pos = [None]
    pos2 = [None]
    index = np.where(np.isin(star_ids, mbpid_ar[ix]))[0]
    # print(index)
    if len(index) == 1: pos = star_pos[index][0]
    if len(index) == 0:
        index = np.where(np.isin(dm_ids, mbpid_ar[ix]))[0]
        if len(index) == 1: pos = dm_pos[index][0]

    index2 = np.where(np.isin(star_ids, mbpidp_ar[ix]))[0]
    # print(index2)
    if len(index2) == 1: pos2 = star_pos[index2][0]
    if len(index2) == 0:
        index2 = np.where(np.isin(dm_ids, mbpidp_ar[ix]))[0]
        if len(index2) == 1: pos2 = dm_pos[index2][0]
    
    # print(pos, pos2)
    # posavg = pos + pos2 #This is the average position of the subhalo

    # In the case of having a position for MBP ID of the merger snapshot and the previous snapshot, 
        # the position would be the average of both positions, else, it is only one of these positions. 
        # It should either be a stellar particle or a DM particle
    posavg = []
    if len(pos) == 3 and len(pos2) == 3:
        posavg = np.array(pos + pos2)/2.
    elif len(pos) ==3 and len(pos2) == 0:
        posavg = np.array(pos)
    elif len(pos2) == 3 and len(pos) == 0:
        posavg = np.array(pos2)

    if len(posavg) == 3:
        return posavg
        # if len(pos_ar) == 0:
        #     pos_ar = posavg.reshape(1, -1)
        # else:
        #     return posavg
            # pos_ar = np.append(pos_ar, posavg.reshape(1, -1), axis = 0)
    else:
        # popix_ar = np.append(popix_ar, ix) #FIXME: #12 Some of the particles are not in the FoF0 particle file
        return None


results = Parallel(n_jobs=32, pre_dispatch='1.5*n_jobs')(delayed(get_positions)(ix) for ix in tqdm(range(len(mbpid_ar))))
# len_before = len(results)
for ix in range(len(results)):
    if results[ix] is None:
        popix_ar = np.append(popix_ar, ix)
    else:
        if len(pos_ar) == 0:
            pos_ar = results[ix].reshape(1, -1)
        else:
            pos_ar = np.append(pos_ar, results[ix].reshape(1, -1), axis = 0)
        # pos_ar = np.append(pos_ar, results[ix].reshape(1, -1), axis = 0)
        
# none_indices = [ix for ix, value in enumerate(results) if value is None]
# results = [value for value in results if value is not None] #Getting rid of all the None entries
# len_after = len(results)
print(f'Number of subhalos being lost are {len(popix_ar)} out of {len(mbpid_ar)}')


df = df.drop(popix_ar)
df['pos_f_ar'] = pos_ar.tolist()
# print(pos_ar)
df['dist_f_ar'] = np.sqrt(np.sum(pos_ar**2, axis=1))



df.to_csv(outpath + fof_str + '_merged_evolved_wmbp_everything.csv', index = False) 



# for (ix, id) in tqdm(enumerate(mbpid_ar)):
#     pos = [None]
#     pos2 = [None]
#     index = np.where(np.isin(star_ids, mbpid_ar[ix]))[0]
#     # print(index)
#     if len(index) == 1: pos = star_pos[index][0]
#     if len(index) == 0:
#         index = np.where(np.isin(dm_ids, mbpid_ar[ix]))[0]
#         if len(index) == 1: pos = dm_pos[index][0]

#     index2 = np.where(np.isin(star_ids, mbpidp_ar[ix]))[0]
#     # print(index2)
#     if len(index2) == 1: pos2 = star_pos[index2][0]
#     if len(index2) == 0:
#         index2 = np.where(np.isin(dm_ids, mbpidp_ar[ix]))[0]
#         if len(index2) == 1: pos2 = dm_pos[index2][0]
    
#     # print(pos, pos2)
#     # posavg = pos + pos2 #This is the average position of the subhalo

#     # In the case of having a position for MBP ID of the merger snapshot and the previous snapshot, 
#         # the position would be the average of both positions, else, it is only one of these positions. 
#         # It should either be a stellar particle or a DM particle
#     posavg = []
#     if len(pos) == 3 and len(pos2) == 3:
#         posavg = np.array(pos + pos2)/2.
#     elif len(pos) ==3 and len(pos2) == 0:
#         posavg = np.array(pos)
#     elif len(pos2) == 3 and len(pos) == 0:
#         posavg = np.array(pos2)

#     if len(posavg) == 3:
#         if len(pos_ar) == 0:
#             pos_ar = posavg.reshape(1, -1)
#         else:
#             pos_ar = np.append(pos_ar, posavg.reshape(1, -1), axis = 0)
#     else:
#         popix_ar = np.append(popix_ar, ix) #FIXME: #12 Some of the particles are not in the FoF0 particle file


# print(f'Number of subhalos being lost are {len(popix_ar)} out of {len(mbpid_ar)}')


# df.to_csv(outpath + 'merged_evolved_fof0_everything_wmbp.csv', index = False)
#=========================
# dm_dist = np.sqrt(np.sum(dm_pos**2, axis=1))

# rpl = np.logspace(1, 3.2, 100)
# Ndm_ar = np.zeros(0) #shmf for merged subhalos
# for (ix, ms) in enumerate(rpl):
#     Ndm_ar = np.append(Ndm_ar, len(dm_dist[dm_dist < ms]))




# =================================
# '''
# file import
# '''
# filepath = '/home/psadh003/tng50/tng_files/'
# baseUrl = 'https://www.tng-project.org/api/TNG50-1/'
# headers = {"api-key":"894f4df036abe0cb9d83561e4b1efcf1"}
# basePath = '/mainvol/jdopp001/L35n2160TNG_fixed/output'

# h = 0.6744

# ages_df = pd.read_csv(filepath + 'ages_tng.csv', comment = '#')

# all_snaps = np.array(ages_df['snapshot'])
# all_redshifts = np.array(ages_df['redshift'])
# all_ages = np.array(ages_df['age(Gyr)'])

# '''
# Following is the dataset of the entire list of subhalos which infalled after z = 3 and survived
# '''
# survived_df = pd.read_csv(filepath + 'sh_survived_after_z3_tng50_1.csv')

# ssh_sfid = survived_df['SubfindID']
# ssh_sfid = np.array([s.strip('[]') for s in ssh_sfid], dtype = int)
# ssh_snap = np.array(survived_df['SnapNum'], dtype = int)
# ssh_ift = all_ages[ssh_snap]


# ssh_sfid1 = survived_df['inf1_subid']
# ssh_sfid1 = np.array([s.strip('[]') for s in ssh_sfid1], dtype = int)
# ssh_snap1 = np.array(survived_df['inf1_snap'], dtype = int)
# ssh_tinf1 = all_ages[ssh_snap1] #This is the infall time 1


# ssh_mstar = survived_df['Mstar']
# ssh_mstar = [s.strip('[]') for s in ssh_mstar]
# ssh_mstar = np.array(ssh_mstar, dtype = float)
# ssh_max_mstar = np.array(survived_df['max_Mstar'], dtype = float)
# ssh_max_mstar_snap = np.array(survived_df['max_Mstar_snap'], dtype = int)
# ssh_max_mstar_id = np.array(survived_df['max_Mstar_id'], dtype = int)



# '''
# Following is the dataset of the entire list of subhalos which infalled after z = 3 and merged
# '''
# merged_df = pd.read_csv(filepath + 'sh_merged_after_z3_tng50_1.csv')

# msh_sfid = merged_df['SubfindID']
# msh_sfid = np.array([s.strip('[]') for s in msh_sfid], dtype = int) #snap ID at infall
# msh_snap = np.array(merged_df['SnapNum'], dtype = int) #Snap at infall
# msh_ift = all_ages[msh_snap]

# msh_sfid1 = merged_df['inf1_subid']
# msh_sfid1 = np.array([s.strip('[]') for s in msh_sfid1], dtype = int)
# msh_snap1 = merged_df['inf1_snap']
# msh_tinf1 = all_ages[msh_snap1] 


# msh_merger_snap = np.array(merged_df['MergerSnapNum'], dtype = int) #SnapNum at the last snapshot of survival
# msh_merger_sfid = np.array(merged_df['MergerSubfindID'], dtype = int) #this is the subfind ID at the last snapshot of survival
# msh_mt = all_ages[msh_merger_snap] #The time of merger
# # print(msh_merger_snap)
# msh_mstar = merged_df['Mstar']
# msh_mstar = [s.strip('[]') for s in msh_mstar]
# msh_mstar = np.array(msh_mstar, dtype = float)
# msh_max_mstar = np.array(merged_df['max_Mstar'], dtype = float)
# msh_max_mstar_snap = np.array(merged_df['max_Mstar_snap'], dtype = int)


# '''
# Following is to get the dataset which has the relevant SubfindIDs
# '''

# # id_df = pd.read_csv('errani_checking_dataset.csv', comment = '#')
# id_df = pd.read_csv(filepath + 'errani_checking_dataset_more1e9msun.csv', comment = '#')

# snap_if_ar = id_df['snap_at_infall']
# sfid_if_ar = id_df['id_at_infall']
# ms_by_mdm = id_df['ms_by_mdm']





# '''
# =============================================================
# Data import ends here
# =============================================================
# '''

# IPython.embed()




# '''
# Following loop is for the subhalos that survive
# '''

# fig, ax = plt.subplots(figsize = (5, 5))


# for ix in tqdm(range(len(snap_if_ar))):
#     '''
#     This is to loop over all the surviving subhalos of big dataset with subhalos in between 1e8.5 and 1e9.5 Msun
#     ''' 
#     subh  = Subhalo(snap = snap_if_ar[ix], sfid = sfid_if_ar[ix])
#     # subh.get_infall_properties()
#     subh.get_star_energy_dist(where = int(snap_if_ar[ix]), plot = True)
#     # plt.xlim(-2, -0.001)
#     plt.show()



# '''
# Following is to just see the mmx values for different NFW virial masses and concentrations
# '''
# Mvir_ar = 10 ** np.arange(5, 12)

# for (ix, mvir) in enumerate(Mvir_ar):
#     subh = NFWProfile(Mvir = mvir, cvir = concentration.concentration(0.6744 * mvir, 'vir', 0, 'ludlow16'), z = 0)
#     vmx, rmx = subh.get_mx_from_vir()
#     mmx = vmx**2 * rmx / (4.3009172706 *1e-6)
#     print(mmx, subh.cvir)
