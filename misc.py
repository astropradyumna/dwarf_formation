'''
This program is to test different functions that are newly being created
'''
import numpy as np
import matplotlib.pyplot as plt 
import os 
# os.environ["USE_LZMA"] = "0"
import pandas as pd
from errani_plus_tng_subhalo import Subhalo
from tqdm import tqdm
# import galpy
import IPython
import illustris_python as il
from matplotlib.backends.backend_pdf import PdfPages
# from subhalo_profiles import ExponentialProfile, NFWProfile
# import warnings
# from colossus.cosmology import cosmology
# from colossus.halo import concentration

# cosmology.setCosmology('planck18')

# Suppress the lzma module warning
# warnings.filterwarnings("ignore", category=UserWarning, module="pandas.compat")
# warnings.simplefilter(action='ignore', category=FutureWarning)


# This is currently being used for finding the position of the MBP 


fof0_path = '/mainvol/psadh003/FoF0_TNG50_redshift0/'
outpath  = '/home/psadh003/tng50/output_files/'

df = pd.read_csv(outpath + 'merged_evolved_fof0.csv', delimiter = ',')
mbpid_ar = df['mbpid_ar']

star_ids = np.load(fof0_path+'Stars_IDs_FoF_0_particle_type_4.npy')
star_pos = np.load(fof0_path+'Stars_POS_FoF_0_particle_type_4.npy')

dm_ids = np.load(fof0_path+'DM_IDs_FoF_0_particle_type_1.npy')
dm_pos = np.load(fof0_path+'DM_POS_FoF_0_particle_type_1.npy')

pos_ar = np.zeros((len(mbpid_ar), 3))

popix_ar = np.zeros(0)

IPython.embed()

for (ix, id) in tqdm(enumerate(mbpid_ar)):
    pos = None
    index = np.where(np.isin(star_ids, mbpid_ar[ix]))[0]
    if len(index) == 1: pos = star_pos[index]
    if len(index) == 0:
        index = np.where(np.isin(dm_ids, mbpid_ar[ix]))[0]
        if len(index) == 1: pos = dm_pos[index]
        if len(pos) == 3:
            pos_ar=  np.append(pos_ar, pos) 
        else:
            popix_ar = np.append(popix_ar, ix) #FIXME: #12 Some of the particles are not in the FoF0 particle file


# np.where(star_ids == )






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
