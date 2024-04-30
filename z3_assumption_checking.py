'''
This code is to analyse the assumption that no big subhalos fall after z = 3
'''

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import illustris_python as il
import sys
from tqdm import tqdm
from tng_subhalo_and_halo import TNG_Subhalo
import IPython

'''
file import
'''
filepath = '/rhome/psadh003/bigdata/tng50/tng_files/'
all_subh_path = filepath + 'not_restricted_by_z3/'
outpath  = '/rhome/psadh003/bigdata/tng50/output_files/'
baseUrl = 'https://www.tng-project.org/api/TNG50-1/'
headers = {"api-key":"894f4df036abe0cb9d83561e4b1efcf1"}
basePath = '/rhome/psadh003/bigdata/L35n2160TNG_fixed/output'

h = 0.6744
min_mstar = 1e3

ages_df = pd.read_csv(filepath + 'ages_tng.csv', comment = '#')

all_snaps = np.array(ages_df['snapshot'])
all_redshifts = np.array(ages_df['redshift'])
all_ages = np.array(ages_df['age(Gyr)'])


fof_no = int(sys.argv[1])
fof_str = 'fof' + str(fof_no)

this_fof = il.groupcat.loadSingle(basePath, 99, haloID = fof_no)
central_sfid_99 = this_fof['GroupFirstSub']
# print(central_sfid_99)

'''
Following is the dataset of the entire list of subhalos which infalled after z = 3 and survived
'''
# survived_df = pd.read_csv(filepath + 'sh_survived_after_z3_tng50_1.csv')
survived_df = pd.read_csv(filepath + fof_str + '_sh_survived_after_z3_tng50_1_everything.csv') #This does not have 100 particle restriction as well


ssh_sfid = survived_df['SubfindID'] #is this at infall?
ssh_sfid = np.array([s.strip('[]') for s in ssh_sfid], dtype = int)
ssh_snap = np.array(survived_df['SnapNum'], dtype = int)
ssh_ift = all_ages[ssh_snap]


ssh_sfid1 = survived_df['inf1_subid']
ssh_sfid1 = np.array([s.strip('[]') for s in ssh_sfid1], dtype = int)
ssh_snap1 = np.array(survived_df['inf1_snap'], dtype = int)
ssh_tinf1 = all_ages[ssh_snap1] #This is the infall time 1


ssh_mstar = survived_df['Mstar']
ssh_mstar = [s.strip('[]') for s in ssh_mstar]
ssh_mstar = np.array(ssh_mstar, dtype = float)
ssh_max_mstar = np.array(survived_df['max_Mstar'], dtype = float)
ssh_max_mstar_snap = np.array(survived_df['max_Mstar_snap'], dtype = int)
ssh_max_mstar_id = np.array(survived_df['max_Mstar_id'], dtype = int)



# '''
# Following is the dataset of the entire list of subhalos which infalled after z = 3 and merged
# '''
# # merged_df = pd.read_csv(filepath + 'sh_merged_after_z3_tng50_1.csv')
# merged_df = pd.read_csv(all_subh_path + fof_str + '_sh_merged_after_z3_tng50_1_everything.csv')

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


mstar_ar_z0 = np.zeros(0)

for ix in tqdm(range(len(ssh_snap))):
    subh = TNG_Subhalo(snap = ssh_snap[ix], sfid = ssh_sfid[ix], last_snap = 99)
    mstar_ar_z0 = np.append(mstar_ar_z0, subh.get_mstar(where = 99, how = 'total'))

# IPython.embed()

fig, ax = plt.subplots(figsize = (6, 6))
ax.plot(all_redshifts[ssh_snap], mstar_ar_z0, 'ko', markersize = 2, alpha = 0.2)
ax.set_yscale('log')
ax.set_xlabel('Redshift of infall', fontsize = 12)
ax.set_ylabel(r'$M_{\star}\,(M_{\odot})$ at z = 0', fontsize = 12)
ax.set_title('FoF0', fontsize = 12)
plt.tight_layout()
plt.savefig('z3_assumption_checking' + fof_str + '_mstar_z0_vs_z_infall.png')
