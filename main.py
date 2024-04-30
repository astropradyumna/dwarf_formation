'''
This is an attempt to organize things in this project
main.py combines everything and generates the data required to generate plots.
'''

import numpy as np
import matplotlib.pyplot as plt 
import os 
os.environ["USE_LZMA"] = "0"
import pandas as pd
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
import sys

# Suppress the lzma module warning
# warnings.filterwarnings("ignore", category=UserWarning, module="pandas.compat")
warnings.simplefilter(action='ignore', category=FutureWarning)


'''
file import
'''
filepath = '/rhome/psadh003/bigdata/tng50/tng_files/'
outpath  = '/rhome/psadh003/bigdata/tng50/output_files/'
baseUrl = 'https://www.tng-project.org/api/TNG50-1/'
headers = {"api-key":"894f4df036abe0cb9d83561e4b1efcf1"}
basePath = '/rhome/psadh003/bigdata/L35n2160TNG_fixed/output'

h = 0.6744
min_mstar = 1e2

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



'''
Following is the dataset of the entire list of subhalos which infalled after z = 3 and merged
'''
# merged_df = pd.read_csv(filepath + 'sh_merged_after_z3_tng50_1.csv')
merged_df = pd.read_csv(filepath + fof_str + '_sh_merged_after_z3_tng50_1_everything.csv')

msh_sfid = merged_df['SubfindID']
msh_sfid = np.array([s.strip('[]') for s in msh_sfid], dtype = int) #snap ID at infall
msh_snap = np.array(merged_df['SnapNum'], dtype = int) #Snap at infall
msh_ift = all_ages[msh_snap]

msh_sfid1 = merged_df['inf1_subid']
msh_sfid1 = np.array([s.strip('[]') for s in msh_sfid1], dtype = int)
msh_snap1 = merged_df['inf1_snap']
msh_tinf1 = all_ages[msh_snap1] 


msh_merger_snap = np.array(merged_df['MergerSnapNum'], dtype = int) #SnapNum at the last snapshot of survival
msh_merger_sfid = np.array(merged_df['MergerSubfindID'], dtype = int) #this is the subfind ID at the last snapshot of survival
msh_mt = all_ages[msh_merger_snap] #The time of merger
# print(msh_merger_snap)
msh_mstar = merged_df['Mstar']
msh_mstar = [s.strip('[]') for s in msh_mstar]
msh_mstar = np.array(msh_mstar, dtype = float)
msh_max_mstar = np.array(merged_df['max_Mstar'], dtype = float)
msh_max_mstar_snap = np.array(merged_df['max_Mstar_snap'], dtype = int)


'''
Following is to get the dataset which has the relevant SubfindIDs
'''

# id_df = pd.read_csv('errani_checking_dataset.csv', comment = '#')
# id_df = pd.read_csv(filepath + 'errani_checking_dataset_more1e9msun.csv', comment = '#')

# snap_if_ar = id_df['snap_at_infall']
# sfid_if_ar = id_df['id_at_infall']
# ms_by_mdm = id_df['ms_by_mdm']





'''
=============================================================
Data import ends here
=============================================================
'''

# IPython.embed()


'''
Surviving subhalos with parallelization. Working!
'''



def save_surviving_subhalos(ix):
    # global vmx_if_ar, rmx_if_ar, mmx_if_ar, vmx_f_ar, rmx_f_ar, mmx_f_ar, sfid_if_ar, snap_if_ar, mstar_max_ar, rh_max_ar, vd_max_ar, mstar_f_ar, rh_f_ar, vd_f_ar, rperi_ar, rapo_ar, torb_ar, tinf_ar, vmx_f_ar_tng, rmx_f_ar_tng, mmx_f_ar_tng, mstar_f_ar_tng, rh_f_ar_tng, vd_f_ar_tng, pos_f_ar, dist_f_ar
    vmx_f_ar, rmx_f_ar, mmx_f_ar, mstar_f_ar, rh_f_ar, vd_f_ar = -1 * np.ones(6, dtype = int)
    mstar_f_pl_ar, rh_f_pl_ar, vd_f_pl_ar = -1 * np.ones(3, dtype = int)
    mstar_f_co_ar, rh_f_co_ar, vd_f_co_ar = -1 * np.ones(3, dtype = int)

    subh = Subhalo(snap = ssh_snap[ix], sfid = ssh_sfid[ix], last_snap = 99, central_sfid_99 = central_sfid_99)
    
    if max(subh.mstar, subh.mstar_co, subh.mstar_pl) < min_mstar or np.array(subh.mstar).size * np.array(subh.mstar_pl).size * np.array(subh.mstar_co).size  == 0 :
        return 
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

    if subh.snap < 25: #These would be the cases where the infall was before z = 3 and we only consider these if they survive with mstar > 5e6 Msun at z = 0.
        mstar_at_0 = subh.get_mstar(where = 99, how = 'total')
        if mstar_at_0 > 5e6:
            vmx_f_ar, rmx_f_ar, mmx_f_ar = vmx_f_ar_tng, rmx_f_ar_tng, mmx_f_ar_tng
            mstar_f_ar, rh_f_ar, vd_f_ar = mstar_f_ar_tng, rh_f_ar_tng, vd_f_ar_tng
            # vmx_f_ar, rmx_f_ar, mmx_f_ar, vd_f_ar, rh_f_ar, mstar_f_ar
        else:
            return None

    # if len(pos_f_ar) == 0:
    pos_f_ar = subh.tree['SubhaloPos'][-1, :] / h
    pos_f_ar = pos_f_ar.reshape(1, -1)
    # else:
    #     this_pos = np.array(subh.tree['SubhaloPos'][-1, :] / h)
    #     pos_f_ar = np.append(pos_f_ar, this_pos.reshape(1, -1), axis=0)

    dist_f_ar = subh.get_dist_from_cen(where=99)
    return vmx_if_ar, rmx_if_ar, mmx_if_ar, vmx_f_ar, rmx_f_ar, mmx_f_ar,sfid_if_ar, snap_if_ar, mstar_max_ar, rh_max_ar, vd_max_ar, mstar_f_ar, rh_f_ar, vd_f_ar,  rperi_ar, rapo_ar, torb_ar, tinf_ar, vmx_f_ar_tng, rmx_f_ar_tng, mmx_f_ar_tng, mstar_f_ar_tng, rh_f_ar_tng, vd_f_ar_tng, pos_f_ar.tolist()[0], dist_f_ar, vd_f_pl_ar, rh_f_pl_ar, mstar_f_pl_ar, vd_f_co_ar, rh_f_co_ar, mstar_f_co_ar, rh_max_pl_ar, rh_max_co_ar, mstar_max_pl_ar, mstar_max_co_ar 






results = Parallel(n_jobs=32, pre_dispatch='1.5*n_jobs')(delayed(save_surviving_subhalos)(ix) for ix in tqdm(range(len(ssh_snap))))
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
    'rh_max_pl_ar', 'rh_max_co_ar', 'mstar_max_pl_ar', 'mstar_max_co_ar' ]

# Create an empty DataFrame with the specified column names
df = pd.DataFrame(columns=column_names)
for ix in range(len(results)):
    df.loc[len(df)] = results[ix]

df.to_csv(outpath + fof_str + '_surviving_evolved_everything.csv', index = False)


'''
Merged subhalos with parallelzation. Writing.
'''
def save_merged_subhalos(ix):
    '''
    This is a function to save return all the parameters of interest for the merged subhalos
    '''
    vmx_f_ar, rmx_f_ar, mmx_f_ar, mstar_f_ar, rh_f_ar, vd_f_ar = -1 * np.ones(6, dtype = int)
    mstar_f_pl_ar, rh_f_pl_ar, vd_f_pl_ar = -1 * np.ones(3, dtype = int)
    mstar_f_co_ar, rh_f_co_ar, vd_f_co_ar = -1 * np.ones(3, dtype = int)

    subh  = Subhalo(snap = int(msh_snap[ix]), sfid = int(msh_sfid[ix]), last_snap = int(msh_merger_snap[ix]), central_sfid_99=central_sfid_99) #these are at infall
    if max(subh.mstar, subh.mstar_co, subh.mstar_pl) < min_mstar or np.array(subh.mstar).size * np.array(subh.mstar_pl).size * np.array(subh.mstar_co).size  == 0: #this would be the mass cutoff at infall for the subhalos
        return None

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

    sfid_if_ar = msh_sfid[ix]
    snap_if_ar = msh_snap[ix]
    rperi_ar = subh.rperi
    rapo_ar = subh.rapo
    torb_ar = subh.torb
    tinf_ar = all_ages[msh_snap[ix]]

    mstar_max_ar = subh.mstar

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

    return vmx_if_ar, rmx_if_ar, mmx_if_ar, vmx_f_ar, rmx_f_ar, mmx_f_ar, sfid_if_ar, snap_if_ar, mstar_max_ar, rh_max_ar, vd_max_ar, mstar_f_ar, rh_f_ar, vd_f_ar, rperi_ar, rapo_ar, torb_ar, tinf_ar, mbpid_ar, mbpidp_ar, vd_f_pl_ar, rh_f_pl_ar, mstar_f_pl_ar, vd_f_co_ar, rh_f_co_ar, mstar_f_co_ar, rh_max_pl_ar, rh_max_co_ar, mstar_max_pl_ar, mstar_max_co_ar


results = Parallel(n_jobs=32, pre_dispatch='1.5*n_jobs')(delayed(save_merged_subhalos)(ix) for ix in tqdm(range(len(msh_snap))))
results = [value for value in results if value is not None] #Getting rid of all the None entries


column_names = ['vmx_if_ar', 'rmx_if_ar', 'mmx_if_ar',
    'vmx_f_ar', 'rmx_f_ar', 'mmx_f_ar',
    'sfid_if_ar', 'snap_if_ar',
    'mstar_max_ar', 'rh_max_ar', 'vd_max_ar',
    'mstar_f_ar', 'rh_f_ar', 'vd_f_ar',
    'rperi_ar', 'rapo_ar', 'torb_ar', 'tinf_ar',
    'mbpid_ar', 'mbpidp_ar',  'vd_f_pl_ar', 'rh_f_pl_ar', 
    'mstar_f_pl_ar', 'vd_f_co_ar', 'rh_f_co_ar', 'mstar_f_co_ar', 
    'rh_max_pl_ar', 'rh_max_co_ar', 'mstar_max_pl_ar', 'mstar_max_co_ar']


df = pd.DataFrame(columns=column_names)
for ix in range(len(results)):
    df.loc[len(df)] = results[ix]

df.to_csv(outpath + fof_str + '_merged_evolved_everything.csv', index = False)







# print(results.shape)
    # print(results[4])


    
    # df = pd.read_csv(outpath + 'temp_surviving.csv')
    # df.[    vmx_if_ar, rmx_if_ar, mmx_if_ar,    vmx_f_ar, rmx_f_ar, mmx_f_ar,    sfid_if_ar, snap_if_ar,    mstar_max_ar, rh_max_ar, vd_max_ar,    mstar_f_ar, rh_f_ar, vd_f_ar,    rperi_ar, rapo_ar, torb_ar, tinf_ar,    vmx_f_ar_tng, rmx_f_ar_tng, mmx_f_ar_tng,    mstar_f_ar_tng, rh_f_ar_tng, vd_f_ar_tng,    pos_f_ar, dist_f_ar]

    
    # rh_max_ar = np.append(rh_max_ar, subh.Rh) #this is the 2d half light radius
    # vd_max_ar = np.append(vd_max_ar, subh.vd) #los vd


    # # print(subh.get_rh(where = 'max')*3./4)
    # # try: #Skip the subhalos which do not have Rh at max Mstar
    # #     rh_max_ar = np.append(rh_max_ar, subh.get_rh(where = 'max')*3./4) #this is the 2d half light radius
    # #     vd_max_ar = np.append(vd_max_ar, subh.get_vd(where = 'max')) #los vd
    # # except ValueError:
    # #     continue

    # # if vmx_if_ar.size == 0:
    # #     vmx_if_ar = subh.vmx0
    # # else:
    # vmx_if_ar = np.append(vmx_if_ar, subh.vmx0)
    # rmx_if_ar = np.append(rmx_if_ar, subh.rmx0)
    # mmx_if_ar = np.append(mmx_if_ar, subh.mmx0) 

    # sfid_if_ar = np.append(sfid_if_ar, subh.sfid)
    # snap_if_ar = np.append(snap_if_ar, subh.snap)
    # rperi_ar = np.append(rperi_ar, subh.rperi)
    # rapo_ar = np.append(rapo_ar, subh.rapo)
    # torb_ar = np.append(torb_ar, subh.torb)
    # tinf_ar = np.append(tinf_ar, all_ages[int(subh.snap)])

    # mstar_max_ar = np.append(mstar_max_ar, subh.mstar)

    
    # # if subh.torb == np.inf: #case of no evolution
    # #     vmxf, rmxf, mmxf, vdf, rhf, mstarf = subh.vmx0, subh.rmx0, subh.mmx0, vd_max_ar[-1], rh_max_ar[-1], subh.mstar
    # # else:
    # vmxf, rmxf, mmxf, vdf, rhf, mstarf = subh.get_model_values(float(tinf_ar[-1]), t) #FIXME: Some orbits are not unbound as galpy reports
    # vmx_f_ar = np.append(vmx_f_ar, vmxf)
    # rmx_f_ar = np.append(rmx_f_ar, rmxf)
    # mmx_f_ar = np.append(mmx_f_ar, mmxf)
    # mstar_f_ar = np.append(mstar_f_ar, mstarf)
    # rh_f_ar = np.append(rh_f_ar, rhf)
    # vd_f_ar = np.append(vd_f_ar, vdf)

    # with warnings.catch_warnings(record=True) as w:
    #     vmxf_tng, rmxf_tng, mmxf_tng = subh.get_mx_values(where = int(99))
    #     if len(w) > 0:
    #         vmxf_tng, rmxf_tng, mmxf_tng = subh.get_rot_curve(where= int(99))
    # # Following are from TNG
    # vmx_f_ar_tng = np.append(vmx_f_ar_tng, vmxf_tng)
    # rmx_f_ar_tng = np.append(rmx_f_ar_tng, rmxf_tng)
    # mmx_f_ar_tng = np.append(mmx_f_ar_tng, mmxf_tng)
    # mstar_f_ar_tng = np.append(mstar_f_ar_tng, subh.get_mstar(where = 99, how = 'total'))
    # try:
    #     rh_f_ar_tng = np.append(rh_f_ar_tng, subh.get_rh(where = 99)*3./4)
    #     vd_f_ar_tng = np.append(vd_f_ar_tng, subh.get_vd(where = 99))
    # except ValueError:
    #     rh_f_ar_tng = np.append(rh_f_ar_tng, 0)
    #     vd_f_ar_tng = np.append(vd_f_ar_tng, 0)
    
    # if len(pos_f_ar) == 0:
    #     pos_f_ar = subh.tree['SubhaloPos'][-1, :]/h
    #     pos_f_ar = pos_f_ar.reshape(1, -1)
    # else:
    #     this_pos = np.array(subh.tree['SubhaloPos'][-1, :]/h)
    #     pos_f_ar = np.append(pos_f_ar, this_pos.reshape(1, -1), axis = 0)

    # dist_f_ar = np.append(dist_f_ar, subh.get_dist_from_cen(where = 99))
    # # print(subh.vmx0)
    # print(vmx_if_ar)
    # return None





'''
Surviving subhalos from FoF0
'''
# vmx_if_ar = np.zeros(0)
# rmx_if_ar = np.zeros(0)
# mmx_if_ar = np.zeros(0)
# vmx_f_ar = np.zeros(0)
# rmx_f_ar = np.zeros(0)
# mmx_f_ar = np.zeros(0)

# sfid_if_ar = np.zeros(0)
# snap_if_ar = np.zeros(0)

# mstar_max_ar = np.zeros(0)
# rh_max_ar = np.zeros(0)
# vd_max_ar = np.zeros(0)

# mstar_f_ar = np.zeros(0)
# rh_f_ar = np.zeros(0)
# vd_f_ar = np.zeros(0)

# rperi_ar = np.zeros(0)
# rapo_ar = np.zeros(0)
# torb_ar = np.zeros(0)
# tinf_ar = np.zeros(0)

# vmx_f_ar_tng = np.zeros(0)
# rmx_f_ar_tng = np.zeros(0)
# mmx_f_ar_tng = np.zeros(0)
# mstar_f_ar_tng = np.zeros(0)
# rh_f_ar_tng = np.zeros(0)
# vd_f_ar_tng = np.zeros(0)

# pos_f_ar = np.zeros(0) #These are the final positions
# dist_f_ar = np.zeros(0)

# def save_surviving_subhalos(snap, sfid, last_snap, central_sfid_99):
#     # global vmx_if_ar, rmx_if_ar, mmx_if_ar, vmx_f_ar, rmx_f_ar, mmx_f_ar, sfid_if_ar, snap_if_ar, mstar_max_ar, rh_max_ar, vd_max_ar, mstar_f_ar, rh_f_ar, vd_f_ar, rperi_ar, rapo_ar, torb_ar, tinf_ar, vmx_f_ar_tng, rmx_f_ar_tng, mmx_f_ar_tng, mstar_f_ar_tng, rh_f_ar_tng, vd_f_ar_tng, pos_f_ar, dist_f_ar

#     subh = Subhalo(snap = snap, sfid = sfid, last_snap=last_snap, central_sfid_99 = central_sfid_99)
#     # print(subh.mstar)

#     if subh.mstar < 1e3 or subh.mstar.size == 0:
#         return None
#     try:
#         t = subh.get_orbit(merged = False, when_te = 'last') #after this, the subhalo has rperi, rapo and torb
#     except Exception as e:
#         print(e) 
#         # ctr = ctr + 1
#         # skipped_ixs = np.append(skipped_ixs, ix)
#         return None
    
#     rh_max_ar = np.append(rh_max_ar, subh.Rh) #this is the 2d half light radius
#     vd_max_ar = np.append(vd_max_ar, subh.vd) #los vd


#     # print(subh.get_rh(where = 'max')*3./4)
#     # try: #Skip the subhalos which do not have Rh at max Mstar
#     #     rh_max_ar = np.append(rh_max_ar, subh.get_rh(where = 'max')*3./4) #this is the 2d half light radius
#     #     vd_max_ar = np.append(vd_max_ar, subh.get_vd(where = 'max')) #los vd
#     # except ValueError:
#     #     continue

#     # if vmx_if_ar.size == 0:
#     #     vmx_if_ar = subh.vmx0
#     # else:
#     vmx_if_ar = np.append(vmx_if_ar, subh.vmx0)
#     rmx_if_ar = np.append(rmx_if_ar, subh.rmx0)
#     mmx_if_ar = np.append(mmx_if_ar, subh.mmx0) 

#     sfid_if_ar = np.append(sfid_if_ar, subh.sfid)
#     snap_if_ar = np.append(snap_if_ar, subh.snap)
#     rperi_ar = np.append(rperi_ar, subh.rperi)
#     rapo_ar = np.append(rapo_ar, subh.rapo)
#     torb_ar = np.append(torb_ar, subh.torb)
#     tinf_ar = np.append(tinf_ar, all_ages[int(subh.snap)])

#     mstar_max_ar = np.append(mstar_max_ar, subh.mstar)

    
#     # if subh.torb == np.inf: #case of no evolution
#     #     vmxf, rmxf, mmxf, vdf, rhf, mstarf = subh.vmx0, subh.rmx0, subh.mmx0, vd_max_ar[-1], rh_max_ar[-1], subh.mstar
#     # else:
#     vmxf, rmxf, mmxf, vdf, rhf, mstarf = subh.get_model_values(float(tinf_ar[-1]), t) #FIXME: Some orbits are not unbound as galpy reports
#     vmx_f_ar = np.append(vmx_f_ar, vmxf)
#     rmx_f_ar = np.append(rmx_f_ar, rmxf)
#     mmx_f_ar = np.append(mmx_f_ar, mmxf)
#     mstar_f_ar = np.append(mstar_f_ar, mstarf)
#     rh_f_ar = np.append(rh_f_ar, rhf)
#     vd_f_ar = np.append(vd_f_ar, vdf)

#     with warnings.catch_warnings(record=True) as w:
#         vmxf_tng, rmxf_tng, mmxf_tng = subh.get_mx_values(where = int(99))
#         if len(w) > 0:
#             vmxf_tng, rmxf_tng, mmxf_tng = subh.get_rot_curve(where= int(99))
#     # Following are from TNG
#     vmx_f_ar_tng = np.append(vmx_f_ar_tng, vmxf_tng)
#     rmx_f_ar_tng = np.append(rmx_f_ar_tng, rmxf_tng)
#     mmx_f_ar_tng = np.append(mmx_f_ar_tng, mmxf_tng)
#     mstar_f_ar_tng = np.append(mstar_f_ar_tng, subh.get_mstar(where = 99, how = 'total'))
#     try:
#         rh_f_ar_tng = np.append(rh_f_ar_tng, subh.get_rh(where = 99)*3./4)
#         vd_f_ar_tng = np.append(vd_f_ar_tng, subh.get_vd(where = 99))
#     except ValueError:
#         rh_f_ar_tng = np.append(rh_f_ar_tng, 0)
#         vd_f_ar_tng = np.append(vd_f_ar_tng, 0)
    
#     if len(pos_f_ar) == 0:
#         pos_f_ar = subh.tree['SubhaloPos'][-1, :]/h
#         pos_f_ar = pos_f_ar.reshape(1, -1)
#     else:
#         this_pos = np.array(subh.tree['SubhaloPos'][-1, :]/h)
#         pos_f_ar = np.append(pos_f_ar, this_pos.reshape(1, -1), axis = 0)

#     dist_f_ar = np.append(dist_f_ar, subh.get_dist_from_cen(where = 99))
#     # print(subh.vmx0)
#     print(vmx_if_ar)
#     return None



# for : #This would run over all the subhalos surviving till z = 0




# df = pd.DataFrame()

# df['vmx_if_ar'] = vmx_if_ar
# df['rmx_if_ar'] = rmx_if_ar
# df['mmx_if_ar'] = mmx_if_ar
# df['vmx_f_ar'] = vmx_f_ar
# df['rmx_f_ar'] = rmx_f_ar
# df['mmx_f_ar'] = mmx_f_ar

# df['sfid_if_ar'] = sfid_if_ar
# df['snap_if_ar'] = snap_if_ar

# df['mstar_max_ar'] = mstar_max_ar
# df['rh_max_ar'] = rh_max_ar
# df['vd_max_ar'] = vd_max_ar

# df['mstar_f_ar'] = mstar_f_ar
# df['rh_f_ar'] = rh_f_ar
# df['vd_f_ar'] = vd_f_ar

# df['rperi_ar'] = rperi_ar
# df['rapo_ar'] = rapo_ar
# df['torb_ar'] = torb_ar
# df['tinf_ar'] = tinf_ar

# df['vmx_f_ar_tng'] = vmx_f_ar_tng
# df['rmx_f_ar_tng'] = rmx_f_ar_tng
# df['mmx_f_ar_tng'] = mmx_f_ar_tng
# df['mstar_f_ar_tng'] = mstar_f_ar_tng
# df['rh_f_ar_tng'] = rh_f_ar_tng
# df['vd_f_ar_tng'] = vd_f_ar_tng

# df['pos_f_ar'] = pos_f_ar.tolist()  #These are the final positions
# df['dist_f_ar'] = dist_f_ar #This is the distance of the subhalo at z = 0

# # df.to_csv(outpath + 'surviving_evolved_fof0_everything.csv', index = False)
# df.to_csv(outpath + 'surviving_evolved_fof0.csv', index = False)


# # IPython.embed()

# # '''
# # Merging subhalos from FoF0
# # '''

# vmx_if_ar = np.zeros(0)
# rmx_if_ar = np.zeros(0)
# mmx_if_ar = np.zeros(0)
# vmx_f_ar = np.zeros(0)
# rmx_f_ar = np.zeros(0)
# mmx_f_ar = np.zeros(0)

# sfid_if_ar = np.zeros(0)
# snap_if_ar = np.zeros(0)

# mstar_max_ar = np.zeros(0)
# rh_max_ar = np.zeros(0)
# vd_max_ar = np.zeros(0)

# mstar_f_ar = np.zeros(0)
# rh_f_ar = np.zeros(0)
# vd_f_ar = np.zeros(0)

# rperi_ar = np.zeros(0)
# rapo_ar = np.zeros(0)
# torb_ar = np.zeros(0)
# tinf_ar = np.zeros(0)

# mbpid_ar = np.zeros(0)
# mbpidp_ar = np.zeros(0) #this is the MBP ID of the previous snapshot


# for ix in tqdm(range(len(msh_snap))):
#     '''
#     This is to loop over all the merging subhalos of big dataset with subhalos in between 1e8.5 and 1e9.5 Msun
#     '''
#     if ix in [5, 9, 14, 22]: continue #takes lot of time to get compiled

#     # if ix < 10:
#     #     continue

#     subh  = Subhalo(snap = int(msh_snap[ix]), sfid = int(msh_sfid[ix]), last_snap = int(msh_merger_snap[ix]), central_sfid_99=0) #these are at infall


#     if subh.mstar < 1e3 or subh.mstar.size == 0: #this would be the mass cutoff at infall for the subhalos
#         continue

#     try:
#         t = subh.get_orbit(merged = True, when_te = 'last') #after this, the subhalo has rperi, rapo and torb
#     except Exception as e:
#         print(e)
#         # ctr = ctr + 1
#         # skipped_ixs = np.append(skipped_ixs, ix)
#         continue

#     rh_max_ar = np.append(rh_max_ar, subh.Rh) #this is the 2d half light radius
#     vd_max_ar = np.append(vd_max_ar, subh.vd) #los vd

#     # print(subh.get_rh(where = 'max')*3./4)
#     # try: #If we do not have Rh at maximum stellar mass, then we are just skipping the subhalo
#     #     rh_max_ar = np.append(rh_max_ar, subh.get_rh(where = 'max')*3./4) #this is the 2d half light radius
#     #     vd_max_ar = np.append(vd_max_ar, subh.get_vd(where = 'max')) #los vd
#     # except ValueError:
#     #     continue


#     vmx_if_ar = np.append(vmx_if_ar, subh.vmx0)
#     rmx_if_ar = np.append(rmx_if_ar, subh.rmx0)
#     mmx_if_ar = np.append(mmx_if_ar, subh.mmx0)

#     sfid_if_ar = np.append(sfid_if_ar, msh_sfid[ix])
#     snap_if_ar = np.append(snap_if_ar, msh_snap[ix])
#     rperi_ar = np.append(rperi_ar, subh.rperi)
#     rapo_ar = np.append(rapo_ar, subh.rapo)
#     torb_ar = np.append(torb_ar, subh.torb)
#     tinf_ar = np.append(tinf_ar, all_ages[msh_snap[ix]])

#     mstar_max_ar = np.append(mstar_max_ar, subh.mstar)



#     # if subh.torb == np.inf:
#     #     vmxf, rmxf, mmxf, vdf, rhf, mstarf = subh.vmx0, subh.rmx0, subh.mmx0, vd_max_ar[-1], rh_max_ar[-1], subh.mstar
#     # else:
#     vmxf, rmxf, mmxf, vdf, rhf, mstarf = subh.get_model_values(float(tinf_ar[-1]), t)

#     vmx_f_ar = np.append(vmx_f_ar, vmxf)
#     rmx_f_ar = np.append(rmx_f_ar, rmxf)
#     mmx_f_ar = np.append(mmx_f_ar, mmxf)
#     mstar_f_ar = np.append(mstar_f_ar, mstarf)
#     rh_f_ar = np.append(rh_f_ar, rhf)
#     vd_f_ar = np.append(vd_f_ar, vdf)

#     mbpid_ar = np.append(mbpid_ar, subh.get_mbpid(where = subh.last_snap)) #Get the MBP ID at infall
#     mbpidp =  np.array(subh.get_mbpid(where = subh.last_snap-1))
#     # print(mbpid_ar[-1], mbpidp)
#     # print(len(mbpidp), mbpidp.shape)
#     if len(mbpidp) != 0:
#         mbpidp_ar = np.append(mbpidp_ar, mbpidp)
#     else:
#         mbpidp_ar = np.append(mbpidp_ar, -1)
#     # print(mbpidp_ar[-1])



# df = pd.DataFrame()

# df['vmx_if_ar'] = vmx_if_ar
# df['rmx_if_ar'] = rmx_if_ar
# df['mmx_if_ar'] = mmx_if_ar
# df['vmx_f_ar'] = vmx_f_ar
# df['rmx_f_ar'] = rmx_f_ar
# df['mmx_f_ar'] = mmx_f_ar

# df['sfid_if_ar'] = sfid_if_ar
# df['snap_if_ar'] = snap_if_ar

# df['mstar_max_ar'] = mstar_max_ar
# df['rh_max_ar'] = rh_max_ar
# df['vd_max_ar'] = vd_max_ar

# df['mstar_f_ar'] = mstar_f_ar
# df['rh_f_ar'] = rh_f_ar
# df['vd_f_ar'] = vd_f_ar

# df['rperi_ar'] = rperi_ar
# df['rapo_ar'] = rapo_ar
# df['torb_ar'] = torb_ar
# df['tinf_ar'] = tinf_ar

# df['mbpid_ar'] = mbpid_ar  #These are the final positions
# df['mbpidp_ar'] = mbpidp_ar #MBP IDs of the snapshot before the final snapshot

# df.to_csv(outpath + 'merged_evolved_fof0.csv', index = False)
# # df.to_csv(outpath + 'merged_evolved_fof0_everything.csv', index = False)




# ==================================================









# '''
# Following are the analyses which we used in the past to generate all the plots required. 
# Moving on, we will be using plots.py to make the plots and do only the calculations here to generate the data required
# '''

# """
# ===================================================================================
# SURVIVING SUBHALOS
# ===================================================================================
# """

# '''
# Plot 0.1: This is to plot all the half mass radius of stars according to TNG, and the one assumed in the model
# '''

# fig, ax = plt.subplots(figsize = (5, 5))


# for ix in tqdm(range(len(ssh_snap))):
#     '''
#     This is to loop over all the surviving subhalos of big dataset with subhalos in between 1e8.5 and 1e9.5 Msun
#     ''' 
#     subh  = Subhalo(snap = ssh_snap[ix], sfid = ssh_sfid[ix], last_snap = 99 )
#     # subh.get_infall_properties()
#     try:
#         Rh_tng_max = subh.get_rh(where = 'max')/np.sqrt(2)
#         Rh_model = subh.get_rh0byrmx0() * subh.rmx0 #This is the assumption
#     except:
#         continue
#     # if subh.get_rh0byrmx0() == 0.5:
#     #     print('Jai')
#     # ax.plot(Rh_model/subh.rmx0, Rh_tng_max/subh.rmx0, 'ko', ms = 3.5)
#     ax.plot(Rh_model, Rh_tng_max, 'ko', ms = 3.5)
#     # if ix in [30, 49, 56, 68, 84, 85, 89, 95]: 
#     #     ax.plot(Rh_model, Rh_tng_max, 'ro')
#     #     if ix == 30:ax.plot(Rh_model, Rh_tng_max, 'ro', label = r'Low $f_{\bigstar}$')
#     # elif ix in [83, 93, 94, 98, 100, 102, 103, 106]:
#     #     ax.plot(Rh_model, Rh_tng_max, 'bo')
#     #     if ix == 83: ax.plot(Rh_model, Rh_tng_max, 'bo', label = r'High $f_{\bigstar}$')

# ax.set_xlabel(r'$R_{\rm{h, inf}}/r_{\rm{mx0}}$ model (kpc)')
# ax.set_ylabel(r'$R_{\rm{h, max\, M_\bigstar}}/r_{\rm{mx0}}$ from TNG (kpc)')
# # ax.set_xlim(0, 8)
# # ax.set_ylim(0, 8)
# x_vals = np.array(ax.get_xlim())    
# ax.plot(x_vals, x_vals, 'k--', lw = 0.5)
# ax.legend(fontsize = 8)
# plt.loglog()
# plt.tight_layout()
# plt.show()


# '''
# Plot 0.2: This is to plot the mass of the ones with no stellar mass remaining
# This is to compare with the exponential profile and test if it is a good enough approximation for the stellar profile
# '''
# pdf_file2 = outpath + "mass profiles_tng50_more1e9msun_tsah.pdf"
# pdf_pages2 = PdfPages(pdf_file2)

# for ix in tqdm(range(len(snap_if_ar))):
# # for ix in tqdm([30, 49, 56, 68, 84, 85, 89, 95]):
# # for ix in tqdm([17, 19]): # These are the indices which has too high rmx0 from the roation curve
#     subh  = Subhalo(snap = snap_if_ar[ix], sfid = sfid_if_ar[ix], last_snap = 99)
#     subh.get_mass_profiles(where = int(snap_if_ar[ix]), plot = True)
#     exp_check = ExponentialProfile(subh.get_mstar(where = 'max', how = 'total'), np.sqrt(2) * subh.get_rh0byrmx0() * subh.rmx0) #This is an instance to check the exponential profile
    
#     vmx, rmx, mmx = subh.get_mx_values(where = int(snap_if_ar[ix])) #this is from forcing the rotation curve to pass through the mass points
#     try:
#         nfw_check = NFWProfile(rmx = subh.rmx0, vmx = subh.vmx0, z = subh.get_z(where = int(snap_if_ar[ix])))
#         plt.plot(rpl, nfw_check.mass(rpl), label = 'NFW assumed from RC', color = 'gray', ls = '-.')
#     except Exception as e:
#         print(e)
#         pass

    
    
#     try:
#         nfw_check2 = NFWProfile(rmx= rmx, vmx = vmx, z = subh.get_z(where = int(snap_if_ar[ix])))
#         plt.plot(rpl, nfw_check2.mass(rpl), label = 'NFW assumed from pts', color = 'gray', ls = ':')
#     except Exception as e:
#         print(e)
#         pass

#     left, right = plt.gca().get_xlim()
#     rpl = np.logspace(np.log10(left), np.log10(right), 100)
#     plt.plot(rpl, exp_check.mass(rpl), label = 'Expoential assumed', color = 'salmon', ls = '-.')
    
    
#     # plt.axvline(subh.get_rh(where = int(snap_if_ar[ix])), lw = 0.3, color = 'gray')
#     # plt.axvline(2 * subh.get_rh(where = int(snap_if_ar[ix])), lw = 0.3, color = 'turquoise')
#     plt.plot(subh.rmx0, subh.mmx0, 'ko', label = 'mx')
#     plt.legend(fontsize = 8)
#     plt.tight_layout()
#     pdf_pages2.savefig()
#     # plt.show()
#     plt.close()

# pdf_pages2.close()


# '''
# Plot 0.3: This is to plot the rotation curve and see why rmx0 values are so less
# '''

# pdf_file3 = outpath + "rot_curves_tng50_more1e9msun_tsah.pdf"
# pdf_pages3 = PdfPages(pdf_file3)

# # for ix in tqdm([30, 49, 56, 68, 84, 85, 89, 95]):
# for ix in tqdm(range(len(snap_if_ar))):
# # for ix in tqdm([17, 19]): # These are the indices which has too high rmx0 from the roation curve
#     subh  = Subhalo(snap = int(snap_if_ar[ix]), sfid = int(sfid_if_ar[ix]), last_snap= 99)
#     print(subh.get_rot_curve(where = int(snap_if_ar[ix]), plot = True))
#     left, right = plt.gca().get_xlim()
#     # print(left, right)
#     rpl = np.logspace(-1, np.log10(right), 100)
#     vmx, rmx, mmx = subh.get_mx_values(where = int(snap_if_ar[ix])) #this is from forcing the rotation curve to pass through the mass points
#     try:
#         nfw_check = NFWProfile(rmx = subh.rmx0, vmx = subh.vmx0, z = subh.get_z(where = int(snap_if_ar[ix])))
#         plt.plot(rpl, nfw_check.velocity(rpl), label = 'NFW assumed frm RC', color = 'gray', ls = '-.', zorder = 100)

#     except Exception as e:
#         print(e)
#         pass
    
#     try:
#         nfw_check2 = NFWProfile(rmx= rmx, vmx = vmx, z = all_redshifts[snap_if_ar[ix]])
#         plt.plot(rpl, nfw_check2.velocity(rpl), label = 'NFW assumed frm pts', color = 'gray', ls = ':', zorder = 100)
#     except Exception as e:
#         print(e)
#         pass
#     # print(nfw_check.velocity(rpl))
    
#     plt.plot(rmx, vmx, marker = 'o', color = 'gray')
#     # nfw_check.velocity(rpl)
#     # exp_check = ExponentialProfile(subh.get_mstar(where = 'max', how = 'total'), np.sqrt(2) * subh.get_rh0byrmx0() * subh.rmx0) #This is an instance to check the exponential profile
#     # left, right = plt.gca().get_xlim()
#     # rpl = np.logspace(np.log10(left), np.log10(right), 100)
#     # plt.plot(rpl, exp_check.mass(rpl), label = 'Expoential assumed', color = 'salmon', ls = ':')
#     plt.tight_layout()
#     plt.legend(fontsize = 8)
#     pdf_pages3.savefig()
#     # plt.show()
#     plt.close()

# pdf_pages3.close()





    


# # ======================================
# '''
# Thie big one, loop which evolves our subhalos
# Surviving
# '''
# ctr = 0
# skipped_ixs = np.zeros(0)
# frem_ar = np.zeros(0) #this is the remnant masss fraction from the SAM
# mmx_ar = np.zeros(0)
# frem_tng_ar = np.zeros(0)
# mmx_tng_ar = np.zeros(0)
# subh_m200_ar = np.zeros(0) #This is the virial mass at infall for the subhalos
# subh_mstar_99_ar = np.zeros(0) #stellar mass at z = 0 for the subhalos being considered
# subh_tinf_ar = np.zeros(0) #This is the array of infall times of the subhalo
# subh_fstar_model_ar = np.zeros(0) #This is the array of stellar mass fraction remaining in the model
# subh_fstar_tng_ar = np.zeros(0)
# subh_fstar_model_from_tngfrem_ar = np.zeros(0) #This is the array of stellar mass fraction that is remaining assuming Mmx/Mmx0 from TNG
# subh_mstar_model_from_tngfrem_ar = np.zeros(0) #This is the array of stellar mass that is remaining assuming Mmx/Mmx0 from TNG
# subh_mstar_model_ar = np.zeros(0) #this is the stellar mass remaining at the end for the model
# subh_mstar_tng_ar = np.zeros(0) #This is the stellar mass remaining at the end for TNG
# subh_parlen_99_ar = np.zeros(0) #This is the length of particles at z = 0
# subh_fbar_tng_ar = np.zeros(0) #This is the ratio of stars to dark matter in 2Rh for the subhalos at infall
# subh_fdm_ar = np.zeros(0) #This is the total remnant dark matter mass to test the Smith+16 paper results
# subh_id99_ar = np.zeros(0) #this is the array of SubfindIDs at z = 0 for the subhalos for labeling
# rperi_ar = np.zeros(0)
# rapo_ar = np.zeros(0)

# # Both the following indices are for the frem_ar, etc. (there are some subhalos that are missed for having an Unbound Orbit)
# ixs_low_mstar = np.zeros(0) #This is the list of indices of the subhalos that have low stellar mass than that in TNG
# ixs_high_mstar = np.zeros(0) #This is the list of indices which have high mstar in the model.
# ixs_unresol = np.zeros(0, dtype = int)
# subh_ixs_low_mstar = np.zeros(0, dtype = int) #This is the index in the total array 
# '''
# Run this loop for doing everything after evolution
# '''



# for ix in tqdm(range(len(snap_if_ar))):
#     '''
#     This is to loop over all the surviving subhalos of big dataset with subhalos in between 1e8.5 and 1e9.5 Msun
#     '''
    
#     # if ix>15: break

#     subh  = Subhalo(snap = int(snap_if_ar[ix]), sfid = int(sfid_if_ar[ix]), last_snap = int(99))

#     # if ix in [30, 49, 56, 68, 84, 85, 89, 95]: 
    
#     # subh.get_infall_properties()
#     # t = subh.get_orbit(merged = False, when_te = 'last') #after this, the subhalo has rperi, rapo and torb
#     try:
#         t = subh.get_orbit(merged = False, when_te = 'last') #after this, the subhalo has rperi, rapo and torb
#     except Exception as e:
#         # print(e)
#         ctr = ctr + 1
#         skipped_ixs = np.append(skipped_ixs, ix)
#         continue

#     subh_fbar_tng_ar = np.append(subh_fbar_tng_ar, subh.get_mstar(where = int(subh.snap), how = 'rh')/ subh.get_mdm(where = int(subh.snap), how = 'rh'))
#     subh_fdm_ar = np.append(subh_fdm_ar, subh.get_mdm(where = 99, how = 'total') / subh.get_mdm(where = int(snap_if_ar[ix]), how = 'total'))
#     subh_mstar_99 = subh.get_mstar(where=99, how = 'total')
#     subh_mstar_99_ar = np.append(subh_mstar_99_ar, subh_mstar_99)
#     subh_id99_ar = np.append(subh_id99_ar, subh.get_sfid(where = 99))
#     subh_tinf_ar = np.append(subh_tinf_ar, all_ages[snap_if_ar[ix]])
#     subh_m200_ar = np.append(subh_m200_ar, subh.get_m200(where = int(snap_if_ar[ix] - 1))) #Append only when there are no errors
#     vmxf, rmxf, mmxf, vdf, rhf, mstarf = subh.get_model_values(float(all_ages[subh.snap]), t) #FIXME: Some orbits are not unbound as galpy reports
#     mmx_ar = np.append(mmx_ar, mmxf)
#     frem = mmxf/subh.mmx0
#     frem_tng = subh.get_tng_values(where = int(99))[2]/subh.mmx0
#     mmx_tng_ar = np.append(mmx_tng_ar, frem_tng * subh.mmx0)
#     frem_tng_ar = np.append(frem_tng_ar, frem_tng)
#     subh_mstar_model_ar = np.append(subh_mstar_model_ar, mstarf)
#     subh_fstar_model_ar = np.append(subh_fstar_model_ar, subh_mstar_model_ar[-1]/subh.mstar)

#     # subh_mstar_model_from_tngfrem_ar = np.append(subh_mstar_model_from_tngfrem_ar, subh.get_starprops_model(frem = frem_tng)[0])
#     # subh_fstar_model_from_tngfrem_ar = np.append(subh_fstar_model_from_tngfrem_ar, subh_mstar_model_from_tngfrem_ar[-1]/subh.mstar)

#     subh_mstar_tng_ar = np.append(subh_mstar_tng_ar, subh_mstar_99)
#     subh_fstar_tng_ar = np.append(subh_fstar_tng_ar, subh_mstar_99 / subh.get_mstar(where = 'max', how = 'total'))
#     subh_parlen_99_ar = np.append(subh_parlen_99_ar, subh.get_len(where = subh.last_snap, which = 'dm') + subh.get_len(where = 99, which = 'stars'))
#     if subh_parlen_99_ar[-1] < 3000:
#         ixs_unresol = np.append(ixs_unresol, int(len(frem_tng_ar) - 1))
#     if subh_fstar_model_ar[-1] > 0.8 and subh_fstar_tng_ar[-1] < 0.45 : #Checking why the masses are too low for the Errani
#         ixs_high_mstar = np.append(ixs_high_mstar, int(len(frem_tng_ar) - 1))
#     elif subh_fstar_model_ar[-1] < 0.1 and subh_fstar_tng_ar[-1] > 0.2:
#         subh_ixs_low_mstar = np.append(subh_ixs_low_mstar, ix)
#         ixs_low_mstar = np.append(ixs_low_mstar, int(len(frem_tng_ar) - 1))
#     frem_ar = np.append(frem_ar, frem)
#     rperi_ar = np.append(rperi_ar, subh.rperi)
#     rapo_ar = np.append(rapo_ar, subh.rapo)




# '''
# Plot 1: This is the comparison between  dark matter masses
# '''
# ixs_high_mstar = ixs_high_mstar.astype('i')
# ixs_low_mstar = ixs_low_mstar.astype('i')
# ixs_unresol = ixs_unresol.astype('i')
# fig, ax = plt.subplots(figsize = (7, 6))
# # dummy = np.linspace(0.001, 1, 3)
# sc = ax.scatter(mmx_ar, mmx_tng_ar, c=subh_tinf_ar, cmap='viridis', marker='o', zorder = 20)
# for six in range(len(mmx_ar)):
#     ax.text(0.95 * mmx_ar[six], mmx_tng_ar[six], str(int(subh_id99_ar[six])), fontsize = 4, color = 'gray', ha = 'right', va = 'center')
# ax.scatter(mmx_ar[ixs_low_mstar], mmx_tng_ar[ixs_low_mstar], label = r'Low $f_{\bigstar}$', edgecolors = 'red', s = 100, facecolors = 'white', zorder = 10)
# ax.scatter(mmx_ar[ixs_high_mstar], mmx_tng_ar[ixs_high_mstar], label = r'High $f_{\bigstar}$', edgecolors = 'blue', s = 100, facecolors = 'white', zorder = 10)
# # ax.scatter(mmx_ar[ixs_unresol], mmx_tng_ar[ixs_unresol], label = r'Unresolved', marker = 'x', s = 100, color = 'black', zorder = 40)

# cbar = plt.colorbar(sc)
# # cbar.set_label(r'$\log M_{\rm{200, infall}}$')
# cbar.set_label(r'$t_{\rm{inf}}$')
# # cbar.set_label(r'$\log f_{\rm{\bigstar, inf}}$')
# # cbar.set_label(r'$M_{\rm{\bigstar, z=0}}$')

# dummy = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 3) 
# ax.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)

# ax.set_xlabel(r'$M_{\rm{mx}}$ from model')
# ax.set_ylabel(r'$M_{\rm{mx}}$ from TNG')
# # ax.set_xlim(-0.1, 1)
# # ax.set_ylim(-0.1, 1)
# ax.legend(fontsize = 8)
# plt.loglog()
# plt.tight_layout()
# plt.show()

# # def plot_fstar():

# '''
# Plot 1.1: This is to plot the remnant dark matter fraction
# '''
# ixs_high_mstar = ixs_high_mstar.astype('i')
# ixs_low_mstar = ixs_low_mstar.astype('i')
# ixs_unresol = ixs_unresol.astype('i')
# fig, ax = plt.subplots(figsize = (7, 6))
# dummy = np.linspace(0.001, 1, 3)
# sc = ax.scatter(frem_ar, frem_tng_ar, c=subh_tinf_ar, cmap='viridis', marker='o', zorder = 20)
# for six in range(len(mmx_ar)):
#     ax.text(0.95 * frem_ar[six], frem_tng_ar[six], str(int(subh_id99_ar[six])), fontsize = 4, color = 'gray', ha = 'right', va = 'center', zorder = 60)
# ax.scatter(frem_ar[ixs_low_mstar], frem_tng_ar[ixs_low_mstar], label = r'Low $f_{\bigstar}$', edgecolors = 'red', s = 100, facecolors = 'white', zorder = 10)
# ax.scatter(frem_ar[ixs_high_mstar], frem_tng_ar[ixs_high_mstar], label = r'High $f_{\bigstar}$', edgecolors = 'blue', s = 100, facecolors = 'white', zorder = 10)
# # ax.scatter(frem_ar[ixs_unresol], frem_tng_ar[ixs_unresol], label = r'Unresolved', marker = 'x', s = 100, color = 'black', zorder = 40)

# cbar = plt.colorbar(sc)
# # cbar.set_label(r'$\log M_{\rm{200, infall}}$')
# cbar.set_label(r'$t_{\rm{inf}}$')
# # cbar.set_label(r'$\log f_{\rm{\bigstar, inf}}$')
# # cbar.set_label(r'$M_{\rm{\bigstar, z=0}}$')
# ax.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)
# ax.set_xlabel(r'$M_{\rm{mx}}/M_{\rm{mx0}}$ from model')
# ax.set_ylabel(r'$M_{\rm{mx}}/M_{\rm{mx0}}$ from TNG')
# # ax.set_xlim(-0.1, 1)
# # ax.set_ylim(-0.1, 1)
# ax.legend(fontsize = 8)
# plt.loglog()
# plt.tight_layout()
# plt.show()


# '''
# Plot 1.2: This is to gauge the variation of remanant fraction w.r.t. rperi or rapo
# '''
# ixs_high_mstar = ixs_high_mstar.astype('i')
# ixs_low_mstar = ixs_low_mstar.astype('i')
# ixs_unresol = ixs_unresol.astype('i')

# fig, ax = plt.subplots(figsize = (7, 6))
# sc = ax.scatter(rapo_ar, frem_tng_ar/frem_ar, c=subh_tinf_ar, cmap='viridis', marker='o', zorder = 20)
# ax.scatter(rapo_ar[ixs_low_mstar], frem_tng_ar[ixs_low_mstar]/frem_ar[ixs_low_mstar], label = r'Low $f_{\bigstar}$', edgecolors = 'red', s = 100, facecolors = 'white', zorder = 10)

# cbar = plt.colorbar(sc)
# # cbar.set_label(r'$\log M_{\rm{200, infall}}$')
# cbar.set_label(r'$t_{\rm{inf}}$')
# ax.set_yscale('log')
# ax.set_xlabel(r'$r_{\rm{apo}}$ (kpc)')
# ax.set_ylabel(r'$M_{\rm{mx, TNG}}/M_{\rm{mx, model}}$')
# ax.axhline(1, ls = '--', color = 'gray')
# ax.legend(fontsize = 8)
# plt.tight_layout()
# plt.show()



# '''
# Plot 2: this is a comparison of the stellar fractions between the model and TNG
# '''
# ixs_high_mstar = ixs_high_mstar.astype('i')
# ixs_low_mstar = ixs_low_mstar.astype('i')
# ixs_unresol = ixs_unresol.astype('i')
# fig, ax = plt.subplots(figsize = (7, 6))
# dummy = np.linspace(0, 1, 3)
# sc = ax.scatter(subh_fstar_model_ar, subh_fstar_tng_ar, c=subh_tinf_ar, cmap='viridis', marker='o', zorder = 20)
# # sc = ax.scatter(subh_fstar_model_from_tngfrem_ar, subh_fstar_tng_ar, c=subh_tinf_ar, cmap='viridis', marker='o', zorder = 20)
# # sc = ax.scatter(1 - np.exp(-14.20 * subh_fdm_ar), subh_fstar_tng_ar, c=(subh_tinf_ar), cmap='viridis', marker='s', zorder = 20)
# cbar = plt.colorbar(sc)
# ax.scatter(subh_fstar_model_ar[ixs_low_mstar], subh_fstar_tng_ar[ixs_low_mstar], label = r'Low $f_{\bigstar}$', edgecolors = 'red', s = 100, facecolors = 'white', zorder = 10)
# ax.scatter(subh_fstar_model_ar[ixs_high_mstar], subh_fstar_tng_ar[ixs_high_mstar], label = r'High $f_{\bigstar}$', edgecolors = 'blue', s = 100, facecolors = 'white', zorder = 10)

# for six in range(len(mmx_ar)):
#     ax.text(0.95 * subh_fstar_model_ar[six], subh_fstar_tng_ar[six], str(int(subh_id99_ar[six])), fontsize = 4, color = 'gray', ha = 'right', va = 'center', zorder = 60)
# # cbar.set_label(r'$\log M_{\rm{200, infall}}$')
# # cbar.set_label(r'$t_{\rm{inf}}$')
# # cbar.set_label(r'$ \log(M_{\rm{mx}}/M_{\rm{mx0}})$ from model')
# cbar.set_label(r'$t_{\rm{inf}}$')
# # cbar.set_label(r'$M_{\rm{\bigstar, z=0}}$')
# ax.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)
# ax.set_xlabel(r'$M_{\rm{\bigstar, z=0}}/M_{\rm{\bigstar, max}}$ from model')
# ax.set_ylabel(r'$M_{\rm{\bigstar, z=0}}/M_{\rm{\bigstar, max}}$ from TNG')
# # ax.set_xlim(0, 1)
# # ax.set_ylim(0, 1)
# # ax.set_title(r'Model uses $M_{\rm{mx}}/M_{\rm{mx0}}$ from TNG', color = 'red')
# # plt.loglog()
# plt.tight_layout()
# plt.show()


# '''
# Plot 2.1: This is a plot of the stellar mass remaining in the TNG as compared to the one in the model
# '''
# ixs_high_mstar = ixs_high_mstar.astype('i')
# ixs_low_mstar = ixs_low_mstar.astype('i')
# ixs_unresol = ixs_unresol.astype('i')
# fig, ax = plt.subplots(figsize = (7, 6))
# # dummy = np.linspace(0, 1, 3)
# sc = ax.scatter(subh_mstar_model_ar, subh_mstar_tng_ar, c=np.log10(frem_ar), cmap='viridis', marker='o', zorder = 20)
# # sc = ax.scatter(subh_mstar_model_from_tngfrem_ar, subh_mstar_tng_ar, c=subh_tinf_ar, cmap='viridis', marker='o', zorder = 20)
# cbar = plt.colorbar(sc)
# ax.scatter(subh_mstar_model_ar[ixs_low_mstar], subh_mstar_tng_ar[ixs_low_mstar], label = r'Low $f_{\bigstar}$', edgecolors = 'red', s = 100, facecolors = 'white', zorder = 10)
# ax.scatter(subh_mstar_model_ar[ixs_high_mstar], subh_mstar_tng_ar[ixs_high_mstar], label = r'High $f_{\bigstar}$', edgecolors = 'blue', s = 100, facecolors = 'white', zorder = 10)
# # ax.scatter(subh_mstar_model_ar[ixs_unresol], subh_mstar_tng_ar[ixs_unresol], label = r'Unresolved', marker = 'x', s = 100, color = 'black', zorder = 40)
# for six in range(len(mmx_ar)):
#     ax.text(0.95 * subh_mstar_model_ar[six], subh_mstar_tng_ar[six], str(int(subh_id99_ar[six])), fontsize = 4, color = 'gray', ha = 'right', va = 'center', zorder = 60)
# # cbar.set_label(r'$\log M_{\rm{200, infall}}$')
# # cbar.set_label(r'$t_{\rm{inf}}$')
# cbar.set_label(r'$ \log(M_{\rm{mx}}/M_{\rm{mx0}})$ from model')
# # cbar.set_label(r'$t_{\rm{inf}}$')
# # cbar.set_label(r'$M_{\rm{\bigstar, z=0}}$')
# dummy = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 3) 
# ax.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)
# ax.set_xlabel(r'$M_{\rm{\bigstar, z=0}}$ from model')
# ax.set_ylabel(r'$M_{\rm{\bigstar, z=0}}$ from TNG')
# # ax.set_xlim(left = 1e5)
# ax.legend(fontsize = 8)
# # ax.set_title(r'Model uses $M_{\rm{mx}}/M_{\rm{mx0}}$ from TNG', color = 'red')

# # ax.set_ylim(0, 1)
# plt.loglog()
# plt.tight_layout()
# plt.show()



# '''
# Plot 3: Plot all the orbits for the subhalos considered
# '''
# pdf_file = outpath + "orbits_tng50_more1e9msun_epts.pdf"
# pdf_pages = PdfPages(pdf_file)
# subh_ixs_low_mstar = subh_ixs_low_mstar.astype('i')



# # for ix in tqdm(range(len(snap_if_ar))):
# # for ix in tqdm([30, 49, 56, 68, 84, 85, 89, 95]):
# # for ix in tqdm([83, 93, 94, 98, 100, 102, 103, 106]):
# for ix in tqdm(subh_ixs_low_mstar):
#     '''
#     This is to loop over all the surviving subhalos of big dataset with subhalos in between 1e8.5 and 1e9.5 Msun
#     ''' 
#     subh  = Subhalo(snap = snap_if_ar[ix], sfid = sfid_if_ar[ix], last_snap = 99)
#     try:
#         t = subh.get_orbit(merged = False, when_te = 'last') #after this, the subhalo has rperi, rapo and torb
#     except Exception as e:
#         pass
#     subh.plot_orbit_comprehensive(merged = False)
#     plt.tight_layout()
#     pdf_pages.savefig()
#     # plt.show()
#     plt.close()   

    
# pdf_pages.close()


# '''
# Plot 4: This plot is to compare the NFW fits from rotation curve and from the mass points
# Special highlight on the objects with low fstar to check if there is any big difference
# '''
# subh_ixs_low_mstar = subh_ixs_low_mstar.astype('i')


# fig, [ax, ax2, ax3] = plt.subplots(nrows = 1, ncols = 3, figsize = (15, 5))
# labvar = False

# for ix in tqdm(range(len(snap_if_ar))):
#     '''
#     This is to loop over all the surviving subhalos of big dataset with subhalos in between 1e8.5 and 1e9.5 Msun
#     ''' 
#     subh  = Subhalo(snap = snap_if_ar[ix], sfid = sfid_if_ar[ix], last_snap = 99 )
#     vmx, rmx, mmx = subh.get_mx_values(where = int(subh.snap)) #These are the values from the 
#     ax.plot(subh.rmx0, rmx, 'ko', ms = 3, alpha = 0.5)
#     ax2.plot(subh.vmx0, vmx, 'ko', ms = 3, alpha = 0.5)
#     ax3.plot(subh.mmx0, mmx, 'ko', ms = 3, alpha = 0.5)
#     if subh.rmx0 > 20 and rmx < 15:
#         print(ix)
#     # if ix in subh_ixs_low_mstar:
#     #     if not labvar:
#     #         ax.plot(subh.rmx0, rmx, 'ro', ms = 3, alpha = 0.5, label = r'low $f_\bigstar$')
#     #         ax2.plot(subh.vmx0, vmx, 'ro', ms = 3, alpha = 0.5, label = r'low $f_\bigstar$')
#     #         ax3.plot(subh.mmx0, mmx, 'ro', ms = 3, alpha = 0.5, label = r'low $f_\bigstar$')
#     #         labvar = True
#     #     else:
#     #         ax.plot(subh.rmx0, rmx, 'ro', ms = 3, alpha = 0.5)
#     #         ax2.plot(subh.vmx0, vmx, 'ro', ms = 3, alpha = 0.5) 
#     #         ax3.plot(subh.mmx0, mmx, 'ro', ms = 3, alpha = 0.5) 

# ax.set_xlabel(r'$r_{\rm{mx0}}$ (kpc) from RC')
# ax.set_ylabel(r'$r_{\rm{mx0}}$ (kpc) from two mass points')
# # ax.set_xlim(0, 8)
# # ax.set_ylim(0, 8)
# x_vals = np.array(ax.get_xlim())    
# ax.plot(x_vals, x_vals, 'k--', lw = 0.5)
# ax.legend(fontsize = 8)

# ax2.set_xlabel(r'$v_{\rm{mx0}}$ (kpc) from RC')
# ax2.set_ylabel(r'$v_{\rm{mx0}}$ (kpc) from two mass points')
# # ax.set_xlim(0, 8)
# # ax.set_ylim(0, 8)
# x_vals = np.array(ax.get_xlim())    
# ax2.plot(x_vals, x_vals, 'k--', lw = 0.5)
# ax2.legend(fontsize = 8)

# plt.tight_layout()
# plt.show()

# '''
# Plot 5: This is to plot the ratio of Rh to rmx0 to check if the outliers that we have are showing up due to some reason
# '''

# fig, ax = plt.subplots(figsize = (5, 5))


# for ix in tqdm(range(len(snap_if_ar))):
#     '''
#     This is to loop over all the surviving subhalos of big dataset with subhalos in between 1e8.5 and 1e9.5 Msun
#     ''' 
#     subh  = Subhalo(snap = snap_if_ar[ix], sfid = sfid_if_ar[ix], last_snap = 99 )
#     # subh.get_infall_properties()

#     Rh_tng_max = subh.get_rh(where = 'max')/np.sqrt(2)
#     Rh_model = subh.get_rh0byrmx0() * subh.rmx0 #This is the assumption
#     # if subh.get_rh0byrmx0() == 0.5:
#     #     print('Jai')
#     ax.plot(Rh_model/subh.rmx0, Rh_tng_max/subh.rmx0, 'ko', ms = 3.5)
#     if ix in subh_ixs_low_mstar:
#         ax.plot(Rh_model/subh.rmx0, Rh_tng_max/subh.rmx0, 'ro', ms = 3.5)
#     # if ix in [30, 49, 56, 68, 84, 85, 89, 95]: 
#     #     ax.plot(Rh_model, Rh_tng_max, 'ro')
#     #     if ix == 30:ax.plot(Rh_model, Rh_tng_max, 'ro', label = r'Low $f_{\bigstar}$')
#     # elif ix in [83, 93, 94, 98, 100, 102, 103, 106]:
#     #     ax.plot(Rh_model, Rh_tng_max, 'bo')
#     #     if ix == 83: ax.plot(Rh_model, Rh_tng_max, 'bo', label = r'High $f_{\bigstar}$')

# ax.set_xlabel(r'$R_{\rm{h, inf}}/r_{\rm{mx0}}$ model (kpc)')
# ax.set_ylabel(r'$R_{\rm{h, max\, M_\bigstar}}/r_{\rm{mx0}}$ from TNG (kpc)')
# # ax.set_xlim(0, 8)
# # ax.set_ylim(0, 8)
# x_vals = np.array(ax.get_xlim())    
# ax.plot(x_vals, x_vals, 'k--', lw = 0.5)
# ax.legend(fontsize = 8)
# plt.tight_layout()
# plt.show()



# '''
# Plot 6: This is to plot the stellar mass vs dark matter mass in half light radius to check if the outliers that we have are showing up due to some reason
# '''

# fig, ax = plt.subplots(figsize = (5, 5))


# for ix in tqdm(range(len(snap_if_ar))):
#     '''
#     This is to loop over all the surviving subhalos of big dataset with subhalos in between 1e8.5 and 1e9.5 Msun
#     ''' 
#     subh  = Subhalo(snap = snap_if_ar[ix], sfid = sfid_if_ar[ix], last_snap = 99 )

#     Mstar_rh = subh.get_mstar(where = 'max', how = 'rh')
#     Mdm_rh = subh.get_mdm(where = 'max', how = 'rh')
#     ax.plot(Mstar_rh, Mdm_rh, 'ko', ms = 3.5)
#     if ix in subh_ixs_low_mstar:
#             ax.plot(Mstar_rh, Mdm_rh, 'ro', ms = 3.5)


# ax.set_xlabel(r'$M_{\rm{\bigstar}}(<r_h)\,\rm{(M_\odot)}$')
# ax.set_ylabel(r'$M_{\rm{DM}}(<r_h)\,\rm{(M_\odot)}$')
# x_vals = np.array(ax.get_xlim())    
# ax.plot(x_vals, x_vals, 'k--', lw = 0.5)
# ax.legend(fontsize = 8)
# plt.loglog()
# plt.tight_layout()
# plt.show()




# # '''
# # Plot 3: This is Mmx and 
# # '''

# # for ix in tqdm([30, 49, 56, 68, 84, 85, 89, 95]):
# #     subh  = Subhalo(snap = snap_if_ar[ix], sfid = sfid_if_ar[ix])
# #     subh.get_mass_profiles(where = 'max', plot = True)
# #     exp_check = ExponentialProfile(subh.get_mstar(where = 'max', how = 'total'), np.sqrt(2) * subh.get_rh0byrmx0() * subh.rmx0) #This is an instance to check the exponential profile
# #     left, right = plt.gca().get_xlim()
# #     rpl = np.logspace(np.log10(left), np.log10(right), 100)
# #     plt.plot(rpl, exp_check.mass(rpl), label = 'Expoential assumed', color = 'salmon', ls = ':')
# #     plt.legend()
# #     pdf_pages2.savefig()
# #     plt.show()
# #     plt.close()


# """
# ================================================================
# MERGED SUBHALOS
# ================================================================
# """
# '''
# Plot 0.1: This is to plot all the half mass radius of stars according to TNG, and the one assumed in the model
# '''

# fig, ax = plt.subplots(figsize = (5, 5))


# for ix in tqdm(range(len(msh_snap))):
#     '''
#     This is to loop over all the surviving subhalos of big dataset with subhalos in between 1e8.5 and 1e9.5 Msun
#     ''' 
#     subh  = Subhalo(snap = int(msh_snap[ix]), sfid = int(msh_sfid[ix]), last_snap = msh_merger_snap[ix])
#     # subh.get_infall_properties()

#     Rh_tng_max = subh.get_rh(where = 'max')/np.sqrt(2)
#     Rh_model = subh.get_rh0byrmx0() * subh.rmx0 #This is the assumption
#     ax.plot(Rh_model, Rh_tng_max, 'ko')

# ax.set_xlabel(r'$R_{\rm{h, inf}}$ model (kpc)')
# ax.set_ylabel(r'$R_{\rm{h, max\, M_\bigstar}}$ from TNG (kpc)')
# ax.set_xlim(0, 8)
# ax.set_ylim(0, 8)
# x_vals = np.array(ax.get_xlim())    
# ax.plot(x_vals, x_vals, 'k--', lw = 0.5)
# ax.legend(fontsize = 8)
# plt.tight_layout()
# plt.show()


# '''
# Plot 0.2: This is to plot the mass of the ones with no stellar mass remaining
# This is to compare with the exponential profile and test if it is a good enough approximation for the stellar profile
# '''
# pdf_file2 = outpath + "mass profiles_tng50_more1e9msun_tsah.pdf"
# pdf_pages2 = PdfPages(pdf_file2)

# for ix in tqdm(range(len(snap_if_ar))):
# # for ix in tqdm([30, 49, 56, 68, 84, 85, 89, 95]):
#     subh  = Subhalo(snap = snap_if_ar[ix], sfid = sfid_if_ar[ix])
#     subh.get_mass_profiles(where = 'max', plot = True)
#     exp_check = ExponentialProfile(subh.get_mstar(where = 'max', how = 'total'), np.sqrt(2) * subh.get_rh0byrmx0() * subh.rmx0) #This is an instance to check the exponential profile
#     left, right = plt.gca().get_xlim()
#     rpl = np.logspace(np.log10(left), np.log10(right), 100)
#     plt.plot(rpl, exp_check.mass(rpl), label = 'Expoential assumed', color = 'salmon', ls = ':')
#     plt.legend()
#     pdf_pages2.savefig()
#     # plt.show()
#     plt.close()

# pdf_pages2.close()





# '''
# Thie big one, loop which evolves our subhalos
# MERGING SUBHALOS
# '''
# ctr = 0
# skipped_ixs = np.zeros(0)
# frem_ar = np.zeros(0) #this is the remnant masss fraction from the SAM
# mmx_ar = np.zeros(0)
# frem_tng_ar = np.zeros(0)
# mmx_tng_ar = np.zeros(0)
# subh_m200_ar = np.zeros(0) #This is the virial mass at infall for the subhalos
# subh_mstar_99_ar = np.zeros(0) #stellar mass at z = 0 for the subhalos being considered
# subh_tinf_ar = np.zeros(0) #This is the array of infall times of the subhalo
# subh_fstar_model_ar = np.zeros(0) #This is the array of stellar mass fraction remaining in the model
# subh_fstar_tng_ar = np.zeros(0)
# subh_fstar_model_from_tngfrem_ar = np.zeros(0) #This is the array of stellar mass fraction that is remaining assuming Mmx/Mmx0 from TNG
# subh_mstar_model_from_tngfrem_ar = np.zeros(0) #This is the array of stellar mass that is remaining assuming Mmx/Mmx0 from TNG
# subh_mstar_model_ar = np.zeros(0) #this is the stellar mass remaining at the end for the model
# subh_mstar_tng_ar = np.zeros(0) #This is the stellar mass remaining at the end for TNG
# subh_parlen_99_ar = np.zeros(0) #This is the length of particles at z = 0
# subh_fbar_tng_ar = np.zeros(0) #This is the ratio of stars to dark matter in 2Rh for the subhalos at infall
# subh_fdm_ar = np.zeros(0) #This is the total remnant dark matter mass to test the Smith+16 paper results
# subh_id99_ar = np.zeros(0) #this is the array of SubfindIDs at z = 0 for the subhalos for labeling
# subh_rh_model_ar = np.zeros(0)

# subh_rh_initial_ar = np.zeros(0) #inital rh of the galaxies
# subh_mstar_initial_ar = np.zeros(0) #Initial stellar masses of the galaxies


# # Both the following indices are for the frem_ar, etc. (there are some subhalos that are missed for having an Unbound Orbit)
# ixs_low_mstar = np.zeros(0) #This is the list of indices of the subhalos that have low stellar mass than that in TNG
# ixs_high_mstar = np.zeros(0) #This is the list of indices which have high mstar in the model.
# ixs_high_rh = np.zeros(0, dtype = int)
# subh_ixs_high_rh = np.zeros(0, dtype = int)
# '''
# Run this loop for doing everything after evolution
# '''



# for ix in tqdm(range(len(msh_snap))):
#     '''
#     This is to loop over all the merging subhalos of big dataset with subhalos in between 1e8.5 and 1e9.5 Msun
#     '''
#     if ix in [5, 9, 14, 22]: continue #takes lot of time to get compiled
#     # if ix>25: break

#     subh  = Subhalo(snap = int(msh_snap[ix]), sfid = int(msh_sfid[ix]), last_snap = int(msh_merger_snap[ix])) #these are at infall

#     # if ix in [30, 49, 56, 68, 84, 85, 89, 95]: 
    
#     # subh.get_infall_properties()
#     # t = subh.get_orbit(merged = False, when_te = 'last') #after this, the subhalo has rperi, rapo and torb
#     # t = subh.get_orbit(merged = True, when_te = 'last') #after this, the subhalo has rperi, rapo and torb
#     try:
#         t = subh.get_orbit(merged = True, when_te = 'last') #after this, the subhalo has rperi, rapo and torb
#     except Exception as e:
#         # print(e)
#         ctr = ctr + 1
#         skipped_ixs = np.append(skipped_ixs, ix)
#         continue

#     try:
#         subh_rh_initial_ar = np.append(subh_rh_initial_ar, subh.get_rh(where = 'max'))
#     except ValueError:
#         continue

#     subh_mstar_initial_ar = np.append(subh_mstar_initial_ar, subh.mstar)

#     # subh_fbar_tng_ar = np.append(subh_fbar_tng_ar, subh.get_mstar(where = int(subh.snap), how = '2rh')/ subh.get_mdm(where = int(subh.snap), how = '2rh'))
#     # subh_fdm_ar = np.append(subh_fdm_ar, subh.get_mdm(where = subh.last_snap, how = 'total') / subh.get_mdm(where = int(snap_if_ar[ix]), how = 'total'))
#     # subh_mstar_99 = subh.get_mstar(where=subh.last_snap, how = 'total') #99 is not at snap 99 always
#     # subh_mstar_99_ar = np.append(subh_mstar_99_ar, subh_mstar_99)
#     # subh_id99_ar = np.append(subh_id99_ar, subh.get_sfid(where = subh.last_snap))
#     # subh_tinf_ar = np.append(subh_tinf_ar, all_ages[snap_if_ar[ix]])
#     fields = ['SubhaloMassInRadType', 'SubhaloGrNr', 'SnapNum', 'GroupNsubs', 'SubhaloPos', 'Group_R_Crit200', 'Group_M_Crit200', 'SubhaloVel']
#     # subh_m200_ar = np.append(subh_m200_ar, subh.get_m200(where = int(snap_if_ar[ix] - 1))) #Append only when there are no errors
#     try:
#         vmxf, rmxf, mmxf, vdf, rhf, mstarf = subh.get_model_values(float(all_ages[subh.snap]), t) #FIXME: Some orbits are not unbound as galpy reports
#     except Exception as e:
#         print(e)
#         continue
#     # print(vmxf, rmxf, mmxf, vdf, rhf, mstarf)
#     if rhf > subh_rh_initial_ar[-1]:
#         subh_ixs_high_rh = np.append(subh_ixs_high_rh, int(len(subh_mstar_initial_ar) - 1))
#         ixs_high_rh = np.append(ixs_high_rh, ix)

#     mmx_ar = np.append(mmx_ar, mmxf)
#     frem = mmxf/subh.mmx0
#     # frem_tng = subh.get_tng_values(where = subh.last_snap)[2]/subh.mmx0
#     # mmx_tng_ar = np.append(mmx_tng_ar, frem_tng * subh.mmx0)
#     # frem_tng_ar = np.append(frem_tng_ar, frem_tng)
#     subh_mstar_model_ar = np.append(subh_mstar_model_ar, mstarf)
#     subh_rh_model_ar = np.append(subh_rh_model_ar, rhf)
#     subh_fstar_model_ar = np.append(subh_fstar_model_ar, subh_mstar_model_ar[-1]/subh.mstar)

#     # subh_mstar_model_from_tngfrem_ar = np.append(subh_mstar_model_from_tngfrem_ar, subh.get_starprops_model(frem = frem_tng)[0])
#     # subh_fstar_model_from_tngfrem_ar = np.append(subh_fstar_model_from_tngfrem_ar, subh_mstar_model_from_tngfrem_ar[-1]/subh.mstar)

#     # subh_mstar_tng_ar = np.append(subh_mstar_tng_ar, subh_mstar_99)
#     # subh_fstar_tng_ar = np.append(subh_fstar_tng_ar, subh_mstar_99 / subh.get_mstar(where = 'max', how = 'total'))
#     # subh_parlen_99_ar = np.append(subh_parlen_99_ar, subh.get_len(where = subh.last_snap, which = 'dm') + subh.get_len(where = subh.last_snap, which = 'stars'))
#     # if subh_parlen_99_ar[-1] < 3000:
#     #     ixs_unresol = np.append(ixs_unresol, int(len(frem_tng_ar) - 1))
#     # if subh_fstar_model_ar[-1] > 0.8 and subh_fstar_tng_ar[-1] < 0.45 : #Checking why the masses are too low for the Errani
#     #     ixs_high_mstar = np.append(ixs_high_mstar, int(len(frem_tng_ar) - 1))
#     # elif subh_fstar_model_ar[-1] < 0.1 and subh_fstar_tng_ar[-1] > 0.2:
#     #     subh_ixs_low_mstar = np.append(subh_ixs_low_mstar, ix)
#     #     ixs_low_mstar = np.append(ixs_low_mstar, int(len(frem_tng_ar) - 1))
#     frem_ar = np.append(frem_ar, frem)




# '''
# Plot 1: Plotting the Stellar vs Dark matter masses for the subhalos that got merged
# '''
# fig, ax = plt.subplots(figsize = (6, 5))
# ax.scatter(mmx_ar, subh_mstar_model_ar, marker = 'o', color = 'black', alpha = 0.5, s = 3)
# ax.set_xlabel(r'$M_{\rm{mx}}\,(M_\odot)$')
# ax.set_ylabel(r'$M_{\rm{\bigstar}}\,(M_\odot)$')
# ax.set_title('From model')
# plt.loglog()
# plt.tight_layout()
# plt.show()


# '''
# Plot 2: Plotting half light radius vs the stellar mass for the merged subhalos
# '''
# fig, ax = plt.subplots(figsize = (6, 5))
# # for ix in range(len(subh_mstar_model_ar)):
# #     ax.plot([subh_mstar_model_ar[ix], subh_mstar_initial_ar[ix]], [1e3 * subh_rh_model_ar[ix], 1e3 * subh_rh_initial_ar[ix]], alpha = 0.5, lw = 0.3, color = 'gray')
# ax.scatter(subh_mstar_model_ar, 1e3 * subh_rh_model_ar, marker = 'o', color = 'black', alpha = 0.5, s = 3, label = 'z=0 from model')
# # ax.scatter(subh_mstar_initial_ar, 1e3 * subh_rh_initial_ar, marker = 'o', color = 'red', alpha = 0.5, s = 3, label = 'initial')
# ax.set_xlabel(r'$M_{\rm{star}}\,\rm{(M_\odot)}$')
# ax.set_ylabel(r'$R_h\,\rm{(pc)}$')
# ax.set_title('160 subhalos merging into central of FoF0 - TNG50-1', fontsize = 10)
# ax.legend(fontsize = 8)
# plt.loglog()
# plt.tight_layout()
# plt.show()




# '''
# Plot 3: Plotting the orbits of the subhalos that get merged
# '''


# pdf_file = outpath + "merged_orbits_tng50_more1e9msun_epts.pdf"
# pdf_pages = PdfPages(pdf_file)
# # subh_ixs_low_mstar = subh_ixs_low_mstar.astype('i')



# for ix in tqdm(range(len(msh_sfid))):
# # for ix in tqdm([30, 49, 56, 68, 84, 85, 89, 95]):
# # for ix in tqdm([83, 93, 94, 98, 100, 102, 103, 106]):
# # for ix in tqdm(ixs_high_rh):
#     '''
#     This is to loop over all the surviving subhalos of big dataset with subhalos in between 1e8.5 and 1e9.5 Msun
#     ''' 
#     # if ix != 3:
#     #     continue
#     if ix in [9, 14, 22]: #weird index, FIXME: #8 Check later
#         continue
#     # if ix > 25:
#     #     break
#     subh  = Subhalo(snap = int(msh_snap[ix]), sfid = int(msh_sfid[ix]), last_snap = msh_merger_snap[ix]) #these are at infall
    
#     try:
#         t = subh.get_orbit(merged = True, when_te = 'last') #after this, the subhalo has rperi, rapo and torb
#         # print(t)
#     except Exception as e:
#         print(e)
#         pass
#     subh.plot_orbit_comprehensive(merged = True)
#     plt.tight_layout()
#     pdf_pages.savefig()
#     # plt.show()
#     plt.close()   

    
# pdf_pages.close()


