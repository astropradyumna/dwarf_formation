'''
This is to generate the plots that Rapha asked for
-o- Vmx/Vmx0 vs Rmx/Rmx0
-o- Mstar/Mstar0 vs Mmx/Mmx0
'''
import numpy as np
import matplotlib.pyplot as plt 
import os 
os.environ["USE_LZMA"] = "0"
import pandas as pd
from errani_plus_tng_subhalo import Subhalo
from tng_subhalo_and_halo import TNG_Subhalo
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
import h5py
from testing_errani import get_LbyL0

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
min_mstar = 1e3

ages_df = pd.read_csv(filepath + 'ages_tng.csv', comment = '#')

all_snaps = np.array(ages_df['snapshot'])
all_redshifts = np.array(ages_df['redshift'])
all_ages = np.array(ages_df['age(Gyr)'])


# id_df = pd.read_csv('errani_checking_dataset.csv', comment = '#')
id_df = pd.read_csv(filepath + 'errani_checking_dataset_more1e9msun.csv', comment = '#')

snap_if_ar = id_df['snap_at_infall']
sfid_if_ar = id_df['id_at_infall']
ms_by_mdm = id_df['ms_by_mdm']

def get_vmxbyvmx0(rmxbyrmx0, alpha = 0.4, beta = 0.65):
    '''
    This function is Eq. 4 in the paper. Expected tidal track for only DM which is NFW
    '''
    return 2**alpha*rmxbyrmx0**beta*(1 + rmxbyrmx0**2)**(-alpha)




# # fig, ax = plt.subplots(figsize = (6, 6))
# fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (12, 12))
# fig3, axn = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 6)) #This would be for the ratio of differences in TNG and model versus Mmx/Mmx0

# rpl = np.logspace(-1.4, 0, 100)


# for ix in tqdm(range(len(snap_if_ar))):
#     '''
#     Assuming that this is for FoF0 as of now. Loops over all the subhalos with cuts defined above.
#     '''
#     # if ix>10:
#     #     continue
#     subh = Subhalo(snap = snap_if_ar[ix], sfid = sfid_if_ar[ix], last_snap = 99, central_sfid_99 = 0)
#     Rh0byrmx0 = subh.get_rh0byrmx0() #This gives the initial Rh0/rmx0
#     mstar = max(subh.tree['SubhaloMassType'][:, 4][subh.tree['SnapNum'] >= subh.snap]) * 1e10 / h
#     vmx_ar = np.zeros(0)
#     rmx_ar = np.zeros(0)
#     mmx_ar = np.zeros(0)
#     mstar_ar = np.zeros(0)
#     Dms_tng_ar = np.zeros(0) #This is the change in TNG
#     Dms_model_ar = np.zeros(0) #This is the change in the model
#     for snap in range(subh.snap, 100, 3):
#         try:
#             os.remove(filepath + 'cutout_files/cutout_'+str(shid)+'_'+str(shsnap)+'.hdf5') #putting this remove because we did not have IDs before
#             vmx, rmx, mmx = subh.get_rot_curve(where = snap)
#             mmx_ar = np.append(mmx_ar, mmx)
#             vmx_ar = np.append(vmx_ar, vmx)
#             rmx_ar = np.append(rmx_ar, rmx)
#             mstar_ar = np.append(mstar_ar, subh.get_mstar(where = snap, how = 'total'))
#         except Exception as e:
#             print(e)
#             continue
#             # vmx, rmx, mmx = subh.get_mx_values(where = snap, typ = 'dm_dominated')
#             # vmx_ar = np.append(vmx_ar, vmx)
#             # rmx_ar = np.append(rmx_ar, vmx)
#         # ax.plot(np.log10(rmx_ar[vmx_ar>1e-2 * vmx_ar[0]]/rmx_ar[0]), np.log10(vmx_ar[vmx_ar>1e-2 * vmx_ar[0]]/vmx_ar[0]), color = 'lightblue', lw = 0.2, alpha = 0.1)
#         # Dms_tng_ar = np.append(Dms_tng_ar, mstar - mstar_ar[0])
#         # Dms_model_ar = np.append(Dms_model_ar, (get_LbyL0(mmx/mmx_ar[0], Rh0byrmx0) - 1)*mstar_ar[0]) #This is change in the model
#         Dms_tng_ar = np.append(Dms_tng_ar, mstar_ar[-1])
#         Dms_model_ar = np.append(Dms_model_ar, (get_LbyL0(mmx/mmx_ar[0], Rh0byrmx0))*mstar_ar[0]) #This is change in the model
        
#         if Rh0byrmx0 == 1/2:
#             ax1.plot(np.log10(mmx_ar/mmx_ar[0]), np.log10(mstar_ar/mstar_ar[0]), color = 'red', lw = 0.2, alpha = 0.1)
#         elif Rh0byrmx0 == 1/4:
#             ax2.plot(np.log10(mmx_ar/mmx_ar[0]), np.log10(mstar_ar/mstar_ar[0]), color = 'orange', lw = 0.2, alpha = 0.1)
#         elif Rh0byrmx0 == 1/8:
#             ax3.plot(np.log10(mmx_ar/mmx_ar[0]), np.log10(mstar_ar/mstar_ar[0]), color = 'skyblue', lw = 0.2, alpha = 0.1)
#         elif Rh0byrmx0 == 1/16:
#             ax4.plot(np.log10(mmx_ar/mmx_ar[0]), np.log10(mstar_ar/mstar_ar[0]), color = 'darkblue', lw = 0.2, alpha = 0.1)

#     axn.plot(np.log10(mmx_ar/mmx_ar[0]), np.log10(Dms_model_ar/Dms_tng_ar), color = 'gray', lw = 0.35, alpha = 0.5)


# # ax1.set_xlim(left = 1.1* ax1.get_xlim()[0])
# # ax2.set_xlim(left = 1.1* ax2.get_xlim()[0])
# # ax3.set_xlim(left = 1.1* ax3.get_xlim()[0])
# # ax4.set_xlim(left = 1.1* ax4.get_xlim()[0])
# # ax.plot(np.log10(rpl), np.log10(get_vmxbyvmx0(rpl)), ls = '--', color = 'black')
# # ax.set_xlabel(r'$\log_{10} r_{\rm{mx}}/r_{\rm{mx0}}$')
# # ax.set_ylabel(r'$\log_{10} v_{\rm{mx}}/v_{\rm{mx0}}$')
# mxpl1 = np.logspace(0.9*ax1.get_xlim()[0], 0, 25)
# mxpl2 = np.logspace(0.9* ax2.get_xlim()[0], 0, 25)
# mxpl3 = np.logspace(0.9* ax3.get_xlim()[0], 0, 25)
# mxpl4 = np.logspace(0.9* ax4.get_xlim()[0], 0, 25)
# ax1.plot(np.log10(mxpl1), np.log10(get_LbyL0(mxpl1, 1/2)), ls = '--', color = 'red', lw = 1.3)
# ax2.plot(np.log10(mxpl2), np.log10(get_LbyL0(mxpl2, 1/4)), ls = '--', color = 'orange', lw = 1.3)
# ax3.plot(np.log10(mxpl3), np.log10(get_LbyL0(mxpl3, 1/8)), ls = '--', color = 'skyblue', lw = 1.3)
# ax4.plot(np.log10(mxpl4), np.log10(get_LbyL0(mxpl4, 1/16)), ls = '--', color = 'darkblue', lw = 1.3)

# axn.axhline(y = 0, color = 'black', ls = '--')
# axn.set_xlabel(r'$\log_{10} M_{\rm{mx}}/M_{\rm{mx0}}$')
# axn.set_ylabel(r'$\log_{10}  M_{\bigstar,\rm{model}}/ M_{\bigstar,\rm{TNG}}$')
# fig3.tight_layout()
# fig3.savefig('tidal_tracks_Dmstar_ratio_vs_mmx.png')


# ax1.set_xlabel(r'$\log_{10} M_{\rm{mx}}/M_{\rm{mx0}}$')
# ax1.set_ylabel(r'$\log_{10} M_{\bigstar}/M_{\bigstar0}$')
# ax1.set_title(r'$R_{\rm{h0}}/r_{\rm{mx0}} = 1/2$')
# ax2.set_xlabel(r'$\log_{10} M_{\rm{mx}}/M_{\rm{mx0}}$')
# ax2.set_ylabel(r'$\log_{10} M_{\bigstar}/M_{\bigstar0}$')
# ax2.set_title(r'$R_{\rm{h0}}/r_{\rm{mx0}} = 1/4$')
# ax3.set_xlabel(r'$\log_{10} M_{\rm{mx}}/M_{\rm{mx0}}$')
# ax3.set_ylabel(r'$\log_{10} M_{\bigstar}/M_{\bigstar0}$')
# ax3.set_title(r'$R_{\rm{h0}}/r_{\rm{mx0}} = 1/8$')
# ax4.set_xlabel(r'$\log_{10} M_{\rm{mx}}/M_{\rm{mx0}}$')
# ax4.set_ylabel(r'$\log_{10} M_{\bigstar}/M_{\bigstar0}$')
# ax4.set_title(r'$R_{\rm{h0}}/r_{\rm{mx0}} = 1/16$')

# fig2.suptitle(r'Subhalos from FoF0 with $\log M_\bigstar \in [8.5, 9.5]$')

# fig2.tight_layout()
# fig2.savefig('tidal_tracks_mstar_vs_mmx.png')
    



# This part is only using the stars which existed at infall. Tring for the first time
fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (12, 12))
fig3, axn = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 6)) #This would be for the ratio of differences in TNG and model versus Mmx/Mmx0



for ix in tqdm(range(len(snap_if_ar))): #This loop runs over all the subhalos that we have
    '''
    Assuming that this is for FoF0 as of now. Loops over all the subhalos with cuts defined above.
    '''
    if ix > 19:
        continue
    subh = Subhalo(snap = snap_if_ar[ix], sfid = sfid_if_ar[ix], last_snap = 99, central_sfid_99 = 0)
    subh.get_star_energy_dist(plot=True)
    Rh0byrmx0 = subh.get_rh0byrmx0() #This gives the initial Rh0/rmx0
    mstar = max(subh.tree['SubhaloMassType'][:, 4][subh.tree['SnapNum'] >= subh.snap]) * 1e10 / h
    vmx_ar = np.zeros(0)
    rmx_ar = np.zeros(0)
    mmx_ar = np.zeros(0)
    mstar_ar = np.zeros(0)
    Dms_tng_ar = np.zeros(0) #This is the change in TNG
    Dms_model_ar = np.zeros(0) #This is the change in the model
    for snap in range(subh.snap, 100, 3): #Just for these snapshots, we have the particle IDs as well
        try:
            vmx, rmx, mmx = subh.get_rot_curve(where = snap)
            mmx_ar = np.append(mmx_ar, mmx)
            vmx_ar = np.append(vmx_ar, vmx)
            rmx_ar = np.append(rmx_ar, rmx)
            
        except Exception as e:
            print(e)
            continue
        
        if snap == subh.snap: #This would be the infall snapshot. Note all the IDs of the particles at this snapshot
            filename = filepath + 'cutout_files/cutout_'+str(subh.sfid)+'_'+str(subh.snap)+'.hdf5' #This woulld correspond to the infall 
            f = h5py.File(filename, 'r') #This is to read the cutout file
            infall_ids = f['PartType4']['ParticleIDs'] #These are the IDs of the stars at infall
        
        filename = filepath + 'cutout_files/cutout_'+str(subh.get_sfid(where = snap)[0])+'_'+str(snap)+'.hdf5' #This would be the filename for current snapshot
        f = h5py.File(filename, 'r')
        this_snap_ids = f['PartType4']['ParticleIDs']
        this_snap_masses = np.array(f['PartType4']['Masses'])
        common_indices = np.intersect1d(this_snap_ids, infall_ids, assume_unique=True, return_indices = True)[1]
        this_snap_mstar = np.sum(this_snap_masses[common_indices])
        mstar_ar = np.append(mstar_ar, this_snap_mstar) #Appending the stellar mass of only the infall stars at this snapshot
        Dms_tng_ar = np.append(Dms_tng_ar, mstar_ar[-1])
        Dms_model_ar = np.append(Dms_model_ar, (get_LbyL0(mmx/mmx_ar[0], Rh0byrmx0))*mstar_ar[0]) #This is change in the model

        if Rh0byrmx0 == 1/2:
            ax1.plot(np.log10(mmx_ar/mmx_ar[0]), np.log10(mstar_ar/mstar_ar[0]), color = 'red', lw = 0.2, alpha = 0.1)
        elif Rh0byrmx0 == 1/4:
            ax2.plot(np.log10(mmx_ar/mmx_ar[0]), np.log10(mstar_ar/mstar_ar[0]), color = 'orange', lw = 0.2, alpha = 0.1)
        elif Rh0byrmx0 == 1/8:
            ax3.plot(np.log10(mmx_ar/mmx_ar[0]), np.log10(mstar_ar/mstar_ar[0]), color = 'skyblue', lw = 0.2, alpha = 0.1)
        elif Rh0byrmx0 == 1/16:
            ax4.plot(np.log10(mmx_ar/mmx_ar[0]), np.log10(mstar_ar/mstar_ar[0]), color = 'darkblue', lw = 0.2, alpha = 0.1)

    axn.plot(np.log10(mmx_ar/mmx_ar[0]), np.log10(Dms_model_ar/Dms_tng_ar), color = 'gray', lw = 0.35, alpha = 0.5)
    

mxpl1 = np.logspace(0.9*ax1.get_xlim()[0], 0, 25)
mxpl2 = np.logspace(0.9* ax2.get_xlim()[0], 0, 25)
mxpl3 = np.logspace(0.9* ax3.get_xlim()[0], 0, 25)
mxpl4 = np.logspace(0.9* ax4.get_xlim()[0], 0, 25)
ax1.plot(np.log10(mxpl1), np.log10(get_LbyL0(mxpl1, 1/2)), ls = '--', color = 'red', lw = 1.3)
ax2.plot(np.log10(mxpl2), np.log10(get_LbyL0(mxpl2, 1/4)), ls = '--', color = 'orange', lw = 1.3)
ax3.plot(np.log10(mxpl3), np.log10(get_LbyL0(mxpl3, 1/8)), ls = '--', color = 'skyblue', lw = 1.3)
ax4.plot(np.log10(mxpl4), np.log10(get_LbyL0(mxpl4, 1/16)), ls = '--', color = 'darkblue', lw = 1.3)

ax1.set_ylim(bottom = -0.75)
ax2.set_ylim(bottom = -1)
ax3.set_ylim(bottom = -0.6)
# ax4.set_ylim(bottom = -0.6)

ax1.set_xlabel(r'$\log_{10} M_{\rm{mx}}/M_{\rm{mx0}}$')
ax1.set_ylabel(r'$\log_{10} M_{\bigstar}/M_{\bigstar0}$')
ax1.set_title(r'$R_{\rm{h0}}/r_{\rm{mx0}} = 1/2$')
ax2.set_xlabel(r'$\log_{10} M_{\rm{mx}}/M_{\rm{mx0}}$')
ax2.set_ylabel(r'$\log_{10} M_{\bigstar}/M_{\bigstar0}$')
ax2.set_title(r'$R_{\rm{h0}}/r_{\rm{mx0}} = 1/4$')
ax3.set_xlabel(r'$\log_{10} M_{\rm{mx}}/M_{\rm{mx0}}$')
ax3.set_ylabel(r'$\log_{10} M_{\bigstar}/M_{\bigstar0}$')
ax3.set_title(r'$R_{\rm{h0}}/r_{\rm{mx0}} = 1/8$')
ax4.set_xlabel(r'$\log_{10} M_{\rm{mx}}/M_{\rm{mx0}}$')
ax4.set_ylabel(r'$\log_{10} M_{\bigstar}/M_{\bigstar0}$')
ax4.set_title(r'$R_{\rm{h0}}/r_{\rm{mx0}} = 1/16$')

fig2.suptitle(r'Subhalos from FoF0 with $\log M_\bigstar \in [8.5, 9.5]$')

fig2.tight_layout()
fig2.savefig('tidal_tracks_mstar_vs_mmx.png')


axn.axhline(y = 0, color = 'black', ls = '--')
axn.set_xlabel(r'$\log_{10} M_{\rm{mx}}/M_{\rm{mx0}}$')
axn.set_ylabel(r'$\log_{10}  M_{\bigstar,\rm{model}}/ M_{\bigstar,\rm{TNG}}$')
axn.set_ylim(top = 1)
fig3.tight_layout()
fig3.savefig('tidal_tracks_Dmstar_ratio_vs_mmx.png')





# results = Parallel(n_jobs=32, pre_dispatch='1.5*n_jobs')(delayed(save_surviving_subhalos)(ix) for ix in tqdm(range(len(ssh_snap))))
    