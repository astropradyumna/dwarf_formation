
import numpy as np
import matplotlib.pyplot as plt 
import os 
os.environ["USE_LZMA"] = "0"
import pandas as pd
# from errani_plus_tng_subhalo import Subhalo
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


fig, ax = plt.subplots(figsize = (6, 6))
fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2)

rpl = np.logspace(-1.4, 0, 100)


for ix in tqdm(range(len(snap_if_ar))):
    subh = TNG_Subhalo(snap = snap_if_ar[ix], sfid = sfid_if_ar[ix], last_snap = 99)
    mstar = max(subh.tree['SubhaloMassType'][:, 4][subh.tree['SnapNum'] >= subh.snap]) * 1e10 / h
    vmx_ar = np.zeros(0)
    rmx_ar = np.zeros(0)
    for snap in range(subh.snap, 100):
        try:
            vmx, rmx, mmx = subh.get_mx_values(where = snap, typ = 'star_dominated')
            vmx_ar = np.append(vmx_ar, vmx)
            rmx_ar = np.append(rmx_ar, rmx)
        except Exception as e:
            print(e)
            continue
            # vmx, rmx, mmx = subh.get_mx_values(where = snap, typ = 'dm_dominated')
            # vmx_ar = np.append(vmx_ar, vmx)
            # rmx_ar = np.append(rmx_ar, vmx)
        ax.plot(np.log10(rmx_ar[vmx_ar>1e-2 * vmx_ar[0]]/rmx_ar[0]), np.log10(vmx_ar[vmx_ar>1e-2 * vmx_ar[0]]/vmx_ar[0]), color = 'lightblue', lw = 0.2, alpha = 0.1)

ax.plot(np.log10(rpl), np.log10(get_vmxbyvmx0(rpl)), ls = '--', color = 'black')
ax.set_xlabel(r'$\log_{10} r_{\rm{mx}}/r_{\rm{mx0}}$')
ax.set_ylabel(r'$\log_{10} v_{\rm{mx}}/v_{\rm{mx0}}$')
plt.tight_layout()
plt.show()
    



    