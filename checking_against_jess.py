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




    # Read a .npy file
# jess_data = np.load('/rhome/psadh003/shared/for_prady/sfid_163_snap_99_energy_info.npy', allow_pickle = True)
# print(jess_data.shape)

subh = Subhalo(snap = 99, sfid = 163, last_snap = 99, central_sfid_99 = 0)
subh.get_star_energy_dist(plot=True)