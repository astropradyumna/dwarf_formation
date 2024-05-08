# from errani_plus_tng_subhalo import Subhalo
import numpy as np
from tqdm import tqdm
# import galpy
import IPython
import illustris_python as il
from matplotlib.backends.backend_pdf import PdfPages
import sys
import ast
import h5py
import pandas as pd
from joblib import Parallel, delayed #This is to parallelize the code
from errani_plus_tng_subhalo import Subhalo
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# from subhalo_profiles import ExponentialProfile, NFWProfile
# import warnings
# from colossus.cosmology import cosmology
# from colossus.halo import concentration

# cosmology.setCosmology('planck18')

# Suppress the lzma module warning
# warnings.filterwarnings("ignore", category=UserWarning, module="pandas.compat")
# warnings.simplefilter(action='ignore', category=FutureWarning)

filepath = '/rhome/psadh003/bigdata/tng50/tng_files/'
ages_df = pd.read_csv(filepath + 'ages_tng.csv', comment = '#')

all_snaps = np.array(ages_df['snapshot'])
all_redshifts = np.array(ages_df['redshift'])
all_ages = np.array(ages_df['age(Gyr)'])
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



merged_df = pd.read_csv(filepath + fof_str + '_sh_merged_after_z3_tng50_1_everything.csv')

msh_sfid = merged_df['SubfindID']
msh_sfid = np.array([s.strip('[]') for s in msh_sfid], dtype = int) #subfind ID at infall
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




# df = pd.read_csv(outpath + 'merged_evolved_fof0_everything.csv', delimiter = ',')
df = pd.read_csv(outpath + fof_str +'_merged_evolved_everything.csv', delimiter = ',', low_memory=False)  
df = df.applymap(convert_to_float)
main_sfid = np.array(df['sfid_if_ar'])
main_snap = np.array(df['snap_if_ar'])
mbpid_ar = np.array(df['mbpid_ar'])
mbpidp_ar = np.array(df['mbpidp_ar']) #MBP ID of one snapshot before 


df1 = pd.read_csv(outpath + fof_str +'_merged_evolved_wmbp_everything.csv', delimiter = ',', low_memory=False)  
df1 = df1.applymap(convert_to_float)
main_sfid1 = np.array(df1['sfid_if_ar'])
main_snap1 = np.array(df1['snap_if_ar'])

star_ids = np.load(this_fof_path+'star_ids.npy')
star_pos = np.load(this_fof_path+'star_pos.npy')

dm_ids = np.load(this_fof_path+'dm_ids.npy')
dm_pos = np.load(this_fof_path+'dm_pos.npy')







scount = 0
pdf_file = outpath + "orbits_no_mbp.pdf"
pdf_pages = PdfPages(pdf_file)

for ix in range(len(main_sfid)): 
    if scount == 5: #We will be looking for 10 subhalos that are not in the wmbp file
        break

    jx = np.where((main_sfid[ix] == main_sfid1) & (main_snap[ix] == main_snap1))[0]
    if jx.size != 0:
        continue #This is the case where subhalos is found in the wmbp file
    else:
        
        scount += 1
        kx = np.where((main_sfid[ix] == msh_sfid) & (main_snap[ix] == msh_snap))[0] #Looking for the subhalo in merger file
        subh  = Subhalo(snap = int(msh_snap[kx]), sfid = int(msh_sfid[kx]), last_snap = int(msh_merger_snap[kx]), central_sfid_99=central_sfid_99) #these are at infall
        mstar_infall = subh.get_mstar(where = int(subh.snap), how = 'total')
        print(f'This subhalo has a stellar mass of {mstar_infall} at infall')
        #We not have to download the cutout file for this subhalo
        merger_shid = int(subh.tree['SubfindID'][subh.tree['SnapNum'] == subh.last_snap])
        subh.download_data(shsnap_ar = int(msh_merger_snap[kx]), shid_ar = int(msh_merger_sfid[kx]))
        #After downloading data, lets open the file
        filename = filepath + 'cutout_files/cutout_'+str(int(msh_merger_sfid[kx]))+'_'+str(int(msh_merger_snap[kx]))+'.hdf5'
        f = h5py.File(filename, 'r') #This is to read the cutout file
        if mbpid_ar[ix] in f['PartType1']['ParticleIDs']:
            print('This subhalo has MBP which is a DM particle')
            # f['PartType'][np.where(mbpid_ar[ix] == f['PartType1']['ParticleIDs'])[0]]
        elif mbpid_ar[ix] in f['PartType4']['ParticleIDs']:
            print('This subhalo has MBP which is a star particle')

        this_subh_dmid_ar = np.array(f['PartType1']['ParticleIDs'], dtype = int)
        
        if 'PartType4' in f.keys():
            this_subh_starid_ar = np.array(f['PartType4']['ParticleIDs'], dtype = int)
            this_subh_allid_ar = np.append(this_subh_dmid_ar, this_subh_starid_ar) #This is the array of all the particle IDs in the subhalo
        else:
            this_subh_allid_ar = this_subh_dmid_ar


        if (mbpid_ar[ix] in star_ids) or (mbpid_ar[ix] in dm_ids): #Checking if the MBP is in the FoF because there looks like there is an issue
            print('There is some issue with misc.py!!')

        subh.plot_orbit_comprehensive()
        plt.tight_layout()
        pdf_pages.savefig()


        

        pos_ar = np.zeros(0)
        popix_ar = np.zeros(0)

        def get_positions(lx):
            '''
            This function is for parallelizing the process of finding the positions of the subhalos
            '''
            pos = [None]
            # pos2 = [None]
            index = np.where(np.isin(star_ids, this_subh_allid_ar[lx]))[0]
            # print(index)
            if len(index) == 1: pos = star_pos[index][0]
            if len(index) == 0:
                index = np.where(np.isin(dm_ids, this_subh_allid_ar[lx]))[0]
                if len(index) == 1: pos = dm_pos[index][0]

            if len(pos) == 3:
                return np.array(pos)
            elif len(pos) == 0:
                return None




        results = Parallel(n_jobs=32, pre_dispatch='1.5*n_jobs')(delayed(get_positions)(lx) for lx in tqdm(range(len(this_subh_allid_ar))))
        # len_before = len(results)
        for lx in range(len(results)):
            if results[lx] is None:
                popix_ar = np.append(popix_ar, lx)
            else:
                if len(pos_ar) == 0:
                    pos_ar = results[lx].reshape(1, -1)
                else:
                    pos_ar = np.append(pos_ar, results[lx].reshape(1, -1), axis = 0)
                # pos_ar = np.append(pos_ar, results[ix].reshape(1, -1), axis = 0)
                
        # none_indices = [ix for ix, value in enumerate(results) if value is None]
        # results = [value for value in results if value is not None] #Getting rid of all the None entries
        # len_after = len(results)
        print(f'Number of particles being lost are {len(popix_ar)} out of {len(this_subh_allid_ar)}')



pdf_pages.close()












#Look for the subhalos from df in df1 and get the first ten missing subhalos
#For these missing subhalos, get the MBP ID. 
#Now, download the subhalo particle data at the last snapshot, and get the particle data type

