'''
This code is to extract the Number of DM particles inside a given radius for a given FoF group
'''

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
star_mass = np.load(this_fof_path+'star_mass.npy')

star_dist = np.sqrt(np.sum(star_pos**2, axis=1))

dm_ids = np.load(this_fof_path+'dm_ids.npy')
dm_pos = np.load(this_fof_path+'dm_pos.npy')


dm_dist = np.sqrt(np.sum(dm_pos**2, axis=1))

rpl = np.logspace(1, 3.2, 100) #r = m

def get_Ndm(ix):
    ms = rpl[ix]
    return(len(dm_dist[dm_dist < ms]))

results = Parallel(n_jobs=32, pre_dispatch='1.5*n_jobs')(delayed(get_Ndm)(ix) for ix in tqdm(range(len(rpl))))


print('Ndm', results)


def get_Nstar(ix):
    ms = rpl[ix]
    return(np.sum(star_mass[star_dist < ms]))

results = Parallel(n_jobs=32, pre_dispatch='1.5*n_jobs')(delayed(get_Nstar)(ix) for ix in tqdm(range(len(rpl))))

print('Nstar', results)
# pos_ar = np.zeros(0)

# popix_ar = np.zeros(0)

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

# def get_positions(ix):
#     '''
#     This function is for parallelizing the process of finding the positions of the subhalos
#     '''
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
#         return posavg
#         # if len(pos_ar) == 0:
#         #     pos_ar = posavg.reshape(1, -1)
#         # else:
#         #     return posavg
#             # pos_ar = np.append(pos_ar, posavg.reshape(1, -1), axis = 0)
#     else:
#         # popix_ar = np.append(popix_ar, ix) #FIXME: #12 Some of the particles are not in the FoF0 particle file
#         return None


# len_before = len(results)

# for ix in range(len(results)):
#     if results[ix] is None:
#         popix_ar = np.append(popix_ar, ix)
#     else:
#         if len(pos_ar) == 0:
#             pos_ar = results[ix].reshape(1, -1)
#         else:
#             pos_ar = np.append(pos_ar, results[ix].reshape(1, -1), axis = 0)
        # pos_ar = np.append(pos_ar, results[ix].reshape(1, -1), axis = 0)
        
# none_indices = [ix for ix, value in enumerate(results) if value is None]
# results = [value for value in results if value is not None] #Getting rid of all the None entries
# len_after = len(results)
# print(f'Number of subhalos being lost are {len(popix_ar)} out of {len(mbpid_ar)}')


# df = df.drop(popix_ar)
# df['pos_f_ar'] = pos_ar.tolist()
# # print(pos_ar)
# df['dist_f_ar'] = np.sqrt(np.sum(pos_ar**2, axis=1))



# df.to_csv(outpath + fof_str + '_merged_evolved_wmbp_everything.csv', index = False) 



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





