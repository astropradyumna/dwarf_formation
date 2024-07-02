'''
Niusha's code, was for darkmatter cluster, but changing it to hpcc now
Initially trying with FoF0. If it works, we can try for other FoF groups
'''


import numpy as np
import illustris_python as il
import math
import pdb
import matplotlib.pyplot as plt
# import get_intersect_1d
import matplotlib.cm as cm
import h5py
from scipy import stats

# for dealing with the periodicity:
################
### Function ###
################
def fold_pos(x,Lbox):
    '''
    When distance is greater than Lbox/2, we take the smaller distance to account for the periodicity of the simulation box
    '''
    aux = x > Lbox / 2.
    x[aux] = x[aux] - Lbox
    return x
##################
def fold_pos2(x,Lbox):
    aux = x < - Lbox / 2.
    x[aux] = x[aux] + Lbox
    return x
##################

simbase = '/rhome/psadh003/bigdata/L35n2160TNG_fixed/output'

### load additional files

filein1 = '/bigdata/saleslab/psadh003/tng50/tng_files/time_table_TNG50.dat'
tt = np.loadtxt(filein1)
a_scale = tt[:,0] # scale factor at every snapshot
z_red = tt[:,1] # redshift at every snapshot
t_cosmic = tt[:,2] # time in Gyr at every snapshot


lbox = 35000
snpz0 = 99
h_small = 0.677

print('starting with catalog ...')

cat_halo = il.groupcat.loadHalos(simbase, snpz0, fields = ['Group_R_Crit200', 'Group_M_Crit200', 'GroupFirstSub'])

idf = cat_halo['GroupFirstSub']
mvir = cat_halo['Group_M_Crit200'] * 1.e10/h_small
rvir = cat_halo['Group_R_Crit200'] * 1/h_small

aux = mvir >= 1e8
mvir_group = mvir[aux]
idf_group = idf[aux]
rvir_group = rvir[aux]

cat_sub = il.groupcat.loadSubhalos(simbase, snpz0, fields = ['SubhaloGrNr','SubhaloMassType', 'SubhaloMassInRadType', 'SubhaloMassInHalfRadType','SubhaloHalfmassRadType', 'SubhaloPos'])

pos_sub = cat_sub['SubhaloPos']
r_h_sub = cat_sub['SubhaloHalfmassRadType'][:,4] * 1/h_small
grnr = cat_sub['SubhaloGrNr']
mstr_sub = cat_sub['SubhaloMassType'][:,4] * 1e10/h_small
mdm_sub = cat_sub['SubhaloMassType'][:,1] * 1e10/h_small #Tis is the dark matter mass

sfid = np.arange(len(grnr))

# for i in range (1):
i = 0 #This would be the FoF group number, this was one before, changing this to 0
grnr_group = i
print(i)
sfid_cen = idf_group[i]
print(sfid_cen)
r_h_cen = r_h_sub[sfid_cen]
print('r_h_cen: ', r_h_cen)
r_vir_cen = rvir_group[i]
print('r_vir_cen: ', r_vir_cen)


x_cen_group = pos_sub[sfid_cen,0]
y_cen_group = pos_sub[sfid_cen,1]
z_cen_group = pos_sub[sfid_cen,2]
#target subhalos(these are massive subhalos above 1e8 M_sun)

aux = (grnr == grnr_group) & (mstr_sub > 5e6) #5e6 #1e10
mstr_sub_target = mstr_sub[aux]
mstr_sub_target = np.delete(mstr_sub_target,0)
sfid_target = sfid[aux]
sfid_target = np.delete(sfid_target, 0)
mdm_sub_target = mdm_sub[aux]
mdm_sub_target = np.delete(mdm_sub_target, 0) #This is to get rid of the central subhalo

print('len of subhalos: ', len(mstr_sub_target))

x_pos_sat = []
y_pos_sat = []
z_pos_sat = []
mstr_sat = []

for j in range(len(sfid_target)):
    '''
    This is taking the particle data for each of the subhalos that exists in the TNG. 
    We cannot do this for our evolved subhalos. Think of some other way
    '''
    if mdm_sub_target[j] == 0:
        continue
    star = il.snapshot.loadSubhalo(simbase, snpz0, sfid_target[j], 'stars', fields = ['Coordinates', 'Masses', 'GFM_StellarFormationTime'])
    mass_star = star['Masses'] *1e10/h_small
    pos_star = star['Coordinates']
    a_at_formation_star = star['GFM_StellarFormationTime']
    
    aux = a_at_formation_star > 0
    mass_star = mass_star[aux]
    xstr = pos_star[aux,0]
    ystr = pos_star[aux,1]
    zstr = pos_star[aux,2]

    x_pos = xstr - x_cen_group
    y_pos = ystr - y_cen_group
    z_pos = zstr - z_cen_group

    x_pos = fold_pos(x_pos, lbox) 
    y_pos = fold_pos(y_pos, lbox) 
    z_pos = fold_pos(z_pos, lbox) 

    x_pos = fold_pos2(x_pos, lbox) /h_small
    y_pos = fold_pos2(y_pos, lbox) /h_small
    z_pos = fold_pos2(z_pos, lbox) /h_small

    x_pos_sat.extend(x_pos)
    y_pos_sat.extend(y_pos)
    z_pos_sat.extend(z_pos)
    mstr_sat.extend(mass_star)

    # print('len of x_pos_sat: ', len(x_pos_sat))

pos_stack = np.stack((x_pos_sat, y_pos_sat, z_pos_sat, mstr_sat), axis = 1)
    #replace an appropriate address and uncomment
outpath  = '/rhome/psadh003/bigdata/tng50/output_files/'

np.save(outpath + 'fof0_plot.npy', pos_stack)

