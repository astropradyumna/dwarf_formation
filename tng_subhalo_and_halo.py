import pandas as pd 
import numpy as np
import illustris_python as il
import requests
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import warnings
from scipy.interpolate import UnivariateSpline
warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy.optimize import curve_fit
from scipy.integrate import quad
from functools import partial
import sys
from scipy.optimize import fsolve
from joblib import Parallel, delayed #This is to parallelize the code



h = 0.6774
mass_dm = 3.07367708626464e-05 * 1e10/h #This is for TNG50-1
G = 4.5390823753559603e-39 #This is in kpc, Msun and seconds

# filepath = '/home/psadh003/tng50/tng_files/'
# baseUrl = 'https://www.tng-project.org/api/TNG50-1/'
# headers = {"api-key":"894f4df036abe0cb9d83561e4b1efcf1"}
# basePath = '/mainvol/jdopp001/L35n2160TNG_fixed/output'

filepath = '/rhome/psadh003/bigdata/tng50/tng_files/'
outpath  = '/rhome/psadh003/bigdata/tng50/output_files/'
plotpath  = '/rhome/psadh003/bigdata/tng50/output_plots/'
baseUrl = 'https://www.tng-project.org/api/TNG50-1/'
headers = {"api-key":"894f4df036abe0cb9d83561e4b1efcf1"}
basePath = '/rhome/psadh003/bigdata/L35n2160TNG_fixed/output'

ages_df = pd.read_csv(filepath + 'ages_tng.csv', comment = '#')

all_snaps = np.array(ages_df['snapshot'])
all_redshifts = np.array(ages_df['redshift'])
all_ages = np.array(ages_df['age(Gyr)'])



'''
Following is the dataset of the entire list of subhalos which infalled after z = 3 and survived
'''
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

'''
Following is the dataset of the entire list of subhalos which infalled after z = 3 and merged
'''
# merged_df = pd.read_csv(filepath + 'sh_merged_after_z3_tng50_1_everything.csv')

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




class TNG_Subhalo():
    '''
    This is a general class for ANY TNG subhalo to calcaulate generic stuff
    '''
    def __init__(self, sfid, snap, last_snap):
        self.sfid = sfid
        self.snap = snap 
        self.last_snap = last_snap
        if self.last_snap != 99:
            self.merged = True
        else:
            self.merged = False

        fields = ['SubhaloMassInRadType', 'SubhaloGrNr', 'SnapNum', 'GroupNsubs', 'SubhaloPos', 'Group_R_Crit200', 
        'Group_M_Crit200', 'SubhaloVel', 'SubhaloHalfmassRadType', 'SubhaloMassType', 'SubhaloLenType', 'SubhaloMassInHalfRadType', 'SubhaloVmaxRad', 
        'SubhaloMassInMaxRadType', 'SubhaloIDMostbound', 'SubhaloVelDisp', 'GroupFirstSub', 'SubhaloVmax', 'SubfindID']
        if not self.merged:     
            temp_tree = il.sublink.loadTree(basePath, self.snap, self.sfid, fields = ['SnapNum', 'SubfindID'], onlyMDB = True)
            sfid_99 = temp_tree['SubfindID'][0] #FIXME: This only works for a surviving subhalo
            self.tree = il.sublink.loadTree(basePath, 99, sfid_99, fields = fields, onlyMPB = True)
        else: #If it merged
            tree = il.sublink.loadTree(basePath, self.snap, self.sfid, fields = fields, onlyMDB = True) #From this we obtain all the desencdants of the subhalo at infall
            tree.pop('count') #removing a useless key from the dictionary
            snaps_temp = tree['SnapNum']
            sfids_temp = tree['SubfindID']

            merger_index = np.where(snaps_temp == self.last_snap)[0][-1] #this is the index of the merger in the tree
            self.tree = {key: value[merger_index:] for key, value in tree.items()} #new tree which only runs from final existing snapshot to the infall snapshot
            # infall_ix = np.where((msh_snap == snap) & (msh_sfid == sfid))[0] #This is to get the index of the current subhalo in the merger dataframe
            # # print(infall_ix)
            # msh_last_snap = int(msh_merger_snap[infall_ix]) #This is the infall snapshot
            # msh_last_sfid = int(msh_merger_sfid[infall_ix]) #This is the infall subfind ID

            # tree = il.sublink.loadTree(basePath, msh_last_snap, msh_last_sfid, fields = fields, onlyMPB = True) #Getting all the progenitors from the last snapshot of survival
            # tree.pop('count') #removing a useless key from the dictionary
            # snaps_temp = tree['SnapNum']
            # sfids_temp = tree['SubfindID']
            # msh_if_ix_tree = np.where((snaps_temp == snap) & (sfids_temp == sfid))[0].item() #This is the infall index in the tree
            # self.tree = {key: value[0:msh_if_ix_tree+1] for key, value in tree.items()} #new tree which only runs from final existing snapshot to the infall snapshot


    def __where_to_snap(self, where):
        '''
        This is an internal functions used to calculate the snapshot from the where input to different functions in use.
        '''
        if where == 'last':
            snap_wanted = self.last_snap
        if where == 'max': #maximum stellar mass after the infall
            snaps_after_infall = np.flip(np.arange(self.snap, self.last_snap + 1)) #These are the snapshots after infall
            snap_arr = self.tree['SnapNum']
            mstar_ar = self.tree['SubhaloMassInRadType'][:, 4] * 1e10/h
            ms_after_infall = np.zeros(0)
            for s in snaps_after_infall:
                ms_after_infall = np.append(ms_after_infall, mstar_ar[snap_arr == s])     
            snap_wanted = snaps_after_infall[np.argmax(ms_after_infall)]
        if isinstance(where, int):
            if 0<= where <= self.last_snap:
                snap_wanted = where 
        return snap_wanted


    def get_mx_values(self, where, typ = 'dm_dominated'):
        '''
        This is to get all mx type of values by fitting an NFW to the mass profile
        '''
        snap_wanted = self.__where_to_snap(where)
        rh = self.get_rh(where)
        mdm_rh = self.get_mdm(where, how = 'rh')
        mdm_2rh = self.get_mdm(where, how = '2rh')
        mdm_vmax = self.get_mdm(where, how = 'vmax')
        r_vmax = self.get_rh(where, how = 'vmax')
        dmrh = self.get_rh(where, how = 'dmrh')
        mdmrh = self.get_mdm(where)/2

        if typ == 'dm_dominated':
            r1 = r_vmax
            m1 = mdm_vmax
            r2 = dmrh
            m2 = mdmrh
        elif typ == 'star_dominated':
            r1 = 2 * rh
            m1 = mdm_2rh
            r2 = dmrh
            m2 = mdmrh

        
        # print(rh, mdm_rh, mdm_2rh)
        def nfw_mass(r, lrhos, lrs):
            rhos = 10**lrhos
            rs = 10**lrs
            mass =  4 * np.pi * rhos * rs ** 3 * (np.log(1 + (r/rs))  -  r / (r + rs))
            return mass

        def simul_func(params):
            lrhos, lrs = params
            result = np.array([np.log10(nfw_mass(r1, lrhos, lrs)) - np.log10(float(m1)), np.log10(nfw_mass(r2, lrhos, lrs)) - np.log10(float(m2))]).ravel()
            # result = np.array([np.log10(nfw_mass(rh, lrhos, lrs)) - np.log10(float(mdm_rh)), np.log10(nfw_mass(r_vmax, lrhos, lrs)) - np.log10(float(mdm_vmax)), 
            #                    np.log10(nfw_mass(2 * rh, lrhos, lrs)) - np.log10(float(mdm_2rh))]).ravel()
            # print(f"Params: {params}, Result: {result}")
            return result

        input_values = [2, 0]
        lrhos, lrs = fsolve(simul_func, input_values)
        
        # print(simul_func([lrhos, lrs]))

        rhos = 10**lrhos
        rs = 10**lrs
        vmx = 1.65 * rs * np.sqrt(G * rhos) #kpc/s
        rmx = 2.16 * rs #kpc
        mmx = rmx * vmx**2 / G #Msun
        return vmx * 3.086e16, rmx, mmx #km/s, kpc, Msun
    
    
    

    
    def get_z(self, where):
        snap_wanted = self.__where_to_snap(where)
        return float(all_redshifts[snap_wanted])
        

    def get_m200(self, where = 99):
        '''
        This function returns the M_{200} at a given snapshot
        '''
        snap_wanted = self.__where_to_snap(where)
        m200_tree = self.tree['Group_M_Crit200']*1e10/h
        m200 = m200_tree[snap_wanted == self.tree['SnapNum']]
        return m200

    def get_mstar(self, where, how = '2rh'):
        '''
        This function returns the stellar mass at a given snapshot, if it survives
        FIXME: Currently assumes that the subhalo survives. Requires work such that this works for everything
        '''
        snap_wanted = self.__where_to_snap(where)
        if how == '2rh': mstar_tree = self.tree['SubhaloMassInRadType'][:, 4]*1e10/h
        elif how == 'total': mstar_tree = self.tree['SubhaloMassType'][:, 4]*1e10/h
        elif how == 'rh': mstar_tree = self.tree['SubhaloMassInHalfRadType'][:, 4]*1e10/h
        mstar = mstar_tree[snap_wanted == self.tree['SnapNum']]
        return mstar
    
    def get_mdm(self, where, how = 'total'):
        '''
        This is to get the dark matter mass from the TNG
        '''
        snap_wanted = self.__where_to_snap(where)
        if how == '2rh': mdm_tree = self.tree['SubhaloMassInRadType'][:, 1]*1e10/h
        elif how == 'total': mdm_tree = self.tree['SubhaloMassType'][:, 1]*1e10/h
        elif how == 'rh': mdm_tree = self.tree['SubhaloMassInHalfRadType'][:, 1]*1e10/h
        elif how == 'vmax': mdm_tree = self.tree['SubhaloMassInMaxRadType'][:, 1]*1e10/h
        mdm = mdm_tree[snap_wanted == self.tree['SnapNum']]
        return mdm
    

    def get_mtot(self, where, how = 'total'):
        '''
        This is to get the sum of all the kinds of particles from TNG in a given way
        '''
        snap_wanted = self.__where_to_snap(where)
        if how == '2rh': mtot_tree = np.sum(self.tree['SubhaloMassInRadType'], axis = 1)*1e10/h
        elif how == 'total': mtot_tree = np.sum(self.tree['SubhaloMassType'], axis = 1)*1e10/h
        elif how == 'rh': mtot_tree = np.sum(self.tree['SubhaloMassInHalfRadType'], axis = 1)*1e10/h
        elif how == 'vmax': mtot_tree = np.sum(self.tree['SubhaloMassInMaxRadType'], axis = 1)*1e10/h
        mtot = mtot_tree[snap_wanted == self.tree['SnapNum']]
        return mtot
    
    
    def get_mbpid(self, where):
        '''
        This function is to get the MBP ID of the subhalos that get merged
        '''
        snap_wanted = self.__where_to_snap(where)
        mbpid = self.tree['SubhaloIDMostbound'][snap_wanted == self.tree['SnapNum']]
        return mbpid
    

    def get_dist_from_cen(self, where):
        '''
        This function gets us the distance from the central subhalo
        '''
        snap_wanted = self.__where_to_snap(where)
        subh_x, subh_y, subh_z = self.tree['SubhaloPos'][snap_wanted == self.tree['SnapNum']][0]/h/(1 + all_redshifts[snap_wanted])
        cen_id = self.tree['GroupFirstSub'][snap_wanted == self.tree['SnapNum']]
        cen_x, cen_y, cen_z = il.groupcat.loadSingle(basePath, snap_wanted, subhaloID = cen_id)['SubhaloPos']/h/(1 + all_redshifts[snap_wanted])
        dist = np.sqrt((cen_x - subh_x) ** 2 + (cen_y - subh_y) ** 2 +  (cen_z - subh_z) ** 2)
        return dist
    
    def get_vmax(self, where):
        '''
        This function gives the vmax of the subhalo at a given time
        '''
        snap_wanted = self.__where_to_snap(where)
        vmax = self.tree['SubhaloVmax'][snap_wanted == self.tree['SnapNum']]
        return vmax

    

    def get_rh(self, where, how = 'rh'):
        '''
        This function is to calculate the half light radius. This is the 3D radius. Scale it with 3/4 wherever required for the projected radius
        FIXME: Currently assumes that the subhalo survives. Requires work such that this works for everything
        '''
        snap_wanted = self.__where_to_snap(where)
        if how == 'rh':
            Rh_tree = self.tree['SubhaloHalfmassRadType'][:, 4]
            # print(Rh_tree)
            # print(snap_wanted)
            rh = Rh_tree[snap_wanted == self.tree['SnapNum']]
        elif how == 'vmax': #shouldn't have ben named rh but to save changing this everywhere, not changing function name
            Rh_tree = self.tree['SubhaloVmaxRad']
            # print(Rh_tree)
            # print(snap_wanted)
            rh = Rh_tree[snap_wanted == self.tree['SnapNum']]
        elif how == 'dmrh':
            Rh_tree = self.tree['SubhaloHalfmassRadType'][:, 1]
            # print(Rh_tree)
            # print(snap_wanted)
            rh = Rh_tree[snap_wanted == self.tree['SnapNum']]
        if np.array(rh).shape[0] == 0:
            raise ValueError(f'Rh value not found for {self.sfid} at snap {self.snap}, max mstar snap is {snap_wanted}')
        return rh/(1 + all_redshifts[snap_wanted])/h #Setting it to right units, kpc
    

    def get_vd(self, where):
        snap_wanted = self.__where_to_snap(where)
        vd = self.tree['SubhaloVelDisp'][snap_wanted == self.tree['SnapNum']]
        return vd

    def get_sfid(self, where):
        '''
        This functions returns the SubFind ID from the tree for a given snapshot
        '''
        snap_wanted = self.__where_to_snap(where)
        sfid = self.tree['SubfindID'][snap_wanted == self.tree['SnapNum']]
        return sfid

    def get_len(self, where, which):
        '''
        This function is to get the number of particles 
        '''
        snap_wanted = self.__where_to_snap(where)
        if which == 'dm': len_par = self.tree['SubhaloLenType'][:, 1][snap_wanted == self.tree['SnapNum']]
        elif which == 'stars': len_par = self.tree['SubhaloLenType'][:,4][snap_wanted == self.tree['SnapNum']]
        return len_par
    
    def get(self, path, params=None):
    # make HTTP GET request to path
        r = requests.get(path, params=params, headers=headers)

        # raise exception if response code is not HTTP SUCCESS (200)
        r.raise_for_status()

        if r.headers['content-type'] == 'application/json':
            return r.json() # parse json responses automatically

        if 'content-disposition' in r.headers:
            filename = r.headers['content-disposition'].split("filename=")[1]
            with open(filename, 'wb') as f:
                f.write(r.content)
            return filename # return the filename string

        return r

    def get_link(self, snap, id):
        '''
        Gives you the link for the subhalo
        '''
        return baseUrl+'snapshots/'+str(snap)+'/subhalos/'+str(id)+'/'

    def download_data(self, shsnap_ar, shid_ar):
        '''
        This function downloads the cutout files for the IDs and snaps mentioned in the array
        '''
        if isinstance(shsnap_ar, int):
            shsnap = shsnap_ar
            shid = shid_ar
            snap = self.get(baseUrl+'snapshots/'+str(shsnap)+'/')
            if not os.path.isfile(filepath + 'cutout_'+str(shid)+'_'+str(shsnap)+'.hdf5'): #if it is not downloaded alraedy
                subh_link = self.get_link(shsnap, shid)
                subh = self.get(subh_link, 'r')
                cutout_request = {'gas':'Coordinates,Masses','stars':'Coordinates,Masses,Velocities,ParticleIDs','dm':'Coordinates,ParticleIDs'}
                cutout = self.get(subh_link+"cutout.hdf5", cutout_request)
                os.rename('cutout_'+str(shid)+'.hdf5', filepath + 'cutout_files/cutout_'+str(shid)+'_'+str(shsnap)+'.hdf5') #Renaming the file to include the snapshot in name
        else:
            shsnap_ar = np.array(shsnap_ar, dtype = int)
            shid_ar = np.array(shid_ar, dtype = int)
            nobj = shsnap_ar.shape.item #number of objects to be downloaded
            for ix in tqdm(range(nobj)): #This loops over all the systems of interest
                shsnap = shsnap_ar[ix]
                shid = shid_ar[ix]
                snap = self.get(baseUrl+'snapshots/'+str(shsnap)+'/')
                if not os.path.isfile(fielpath + 'cutout_'+str(shid)+'_'+str(shsnap)+'.hdf5'): #if it is not downloaded alraedy
                    subh_link = self.get_link(shsnap, shid)
                    subh = self.get(subh_link, 'r')
                    cutout_request = {'gas':'Coordinates,Masses','stars':'Coordinates,Masses,ParticleIDs','dm':'Coordinates,ParticleIDs'}
                    cutout = self.get(subh_link+"cutout.hdf5", cutout_request)
                    os.rename('cutout_'+str(shid)+'.hdf5', filepath + 'cutout_files/cutout_'+str(shid)+'_'+str(shsnap)+'.hdf5') #Renaming the file to include the snapshot in name
        return None
    
    def get_mass_profiles(self, where = None, rl = 'linear', h = 0.6774, plot = False):
        '''
        Gets you the mass profiles for the ID and snap metioned
        '''
        if where == None: #If the input is none, then go ahead a nd use the default values
            shsnap = self.snap
            shid = self.sfid
        else: #If there is some input, use that to get the rotation curve
            shsnap = self.__where_to_snap(where)
            shid = int(self.tree['SubfindID'][self.tree['SnapNum'] == shsnap])
        # snap = get(baseUrl+'snapshots/'+str(shsnap)+'/')
        # z = snap['redshift']
        z = all_redshifts[shsnap]
        filename = filepath + 'cutout_files/cutout_'+str(shid)+'_'+str(shsnap)+'.hdf5'
        if not os.path.exists(filename):
            print(f'Downloading data for sfid: {int(shid)} and snap: {int(shsnap)}')
            self.download_data(int(shsnap), int(shid))
        f = h5py.File(filename, 'r') #This is to read the cutout file
        # subh_link = get_link(shsnap, shid)
        subh = il.groupcat.loadSingle(basePath,shsnap,subhaloID=shid) #This loads the subhalo position mentioned in the group catalog
        subh_pos_x = subh['SubhaloPos'][0]/h/(1+z)
        subh_pos_y = subh['SubhaloPos'][1]/h/(1+z)
        subh_pos_z = subh['SubhaloPos'][2]/h/(1+z)
        dm_coords = f['PartType1']['Coordinates'] #This is for individual DM particles
        dm_xcoord = dm_coords[:, 0]/h/(1+z) - subh_pos_x
        dm_ycoord = dm_coords[:, 1]/h/(1+z) - subh_pos_y
        dm_zcoord = dm_coords[:, 2]/h/(1+z) - subh_pos_z


        if 'PartType0' in f.keys():
            cg = 1 #telling that gas is present
            gas_coords = f['PartType0']['Coordinates']
            gas_xcoord = gas_coords[:, 0]/h/(1+z) - subh_pos_x
            gas_ycoord = gas_coords[:, 1]/h/(1+z) - subh_pos_y
            gas_zcoord = gas_coords[:, 2]/h/(1+z) - subh_pos_z
            gas_masses = np.array(f['PartType0']['Masses'])*1e10/h
        else:
            cg = 0 
            

        if 'PartType4' in f.keys():
            cs = 1 #telling that stars are present
            star_coords = f['PartType4']['Coordinates']
            star_xcoord = star_coords[:, 0]/h/(1+z) - subh_pos_x
            star_ycoord = star_coords[:, 1]/h/(1+z) - subh_pos_y
            star_zcoord = star_coords[:, 2]/h/(1+z) - subh_pos_z
            star_masses = np.array(f['PartType4']['Masses'])*1e10/h

            
        else:
            cs = 0

        dm_rad = np.sqrt(dm_xcoord**2 + dm_ycoord**2 + dm_zcoord**2)
        
        if cg == 1: gas_rad = np.sqrt(gas_xcoord**2 + gas_ycoord**2 + gas_zcoord**2)
        if cs == 1: star_rad = np.sqrt(star_xcoord**2 + star_ycoord**2 + star_zcoord**2)
        
        mass_arr_plot_cont = np.zeros(0)
        dm_mass_arr_cont = np.zeros(0)
        gas_mass_arr_cont = np.zeros(0)
        star_mass_arr_cont = np.zeros(0)
        
        if rl == 'log':
            rad_plot_cont = np.logspace(-1, np.log10(max(dm_rad)), 500)
        elif rl == 'outer':
            rad_plot_cont = np.linspace(1, max(dm_rad), 2000)
        else:
            rad_plot_cont = np.logspace(-1, np.log10(max(dm_rad)), 500)

        # print(rad_plot_cont)

        for rad in rad_plot_cont:
            if cg == 1 and cs == 1:
                Min = np.sum(star_masses[star_rad < rad]) + np.sum(gas_masses[gas_rad < rad])
            elif cg ==1 and cs == 0:
                Min = np.sum(gas_masses[gas_rad < rad])
            elif cg == 0 and cs == 1: 
                Min = np.sum(star_masses[star_rad < rad])
            else:
                Min = 0

            Min = Min + mass_dm*len(dm_rad[dm_rad< rad])
            mass_arr_plot_cont = np.append(mass_arr_plot_cont, Min)
            dm_mass_arr_cont= np.append(dm_mass_arr_cont, mass_dm*len(dm_rad[dm_rad< rad]))
            if cg == 1: gas_mass_arr_cont = np.append(gas_mass_arr_cont, np.sum(gas_masses[gas_rad < rad]))
            if cs ==1: star_mass_arr_cont = np.append(star_mass_arr_cont, np.sum(star_masses[star_rad < rad]))
        

        if plot:
            fig, ax = plt.subplots()
            ax.plot(rad_plot_cont, mass_arr_plot_cont, 'k', label='Total Mass')
            ax.plot(rad_plot_cont, dm_mass_arr_cont, 'k--', label='DM')
            if cg == 1: ax.plot(rad_plot_cont, gas_mass_arr_cont, 'b', label='Gas')
            ax.plot(rad_plot_cont, star_mass_arr_cont, 'r', label='Stars')
            ax.axvline(self.get_rh(where = int(self.snap)), color =  'r', ls = '--', label = r'$r_{h, \bigstar}$')
            # ax.set_xlim(0, min(max(dm_rad), 10 * self.get_Rh())
            ax.set_xlabel(r'Radius $[kpc]$')
            ax.set_ylabel(r'Mass $[M_\odot]$')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend(fontsize=8)
            ax.set_ylim(bottom = 100 * mass_dm)
            ax.set_xlim(left = 10 * min(dm_rad))
            ax.set_title('Mass profiles for subhalo ID = '+str(shid)+' at z = '+str(round(z, 2)), fontsize = 10)
            plt.tight_layout()
            plt.savefig(plotpath + 'mass_profiles/mass_profile_'+str(shid)+'_'+str(shsnap)+'.png')

        if cg == 1 and cs == 1: #If there is gas element
            return cg, rad_plot_cont, mass_arr_plot_cont, dm_mass_arr_cont, star_mass_arr_cont, gas_mass_arr_cont
        elif cg == 0 and cs == 1: 
            return cg, rad_plot_cont, mass_arr_plot_cont, dm_mass_arr_cont, star_mass_arr_cont, np.zeros(len(rad_plot_cont))
        elif cg == 1 and cs == 0: 
            return cg, rad_plot_cont, mass_arr_plot_cont, dm_mass_arr_cont, np.zeros(len(rad_plot_cont)), gas_mass_arr_cont
        else: 
            return cg, rad_plot_cont, mass_arr_plot_cont, dm_mass_arr_cont, np.zeros(len(rad_plot_cont)), np.zeros(len(rad_plot_cont))


    def get_rot_curve(self, where = None, plot = False):
        '''
        Plots the rotation curve 

        Args:
        where: the snapshot where the rotation curve is required. 


        Returns:
        vmx, rmx, mmx (The velocity, radius and mass inside of the radius where the velocity curve of the DM peaks)
        '''
        if where == None: #If the input is none, then go ahead a nd use the default values
            shsnap = self.snap
            shid = self.sfid
        else: #If there is some input, use that to get the rotation curve
            shsnap = self.__where_to_snap(where)
            shid = self.tree['SubfindID'][self.tree['SnapNum'] == shsnap]

        cg, rad_plot_cont, mass_arr_plot_cont, dm_mass_arr_cont, star_mass_arr_cont, gas_mass_arr_cont = self.get_mass_profiles(where)
        

        circ_vel_plot_cont = np.sqrt(G*mass_arr_plot_cont/(rad_plot_cont))*3.086e+16
        dm_vel_plot_cont = np.sqrt(G*dm_mass_arr_cont/(rad_plot_cont))*3.086e+16
        if cg == 1: gas_vel_plot_cont = np.sqrt(G*gas_mass_arr_cont/(rad_plot_cont))*3.086e+16
        star_vel_plot_cont = np.sqrt(G*star_mass_arr_cont/(rad_plot_cont))*3.086e+16
        

        # snap = get(baseUrl+'snapshots/'+str(shsnap)+'/')
        # z = snap['redshift']
        z = all_redshifts[shsnap]

        filename = filepath + 'cutout_files/cutout_'+str(int(shid))+'_'+str(int(shsnap))+'.hdf5' 
        f = h5py.File(filename, 'r')
        # subh_link = get_link(shsnap, shid)
        subh = il.groupcat.loadSingle(basePath,shsnap,subhaloID=shid) #This loads the subhalo position mentioned in the group catalog
        subh_pos_x = subh['SubhaloPos'][0]/h/(1+z)
        subh_pos_y = subh['SubhaloPos'][1]/h/(1+z)
        subh_pos_z = subh['SubhaloPos'][2]/h/(1+z)
        dm_coords = f['PartType1']['Coordinates']
        dm_xcoord = dm_coords[:, 0]/h/(1+z) - subh_pos_x
        dm_ycoord = dm_coords[:, 1]/h/(1+z) - subh_pos_y
        dm_zcoord = dm_coords[:, 2]/h/(1+z) - subh_pos_z
        dm_rad = np.sqrt(dm_xcoord**2 + dm_ycoord**2 + dm_zcoord**2)

        ix_max_vel = np.argmax(dm_vel_plot_cont)
        vmx_subh = dm_vel_plot_cont[ix_max_vel]
        rmx_subh = rad_plot_cont[ix_max_vel]
        # Mmx_subh = mass_dm*len(dm_rad[dm_rad<= rmx_subh]) 
        Mmx_subh = rmx_subh * (3.24078e-17 * vmx_subh)**2 / G

        if plot:
            fig, ax = plt.subplots()
            ax.plot(rad_plot_cont[np.argmax(dm_vel_plot_cont)], max(dm_vel_plot_cont), 'go')
            ax.plot(rad_plot_cont, circ_vel_plot_cont, 'k', label='Total Mass')
            ax.plot(rad_plot_cont, dm_vel_plot_cont, 'k--', label='DM')
            if cg == 1: ax.plot(rad_plot_cont, gas_vel_plot_cont, 'b', label='Gas')
            ax.plot(rad_plot_cont, star_vel_plot_cont, 'r', label='Stars')
            ax.set_xlim(0, min(max(dm_rad), 2.5*rmx_subh))
            ax.set_xlabel(r'Radius $[kpc]$')
            ax.set_ylabel(r'Circular Velocity $[km/s]$')
            ax.legend(fontsize=12)
            ax.set_title('Rotation Curve for subhalo ID = '+str(shid)+' at z = '+str(round(z, 2)), fontsize = 10)
            plt.tight_layout()

        return vmx_subh, rmx_subh, Mmx_subh


    def get_vhrh(self, shsnap, shid):
        cg, rad_plot_cont, mass_arr_plot_cont, dm_mass_arr_cont, star_mass_arr_cont, gas_mass_arr_cont = self.get_mass_profiles(shsnap, shid)
        circ_vel_plot_cont = np.sqrt(G*mass_arr_plot_cont/(rad_plot_cont))*3.086e+16
        dm_vel_plot_cont = np.sqrt(G*dm_mass_arr_cont/(rad_plot_cont))*3.086e+16 #km/s?
        if cg == 1: gas_vel_plot_cont = np.sqrt(G*gas_mass_arr_cont/(rad_plot_cont))*3.086e+16
        star_vel_plot_cont = np.sqrt(G*star_mass_arr_cont/(rad_plot_cont))*3.086e+16
        # snap = get(baseUrl+'snapshots/'+str(shsnap)+'/')
        # z = snap['redshift']
        z = all_redshifts[shsnap]
        filename = filepath + 'cutout_files/cutout_'+str(shid)+'_'+str(shsnap)+'.hdf5' 
        f = h5py.File(filename, 'r')
        # subh_link = get_link(shsnap, shid)
        subh = il.groupcat.loadSingle(basePath,shsnap,subhaloID=shid) #This loads the subhalo position mentioned in the group catalog
        subh_pos_x = subh['SubhaloPos'][0]/h/(1+z)
        subh_pos_y = subh['SubhaloPos'][1]/h/(1+z)
        subh_pos_z = subh['SubhaloPos'][2]/h/(1+z)
        dm_coords = f['PartType1']['Coordinates']
        dm_xcoord = dm_coords[:, 0]/h/(1+z) - subh_pos_x
        dm_ycoord = dm_coords[:, 1]/h/(1+z) - subh_pos_y
        dm_zcoord = dm_coords[:, 2]/h/(1+z) - subh_pos_z
        dm_rad = np.sqrt(dm_xcoord**2 + dm_ycoord**2 + dm_zcoord**2)

        rh = subh['SubhaloHalfmassRadType'][4]/h/(1+z) #This is the 3D half mass radius for the stars.
        vh = np.interp(rh, rad_plot_cont, dm_vel_plot_cont)

        # if vh == 0:
        #   print(dm_vel_plot_cont)

        return vh, rh
    
    def get_rh0byrmx0_in(self):
        '''
        This is a function to calculate the initial rh0burmx0 for the subhalo
        '''
        
        Rh0 = self.get_rh(where = 'max')*3./4 #FIXME: This needs to accound for the subhalos without measured Rh
        rmx0 = self.get_mx_values(where = int(self.snap))[1]
        return Rh0/rmx0


    def get_star_energy_dist(self, where = None, rl = 'linear', h = 0.6774, plot = False, spherical = True):
        '''
        This is without the assumption of spherical symmetry. 
        We calculate the potential energy by taking the pairwise potential energy of stellar particles
        '''
        if where == None: #If the input is none, then go ahead a nd use the default values
            shsnap = self.snap
            shid = self.sfid
        else: #If there is some input, use that to get the rotation curve
            shsnap = self.__where_to_snap(where)
            shid = int(self.tree['SubfindID'][self.tree['SnapNum'] == shsnap])

        z = all_redshifts[shsnap]
        filename = filepath + 'cutout_files/cutout_'+str(shid)+'_'+str(shsnap)+'.hdf5'
        f = h5py.File(filename, 'r') #This is to read the cutout file
        subh = il.groupcat.loadSingle(basePath,shsnap,subhaloID=shid) #This loads the subhalo position mentioned in the group catalog
        subh_pos_x = subh['SubhaloPos'][0]/h/(1+z)
        subh_pos_y = subh['SubhaloPos'][1]/h/(1+z)
        subh_pos_z = subh['SubhaloPos'][2]/h/(1+z)
        subh_vel_x = subh['SubhaloVel'][0]
        subh_vel_y = subh['SubhaloVel'][1]
        subh_vel_z = subh['SubhaloVel'][2]
        dm_coords = f['PartType1']['Coordinates'] #This is for individual DM particles
        dm_xcoord = dm_coords[:, 0]/h/(1+z) - subh_pos_x
        dm_ycoord = dm_coords[:, 1]/h/(1+z) - subh_pos_y
        dm_zcoord = dm_coords[:, 2]/h/(1+z) - subh_pos_z


        if 'PartType0' in f.keys():
            cg = 1 #telling that gas is present
            gas_coords = f['PartType0']['Coordinates']
            gas_xcoord = gas_coords[:, 0]/h/(1+z) - subh_pos_x
            gas_ycoord = gas_coords[:, 1]/h/(1+z) - subh_pos_y
            gas_zcoord = gas_coords[:, 2]/h/(1+z) - subh_pos_z
            gas_masses = np.array(f['PartType0']['Masses'])*1e10/h
        else:
            cg = 0 

        star_coords = f['PartType4']['Coordinates']
        star_xcoord = star_coords[:, 0]/h/(1+z) - subh_pos_x
        star_ycoord = star_coords[:, 1]/h/(1+z) - subh_pos_y
        star_zcoord = star_coords[:, 2]/h/(1+z) - subh_pos_z
        star_vels = f['PartType4']['Velocities']
        star_xvel = star_vels[:, 0]*np.sqrt(1 / (1+z)) - subh_vel_x
        star_yvel = star_vels[:, 1]*np.sqrt(1 / (1+z)) - subh_vel_y
        star_zvel = star_vels[:, 2]*np.sqrt(1 / (1+z)) - subh_vel_z
        star_masses = np.array(f['PartType4']['Masses'])*1e10/h


        

        def get_potential_energy(x, y, z):
            '''
            This returns the potential energy of the star at the given position in (km/s)^2
            '''
            G1 = 4.30092e-6 #kpc/Msun * (km/s)^2
            pe_dm = mass_dm * np.sum(1/np.sqrt((x - dm_xcoord)**2 + (y - dm_ycoord)**2 + (z - dm_zcoord)**2))
            pe_gas = 0
            if cg == 1: pe_gas = np.sum(gas_masses/np.sqrt((x - gas_xcoord)**2 + (y - gas_ycoord)**2 + (z - gas_zcoord)**2))
            pe_stars = 0
            for ix in range(len(star_masses)):
                if star_xcoord[ix] == x and star_ycoord[ix] == y and star_zcoord[ix] == z:
                    continue
                pe_stars = pe_stars + star_masses[ix]/np.sqrt((x - star_xcoord[ix])**2 + (y - star_ycoord[ix])**2 + (z - star_zcoord[ix])**2)
            pe = -G1 * (pe_dm + pe_gas + pe_stars)
            return pe

        def get_total_energy(ix):
            ke = 0.5 * (star_xvel[ix]**2 + star_yvel[ix]**2 + star_zvel[ix]**2)
            pe = get_potential_energy(star_xcoord[ix], star_ycoord[ix], star_zcoord[ix])
            te = pe + ke
            return te


        te_ar = Parallel(n_jobs=32, pre_dispatch='1.5*n_jobs')(delayed(get_total_energy)(ix) for ix in (range(len(star_masses))))
        phi0 = phi0_tng = get_potential_energy(0, 0, 0) #This would be the potential energy at the center of the subhalo, no spherical assumption
        # phi0 =  -4.67 * self.vmx0**2 #This is the potential energy at the center of the subhalo assuming NFW profile, for testing
        eps_ar = 1 - te_ar/phi0

        eps_ar = eps_ar[eps_ar > 0]
        bins = np.linspace(max(-3, np.log10(min(eps_ar))), np.log10(max(eps_ar)), 100)
        

        counts, bins, bars = plt.hist(np.log10(eps_ar), bins = bins)
        plt.close()
        bin_centers = 10**((bins[:-1] + bins[1:])/2)
        bin_centers = bin_centers[:-1]
        # bin_width = np.diff(10**bins)
        # dN_by_dE_tng = counts/bin_width/len(star_xvel)
        # dN_by_dE_tng = dN_by_dE_tng[:-1]

        '''
        Following is calculation 2 for getting dN/deps
        '''
        bin_width = np.diff(bins)
        dN_by_dlogE_tng = counts/bin_width/len(star_xvel)
        dN_by_dE_tng = dN_by_dlogE_tng[:-1] / bin_centers / np.log(10)

        # print(bins)
        # print(bin_centers)
        # print(dN_by_dE_tng)

        Rh0byrmx0_in = self.get_rh0byrmx0_in()

        print(f'rmx0 = {self.rmx0}') #What nonsense is this?

        values = [1/2, 1/4, 1/8, 1/16]
        Rh0byrmx0 = min(values, key=lambda x: abs(np.log(x) - np.log(Rh0byrmx0_in)))

        if Rh0byrmx0 == 1/2:
            Es = 0.485
            norm = np.log10(0.0165)
        elif Rh0byrmx0 == 1/4:
            Es = 1/3. 
            norm = np.log10(0.00366)
        elif Rh0byrmx0 == 1/8:
            Es = 0.21 
            norm = np.log10(0.0005788918279649972)
        elif Rh0byrmx0 == 1/16:
            Es = 0.112  
            norm = np.log10(4.68e-5)


        def get_dNs_by_dE(leps, eps_star, alpha = 3, beta = 3):
            '''
            This is Eq. 13 from Errani+22
            Peak of this distribution is at eps_star = eps_s * (alpha / beta)**(1/beta)

            Args:
            eps: The normalized energy which has to be in betwen 0 and 1
            alpha, beta, eps_s: Parameters of the energy dist. function 
            '''
            eps_s = eps_star / ((alpha/beta) ** (1/beta) )
            # print(eps_s)
            eps = 10**leps
            if isinstance(eps, float) or isinstance(eps, int):
                if 0 < eps <= 1:
                    dNs_by_dE = eps**alpha * np.exp(-(eps/eps_s)**beta)
                else:
                    dNs_by_dE = 0

            if isinstance(eps, np.ndarray):
                dNs_by_dE = np.zeros(0)
                for e in eps:
                    if 0 < e <= 1:
                        dNs_by_dE = np.append(dNs_by_dE, e**alpha * np.exp(-(e/eps_s)**beta))
                    else:
                        dNs_by_dE = np.append(dNs_by_dE, 0)

            # print(eps, dNs_by_dE)
            # return ( dNs_by_dE / (eps_s**alpha * np.exp(-1)) )
            return ( dNs_by_dE)

        
        ed_tng = UnivariateSpline(np.log10(bin_centers[bin_centers * dN_by_dE_tng > 0]), np.log10(dN_by_dE_tng[bin_centers * dN_by_dE_tng > 0]), s= 0.1)
        fact = ed_tng(np.log10(Es))

        fact = max(np.log10(dN_by_dE_tng))
        # print(fact)

        



        if spherical: #This is for the potential energy calculation using spherical symmetry assumption
            cg, rad_plot_cont, mass_arr_plot_cont, dm_mass_arr_cont, star_mass_arr_cont, gas_mass_arr_cont = self.get_mass_profiles(where = int(shsnap), plot = False)
            # dm_rad = np.sqrt(dm_xcoord**2 + dm_ycoord**2 + dm_zcoord**2)
            star_rad = np.sqrt(star_xcoord**2 + star_ycoord**2 + star_zcoord**2)
            # if cg == 1: gas_rad = np.sqrt(gas_xcoord**2 + gas_ycoord**2 + gas_zcoord**2)
            rad_potl = (rad_plot_cont[:-1] + rad_plot_cont[1:])/2
            mass_in_bin = np.diff(mass_arr_plot_cont)
            phi02 = -G*np.sum(mass_in_bin/rad_potl)
            pot_ar = np.zeros(0)

            for r in rad_potl:
                pot_r = -G*(np.sum(mass_in_bin[rad_potl<r])/r + np.sum(mass_in_bin[rad_potl>r]/rad_potl[rad_potl>r]))
                pot_ar = np.append(pot_ar, pot_r) #This would be in (kpc/s)^2
                
            phi_spl = UnivariateSpline(np.append(0, rad_potl), np.append(phi02, pot_ar), s = 0.1)

            ke_ar2 = (star_xvel**2 + star_yvel**2 + star_zvel**2)/2 #Kinetic energy of all particles, these are not mutliplied by mass
            pe_ar2 = phi_spl(star_rad) #This would be an array of potential energies of all the particles
            te_ar2 = ke_ar2*(3.24078e-17)**2 + pe_ar2 
            eps_ar2 = 1 - te_ar2/phi02

            # print(te_ar2)
            # print(ke_ar2)
            # print(pe_ar2)

            eps_rh = 1 - phi_spl(self.get_rh(where = int(shsnap)))/(2*phi02) #This is the epsilon at the half mass radius, 2* for the circular orbit
            eps_ar2 = eps_ar2[(eps_ar2<1) & (eps_ar2>0)]#FIXME: Check if other lengths are matching with what you have here
            assert (eps_ar2 > 0).all(), 'Check the potential energy calculations'

            # print(eps_ar2)

            bins2 = np.linspace(np.log10(min(eps_ar2)), np.log10(max(eps_ar2)), 75)
            counts2, bins2, bars2 = plt.hist(np.log10(eps_ar2), bins = bins2)



            plt.close()
            bin_centers2 = 10**((bins2[:-1] + bins2[1:])/2)
            # print(np.log10(bin_centers2))

            bin_centers2 = bin_centers2[:-1]
            bin_width2 = np.diff(bins2)
            dN_by_dE_tng2 = counts2/bin_width2/len(star_xvel)
            dN_by_dE_tng2 = dN_by_dE_tng2[:-1]


            # print(bin_centers2, bins2, counts2)

            bin_centers21 = bin_centers2[bin_centers2 * dN_by_dE_tng2 != 0]
            dN_by_dE_tng21 = dN_by_dE_tng2[bin_centers2 * dN_by_dE_tng2 != 0]

            # print(np.log10(dN_by_dE_tng21))

            mask = ~np.isnan(dN_by_dE_tng2)
            bin_centers21 = bin_centers2[mask]
            dN_by_dE_tng21 = dN_by_dE_tng2[mask]
            
            mask = ~np.isnan(dN_by_dE_tng21)
            bin_centers21 = bin_centers2[mask]
            dN_by_dE_tng21 = dN_by_dE_tng2[mask]

            six = np.argsort(bin_centers21)
            
            ed_tng2 = UnivariateSpline(np.log10(bin_centers21[six]), np.log10(dN_by_dE_tng21[six]))
            fact2 = ed_tng2(np.log10(Es))

            fact2 = max(np.log10(dN_by_dE_tng21))

            star_profile = UnivariateSpline(rad_plot_cont, star_mass_arr_cont, s = 0.5) #This is the spline for the stellar mass profile
            dmstar_by_dr = star_profile.derivative(n = 1) #This is the derivative of the stellar mass profile
            star_density_profile = UnivariateSpline(rad_plot_cont, dmstar_by_dr(rad_plot_cont)/(4 * np.pi * rad_plot_cont**2), s = 0.1) #This is the spline for the stellar density profile    


            def get_density_model(r_ar):
                '''
                This function returns the density profile of the stars for the Errani assumed energy distribution
                This is again assuming spherical symmetry, hence it is placed here.
                '''
                dens_ar = np.zeros(0)
                for r in r_ar:
                    psi =  - phi_spl(r)
                    dens = - phi0 * 4 * np.pi * quad(lambda eps: np.sqrt(2 *(psi + (1 - eps)*phi0)) * get_dNs_by_dE(np.log10(eps), eps_star = Es), 1, 1 + (psi/phi0))[0]
                    dens_ar = np.append(dens_ar, dens)
                return len(star_masses) * dens_ar * np.mean(star_masses)/ norm #This is after taking care of all the normalizing factors




        if plot:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 6))
            eps_pl = np.linspace(min(bin_centers), max(bin_centers), 30)
            ax.plot(np.log10(bin_centers), np.log10(dN_by_dE_tng), 'kx', label = 'Pariwise potential - TNG')
            if spherical:
                # ax.plot(np.log10(bin_centers2), np.log10(dN_by_dE_tng2), 'ko', ms = 2, mfc = 'white', label = 'Spherical symmetry', alpha = 0.5)
                ax.axvline(np.log10(eps_rh), ls = '-.', color = 'gray', lw = 0.5, label = r'$\varepsilon(r_{\rm{h}})$ - TNG')
                phi0_nfw = -4.67 * self.vmx0**2 #This is for testing against the value of phi0 which was from spherical symmetry
                # phi0_sph = phi0*(3.086e+16)**2
                phi0_sph = phi0
                ax.text(0.01, 0.95, f'phi0 = {phi0_tng:.2f} \n -4.67 vmx^2 = {phi0_nfw:.2f}', transform=ax.transAxes, ha = 'left', va = 'top', fontsize = 8)
                # ax2.plot(rad_plot_cont, star_density_profile(rad_plot_cont), 'k--', label = 'TNG Density profile')
                # print(get_density_model(rad_plot_cont))
                

            ax.axvline(np.log10(Es), ls = ':', color = 'gray', lw = 0.5, label = r'$\varepsilon_{\rm{\star}}$ - picked')
            # ax.plot(np.log10(eps_pl), ed_tng(np.log10(eps_pl))-fact, lw = 0.3, color = 'red')
            # if Rh0byrmx0 == 1/2:
            Es = 0.485
            norm = np.log10(0.0165)
            ax.plot(np.log10(eps_pl), np.log10(get_dNs_by_dE(np.log10(eps_pl), eps_star = Es)) - norm, lw = 0.7, color = 'red', label = r'$R_{\rm{h0}}/r_{\rm{mx0}}$ = 1/2')
            # elif Rh0byrmx0 == 1/4:
            Es = 1/3. 
            norm = np.log10(0.00366)
            ax.plot(np.log10(eps_pl), np.log10(get_dNs_by_dE(np.log10(eps_pl), eps_star = Es)) - norm, lw = 0.7, color = 'orange', label = r'$R_{\rm{h0}}/r_{\rm{mx0}}$ = 1/4')
            # elif Rh0byrmx0 == 1/8:
            Es = 0.21 
            norm = np.log10(0.0005788918279649972)
            ax.plot(np.log10(eps_pl), np.log10(get_dNs_by_dE(np.log10(eps_pl), eps_star = Es)) - norm, lw = 0.7, color = 'royalblue', label = r'$R_{\rm{h0}}/r_{\rm{mx0}}$ = 1/8')
            # elif Rh0byrmx0 == 1/16:
            Es = 0.112  
            norm = np.log10(4.68e-5)
            ax.plot(np.log10(eps_pl), np.log10(get_dNs_by_dE(np.log10(eps_pl), eps_star = Es)) - norm, lw = 0.7, color = 'darkblue', label = r'$R_{\rm{h0}}/r_{\rm{mx0}}$ = 1/16')

            ax.set_xlabel(r'$\log \epsilon$')
            ax.set_ylabel(r'$dN/d\epsilon$')
            ax.set_ylim(bottom = -0.25 + min(np.log10(dN_by_dE_tng[dN_by_dE_tng > 0])),
                        top =  (0.75 + max(np.log10(dN_by_dE_tng[dN_by_dE_tng > 0]))))
            # ax.set_xscale('log')
            # ax.set_yscale('log')
            ax.set_title('subhalo ID = '+str(shid)+' at snapshot '+str(shsnap) + ' and ' + r'$R_{\rm{h0}}/r_{\rm{mx0}} = $' + f'{Rh0byrmx0_in[0]:.2f}', fontsize = 10)
            ax.legend(fontsize = 8, loc = 'lower right')

            plt.tight_layout()
            plt.savefig(plotpath + 'energy_dists/energy_dist_'+str(shid)+'_'+str(shsnap)+'.png')
            plt.close()


        if False: #This is both energy distribution and density profile
            fig, (ax, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))
            eps_pl = np.linspace(min(bin_centers), max(bin_centers), 30)
            ax.plot(np.log10(bin_centers), np.log10(dN_by_dE_tng), 'kx', label = 'Pariwise potential')
            if spherical:
                ax.plot(np.log10(bin_centers2), np.log10(dN_by_dE_tng2), 'ko', ms = 2, mfc = 'white', label = 'Spherical symmetry', alpha = 0.5)
                ax.axvline(np.log10(eps_rh), ls = '-.', color = 'gray', lw = 0.5, label = r'$\varepsilon(r_{\rm{h}})$')
                # ax2.plot(rad_plot_cont, star_density_profile(rad_plot_cont), 'k--', label = 'TNG Density profile')
                print(get_density_model(rad_plot_cont))
                ax2.plot(rad_plot_cont, get_density_model(rad_plot_cont), color = 'gray', lw = 0.7, label = 'Model Density profile')

            # ax.plot(np.log10(eps_pl), ed_tng(np.log10(eps_pl))-fact, lw = 0.3, color = 'red')
            ax.plot(np.log10(eps_pl), np.log10(get_dNs_by_dE(np.log10(eps_pl), eps_star = Es)) - norm, lw = 0.7, color = 'gray')
            ax.set_xlabel(r'$\log \epsilon$')
            ax.set_ylabel(r'$dN/d\epsilon$')
            ax.set_ylim(bottom = -0.25 + min(np.log10(dN_by_dE_tng[dN_by_dE_tng > 0])),
                        top =  (0.75 + max(np.log10(dN_by_dE_tng[dN_by_dE_tng > 0]))))
            # ax.set_xscale('log')
            # ax.set_yscale('log')
            ax.axvline(np.log10(Es), ls = ':', color = 'gray', lw = 0.3, label = r'$\varepsilon_{\rm{\star}}$')
            ax.set_title('Stellar Energy Distribution for subhalo ID = '+str(shid)+' at snapshot '+str(shsnap), fontsize = 10)
            ax.legend(fontsize = 8)

            ax2.axvline(self.get_rh(where = int(shsnap)), ls = ':', color = 'gray', lw = 0.3, label = r'$r_{\rm{h}}$')
            ax2.set_xlabel(r'Radius $\rm{(kpc)}$')
            ax2.set_ylabel(r'$\rho_{\star}$ $\rm{(M_\odot/kpc^3)}$')
            ax2.legend(fontsize = 8)
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            # ax2.set_ylim(bottom = 10)
            plt.tight_layout()
            plt.savefig(plotpath + 'energy_dists/energy_dist_'+str(shid)+'_'+str(shsnap)+'.png')
            plt.close()

        return None


    def get_star_energy_dist_sph(self, where = None, rl = 'linear', h = 0.6774, plot = False):
        '''
        THIS ASSUMES SPHERICAL SYMMETRY
        Gets you the stellar energy distribution for the ID and snap metioned 
        This is very specifically to check if equation 13 of Errani+22 is really present in here
        
        Args:
        where: the snapshot where you need the distribution

        Returns:
        dN/deps spline function such that input energy (epsilon) is in between 0 and  1
        '''
        if where == None: #If the input is none, then go ahead a nd use the default values
            shsnap = self.snap
            shid = self.sfid
        else: #If there is some input, use that to get the rotation curve
            shsnap = self.__where_to_snap(where)
            shid = int(self.tree['SubfindID'][self.tree['SnapNum'] == shsnap])

        cg, rad_plot_cont, mass_arr_plot_cont, dm_mass_arr_cont, star_mass_arr_cont, gas_mass_arr_cont = self.get_mass_profiles(where = int(shsnap))
        
        # snap = get(baseUrl+'snapshots/'+str(shsnap)+'/')
        # z = snap['redshift']
        z = all_redshifts[shsnap]
        filename = filepath + 'cutout_files/cutout_'+str(shid)+'_'+str(shsnap)+'.hdf5'

        f = h5py.File(filename, 'r') #This is to read the cutout file
        # subh_link = get_link(shsnap, shid)
        subh = il.groupcat.loadSingle(basePath,shsnap,subhaloID=shid) #This loads the subhalo position mentioned in the group catalog
        subh_pos_x = subh['SubhaloPos'][0]/h/(1+z)
        subh_pos_y = subh['SubhaloPos'][1]/h/(1+z)
        subh_pos_z = subh['SubhaloPos'][2]/h/(1+z)
        subh_vel_x = subh['SubhaloVel'][0]
        subh_vel_y = subh['SubhaloVel'][1]
        subh_vel_z = subh['SubhaloVel'][2]
        dm_coords = f['PartType1']['Coordinates'] #This is for individual DM particles
        dm_xcoord = dm_coords[:, 0]/h/(1+z) - subh_pos_x
        dm_ycoord = dm_coords[:, 1]/h/(1+z) - subh_pos_y
        dm_zcoord = dm_coords[:, 2]/h/(1+z) - subh_pos_z


        if 'PartType0' in f.keys():
            cg = 1 #telling that gas is present
            gas_coords = f['PartType0']['Coordinates']
            gas_xcoord = gas_coords[:, 0]/h/(1+z) - subh_pos_x
            gas_ycoord = gas_coords[:, 1]/h/(1+z) - subh_pos_y
            gas_zcoord = gas_coords[:, 2]/h/(1+z) - subh_pos_z
            gas_masses = np.array(f['PartType0']['Masses'])*1e10/h
        else:
            cg = 0 

        star_coords = f['PartType4']['Coordinates']
        star_xcoord = star_coords[:, 0]/h/(1+z) - subh_pos_x
        star_ycoord = star_coords[:, 1]/h/(1+z) - subh_pos_y
        star_zcoord = star_coords[:, 2]/h/(1+z) - subh_pos_z
        star_vels = f['PartType4']['Velocities']
        star_xvel = star_vels[:, 0]*np.sqrt(1 / (1+z)) - subh_vel_x
        star_yvel = star_vels[:, 1]*np.sqrt(1 / (1+z)) - subh_vel_y
        star_zvel = star_vels[:, 2]*np.sqrt(1 / (1+z)) - subh_vel_z


        star_masses = np.array(f['PartType4']['Masses'])*1e10/h

        dm_rad = np.sqrt(dm_xcoord**2 + dm_ycoord**2 + dm_zcoord**2)
        star_rad = np.sqrt(star_xcoord**2 + star_ycoord**2 + star_zcoord**2)
        if cg == 1: gas_rad = np.sqrt(gas_xcoord**2 + gas_ycoord**2 + gas_zcoord**2)
        

        rad_plot_cont, mass_arr_plot_cont
        rad_potl = (rad_plot_cont[:-1] + rad_plot_cont[1:])/2
        mass_in_bin = np.diff(mass_arr_plot_cont)
        phi0 = -G*np.sum(mass_in_bin/rad_potl)
        pot_ar = np.zeros(0)

        for r in rad_potl:
            pot_r = -G*(np.sum(mass_in_bin[rad_potl<r])/r + np.sum(mass_in_bin[rad_potl>r]/rad_potl[rad_potl>r]))
            pot_ar = np.append(pot_ar, pot_r) #This would be in (kpc/s)^2
            
        phi_spl = UnivariateSpline(np.append(0, rad_potl), np.append(phi0, pot_ar))

        ke_ar = (star_xvel**2 + star_yvel**2 + star_zvel**2)/2 #Kinetic energy of all particles, these are not mutliplied by mass
        pe_ar = phi_spl(star_rad) #This would be an array of potential energies of all the particles
        te_ar = ke_ar*(3.24078e-17)**2 + pe_ar 
        eps_ar = 1 - te_ar/phi0

        eps_rh = 1 - phi_spl(self.get_rh(where = self.snap))/phi0 #This is the epsilon at the half mass radius
        assert (eps_ar > 0).all(), 'Check the potential energy calculations'
        eps_ar = eps_ar[(eps_ar<1) & (eps_ar>0)]#FIXME: Check if other lengths are matching with what you have here

        # plt.plot(star_rad, phi_spl(star_rad), 'ko', ms = 1)
        # plt.plot(star_rad, te_ar, 'bo', ms = 1)
        # plt.axhline(phi0, ls = '--', color = 'gray')
        # plt.show()
        # print(min(eps_ar), max(eps_ar))

        bins = np.logspace(np.log10(min(eps_ar)), np.log10(max(eps_ar)), 100)
        # bins = np.quantile(eps_ar, np.array([0, 0.0001, 0.0002, 0.0003, 0.0004, .0005, 0.0006,0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003, 0.004, .005, 0.006,0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, .05, 0.06, 0.07, 0.08, 0.09, 0.1,0.25,0.40,0.5,0.6,0.75,0.80,0.85,0.90,0.95,1]))
        # np.set_printoptions(threshold=sys.maxsize)
        # print(bins)
        counts, bins, bars = plt.hist(eps_ar, bins = bins)
        plt.close()
        bin_centers = (bins[:-1] + bins[1:])/2
        bin_centers = bin_centers[:-1]
        bin_width = np.diff(bins)
        dN_by_dE_tng = counts/bin_width/len(star_xvel)
        dN_by_dE_tng = dN_by_dE_tng[:-1]


        # print(f'Bin min and max are: {min(bin_centers):.2f} and {max(bin_centers):.2f}')

        def get_dNs_by_dE(leps, A, alpha, beta, eps_star):
            '''
            This is Eq. 13 from Errani+22
            Peak of this distribution is at eps_star = eps_s * (alpha / beta)**(1/beta)

            Args:
            eps: The normalized energy which has to be in betwen 0 and 1
            alpha, beta, eps_s: Parameters of the energy dist. function 
            '''
            eps_s = eps_star / ((alpha/beta) ** (1/beta) )
            # print(eps_s)
            eps = 10**leps
            if isinstance(eps, float) or isinstance(eps, int):
                if 0 < eps <= 1:
                    dNs_by_dE = eps**alpha * np.exp(-(eps/eps_s)**beta)
                else:
                    dNs_by_dE = 0

            if isinstance(eps, np.ndarray):
                dNs_by_dE = np.zeros(0)
                for e in eps:
                    if 0 < e <= 1:
                        dNs_by_dE = np.append(dNs_by_dE, e**alpha * np.exp(-(e/eps_s)**beta))
                    else:
                        dNs_by_dE = np.append(dNs_by_dE, 0)

            # print(eps, dNs_by_dE)
            return (A * dNs_by_dE)
        
        # print(np.isinf( dN_by_dE_tng))
        # print(np.isinf( np.log10(bin_centers)))
        # print(bin_centers)
        # print(np.argmax(dN_by_dE_tng))

        eps_star = bin_centers[np.argmax(dN_by_dE_tng)] #Esp_star is the energy where we have maximum particles
        get_dNs_by_dE_f = partial(get_dNs_by_dE, eps_star = eps_star, alpha = 3, beta = 3)
        # print(get_ldNs_by_dE_f(np.log10(bin_centers), 1, 1, 1e10))
        # popt, pcov = curve_fit(get_dNs_by_dE_f, np.log10(bin_centers),  dN_by_dE_tng, p0 = [15, 6, 1e10], bounds = [[0.1, 0.1, -np.inf],[20, 20, np.inf]]) #This is when both alpha and beta are being considered
        popt, pcov = curve_fit(get_dNs_by_dE_f, np.log10(bin_centers),  dN_by_dE_tng, p0 = [1e10], bounds = [[-np.inf],[np.inf]])  #This is when only the amplitude is being considered
        A = popt

        



        if plot == True:
            fig, ax = plt.subplots()
            eps_pl = np.linspace(min(bin_centers), max(bin_centers), 30)
            ax.plot(np.log10(bin_centers), np.log10(dN_by_dE_tng), 'kx', ms = 3)
            ax.plot(np.log10(eps_pl), np.log10(get_dNs_by_dE_f(np.log10(eps_pl), A)), lw = 0.7, color = 'gray')
            # ax.set_yscale('log')
            ax.set_xlabel(r'$\log \varepsilon$')
            ax.set_ylabel('Counts')
            # ax.set_title(f'alpha = {alpha:.1f}, beta = {beta:.1f} and eps_star = {eps_star:.1f}')
            ax.set_title(f'alpha = 3, beta = 3 and eps_star = {eps_star:.1f}')
            ax.axvline(eps_rh, ls = ':', color = 'gray')
            # ax.plot(star_rad, ke_ar, 'k.', ms = 1, alpha = 0.2)
            # ax.plot(star_rad, pe_ar * (3.086e+16)**2, 'b.', ms = 1, alpha = 0.2)
            plt.tight_layout()
        return None



# class TNG_Halo():
#     '''
#     This class is to get all the halo properties. Initialize by the snapshot and central ID or snapshot and GrNr
#     '''
#     def __init__(self, ):
#         self.