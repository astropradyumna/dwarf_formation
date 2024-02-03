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
from functools import partial
import sys
from scipy.optimize import fsolve



h = 0.6774
mass_dm = 3.07367708626464e-05 * 1e10/h #This is for TNG50-1
G = 4.5390823753559603e-39 #This is in kpc, Msun and seconds

filepath = '/home/psadh003/tng50/tng_files/'
baseUrl = 'https://www.tng-project.org/api/TNG50-1/'
headers = {"api-key":"894f4df036abe0cb9d83561e4b1efcf1"}
basePath = '/mainvol/jdopp001/L35n2160TNG_fixed/output'

ages_df = pd.read_csv(filepath + 'ages_tng.csv', comment = '#')

all_snaps = np.array(ages_df['snapshot'])
all_redshifts = np.array(ages_df['redshift'])
all_ages = np.array(ages_df['age(Gyr)'])



'''
Following is the dataset of the entire list of subhalos which infalled after z = 3 and survived
'''
survived_df = pd.read_csv(filepath + 'sh_survived_after_z3_tng50_1.csv')

ssh_sfid = survived_df['SubfindID']
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
merged_df = pd.read_csv(filepath + 'sh_merged_after_z3_tng50_1.csv')

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




class TNG_Subhalo():
    '''
    This is a general class for ANY TNG subhalo to calcaulate generic stuff
    '''
    def __init__(self, sfid, snap, last_snap):
        self.sfid = sfid
        self.snap = snap 
        self.last_snap = last_snap
        if self.last_snap != 99:
            self.merged = False
        else:
            self.merged = True

        fields = ['SubhaloMassInRadType', 'SubhaloGrNr', 'SnapNum', 'GroupNsubs', 'SubhaloPos', 'Group_R_Crit200', 
        'Group_M_Crit200', 'SubhaloVel', 'SubhaloHalfmassRadType', 'SubhaloMassType', 'SubhaloLenType', 'SubhaloMassInHalfRadType', 'SubhaloVmaxRad', 'SubhaloMassInMaxRadType']
        if not self.merged:     
            temp_tree = il.sublink.loadTree(basePath, self.snap, self.sfid, fields = ['SnapNum', 'SubfindID'], onlyMDB = True)
            sfid_99 = temp_tree['SubfindID'][0] #FIXME: This only works for a surviving subhalo
            self.tree = il.sublink.loadTree(basePath, 99, sfid_99, fields = fields, onlyMPB = True)
        else: #If it merged
            infall_ix = np.where((msh_snap == snap) & (msh_sfid == sfid))[0] #This is to get the index of the current subhalo in the merger dataframe
            msh_last_snap = int(msh_merger_snap[infall_ix]) #This is the infall snapshot
            msh_last_sfid = int(msh_merger_sfid[infall_ix]) #This is the infall subfind ID

            tree = il.sublink.loadTree(basePath, msh_last_snap, msh_last_sfid, fields = fields, onlyMPB = True) #Getting all the progenitors from the last snapshot of survival
            tree.pop('count') #removing a useless key from the dictionary
            snaps_temp = tree['SnapNum']
            sfids_temp = tree['SubfindID']
            msh_if_ix_tree = np.where((snaps_temp == snap) & (sfids_temp == sfid))[0].item() #This is the infall index in the tree
            self.tree = {key: value[0:msh_if_ix_tree+1] for key, value in tree.items()} #new tree which only runs from final existing snapshot to the infall snapshot
        

    def __where_to_snap(self, where):
        '''
        This is an internal functions used to calculate the snapshot from the where input to different functions in use.
        '''
        if where == 'last':
            snap_wanted = self.last_snap
        if where == 'max': #maximum stellar mass after the infall
            snaps_after_infall = np.flip(np.arange(self.snap, self.last_snap)) #These are the snapshots after infall
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


    def get_mx_values(self, where):
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

        
        # print(rh, mdm_rh, mdm_2rh)
        def nfw_mass(r, lrhos, lrs):
            rhos = 10**lrhos
            rs = 10**lrs
            mass =  4 * np.pi * rhos * rs ** 3 * (np.log(1 + (r/rs))  -  r / (r + rs))
            return mass

        def simul_func(params):
            lrhos, lrs = params
            result = np.array([np.log10(nfw_mass(2 * rh, lrhos, lrs)) - np.log10(float(mdm_2rh)), np.log10(nfw_mass(dmrh, lrhos, lrs)) - np.log10(float(mdmrh))]).ravel()
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
        if how == '2rh': mstar_tree = self.tree['SubhaloMassInRadType'][:, 1]*1e10/h
        elif how == 'total': mstar_tree = self.tree['SubhaloMassType'][:, 1]*1e10/h
        elif how == 'rh': mstar_tree = self.tree['SubhaloMassInHalfRadType'][:, 1]*1e10/h
        elif how == 'vmax': mstar_tree = self.tree['SubhaloMassInMaxRadType'][:, 1]*1e10/h
        mstar = mstar_tree[snap_wanted == self.tree['SnapNum']]
        return mstar
    

    def get_rh(self, where, how = 'rh'):
        '''
        This function is to calculate the half light radius. This is the 3D radius. Scale it with sqrt(2) wherever required for the projected radius
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
            if not os.path.isfile('cutout_'+str(shid)+'_'+str(shsnap)+'.hdf5'): #if it is not downloaded alraedy
                subh_link = self.get_link(shsnap, shid)
                subh = self.get(subh_link, 'r')
                cutout_request = {'gas':'Coordinates,Masses','stars':'Coordinates,Masses,Velocities','dm':'Coordinates'}
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
                if not os.path.isfile('cutout_'+str(shid)+'_'+str(shsnap)+'.hdf5'): #if it is not downloaded alraedy
                    subh_link = self.get_link(shsnap, shid)
                    subh = self.get(subh_link, 'r')
                    cutout_request = {'gas':'Coordinates,Masses','stars':'Coordinates,Masses','dm':'Coordinates'}
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

        star_coords = f['PartType4']['Coordinates']
        star_xcoord = star_coords[:, 0]/h/(1+z) - subh_pos_x
        star_ycoord = star_coords[:, 1]/h/(1+z) - subh_pos_y
        star_zcoord = star_coords[:, 2]/h/(1+z) - subh_pos_z


        star_masses = np.array(f['PartType4']['Masses'])*1e10/h

        dm_rad = np.sqrt(dm_xcoord**2 + dm_ycoord**2 + dm_zcoord**2)
        star_rad = np.sqrt(star_xcoord**2 + star_ycoord**2 + star_zcoord**2)
        if cg == 1: gas_rad = np.sqrt(gas_xcoord**2 + gas_ycoord**2 + gas_zcoord**2)
        
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
            if cg == 1:
                Min = np.sum(star_masses[star_rad < rad]) + np.sum(gas_masses[gas_rad < rad])
            else: 
                Min = np.sum(star_masses[star_rad < rad])

            Min = Min + mass_dm*len(dm_rad[dm_rad< rad])
            mass_arr_plot_cont = np.append(mass_arr_plot_cont, Min)
            dm_mass_arr_cont= np.append(dm_mass_arr_cont, mass_dm*len(dm_rad[dm_rad< rad]))
            if cg == 1: gas_mass_arr_cont = np.append(gas_mass_arr_cont, np.sum(gas_masses[gas_rad < rad]))
            star_mass_arr_cont = np.append(star_mass_arr_cont, np.sum(star_masses[star_rad < rad]))
        

        if plot:
            fig, ax = plt.subplots()
            ax.plot(rad_plot_cont, mass_arr_plot_cont, 'k', label='Total Mass')
            ax.plot(rad_plot_cont, dm_mass_arr_cont, 'k--', label='DM')
            if cg == 1: ax.plot(rad_plot_cont, gas_mass_arr_cont, 'b', label='Gas')
            ax.plot(rad_plot_cont, star_mass_arr_cont, 'r', label='Stars')
            # ax.set_xlim(0, min(max(dm_rad), 10 * self.get_Rh())
            ax.set_xlabel(r'Radius $[kpc]$')
            ax.set_ylabel(r'Mass $[M_\odot]$')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend(fontsize=8)
            ax.set_ylim(bottom = 100 * mass_dm)
            ax.set_xlim(left = 10 * min(dm_rad))
            ax.set_title('Mass profiles for subhalo ID = '+str(shid)+' at z = '+str(round(z, 2)), fontsize = 10)

        if cg == 1: #If there is gas element
            return cg, rad_plot_cont, mass_arr_plot_cont, dm_mass_arr_cont, star_mass_arr_cont, gas_mass_arr_cont
        else: 
            return cg, rad_plot_cont, mass_arr_plot_cont, dm_mass_arr_cont, star_mass_arr_cont, np.zeros(len(rad_plot_cont))


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


    def get_star_energy_dist(self, where = None, rl = 'linear', h = 0.6774, plot = False):
        '''
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

        cg, rad_plot_cont, mass_arr_plot_cont, dm_mass_arr_cont, star_mass_arr_cont, gas_mass_arr_cont = self.get_mass_profiles(shsnap, shid)
        
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
        assert (eps_ar > 0).all(), 'Check the potential energy calculations'
        eps_ar = eps_ar[(eps_ar<1) & (eps_ar>0)]#FIXME: Check if other lengths are matching with what you have here

        # plt.plot(star_rad, phi_spl(star_rad), 'ko', ms = 1)
        # plt.plot(star_rad, te_ar, 'bo', ms = 1)
        # plt.axhline(phi0, ls = '--', color = 'gray')
        # plt.show()
        # print(min(eps_ar), max(eps_ar))

        bins = np.logspace(np.log10(min(eps_ar)), np.log10(max(eps_ar)), 20)
        bins = np.quantile(eps_ar, np.array([0, 0.0001, 0.0002, 0.0003, 0.0004, .0005, 0.0006,0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003, 0.004, .005, 0.006,0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, .05, 0.06, 0.07, 0.08, 0.09, 0.1,0.25,0.40,0.5,0.6,0.75,0.80,0.85,0.90,0.95,1]))
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

        def get_dNs_by_dE(leps, alpha, beta, A, eps_star):
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
        get_dNs_by_dE_f = partial(get_dNs_by_dE, eps_star = eps_star)
        # print(get_ldNs_by_dE_f(np.log10(bin_centers), 1, 1, 1e10))
        popt, pcov = curve_fit(get_dNs_by_dE_f, np.log10(bin_centers),  dN_by_dE_tng, p0 = [15, 6, 1e10], bounds = [[0.1, 0.1, -np.inf],[20, 20, np.inf]])
        alpha, beta, A = popt

        



        if plot == True:
            fig, ax = plt.subplots()
            eps_pl = np.linspace(min(bin_centers), max(bin_centers), 30)
            ax.plot(np.log10(bin_centers), np.log10(dN_by_dE_tng), 'ko', ms = 3)
            ax.plot(np.log10(eps_pl), np.log10(get_dNs_by_dE_f(np.log10(eps_pl), alpha, beta, A)), lw = 0.7, color = 'gray')
            # ax.set_yscale('log')
            ax.set_xlabel(r'$\log \varepsilon$')
            ax.set_ylabel('Counts')
            ax.set_title(f'alpha = {alpha:.1f}, beta = {beta:.1f} and eps_star = {eps_star:.1f}')
            # ax.plot(star_rad, ke_ar, 'k.', ms = 1, alpha = 0.2)
            # ax.plot(star_rad, pe_ar * (3.086e+16)**2, 'b.', ms = 1, alpha = 0.2)
        
        return None

