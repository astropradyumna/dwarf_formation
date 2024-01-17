import pandas as pd 
import numpy as np
import illustris_python as il
import requests
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



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


class TNG_Subhalo():
    '''
    This is a general class for ANY TNG subhalo to calcaulate generic stuff
    '''
    def __init__(self, sfid, snap):
        self.sfid = sfid
        self.snap = snap 
        fields = ['SubhaloMassInRadType', 'SubhaloGrNr', 'SnapNum', 'GroupNsubs', 'SubhaloPos', 'Group_R_Crit200', 
        'Group_M_Crit200', 'SubhaloVel', 'SubhaloHalfmassRadType', 'SubhaloMassType', 'SubhaloLenType']
        temp_tree = il.sublink.loadTree(basePath, self.snap, self.sfid, fields = ['SnapNum', 'SubfindID'], onlyMDB = True)
        sfid_99 = temp_tree['SubfindID'][0] #FIXME: This only works for a surviving subhalo
        assert len(temp_tree) <= 100, 'Probably getting merged, you are yet to write the code' #This is probably getting merged
        self.tree = il.sublink.loadTree(basePath, 99, sfid_99, fields = fields, onlyMPB = True)
        

    def __where_to_snap(self, where):
        '''
        This is an internal functions used to calculate the snapshot from the where input to different functions in use.
        '''
        if where == 99 or where == 'last':
            snap_wanted = 99
        if where == 'max':
            snaps_after_infall = np.flip(np.arange(self.snap, 99)) #These are the snapshots after infall
            snap_arr = self.tree['SnapNum']
            mstar_ar = self.tree['SubhaloMassInRadType'][:, 4] * 1e10/h
            ms_after_infall = np.zeros(0)
            for s in snaps_after_infall:
                ms_after_infall = np.append(ms_after_infall, mstar_ar[snap_arr == s])     
            snap_wanted = snaps_after_infall[np.argmax(ms_after_infall)]
        if isinstance(where, int):
            if 0<= where <= 99:
                snap_wanted = where 
        return snap_wanted

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
        mstar = mstar_tree[snap_wanted == self.tree['SnapNum']]
        return mstar
    
    def get_mdm(self, where, how = 'total'):
        '''
        This is to get the dark matter mass from the TNG
        '''
        snap_wanted = self.__where_to_snap(where)
        if how == '2rh': mstar_tree = self.tree['SubhaloMassInRadType'][:, 1]*1e10/h
        elif how == 'total': mstar_tree = self.tree['SubhaloMassType'][:, 1]*1e10/h
        mstar = mstar_tree[snap_wanted == self.tree['SnapNum']]
        return mstar
    

    def get_rh(self, where):
        '''
        This function is to calculate the half light radius. This is the 3D radius. Scale it with sqrt(2) wherever required for the projected radius
        FIXME: Currently assumes that the subhalo survives. Requires work such that this works for everything
        '''
        snap_wanted = self.__where_to_snap(where)
        Rh_tree = self.tree['SubhaloHalfmassRadType'][:, 4]/(1 + all_redshifts[snap_wanted])/h
        rh = Rh_tree[snap_wanted == self.tree['SnapNum']]
        return rh

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
                cutout_request = {'gas':'Coordinates,Masses','stars':'Coordinates,Masses','dm':'Coordinates'}
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
            self.download_data(int(shsnap), int(shid))
        f = h5py.File(filename, 'r') #This is to read the cutout file
        # subh_link = get_link(shsnap, shid)
        subh = il.groupcat.loadSingle(basePath,shsnap,subhaloID=shid) #This loads the subhalo position mentioned in the group catalog
        subh_pos_x = subh['SubhaloCM'][0]/h/(1+z)
        subh_pos_y = subh['SubhaloCM'][1]/h/(1+z)
        subh_pos_z = subh['SubhaloCM'][2]/h/(1+z)
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
            rad_plot_cont = np.logspace(-1, np.log10(2*max(dm_rad)), 500)

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
            ax.legend(fontsize=12)
            ax.set_title('Mass profiles for subhalo ID = '+str(shid)+' at z = '+str(round(z, 2)))

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

        cg, rad_plot_cont, mass_arr_plot_cont, dm_mass_arr_cont, star_mass_arr_cont, gas_mass_arr_cont = self.get_mass_profiles(shsnap, shid)
        

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
        subh_pos_x = subh['SubhaloCM'][0]/h/(1+z)
        subh_pos_y = subh['SubhaloCM'][1]/h/(1+z)
        subh_pos_z = subh['SubhaloCM'][2]/h/(1+z)
        dm_coords = f['PartType1']['Coordinates']
        dm_xcoord = dm_coords[:, 0]/h/(1+z) - subh_pos_x
        dm_ycoord = dm_coords[:, 1]/h/(1+z) - subh_pos_y
        dm_zcoord = dm_coords[:, 2]/h/(1+z) - subh_pos_z
        dm_rad = np.sqrt(dm_xcoord**2 + dm_ycoord**2 + dm_zcoord**2)

        ix_max_vel = np.argmax(dm_vel_plot_cont)
        vmx_subh = dm_vel_plot_cont[ix_max_vel]
        rmx_subh = rad_plot_cont[ix_max_vel]
        Mmx_subh = mass_dm*len(dm_rad[dm_rad< rmx_subh]) 

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
            ax.set_title('Rotation Curve for subhalo ID = '+str(shid)+' at z = '+str(round(z, 2)))
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
        subh_pos_x = subh['SubhaloCM'][0]/h/(1+z)
        subh_pos_y = subh['SubhaloCM'][1]/h/(1+z)
        subh_pos_z = subh['SubhaloCM'][2]/h/(1+z)
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
