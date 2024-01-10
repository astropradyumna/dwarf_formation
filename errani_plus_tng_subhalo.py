'''
This file is to define the subhalo class using the subfind ID and the snaphot
The class must have functions to calculate everything that is required

INPUT HAS TO BE AT INFALL
'''
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import illustris_python as il
from orbit_calculator_preamble import *
from galpy.potential import NFWPotential, TimeDependentAmplitudeWrapperPotential
from galpy.orbit import Orbit
from astropy import units as u
from testing_errani import get_rot_curve, get_rmxbyrmx0, get_vmxbyvmx0, get_mxbymx0, get_LbyL0
from tng_subhalo_and_halo import TNG_Subhalo
from matplotlib import gridspec




h = 0.6774
mass_dm = 3.07367708626464e-05 * 1e10/h #This is for TNG50-1
G = 4.5390823753559603e-39 #This is in kpc, Msun and seconds

filepath = '/home/psadh003/tng50/tng_files/'
outpath  = '/home/psadh003/tng50/output_files/'
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


'''
Getting the position, etc. of the central subhalo
'''
central_id_at99 = 0
central_fields = ['GroupFirstSub', 'SubhaloGrNr', 'SnapNum', 'GroupNsubs', 'SubhaloPos', 'Group_R_Crit200', 'Group_M_Crit200', 'SubhaloVel']
central_tree = il.sublink.loadTree(basePath, 99, central_id_at99, fields = central_fields, onlyMPB = True)
central_snaps = central_tree['SnapNum']
central_redshift = all_redshifts[central_snaps]
central_x =  central_tree['SubhaloPos'][:, 0]/(1 + central_redshift)/h
central_y =  central_tree['SubhaloPos'][:, 1]/(1 + central_redshift)/h
central_z =  central_tree['SubhaloPos'][:, 2]/(1 + central_redshift)/h
central_r200 = central_tree['Group_R_Crit200']/(1 + central_redshift)/h #This is the virial radius of the group
ages_rvir = all_ages[central_snaps] #Ages corresponding to the virial radii
central_grnr = central_tree['SubhaloGrNr']
central_gr_m200 = central_tree['Group_M_Crit200']*1e10/h #This is the M200 of the central group
central_vx = central_tree['SubhaloVel'][:, 0] #km/s
central_vy = central_tree['SubhaloVel'][:, 1]
central_vz = central_tree['SubhaloVel'][:, 2]




class Subhalo(TNG_Subhalo):
    '''
    This class is for a subhalo in general which we are using for evolution, has to be something in TNG
    '''
    def __init__(self, sfid, snap):
        '''
        Args:
        sfid: The subfind ID at infall snapshot
        snap: The snapshot of infall
        '''
        self.sfid = sfid
        self.snap = snap 

        fields = ['SubhaloMassInRadType', 'SubhaloGrNr', 'SubfindID', 'SnapNum', 'GroupNsubs', 'SubhaloPos', 'Group_R_Crit200', 'Group_M_Crit200', 'SubhaloVel', 'SubhaloHalfmassRadType', 'SubhaloMassType']
        temp_tree = il.sublink.loadTree(basePath, self.snap, self.sfid, fields = ['SnapNum', 'SubfindID'], onlyMDB = True)
        sfid_99 = temp_tree['SubfindID'][0] #FIXME: This only works for a surviving subhalo
        self.tree = il.sublink.loadTree(basePath, 99, sfid_99, fields = fields, onlyMPB = True)

        self.vmx0, self.rmx0, self.mmx0 = self.get_rot_curve(where= int(self.snap))
        self.torb = None 
        self.rperi = None 
        self.rapo = None 
        
        assert len(temp_tree) <= 100, 'Probably getting merged, you are yet to write the code' #This is probably getting merged
        self.mstar = max(self.tree['SubhaloMassType'][:, 4][self.tree['SnapNum'] > self.snap]) * 1e10 / h #This is the maximum stellar mass at infall

    def get_infall_properties(self):
        '''
        This function is to obtain the properties of the subhalo at infall and then update it to the Subhalo object.
        Checks if the subhalo cutout file is downloaded. If it is not, it downloads it. 

        Args: None

        Returns: None
        '''
        snap = self.snap
        sfid = self.sfid

        vmx, rmx, mmx = self.get_rot_curve(shid= sfid, shsnap= snap) #FIXME: This assumes that we are passing the infall snapshot and subfindIDs, this might not be always true

        # Following are the infall details
        self.vmx0 = vmx 
        self.rmx0 = rmx 
        self.mmx0 = mmx

        return None


    def get_orbit(self, merged, when_te = 'last'):
        '''
        This function calculates all the parameters related to the orbit.
        To visualize the orbit, use subhalo_time_evolution.py, this function does not plot anything.
        This function uses galpy to find out the orbital parameters. 
        FIXME: Currently assumes constant energy and virial mass which might affect the subhalo evolution

        Args:
        merged (boolean): True if the subhalo merged, false otherwise
        when_te (string): 'last'(default) for the last surviving snapshot, 'infall' for the energy to be taken at infall, 'first_peri' for the energy to be taken at the first pericenter, 'if_r200' for the energy to be taken at the first crossing into virial radius
        
        Returns:
        rperi: The pericenric radius
        rapo: Apocenter
        torb: Orbital time
        '''
        snap = self.snap 
        sfid = self.sfid
        fields = ['SubhaloGrNr', 'GroupFirstSub', 'SnapNum', 'SubfindID', 'SubhaloPos', 'SubhaloVel', 'SubhaloMassType'] #These are the fields that will be taken for the subhalos
        if not merged:
            tree = il.sublink.loadTree(basePath, snap, sfid, fields = fields, onlyMDB = True) #This only works for surviving subhalos
        else: #If it merges
            tree = il.sublink.loadTree(basePath, snap, sfid, fields = fields, onlyMPB = True) #Getting all the progenitors from the last snapshot of survival
            merger_ix = np.where((msh_merger_snap == snap) & (msh_merger_sfid == sfid))[0] #This is to get the index of the current subhalo in the merger dataframe
            msh_if_snap = msh_snap[merger_ix] #This is the infall snapshot
            msh_if_sfid = msh_sfid[merger_ix] #This is the infall subfind ID
            tree.pop('count') #removing a useless key from the dictionary
            snaps_temp = tree['SnapNum']
            sfids_temp = tree['SubfindID']
            msh_if_ix_tree = np.where((snaps_temp == msh_if_snap) & (sfids_temp == msh_if_sfid))[0].item() #This is the infall index in the tree
            tree = {key: value[0:msh_if_ix_tree+1] for key, value in tree.items()} #new tree which only runs from final existing snapshot to the infall snapshot

            # IPython.embed()
        subh_snap = tree['SnapNum']
        subh_redshift = all_redshifts[subh_snap]
        subh_x = tree['SubhaloPos'][:, 0]/(1 + subh_redshift)/h
        subh_y = tree['SubhaloPos'][:, 1]/(1 + subh_redshift)/h
        subh_z = tree['SubhaloPos'][:, 2]/(1 + subh_redshift)/h
        subh_vx = tree['SubhaloVel'][:, 0]
        subh_vy = tree['SubhaloVel'][:, 1]
        subh_vz = tree['SubhaloVel'][:, 2]
        subh_mstar = tree['SubhaloMassType'][:, 4]*1e10/h #Stellar mass of the subhalo


        common_snaps = np.intersect1d(subh_snap, central_snaps) #This is in acsending order. Descending order after flipping
        common_snaps_des = np.flip(common_snaps) #This will be in descending order to get the indices in central_is and subh_ix
        central_ixs = np.where(np.isin(central_snaps, common_snaps))[0] #getting the indices of common snaps in the central suhalo. The indices are in such a way that the snaps are again in descending order.
        subh_ixs = np.where(np.isin(subh_snap, common_snaps))[0]


        subh_z_if = all_redshifts[common_snaps[0]] #This is the redshift of infall
        subh_dist = np.sqrt((subh_x[subh_ixs] - central_x[central_ixs])**2 + (subh_y[subh_ixs] - central_y[central_ixs])**2 + (subh_z[subh_ixs] - central_z[central_ixs])**2)
        subh_ages = np.flip(all_ages[common_snaps]) #The ages after flipping will be in descending order


        
        for ix, sx in enumerate(common_snaps_des):
            '''
            This loop is to go through all the common indices in descending order
            '''
            # print(ix)
            if subh_dist[subh_ixs[common_snaps_des == sx]] < central_r200[central_snaps == sx]: # As weird as it sounds, this is the first when the subhalos distance is below the virial radius (on completing the loop)
                snap_r200_if = sx 
            if ix > 0 and ix < len(common_snaps_des) - 1:
                if subh_dist[subh_ixs[common_snaps_des == common_snaps_des[ix]]] < subh_dist[subh_ixs[common_snaps_des == common_snaps_des[ix - 1]]] and subh_dist[subh_ixs[common_snaps_des == common_snaps_des[ix]]] < subh_dist[subh_ixs[common_snaps_des == common_snaps_des[ix + 1]]]:
                    first_peri_snap = sx

        # This is the snapshot at which the input to pericenter estimation codes wll be taken.
        if when_te == 'last':
            te_snap = common_snaps[-1] #This will be the last snapshot, z = 0 for the surviving ones
        elif when_te == 'infall':
            te_snap = common_snaps[0] #This will be the infall snap
        elif when_te == 'first_peri': 
            te_snap = first_peri_snap #This will be the snapshot of the first pericentric passage
        elif when_te == 'if_r200':
            te_snap = snap_r200_if # This will be the time at which the the infaling subhalo crosses the virial radius
        else:
            raise ValueError('Recheck the input for when_te variable')
        
        te_snap_z = all_redshifts[te_snap] #This is the refshift at the total energy snapshot
        te_time = all_ages[te_snap]
        te_snap_ix = te_snap == common_snaps_des
        # This index has to be input to subh_x itself or central_x itself. Not any subset of it! For e.g. subh_x[te_subh_ix]
        te_subh_ix = subh_ixs[te_snap_ix] #Subhalo index for this infall time. 
        te_central_ix = central_ixs[te_snap_ix]

        subh_vx_cen = subh_vx[te_subh_ix] - central_vx[te_central_ix]
        subh_vy_cen = subh_vy[te_subh_ix] - central_vy[te_central_ix]
        subh_vz_cen = subh_vz[te_subh_ix] - central_vz[te_central_ix]    


        subh_x_cen = subh_x[te_subh_ix] - central_x[te_central_ix]
        subh_y_cen = subh_y[te_subh_ix] - central_y[te_central_ix]
        subh_z_cen = subh_z[te_subh_ix] - central_z[te_central_ix]




        mvir = central_gr_m200[te_central_ix].item() #this would be the virial mass of the FoF halo infall time 
        
        # This is method 3: Using galpy. For codes using other methods, look at subhalo_time_evolution.py
        def get_nfw_at_t(t):
            '''
            This is to mdel the time dependence of the galpy potential.
            -o- potential is a nonlinear function in the virial mass because of the scale radius.
            -o- Using taylor expansion of the NFW potential to have the correction to first order in r/rvir.
            '''
            t = t + te_time
            # A = np.sqrt(t) #testing so that nothing blows up
            # A = (fof_m200_t(t) / (fof_r200_t(t))) * ((fof_r200_t(te_time)) / fof_m200_t(te_time))
            A = 1
            return A

        nfw = NFWPotential(conc=8, mvir=mvir/1e12, wrtcrit = True, overdens = 200 * get_critical_dens(te_snap_z)/get_critical_dens(0))
        potential = TimeDependentAmplitudeWrapperPotential(A = get_nfw_at_t, pot = nfw) #This is to vary the potential with time 

        x, y, z = subh_x_cen, subh_y_cen, subh_z_cen
        # Convert positions to cylindrical coordinates
        R = np.sqrt(x**2 + y**2)
        Phi = np.arctan2(y, x)
        Z = z

        vx, vy, vz = subh_vx_cen, subh_vy_cen, subh_vz_cen

        vR = (x * vx + y * vy) / R
        vPhi = -(-x * vy + y * vx) / R #Extra minus sign because this is a left handed system (galpy)
        vZ = vz

        initial_conditions = [R * u.kpc, vR * u.kilometer/u.second, vPhi * u.kilometer/u.second, Z * u.kpc, vZ * u.kilometer/u.second, Phi * u.radian ] 
        subhalo_orbit = Orbit(initial_conditions)

        # Integrate the orbit
        if when_te == 'last':
            ts = np.linspace(0, -13.8 , 500) #This is for 13.8 Gyr before the given snapshot. 
        else: #Since I am already raising a ValueError before, I am using else directly here
            ts = np.linspace(0, 13.8 - te_time.item(), 500) #This is only for the remaining time

        subhalo_orbit.integrate(ts * u.Gyr, potential, method = 'leapfrog')


        torb = subhalo_orbit.Tr(use_physical = True, type = 'spherical')
        rapo = subhalo_orbit.rap(use_physical = True, type= 'spherical')
        rperi = subhalo_orbit.rperi(use_physical = True, type= 'spherical')

        self.torb = torb
        self.rapo = rapo 
        self.rperi = rperi

        return float(all_ages[99] - all_ages[common_snaps[0]])
    

    def evolve(self, tevol, V0):
        '''
        This is a function that evolves the subhalo using Errani models

        Args:
        t (float): The time for which evolution must take place
        V0: This is a parameter of the host whn paramterized using the isothermal profile
        '''
        rmx0 = self.rmx0
        vmx0 = self.vmx0
        rperi = self.rperi
        rapo = self.rapo

        if any([rmx0, vmx0, rperi, rapo]) == None:
            raise ValueError('Some of the required values are None, recheck if they have been updated in the Object')

        tmx0 = 2 * np.pi *  ( rmx0 / vmx0 ) * 3.086e16 * 3.17098e-8 * 1e-9 #This would be in Gyrs assuming r to be in kpc and v to be in km/s
        tperi = 2 * np.pi * ( rperi / V0) * 3.086e16 * 3.17098e-8 * 1e-9 #This is the tperi that is calculated in Errani+21
        
        x = rapo / rperi
        fecc = (2 * x / (x + 1)) ** 3.2

        torb = self.torb* fecc #Gyr, this is after accounting for the ellipticity of the orbit
        
        def get_tmx_t(t):
            '''
            This is the Tmx value at a given time t. 
            We can calculate the real time t given a Tmx (for a given value of Mmx)
            '''
            
            if tmx0/tperi >= 2/3: #Heavy mass loss regime
                tasy = 0.22 * tperi
                y0 = (tmx0 - tasy) / tperi
                tau_asy = 0.65 * torb
                tau = tau_asy / y0 
                eta = 1 - np.exp( - 2.5 * y0 )
                inner_term = 1 + (t/tau)**eta 
                tmx = tasy + tperi * y0 * (inner_term)**(-1/eta)
            else: #modest mass loss regime
                tasyp =  ( tmx0 / (1 + (tmx0/tperi))**2.2)
                etap = 0.67
                # yp = (tmx - tasyp)/tperi 
                y0p = (tmx0 - tasyp)/tperi 
                taup = torb * 1.2 * (tmx0 / tperi)**(-0.5)
                inner_term = 1 + (t/taup)**etap
                tmx = tasyp + tperi * y0p * (inner_term)**(-1/etap)
            return tmx
        
        def get_tmx(rmx):
            ''' 
            This is to calculate the Tmx value from rmx and vmx
            '''
            vmx = vmx0 * get_vmxbyvmx0(rmx/rmx0)
            return 2 * np.pi * rmx / vmx * 3.086e16 * 3.17098e-8 * 1e-9 
        
        
        def get_time(rmx):
            '''
            I want to obtain the time for a given value of rmx
            '''
            t = fsolve(lambda t: get_tmx_t(t) - get_tmx(rmx), 2)[0]
            return t
        
        tmx = get_tmx_t(tevol)
        rmx = fsolve(lambda rmx: get_tmx(rmx) - tmx, rmx0/100)[0]
        frem = get_mxbymx0(rmx/rmx0) 
        # print(f'frem = {100 * frem:.2f} % ')
        
        return frem
    

    def get_tng_values(self):
        '''
        This function is to get the values that we obtain from the simulation as we proceed
        FIXME: This currently calculates vmx, rmx and Mmx. Update accordingly 
        '''
        subh_tree = il.sublink.loadTree(basePath, self.snap, self.sfid, fields = ['SnapNum', 'SubfindID'], onlyMDB = True)
        subh_snap_99 = subh_tree['SnapNum'][0] #First snap is generally 99, so starting from there
        subh_sfid_99 = subh_tree['SubfindID'][0]
        vmx, rmx, mmx = get_rot_curve(shid= subh_sfid_99, shsnap= subh_snap_99)

        return vmx, rmx, mmx

    def get_rh0byrmx0(self):
        '''
        This is an internal function to calculate the initial rh0burmx0 for the subhalo
        '''
        values = [1/2, 1/4, 1/8, 1/16]
        Rh0 = self.get_rh(where = int(self.snap))/np.sqrt(2)
        
        closest_value = min(values, key=lambda x: abs(x - Rh0/self.rmx0))
        # print(closest_value)
        return closest_value

    def get_mstar_model(self, tevol = None, frem = None): #FIXME: The fraction rh0byrmx0 has to be accounted for
        '''
        This function calculates the stellar mass from the model at a given time from the 
        '''
        rh0byrmx0 = self.get_rh0byrmx0()
        if tevol is not None:
            frem = self.evolve(tevol, V0 = 978.59) #FIXME: This needs to be updated for other halos
            mstar_now = get_LbyL0(frem, rh0byrmx0) * self.mstar
        if frem is not None:
            mstar_now = get_LbyL0(frem, rh0byrmx0) * self.mstar
        return mstar_now


    def plot_orbit_comprehensive(self, merged, when_te = 'last', show = False):
        '''
        This function plots the orbit comprehensively with energy and masses variation with time
        
        This function plots the orbit, the velocities in x, y and z, and the total energy; FIXME: has to be shortened further after testing different techniques. 

        Args:
        snap and sfid: THIS HAS TO BE THE LAST SURVIVING SNAPSHOT FOR SUBHALOS THAT MERGE and the infall snapshot for the surviving subhalos
        ax_ar: This is an array of plotting axes. Please pass 5 axes
        merged(boolean): If the subhalo merged, then please give this as True

        Returns None because this is only a plotting routine
        ''' 

        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1]) #Generating one big image to the left and four small images to the right, from ChatGPT
        ax = plt.subplot(gs[:, 0])
        ax_sub1 = plt.subplot(gs[0, 1]) #For energy
        ax_sub2 = plt.subplot(gs[1, 1]) #For masses


        snap = self.snap
        sfid = self.sfid
        fields = ['SubhaloGrNr', 'GroupFirstSub', 'SnapNum', 'SubfindID', 'SubhaloPos', 'SubhaloVel', 'SubhaloMassType', 'SubhaloMassInRadType'] #These are the fields that will be taken for the subhalos
        if not merged:
            tree = il.sublink.loadTree(basePath, snap, sfid, fields = fields, onlyMDB = True) #This only works for surviving subhalos
        else: #If it merges
            tree = il.sublink.loadTree(basePath, snap, sfid, fields = fields, onlyMPB = True) #Getting all the progenitors from the last snapshot of survival
            merger_ix = np.where((msh_merger_snap == snap) & (msh_merger_sfid == sfid))[0] #This is to get the index of the current subhalo in the merger dataframe
            msh_if_snap = msh_snap[merger_ix] #This is the infall snapshot
            msh_if_sfid = msh_sfid[merger_ix] #This is the infall subfind ID
            tree.pop('count') #removing a useless key from the dictionary
            snaps_temp = tree['SnapNum']
            sfids_temp = tree['SubfindID']
            msh_if_ix_tree = np.where((snaps_temp == msh_if_snap) & (sfids_temp == msh_if_sfid))[0].item() #This is the infall index in the tree
            tree = {key: value[0:msh_if_ix_tree+1] for key, value in tree.items()} #new tree which only runs from final existing snapshot to the infall snapshot

        subh_snap = tree['SnapNum']
        subh_sfid = tree['SubfindID']
        subh_redshift = all_redshifts[subh_snap]
        subh_x = tree['SubhaloPos'][:, 0]/(1 + subh_redshift)/h
        subh_y = tree['SubhaloPos'][:, 1]/(1 + subh_redshift)/h
        subh_z = tree['SubhaloPos'][:, 2]/(1 + subh_redshift)/h
        subh_vx = tree['SubhaloVel'][:, 0]
        subh_vy = tree['SubhaloVel'][:, 1]
        subh_vz = tree['SubhaloVel'][:, 2]
        subh_mstar = tree['SubhaloMassType'][:, 4]*1e10/h #Stellar mass of the subhalo


        common_snaps = np.intersect1d(subh_snap, central_snaps) #This is in acsending order. Descending order after flipping
        common_snaps_des = np.flip(common_snaps) #This will be in descending order to get the indices in central_ix and subh_ix
        central_ixs = np.where(np.isin(central_snaps, common_snaps))[0] #getting the indices of common snaps in the central suhalo. The indices are in such a way that the snaps are again in descending order.
        subh_ixs = np.where(np.isin(subh_snap, common_snaps))[0]


        subh_z_if = all_redshifts[common_snaps[0]] #This is the redshift of infall
        subh_dist = np.sqrt((subh_x[subh_ixs] - central_x[central_ixs])**2 + (subh_y[subh_ixs] - central_y[central_ixs])**2 + (subh_z[subh_ixs] - central_z[central_ixs])**2)
        subh_ages = np.flip(all_ages[common_snaps]) #The ages after flipping will be in descending order



        for ix, sx in enumerate(common_snaps_des):
            '''
            This loop is to go through all the common indices in descending order
            '''
            # print(ix)
            if subh_dist[subh_ixs[common_snaps_des == sx]] < central_r200[central_snaps == sx]: # As weird as it sounds, this is the first when the subhalos distance is below the virial radius (on completing the loop)
                snap_r200_if = sx 
            if ix > 0 and ix < len(common_snaps_des) - 1:
                if subh_dist[subh_ixs[common_snaps_des == common_snaps_des[ix]]] < subh_dist[subh_ixs[common_snaps_des == common_snaps_des[ix - 1]]] and subh_dist[subh_ixs[common_snaps_des == common_snaps_des[ix]]] < subh_dist[subh_ixs[common_snaps_des == common_snaps_des[ix + 1]]]:
                    first_peri_snap = sx

        # This is the snapshot at which the input to pericenter estimation codes wll be taken.
        if when_te == 'last':
            te_snap = common_snaps[-1] #This will be the last snapshot, z = 0 for the surviving ones
        elif when_te == 'infall':
            te_snap = common_snaps[0] #This will be the infall snap
        elif when_te == 'first_peri': 
            te_snap = first_peri_snap #This will be the snapshot of the first pericentric passage
        elif when_te == 'if_r200':
            te_snap = snap_r200_if # This will be the time at which the the infaling subhalo crosses the virial radius
        else:
            raise ValueError('Recheck the input for when_te variable')
        

        te_snap_z = all_redshifts[te_snap] #This is the refshift at the total energy snapshot
        te_time = all_ages[te_snap] #This is the time where we are inputting the energy to galpy
        te_snap_ix = te_snap == common_snaps_des
        # Discovery of the day: This index has to be input to subh_x itself or central_x itself. Not any subset of it!
        te_subh_ix = subh_ixs[te_snap_ix] #Subhalo index for this infall time. 
        te_central_ix = central_ixs[te_snap_ix]

        subh_vx_cen = subh_vx[te_subh_ix] - central_vx[te_central_ix]
        subh_vy_cen = subh_vy[te_subh_ix] - central_vy[te_central_ix]
        subh_vz_cen = subh_vz[te_subh_ix] - central_vz[te_central_ix]    


        subh_x_cen = subh_x[te_subh_ix] - central_x[te_central_ix]
        subh_y_cen = subh_y[te_subh_ix] - central_y[te_central_ix]
        subh_z_cen = subh_z[te_subh_ix] - central_z[te_central_ix]


        mvir = central_gr_m200[te_central_ix].item() 

        # Following part is for galpy


        def get_nfw_at_t(t):
            '''
            This is to mdel the time dependence of the galpy potential
            -o- potential is a nonlinear function in the virial mass because of the scale radius
            -o- we will be divding the potential calculated the current time with the potential
            '''
            t = t + te_time
            # A = np.sqrt(t) #testing so that nothing blows up
            # A = (fof_m200_t(t) / (fof_r200_t(t))) * ((fof_r200_t(te_time)) / fof_m200_t(te_time))
            A = 1
            return A

        nfw = NFWPotential(conc=8, mvir=mvir/1e12, wrtcrit = True, overdens = 200 * get_critical_dens(te_snap_z)/get_critical_dens(0))
        potential = TimeDependentAmplitudeWrapperPotential(A = get_nfw_at_t, pot = nfw) #This is to vary the time 

        # potential = NFWPotential(conc=8, mvir=mvir/1e12, wrtcrit = True, overdens = 200 * get_critical_dens(te_snap_z)/get_critical_dens(0))
        x, y, z = subh_x_cen, subh_y_cen, subh_z_cen
        # Convert positions to cylindrical coordinates
        R = np.sqrt(x**2 + y**2)
        Phi = np.arctan2(y, x)
        Z = z

        vx, vy, vz = subh_vx_cen, subh_vy_cen, subh_vz_cen

        vR = (x * vx + y * vy) / R
        vPhi = -(-x * vy + y * vx) / R #Extra minus sign because this is a left handed system (galpy)
        vZ = vz

        initial_conditions = [R * u.kpc, vR * u.kilometer/u.second, vPhi * u.kilometer/u.second, Z * u.kpc, vZ * u.kilometer/u.second, Phi * u.radian ] 
        subhalo_orbit = Orbit(initial_conditions)

        # Integrate the orbit
        # Integrate the orbit
        if when_te == 'last':
            ts = np.linspace(0, -12 , 500) #This is for 13.8 Gyr before the given snapshot. 
        else: #Since I am already raising a ValueError before, I am using else directly here
            ts = np.linspace(0, 13.8 - te_time.item(), 500) #This is only for the remaining time

        subhalo_orbit.integrate(ts * u.Gyr, potential, method = 'leapfrog')

        # Following is a weird way to obtain the orbit data since this does not work directly in galpy 1.7
        fig, = subhalo_orbit.plot(d1 = 't', d2 = 'x')
        plt.close()
        fig2, = subhalo_orbit.plot(d1 = 'y', d2 = 'z')
        plt.close()
        fig3, = subhalo_orbit.plot(d1 = 'vx', d2 = 'vy')
        plt.close()
        fig4, = subhalo_orbit.plot(d1 = 'vz', d2 = 'E')
        plt.close()

        t_gp = fig.get_xdata() + te_time
        x_gp = fig.get_ydata()
        y_gp = fig2.get_xdata()
        z_gp = fig2.get_ydata()
        vx_gp = fig3.get_xdata()
        vy_gp = fig3.get_ydata()
        vz_gp = fig4.get_xdata()
        E_gp = fig4.get_ydata()

        dist_gp = np.sqrt(x_gp**2 + y_gp**2 + z_gp**2)


        # ========== PLOTTING PART ======================== 
        

        ax.plot(np.flip(subh_ages), np.flip(subh_dist), lw = 1, color = 'blue', label = r'Orbit')
        ax.plot(all_ages[central_snaps[central_ixs]], central_r200[central_ixs], c = 'gray', ls = '--',label = r'$R_{200}$', lw = 0.3)
        ax.plot(t_gp, dist_gp, lw = 0.5, alpha = 0.8, label = 'galpy orbit', color = 'r')
        ax.set_ylabel('Cluster-centric distance (kpc)')
        ax.set_xlabel('Age (Gyr)')
        ax.set_xlim(left = 0.9*all_ages[common_snaps[0]], right = 1.1*all_ages[99])
        
        subh_max_mstar = round(np.log10(max(subh_mstar)), 2)
        ax.set_title('ID at z=0 is '+str(tree['SubfindID'][0])+r'    $\rm{\log_{10}M_\bigstar} = $'+str(subh_max_mstar), fontsize = 10)
        # ax.text(1.05, 0.25, r'$\rm{\log_{10}M_\bigstar} = $'+str(subh_max_mstar), transform=ax.transAxes, fontsize = 11)

        

        

        ax.axhline(max(dist_gp), c = 'red', ls = ':', alpha = 0.5, label = 'peri/apo from galpy')
        ax.axhline(min(dist_gp), c = 'red', ls = ':', alpha = 0.5)

        # plt.legend()
        # ax.legend(fontsize = 10, loc='upper left', bbox_to_anchor=(1, 1))
        ax.legend(fontsize = 8)

        

        '''
        Here, we calculate the energy as it evolves. The energy being calculated for the orbit has mass changing with time and critical density also changes
        '''
        pe_ar = np.zeros(0) 
        pe_ar_gp = np.zeros(0) #This is calculation of the potential energy for galpy orbit same as above
        for (ix, dist) in enumerate(subh_dist):
            snap = common_snaps_des[ix]
            pe_ar = np.append(pe_ar, get_nfw_potential(dist, central_gr_m200[central_ixs[ix]], 8, all_redshifts[snap]))

        dist_gp = np.sqrt(x_gp**2 + y_gp**2 + z_gp**2)
        # print(t_gp)
        for (ix, t) in enumerate(t_gp): #This is a loop over galpy
            snap = np.searchsorted(all_ages, t) - 1 #index of the snapshot that has age lower than this time
            # print(f'{len(pe_ar_gp)} out of {len(t_gp)}')
            pe_ar_gp = np.append(pe_ar_gp, get_nfw_potential(dist_gp[ix], central_gr_m200[central_snaps == snap], 8, all_redshifts[snap])) 

            

        ke_ar = 0.5 *((subh_vx[subh_ixs] - central_vx[central_ixs])**2 + (subh_vy[subh_ixs] - central_vy[central_ixs])**2 + (subh_vz[subh_ixs] - central_vz[central_ixs])**2)
        ke_ar_gp = 0.5 * (vx_gp**2 + vy_gp**2 + vz_gp**2)

        ax_sub1.plot(t_gp, E_gp, color = 'red', alpha = 0.5, label = 'galpy')
        # ax_sub1.axhline(subhalo_orbit.E(use_physical = True), color = 'red', alpha = 0.5, label = 'e2')
        ax_sub1.plot(t_gp, ke_ar_gp, color = 'hotpink', ls = ':')
        ax_sub1.plot(t_gp, pe_ar_gp, color = 'hotpink', ls = '-.')
        ax_sub1.plot(t_gp, pe_ar_gp + ke_ar_gp, color = 'hotpink', ls = '-')

        ax_sub1.plot(np.flip(subh_ages), np.flip(ke_ar), color = 'blue', ls = ':', label = 'KE')
        ax_sub1.plot(np.flip(subh_ages), np.flip(pe_ar), color = 'blue', ls = '-.', label = 'PE')
        ax_sub1.plot(np.flip(subh_ages), np.flip(ke_ar + pe_ar), color = 'blue', ls = '-', label = 'TE')
        ax_sub1.axhline(0, ls = '--', color = 'gray', alpha = 0.2)
        ax_sub1.set_xlim(left = 0.9*all_ages[common_snaps[0]], right = 1.1*all_ages[99])
        ax_sub1.set_xlabel('Age (Gyr)')
        ax_sub1.set_ylabel(r'$E \, \rm{(km/s)^2}$')
        ax_sub1.legend(fontsize = 6)

       
        #This is now for the third subplot:
        #The plan is to plot the mass variation with time
        Mdm = tree['SubhaloMassInRadType'][:, 1][subh_ixs] * 1e10/h #This would be such that the snaps are in decreasing order and also after the infall only
        Mstar = tree['SubhaloMassInRadType'][:, 4][subh_ixs] * 1e10/h #This would be such that the snaps are in decreasing order and also after the infall only

        totMdm = tree['SubhaloMassType'][:, 1][subh_ixs] * 1e10/h #This would be such that the snaps are in decreasing order and also after the infall only
        totMstar = tree['SubhaloMassType'][:, 4][subh_ixs] * 1e10/h #This would be such that the snaps are in decreasing order and also after the infall only
        
        

        if self.torb != None:
            max_time = 13.8 - subh_ages[-1]
            tpl = np.linspace(0, max_time, 25) #This is for plotting the results from Errani models
            mstar_model = np.zeros(0) #This is the stellar mass that we obtain from the model
            mmx_model = np.zeros(0) #This is the Mmx that the model gives
            for tevol in tpl:
                frem = self.evolve(tevol, 978) #This is Mmx/Mmx0 from the model
                mstar_model = np.append(mstar_model, self.get_mstar_model(tevol, frem) )
                mmx_model = np.append(mmx_model, frem * self.mmx0)
            ax_sub2.plot(tpl + all_ages[common_snaps[0]], np.log10(mstar_model), 'r--', label = 'model stellar mass')
            ax_sub2.plot(tpl + all_ages[common_snaps[0]], np.log10(mmx_model), 'r-', label = 'model Mmx')
        
        lab_var = True
        for ix in range(len(common_snaps)):
            # plot only if the file is available
            snap = int(common_snaps[ix])
            shid = int(np.flip(subh_sfid[subh_ixs])[ix]) #flipping because snaps would be in descending order o.w.
            filename = filepath + 'cutout_files/cutout_'+str(shid)+'_'+str(snap)+'.hdf5'
             #This is just to set the label of the 
            if os.path.exists(filename): 
                mmx_here = self.get_rot_curve(where = int(snap))[2]
                if lab_var:
                    ax_sub2.plot(all_ages[snap], np.log10(mmx_here), 'bo', label = 'Mdm from TNG')
                    lab_var = False
                else:
                    ax_sub2.plot(all_ages[snap], np.log10(mmx_here), 'bo')

        # ax_sub2.plot(np.flip(subh_ages), np.flip(np.log10(Mdm)), 'b-', label = r'Dark mattter in $2R_h$')
        # ax_sub2.plot(np.flip(subh_ages), np.flip(np.log10(Mstar)), 'b--', label = r'Stars in $2R_h$')
        # ax_sub2.plot(np.flip(subh_ages), np.flip(np.log10(totMdm)), 'g-', label = r'Dark mattter total')
        ax_sub2.plot(np.flip(subh_ages), np.flip(np.log10(totMstar)), 'b--', label = r'Stars in total')
        
        ax_sub2.set_ylabel(r'Mass ($\log M_\odot$)')
        # ax_sub2.set_yscale('log')
        ax_sub2.set_xlabel(r'Time (Gyr)')
        ax_sub2.legend(fontsize = 6)

        if show == True: plt.show()





