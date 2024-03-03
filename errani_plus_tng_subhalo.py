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
from testing_errani import get_rot_curve, get_rmxbyrmx0, get_vmxbyvmx0, get_mxbymx0, get_LbyL0, l10rbyrmx0_1by4_spl,l10rbyrmx0_1by2_spl, l10rbyrmx0_1by8_spl, l10rbyrmx0_1by16_spl, l10vbyvmx0_1by2_spl, l10vbyvmx0_1by4_spl, l10vbyvmx0_1by8_spl, l10vbyvmx0_1by16_spl
from tng_subhalo_and_halo import TNG_Subhalo
from matplotlib import gridspec
from colossus.cosmology import cosmology
from colossus.halo import concentration
from scipy.signal import argrelmin
import warnings
from populating_stars import *


cosmology.setCosmology('planck18')

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



class ErraniSubhalo():
    def __init__(self, torb, rperi, rapo, Rh, vmx0, rmx0, mmx0, mstar0):
        self.torb = torb 
        self.rperi = rperi 
        self.rapo = rapo 
        self.vmx0 = vmx0
        self.rmx0 = rmx0
        self.mmx0 = mmx0
        self.mstar0 = mstar0
        self.Rh = Rh
        
    def get_rh0byrmx0(self):
        '''
        This is a function to calculate the initial rh0burmx0 for the subhalo
        '''
        values = [1/2, 1/4, 1/8, 1/16]
        Rh0 = self.Rh
        
        closest_value = min(values, key=lambda x: abs(np.log10(x) - np.log10(Rh0/self.rmx0)))
        # print(closest_value)
        return closest_value
    

    # def get_rh_vd(self, frem):
    #     '''
    #     This is to obtain the Rh and vd from the frem
    #     '''
        
    #     return rh_now, vd_now
    


    def evolve(self, tevol, V0):
        '''
        This is a function that evolves the subhalo using Errani models

        Args:
        tinf: The infall time in Gyr of the subhalo  
        tevol (float): The time for which evolution must take place
        V0: This is a parameter of the host whn paramterized using the isothermal profile

        '''
        rmx0 = self.rmx0
        vmx0 = self.vmx0
        rperi = self.rperi
        rapo = self.rapo

        
        
        if rmx0 != rmx0:
            raise ValueError('rmx0 is NaN!')
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
                # print('Heavy mass loss regime')
                tasy = 0.22 * tperi
                y0 = (tmx0 - tasy) / tperi
                tau_asy = 0.65 * torb
                tau = tau_asy / y0 
                eta = 1 - np.exp( - 2.5 * y0 )
                inner_term = 1 + (t/tau)**eta 
                tmx = tasy + tperi * y0 * (inner_term)**(-1/eta)
            else: #modest mass loss regime
                # print('Modest mass loss regime')
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
        # print(tmx, rmx0)
        with warnings.catch_warnings(record=True) as w:
            rmx = fsolve(lambda rmx: get_tmx(rmx) - tmx, rmx0/1e2)[0]
            if len(w) > 0:
                with warnings.catch_warnings(record=True) as w:
                    rmx = fsolve(lambda rmx: get_tmx(rmx) - tmx, rmx0/1e3)[0]
                    if len(w) > 0:
                        with warnings.catch_warnings(record=True) as w:
                            rmx = fsolve(lambda rmx: get_tmx(rmx) - tmx, rmx0/1e4)[0]
                            if len(w) > 0:
                                rmx = fsolve(lambda rmx: get_tmx(rmx) - tmx, rmx0/1e5)[0]
            
        # print(rmx)
        frem = get_mxbymx0(rmx/rmx0) 
        
        mmx = frem * self.mmx0
        vmx = get_vmxbyvmx0(rmx/rmx0) * self.vmx0
        # print(f'frem = {100 * frem:.2f} % ')
        mstar = get_LbyL0(frem, self.get_rh0byrmx0()) * self.mstar0
        
        # rh0byrmx0 = self.get_rh0byrmx0()
        # rh_now, vd_now = self.get_rh_vd(frem)

        # The following part is to get the vd and rh
        rh0byrmx0 = self.get_rh0byrmx0()

        vd_diff = (get_vmxbyvmx0(get_rmxbyrmx0(10**-2.5)) * self.vmx0) - ((10 ** l10vbyvmx0_1by4_spl(-2.5)) * self.vmx0) #Calculating the difference at frem = 10**-2.5 and assuming it to be a constant

        if np.log10(frem) >= -2.5:
            if rh0byrmx0 == 0.25:
                rh_now = 10 ** (l10rbyrmx0_1by4_spl(np.log10(frem))) * self.rmx0
                vd_now = 10 ** (l10vbyvmx0_1by4_spl(np.log10(frem))) * self.vmx0
            elif rh0byrmx0 == 0.125:
                rh_now = 10 ** (l10rbyrmx0_1by8_spl(np.log10(frem))) * self.rmx0
                vd_now = 10 ** (l10vbyvmx0_1by8_spl(np.log10(frem))) * self.vmx0
            elif rh0byrmx0 == 0.5:
                rh_now = 10 ** (l10rbyrmx0_1by2_spl(np.log10(frem))) * self.rmx0
                vd_now = 10 ** (l10vbyvmx0_1by2_spl(np.log10(frem))) * self.vmx0
            elif rh0byrmx0 == 0.0625:
                rh_now = 10 ** (l10rbyrmx0_1by16_spl(np.log10(frem))) * self.rmx0
                vd_now = 10 ** (l10vbyvmx0_1by16_spl(np.log10(frem))) * self.vmx0


        elif np.log10(frem) < -2.5:
            rh_now = rmx 
            vd_now = vmx - vd_diff




        if mstar < 10: #FIXME:Decide #10 on some limit for stellar mass from Errani model
            mstar = 0
            rh_now = 0

        

        return  vmx, rmx, mmx, vd_now, rh_now, mstar



class Subhalo(TNG_Subhalo):
    '''
    This class is for a subhalo in general which we are using for evolution, has to be something in TNG
    '''
    #FIXME: #9 Extend Subhalo class and the TNG_Subhalo class to non FoF0 cases
    def __init__(self, sfid, snap, last_snap, central_sfid_99):
        '''
        Args:
        sfid: The subfind ID at infall snapshot
        snap: The snapshot of infall
        last_snap: has to be passed, this is the last snapshot of survival for the subhalo
        central_sfid_99: This is the SubfindID of the subhalos at z = 0. Used to get all the FoF properties for orbit calc, etc.
        '''
        self.sfid = sfid
        self.snap = snap 
        self.last_snap = last_snap
        self.central_sfid_99 = central_sfid_99
        self.__initialize_central_props() #this is to have all the central subhalos properties as instance attributes

        if last_snap != 99:
            self.merged = True
        else:
            self.merged = False

        fields = ['SubhaloMassInRadType', 'SubhaloGrNr', 'SubfindID', 'SnapNum', 'GroupNsubs', 'SubhaloMassInHalfRadType',
            'SubhaloPos', 'Group_R_Crit200', 'Group_M_Crit200', 'SubhaloVel', 'SubhaloHalfmassRadType', 'SubhaloMassType', 'SubhaloLenType', 'SubhaloVmaxRad', 
            'SubhaloMassInMaxRadType', 'SubhaloIDMostbound', 'SubhaloVelDisp', 'GroupFirstSub', 'SubhaloVmax']
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

        
        self.torb = None 
        self.rperi = None 
        self.rapo = None 
        
        self.mstar = max(self.tree['SubhaloMassType'][:, 4][self.tree['SnapNum'] >= self.snap]) * 1e10 / h #This is the maximum stellar mass at infall
        # try:
        #     self.vmax = self.get_vmax(where = 'max')
        #     self.Rh = self.get_rh(where = 'max')
        # except Exception as e:
        #     print(e)
        #     self.vmax = self.get_vmax(where = 'max')
        #     self.Rh = self.get_rh(where = 'max')


        if self.mstar >= 5e6: #If we have stellar mass < 5e6 Msun, we take stellar mass from Vmax
            #in this case, assuming there are sufficient stars to give us valid values for Rh and M*
            try:
                self.Rh = self.get_rh(where = 'max')*3/4
                self.vd = self.get_vd(where = 'max')
            except Exception as e:
                print(e)
                self.Rh = self.get_rh(where = int(self.snap))*3/4
                self.vd = self.get_vd(where = int(self.snap))

            with warnings.catch_warnings(record=True) as w:
                self.vmx0, self.rmx0, self.mmx0 = self.get_mx_values(where = int(self.snap))
                if len(w) > 0:
                    print(w)
                    self.vmx0, self.rmx0, self.mmx0 = self.get_rot_curve(where= int(self.snap))
        else:
            try:
                self.vmax = self.get_vmax(where = 'max')
            except Exception as e:
                print(e)
                self.vmax = self.get_vmax(where = int(self.snap))
            self.mstar = get_mstar_co_wsc(np.log10(self.vmax)) #This would be the stellar mass in Msun after application of scatter
            if self.mstar < 0:
                self.mstar = np.array([1e-20])
            self.Rh = get_rh_wsc(np.log10(self.mstar)) #This would be the half light radius. 
            self.vd = self.get_vd(where = int(self.snap)) #guessing we wouldn't have a max for some subhalos, just use it at infall
            with warnings.catch_warnings(record=True) as w:
                self.vmx0, self.rmx0, self.mmx0 = self.get_mx_values(where = int(self.snap))
                if len(w) > 0:
                    for warning in w:
                        print(warning.message)
                    self.vmx0 = self.get_vmax(where = int(self.snap))
                    self.rmx0 = self.get_rh(where = int(self.snap), how = 'vmax')
                    self.mmx0 = self.get_mdm(where = int(self.snap), how = 'vmax')
        




    def __initialize_central_props(self):
        central_id_at99 = self.central_sfid_99


        self.central_fields = ['GroupFirstSub', 'SubhaloGrNr', 'SnapNum', 'GroupNsubs', 'SubhaloPos', 'Group_R_Crit200', 'Group_M_Crit200', 'SubhaloVel']
        self.central_tree = il.sublink.loadTree(basePath, 99, central_id_at99, fields = self.central_fields, onlyMPB = True)
        self.central_snaps = self.central_tree['SnapNum']
        self.central_redshift = all_redshifts[self.central_snaps]
        self.central_x =  self.central_tree['SubhaloPos'][:, 0]/(1 + self.central_redshift)/h
        self.central_y =  self.central_tree['SubhaloPos'][:, 1]/(1 + self.central_redshift)/h
        self.central_z =  self.central_tree['SubhaloPos'][:, 2]/(1 + self.central_redshift)/h
        self.central_r200 = self.central_tree['Group_R_Crit200']/(1 + self.central_redshift)/h #This is the virial radius of the group
        self.ages_rvir = all_ages[self.central_snaps] #Ages corresponding to the virial radii
        self.central_grnr = self.central_tree['SubhaloGrNr']
        self.central_gr_m200 = self.central_tree['Group_M_Crit200']*1e10/h #This is the M200 of the central group
        self.central_vx = self.central_tree['SubhaloVel'][:, 0] #km/s
        self.central_vy = self.central_tree['SubhaloVel'][:, 1]
        self.central_vz = self.central_tree['SubhaloVel'][:, 2]
        self.central_v0 = np.sqrt(4.3e-6 * self.central_gr_m200 / self.central_r200) #this is the isothermal speed of the FoF halo for all snapshots

        return None


  
    def get_rh0byrmx0(self):
        '''
        This is a function to calculate the initial rh0burmx0 for the subhalo
        '''
        values = [1/2, 1/4, 1/8, 1/16]
        
        Rh0 = self.get_rh(where = 'max')*3./4 #FIXME: This needs to accound for the subhalos without measured Rh

        
        closest_value = min(values, key=lambda x: abs(np.log(x) - np.log(Rh0/self.rmx0)))
        # print(closest_value)
        return closest_value



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
            tree = il.sublink.loadTree(basePath, self.snap, self.sfid, fields = fields, onlyMDB = True) #From this we obtain all the desencdants of the subhalo at infall
            tree.pop('count') #removing a useless key from the dictionary
            snaps_temp = tree['SnapNum']
            sfids_temp = tree['SubfindID']

            merger_index = np.where(snaps_temp == self.last_snap)[0][-1] #this is the index of the merger in the tree
            # msh_if_ix_tree = np.where((snaps_temp == snap) & (sfids_temp == sfid))[0].item() #This is the infall index in the tree
            tree = {key: value[merger_index:] for key, value in tree.items()} #new tree which only runs from final existing snapshot to the infall snapshot

            # infall_ix = np.where((msh_snap == snap) & (msh_sfid == sfid))[0] #This is to get the index of the current subhalo in the merger dataframe
            # msh_last_snap = msh_merger_snap[infall_ix] #This is the infall snapshot
            # msh_last_sfid = msh_merger_sfid[infall_ix] #This is the infall subfind ID

            # tree = il.sublink.loadTree(basePath, int(msh_last_snap), int(msh_last_sfid), fields = fields, onlyMPB = True) #Getting all the progenitors from the last snapshot of survival
            # tree.pop('count') #removing a useless key from the dictionary
            # snaps_temp = tree['SnapNum']
            # sfids_temp = tree['SubfindID']
            # msh_if_ix_tree = np.where((snaps_temp == snap) & (sfids_temp == sfid))[0].item() #This is the infall index in the tree
            # tree = {key: value[0:msh_if_ix_tree+1] for key, value in tree.items()} #new tree which only runs from final existing snapshot to the infall snapshot

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


        common_snaps = np.intersect1d(subh_snap, self.central_snaps) #This is in acsending order. Descending order after flipping
        common_snaps_des = np.flip(common_snaps) #This will be in descending order to get the indices in self.central_is and subh_ix
        central_ixs = np.where(np.isin(self.central_snaps, common_snaps))[0] #getting the indices of common snaps in the central suhalo. The indices are in such a way that the snaps are again in descending order.
        subh_ixs = np.where(np.isin(subh_snap, common_snaps))[0]


        subh_z_if = all_redshifts[common_snaps[0]] #This is the redshift of infall
        subh_dist = np.sqrt((subh_x[subh_ixs] - self.central_x[central_ixs])**2 + (subh_y[subh_ixs] - self.central_y[central_ixs])**2 + (subh_z[subh_ixs] - self.central_z[central_ixs])**2)
        subh_ages = np.flip(all_ages[common_snaps]) #The ages after flipping will be in descending order


        snap_r200_if = None
        for ix, sx in enumerate(common_snaps_des):
            '''
            This loop is to go through all the common indices in descending order
            '''
            # print(ix)
            if subh_dist[subh_ixs[common_snaps_des == sx]] < self.central_r200[self.central_snaps == sx]: # As weird as it sounds, this is the first when the subhalos distance is below the virial radius (on completing the loop)
                snap_r200_if = sx 
            if ix > 0 and ix < len(common_snaps_des) - 1:
                if subh_dist[subh_ixs[common_snaps_des == common_snaps_des[ix]]] < subh_dist[subh_ixs[common_snaps_des == common_snaps_des[ix - 1]]] and subh_dist[subh_ixs[common_snaps_des == common_snaps_des[ix]]] < subh_dist[subh_ixs[common_snaps_des == common_snaps_des[ix + 1]]]:
                    first_peri_snap = sx

        if snap_r200_if == None:
            raise ValueError('This subhalo does not enter Rvir') #FIXME: #14 Some subhalos do not enter the virial radius
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

        subh_vx_cen = subh_vx[te_subh_ix] - self.central_vx[te_central_ix]
        subh_vy_cen = subh_vy[te_subh_ix] - self.central_vy[te_central_ix]
        subh_vz_cen = subh_vz[te_subh_ix] - self.central_vz[te_central_ix]    


        subh_x_cen = subh_x[te_subh_ix] - self.central_x[te_central_ix]
        subh_y_cen = subh_y[te_subh_ix] - self.central_y[te_central_ix]
        subh_z_cen = subh_z[te_subh_ix] - self.central_z[te_central_ix]




        mvir = self.central_gr_m200[te_central_ix].item() #this would be the virial mass of the FoF halo infall time 
        
        # This is method 3: Using galpy. For codes using other methods, look at subhalo_time_evolution.py
        # def get_nfw_at_t(t):
        #     '''
        #     This is to mdel the time dependence of the galpy potential.
        #     -o- potential is a nonlinear function in the virial mass because of the scale radius.
        #     -o- Using taylor expansion of the NFW potential to have the correction to first order in r/rvir.
        #     '''
        #     t = t + te_time
        #     # A = np.sqrt(t) #testing so that nothing blows up
        #     # A = (fof_m200_t(t) / (fof_r200_t(t))) * ((fof_r200_t(te_time)) / fof_m200_t(te_time))
        #     A = 1
        #     return A

        potential = NFWPotential(conc=concentration.concentration(0.6744 * mvir, 'vir', 0, 'ludlow16'), mvir=mvir/1e12, wrtcrit = True, overdens = 200 * get_critical_dens(te_snap_z)/get_critical_dens(0))
        # potential = TimeDependentAmplitudeWrapperPotential(A = get_nfw_at_t, pot = nfw) #This is to vary the potential with time 

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
        subhalo_orbit2 = Orbit(initial_conditions)

        # Integrate the orbit
        if when_te == 'last':
            ts = np.linspace(0, -10 , 1000) #This is for 13.8 Gyr before the given snapshot. 
        else: #Since I am already raising a ValueError before, I am using else directly here
            ts = np.linspace(0, 10 - te_time.item(), 500) #This is only for the remaining time

        subhalo_orbit.integrate(ts * u.Gyr, potential, method = 'leapfrog')

        try:
            torb = subhalo_orbit.Tr(use_physical = True, type = 'spherical')
            rapo = subhalo_orbit.rap(use_physical = True, type= 'spherical')
            rperi = subhalo_orbit.rperi(use_physical = True, type= 'spherical')
        except Exception as e:
            # Following is a weird way to obtain the orbit data since this does not work directly in galpy 1.7
            # try:
            ts2 = np.linspace(0, -10 , 1000)
            subhalo_orbit2.integrate(ts2 * u.Gyr, potential, method = 'leapfrog')
            fig, = subhalo_orbit2.plot(d1 = 't', d2 = 'x')
            plt.close()
            fig2, = subhalo_orbit2.plot(d1 = 'y', d2 = 'z')
            plt.close()

            t_gp = fig.get_xdata() + te_time
            x_gp = fig.get_ydata()
            y_gp = fig2.get_xdata()
            z_gp = fig2.get_ydata()

            dist_gp = np.sqrt(x_gp**2 + y_gp**2 + z_gp**2)
            rapo = max(dist_gp)
            rperi = min(dist_gp) 
            minima_indices = argrelmin(dist_gp)[0]
            time_of_min = t_gp[minima_indices]

            if len(minima_indices) <= 1: #If we have <= 1 element as list of minima, we cannot find the time period again 
                ts2 = np.linspace(0, -27 , 2700)
                subhalo_orbit2.integrate(ts2 * u.Gyr, potential, method = 'leapfrog')
                fig, = subhalo_orbit2.plot(d1 = 't', d2 = 'x')
                plt.close()
                fig2, = subhalo_orbit2.plot(d1 = 'y', d2 = 'z')
                plt.close()

                t_gp = fig.get_xdata() + te_time
                x_gp = fig.get_ydata()
                y_gp = fig2.get_xdata()
                z_gp = fig2.get_ydata()

                dist_gp = np.sqrt(x_gp**2 + y_gp**2 + z_gp**2)
                rapo = max(dist_gp)
                rperi = min(dist_gp) 
                minima_indices = argrelmin(dist_gp)[0]

                # print(rperi)
                # print(np.diff(t_gp[minima_indices]))
                time_of_min = t_gp[minima_indices]

            if len(time_of_min) > 1: #In this case, we can find the time period because we have measured two local minima   
                torb = -1 * np.mean(np.diff(time_of_min))
            else:
                torb = np.inf #This is the case where orbital time is >= 13.8 Gyrs, we are just going to assume in this case that this is not going to evolve.

                # print(rperi)
                # print(np.diff(t_gp[minima_indices]))
                # torb = -1 * np.mean(np.diff(t_gp[minima_indices]))
                
            # print(dist_gp[minima_indices])
            # print()
        assert ((torb == torb) and (torb != None)), 'This is unexpected, check torb calculations'
            
            # raise ValueError('Orbital time not currently available')
            

        self.torb = torb
        self.rapo = rapo 
        self.rperi = rperi

        # return float(all_ages[99] - all_ages[common_snaps[0]])
        return float(all_ages[99] - all_ages[snap_r200_if])


    def get_model_values(self, tinf, tevol):
        if self.torb == np.inf: #If the orbital time is ttoo big, then just assume there is evolution
            return self.vmx0, self.rmx0, self.mmx0, self.vd, self.Rh, self.mstar
        errani_start_subh = ErraniSubhalo(self.torb, self.rperi, self.rapo, self.Rh, self.vmx0, self.rmx0, self.mmx0, self.mstar)
        tinf = 10 #FIXME: update this after you are done with testing
        if tinf > 9:
            # vmxf, rmxf, mmxf, vdf, rhf, mstarf =  errani_start_subh.evolve(tevol, V0 = central_v0[0]) #just assume a constant V0
            avg_v0_b4_9 = (self.central_v0[self.central_snaps == self.snap] + self.central_v0[self.central_snaps == 99])/2.
            # print(avg_v0_b4_9)
            vmxf, rmxf, mmxf, vdf, rhf, mstarf =  errani_start_subh.evolve(tevol, V0 = avg_v0_b4_9) #just assume a constant V0, but we are taking an average here 
        elif tevol + tinf <  9:
            avg_v0_b4_9 = (self.central_v0[self.central_snaps == self.snap] + self.central_v0[self.central_snaps == np.searchsorted(all_ages, tevol+tinf) - 1])/2.
            vmxf, rmxf, mmxf, vdf, rhf, mstarf = errani_start_subh.evolve(tevol, V0 = avg_v0_b4_9)
        else:
            avg_v0_b4_9 = (self.central_v0[self.central_snaps == self.snap] + self.central_v0[self.central_snaps == np.searchsorted(all_ages, tevol+tinf) - 1])/2.
            vmx1, rmx1, mmx1, vd1, rh1, mstar1 = errani_start_subh.evolve(9 - tinf, V0 = avg_v0_b4_9)
            errani_last_subh = ErraniSubhalo(self.torb, self.rperi, self.rapo, rh1, vmx1, rmx1, mmx1, mstar1)
            vmxf, rmxf, mmxf, vdf, rhf, mstarf = errani_last_subh.evolve(tevol - (9 - tinf), V0 = self.central_v0[0])
            # frem = mmxf/self.mmx0
            #first run for the first half,
            #then use that output for the input of second half
            # errani_last_subh = ErraniSubalo()
            #evolve the last subh again
            #get the required values out of this
            #do not use Mmx/Mmx0 to calculate anything because it is nomore a single evolution

        return vmxf, rmxf, mmxf, vdf, rhf, mstarf
    

    def get_tng_values(self, where):
        '''
        This function is to get the values that we obtain from the simulation as we proceed
        FIXME: This currently calculates vmx, rmx and Mmx. Update accordingly 
        '''
        subh_tree = il.sublink.loadTree(basePath, self.snap, self.sfid, fields = ['SnapNum', 'SubfindID'], onlyMDB = True)
        subh_snap_99 = subh_tree['SnapNum'][0] #First snap is generally 99, so starting from there
        subh_sfid_99 = subh_tree['SubfindID'][0]
        vmx, rmx, mmx = self.get_rot_curve(where)

        return vmx, rmx, mmx

   
    
    
    


    def plot_orbit_comprehensive(self, merged, when_te = 'last', show = False):
        '''
        This function plots the orbit comprehensively with energy and masses variation with time
        
        This function plots the orbit, the velocities in x, y and z, and the total energy; FIXME: has to be shortened further after testing different techniques. 

        Args:
        snap and sfid: THIS HAS TO BE THE INFALL SNAPSHOT FOR SUBHALOS THAT MERGE and the infall snapshot for the surviving subhalos
        ax_ar: This is an array of plotting axes. Please pass 5 axes
        merged(boolean): If the subhalo merged, then please give this as True

        Returns None because this is only a plotting routine
        ''' 

        fig = plt.figure(figsize=(15, 6))
        gs = gridspec.GridSpec(2, 3, width_ratios=[3, 1.5, 1.5]) #Generating one big image to the left and four small images to the right, from ChatGPT
        ax_sub2 = plt.subplot(gs[:, 0])
        ax = plt.subplot(gs[0, 1]) #For energy
        ax_sub1 = plt.subplot(gs[1, 1]) #For masses
        ax_sub3 = plt.subplot(gs[0, 2]) #For radii
        ax_sub4 = plt.subplot(gs[1, 2]) #For velocity dispersion


        snap = self.snap
        sfid = self.sfid
        fields = ['SubhaloGrNr', 'GroupFirstSub', 'SnapNum', 'SubfindID', 'SubhaloPos', 'SubhaloVel', 'SubhaloMassType', 'SubhaloMassInRadType', 'SubhaloHalfmassRadType', 'SubhaloVelDisp', 'SubhaloMassInHalfRadType'] #These are the fields that will be taken for the subhalos
        if not merged:
            tree = il.sublink.loadTree(basePath, snap, sfid, fields = fields, onlyMDB = True) #This only works for surviving subhalos
        else: #If it merges
            tree = il.sublink.loadTree(basePath, self.snap, self.sfid, fields = fields, onlyMDB = True) #From this we obtain all the desencdants of the subhalo at infall
            tree.pop('count') #removing a useless key from the dictionary
            snaps_temp = tree['SnapNum']
            sfids_temp = tree['SubfindID']

            merger_index = np.where(snaps_temp == self.last_snap)[0][-1] #this is the index of the merger in the tree
            # msh_if_ix_tree = np.where((snaps_temp == snap) & (sfids_temp == sfid))[0].item() #This is the infall index in the tree
            tree = {key: value[merger_index:] for key, value in tree.items()} #new tree which only runs from final existing snapshot to the infall snapshot

            # infall_ix = np.where((msh_snap == snap) & (msh_sfid == sfid))[0] #This is to get the index of the current subhalo in the merger dataframe
            # msh_last_snap = msh_merger_snap[infall_ix] #This is the infall snapshot
            # msh_last_sfid = msh_merger_sfid[infall_ix] #This is the infall subfind ID

            tree = il.sublink.loadTree(basePath, int(msh_last_snap), int(msh_last_sfid), fields = fields, onlyMPB = True) #Getting all the progenitors from the last snapshot of survival
            tree.pop('count') #removing a useless key from the dictionary
            snaps_temp = tree['SnapNum']
            sfids_temp = tree['SubfindID']
            msh_if_ix_tree = np.where((snaps_temp == snap) & (sfids_temp == sfid))[0].item() #This is the infall index in the tree
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


        common_snaps = np.intersect1d(subh_snap, self.central_snaps) #This is in acsending order. Descending order after flipping
        common_snaps_des = np.flip(common_snaps) #This will be in descending order to get the indices in central_ix and subh_ix
        central_ixs = np.where(np.isin(self.central_snaps, common_snaps))[0] #getting the indices of common snaps in the central suhalo. The indices are in such a way that the snaps are again in descending order.
        subh_ixs = np.where(np.isin(subh_snap, common_snaps))[0]


        subh_z_if = all_redshifts[common_snaps[0]] #This is the redshift of infall
        subh_dist = np.sqrt((subh_x[subh_ixs] - self.central_x[central_ixs])**2 + (subh_y[subh_ixs] - self.central_y[central_ixs])**2 + (subh_z[subh_ixs] - self.central_z[central_ixs])**2)
        subh_ages = np.flip(all_ages[common_snaps]) #The ages after flipping will be in descending order



        for ix, sx in enumerate(common_snaps_des):
            '''
            This loop is to go through all the common indices in descending order
            '''
            # print(ix)
            if subh_dist[subh_ixs[common_snaps_des == sx]] < self.central_r200[self.central_snaps == sx]: # As weird as it sounds, this is the first when the subhalos distance is below the virial radius (on completing the loop)
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
        

        # print(f'te_snap = {te_snap}')

        te_snap_z = all_redshifts[te_snap] #This is the refshift at the total energy snapshot
        te_time = all_ages[te_snap] #This is the time where we are inputting the energy to galpy
        te_snap_ix = te_snap == common_snaps_des
        # Discovery of the day: This index has to be input to subh_x itself or central_x itself. Not any subset of it!
        te_subh_ix = subh_ixs[te_snap_ix] #Subhalo index for this infall time. 
        te_central_ix = self.central_ixs[te_snap_ix]

        subh_vx_cen = subh_vx[te_subh_ix] - self.central_vx[te_central_ix]
        subh_vy_cen = subh_vy[te_subh_ix] - self.central_vy[te_central_ix]
        subh_vz_cen = subh_vz[te_subh_ix] - self.central_vz[te_central_ix]    


        subh_x_cen = subh_x[te_subh_ix] - self.central_x[te_central_ix]
        subh_y_cen = subh_y[te_subh_ix] - self.central_y[te_central_ix]
        subh_z_cen = subh_z[te_subh_ix] - self.central_z[te_central_ix]


        mvir = self.central_gr_m200[te_central_ix].item() 

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

        potential = NFWPotential(conc=concentration.concentration(0.6744 * mvir, 'vir', 0, 'ludlow16'), mvir=mvir/1e12, wrtcrit = True, overdens = 200 * get_critical_dens(te_snap_z)/get_critical_dens(0))
        # potential = TimeDependentAmplitudeWrapperPotential(A = get_nfw_at_t, pot = nfw) #This is to vary the time 

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
        
        midway = False
        # Integrate the orbit
        # Integrate the orbit
        if when_te == 'last':
            ts = np.arange(0, all_ages[common_snaps[0] - 1] -all_ages[te_snap] , -0.1) #This is for 13.8 Gyr before the given snapshot. 
            subhalo_orbit2 = Orbit(initial_conditions)
            if all_ages[-1] > all_ages[te_snap]:
                midway = True
                ts2 = np.arange(0, all_ages[-1] - all_ages[te_snap], 0.1)
                subhalo_orbit2.integrate(ts2 * u.Gyr, potential, method = 'leapfrog')
                fig, = subhalo_orbit2.plot(d1 = 't', d2 = 'x')
                plt.close()
                fig2, = subhalo_orbit2.plot(d1 = 'y', d2 = 'z')
                plt.close()
                fig3, = subhalo_orbit2.plot(d1 = 'vx', d2 = 'vy')
                plt.close()
                fig4, = subhalo_orbit2.plot(d1 = 'vz', d2 = 'E')
                plt.close()

                t2_gp = fig.get_xdata() + te_time
                x2_gp = fig.get_ydata()
                y2_gp = fig2.get_xdata()
                z2_gp = fig2.get_ydata()
                vx2_gp = fig3.get_xdata()
                vy2_gp = fig3.get_ydata()
                vz2_gp = fig4.get_xdata()
                E2_gp = fig4.get_ydata()

                # print(t2_gp)
                dist2_gp = np.sqrt(x2_gp**2 + y2_gp**2 + z2_gp**2)

        else: #Since I am already raising a ValueError before, I am using else directly here
            ts = np.linspace(0, 13.8 - te_time.item(), 100) #This is only for the remaining time

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

        # print(t_gp)

        dist_gp = np.sqrt(x_gp**2 + y_gp**2 + z_gp**2)


        if midway:
            t_gp = np.append(np.flip(t2_gp), t_gp)
            x_gp = np.append(np.flip(x2_gp), x_gp)
            y_gp = np.append(np.flip(y2_gp), y_gp)
            z_gp = np.append(np.flip(z2_gp), z_gp)
            vx_gp = np.append(np.flip(vx2_gp), vx_gp)
            vy_gp = np.append(np.flip(vy2_gp), vy_gp)
            vz_gp = np.append(np.flip(vz2_gp), vz_gp)
            E_gp = np.append(np.flip(E2_gp), E_gp)

            dist_gp = np.sqrt(x_gp**2 + y_gp**2 + z_gp**2)


        # print(f't_gp = {t_gp}')
        # ========== PLOTTING PART ======================== 
        

        ax.plot(np.flip(subh_ages), np.flip(subh_dist), lw = 1, color = 'blue', label = r'Orbit')
        ax.plot(all_ages[self.central_snaps[central_ixs]], self.central_r200[central_ixs], c = 'gray', ls = '--',label = r'$R_{200}$', lw = 0.3)
        ax.plot(t_gp, dist_gp, lw = 0.5, alpha = 0.8, label = 'galpy orbit', color = 'r')
        ax.set_ylabel('Distance (kpc)')
        ax.set_xlabel('Time (Gyr)')
        ax.set_xlim(left = 0.95*all_ages[common_snaps[0]], right = 1.05*all_ages[99])
        
        subh_max_mstar = round(np.log10(max(subh_mstar)), 2)
        ax.set_title(f'ID at snap {int(self.last_snap)} is '+str(tree['SubfindID'][0])+r'    $\rm{\log_{10}M_\bigstar} = $'+str(subh_max_mstar), fontsize = 10)
        # ax.text(1.05, 0.25, r'$\rm{\log_{10}M_\bigstar} = $'+str(subh_max_mstar), transform=ax.transAxes, fontsize = 11)

        

        

        ax.axhline(max(dist_gp), c = 'red', ls = ':', alpha = 0.5, label = 'peri/apo from galpy')
        ax.axhline(min(dist_gp), c = 'red', ls = ':', alpha = 0.5)

        # plt.legend()
        # ax.legend(fontsize = 10, loc='upper left', bbox_to_anchor=(1, 1))
        ax.legend(fontsize = 6)

        

        '''
        Here, we calculate the energy as it evolves. The energy being calculated for the orbit has mass changing with time and critical density also changes
        '''
        pe_ar = np.zeros(0) 
        pe_ar_gp = np.zeros(0) #This is calculation of the potential energy for galpy orbit same as above
        for (ix, dist) in enumerate(subh_dist):
            snap = common_snaps_des[ix]
            pe_ar = np.append(pe_ar, get_nfw_potential(dist, self.central_gr_m200[central_ixs[ix]], 8, all_redshifts[snap]))

        dist_gp = np.sqrt(x_gp**2 + y_gp**2 + z_gp**2)
        # print(t_gp)
        for (ix, t) in enumerate(t_gp): #This is a loop over galpy
            snap = np.searchsorted(all_ages, t) - 1 #index of the snapshot that has age lower than this time
            # print(f'{len(pe_ar_gp)} out of {len(t_gp)}')
            pe_ar_gp = np.append(pe_ar_gp, get_nfw_potential(dist_gp[ix], self.central_gr_m200[self.central_snaps == snap], 8, all_redshifts[snap])) 

            

        ke_ar = 0.5 *((subh_vx[subh_ixs] - self.central_vx[central_ixs])**2 + (subh_vy[subh_ixs] - self.central_vy[central_ixs])**2 + (subh_vz[subh_ixs] - self.central_vz[central_ixs])**2)
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
        ax_sub1.set_xlabel('Time (Gyr)')
        ax_sub1.set_ylabel(r'$E \, \rm{(km/s)^2}$')
        ax_sub1.legend(fontsize = 6)

       
        #This is now for the third subplot:
        #The plan is to plot the mass variation with time
        Mdm = tree['SubhaloMassInRadType'][:, 1][subh_ixs] * 1e10/h #This would be such that the snaps are in decreasing order and also after the infall only
        Mstar = tree['SubhaloMassInRadType'][:, 4][subh_ixs] * 1e10/h #This would be such that the snaps are in decreasing order and also after the infall only

        totMdm = tree['SubhaloMassType'][:, 1][subh_ixs] * 1e10/h #This would be such that the snaps are in decreasing order and also after the infall only
        totMstar = tree['SubhaloMassType'][:, 4][subh_ixs] * 1e10/h #This would be such that the snaps are in decreasing order and also after the infall only
        rh_tng = tree['SubhaloHalfmassRadType'][:, 4][subh_ixs]/h/(np.ones(len(subh_ixs)) + all_redshifts[common_snaps_des] )
        vd_tng = tree['SubhaloVelDisp'][subh_ixs] #this is in km/s


        if self.torb != None: #this is for the model part
            # print(f'torb = {self.torb}')
            max_time = 13.8 - subh_ages[-1]
            tpl = np.linspace(0, max_time, 25) #This is for plotting the results from Errani models
            mstar_model = np.zeros(0) #This is the stellar mass that we obtain from the model
            mmx_model = np.zeros(0) #This is the Mmx that the model gives
            rmx_model = np.zeros(0)
            vmx_model = np.zeros(0)    

            rh_model = np.zeros(0) #The projected half light radius from the model
            vd_model = np.zeros(0) #The projected los velocity dispersion from the model
            for tevol in tpl:
                vmx, rmx, mmx, vd, rh, mstar = self.get_model_values(float(all_ages[self.snap]), tevol) #This is Mmx/Mmx0 from the model
                # mstar, rh, vd = self.get_starprops_model(tevol, frem)
                mstar_model = np.append(mstar_model, mstar)
                rh_model = np.append(rh_model, rh)
                vd_model = np.append(vd_model, vd)
                mmx_model = np.append(mmx_model, mmx)
                # print(type(np.asscalar(frem)))
                rmx_model = np.append(rmx_model, rmx)
                vmx_model = np.append(vmx_model, vmx)
            ax_sub2.plot(tpl + all_ages[common_snaps[0]], np.log10(mstar_model), 'r--', label = 'model stellar mass')
            ax_sub2.plot(tpl + all_ages[common_snaps[0]], np.log10(mmx_model), 'r-', label = 'model Mmx')

            ax_sub3.plot(tpl + all_ages[common_snaps[0]], rh_model, 'r--', label = r'$R_h$ model')
            ax_sub3.plot(tpl + all_ages[common_snaps[0]], rmx_model, 'r-', label = r'$r_{\rm{mx}}$ model')

            ax_sub4.plot(tpl + all_ages[common_snaps[0]], vd_model, 'r--', label = r'$\sigma$ model')
            ax_sub4.plot(tpl + all_ages[common_snaps[0]], vmx_model, 'r-', label = r'$v_{\rm{mx}}$ model')
        
        lab_var = True
        # Uncomment the following line for analyzing the cutout files again
        for ix in range(len(common_snaps)):
            # plot only if the file is available
            snap = int(common_snaps[ix])
            shid = int(np.flip(subh_sfid[subh_ixs])[ix]) #flipping because snaps would be in descending order o.w.
            filename = filepath + 'cutout_files/cutout_'+str(shid)+'_'+str(snap)+'.hdf5'
             #This is just to set the label of the 
            if os.path.exists(filename): 
                vmx_here, rmx_here, mmx_here = self.get_rot_curve(where = int(snap))
                # print(vmx_here, rmx_here, mmx_here)
                # print(f'vmx/vmx0 = {vmx_here/self.vmx0:.2f}')
                # print(f'rmx/rmx0 = {rmx_here/self.rmx0:.2f}')
                if lab_var:
                    ax_sub2.plot(all_ages[snap], np.log10(mmx_here), 'bo', label = r'$M_{\rm{mx}}$ from TNG')
                    ax_sub3.plot(all_ages[snap], rmx_here, 'bo', label = r'$r_{\rm{mx}}$ from TNG')
                    ax_sub4.plot(all_ages[snap], vmx_here, 'bo', label = r'$v_{\rm{mx}}$ from TNG')
                    lab_var = False
                else:
                    ax_sub2.plot(all_ages[snap], np.log10(mmx_here), 'bo')
                    ax_sub3.plot(all_ages[snap], rmx_here, 'bo')
                    ax_sub4.plot(all_ages[snap], vmx_here, 'bo')

        ax_sub2.plot(np.flip(subh_ages), np.flip(np.log10(totMdm)), color = 'chocolate', label = r'Dark mattter total')
        ax_sub2.plot(np.flip(subh_ages), np.flip(np.log10(totMstar)), 'b--', label = r'Stars in total')
        
        ax_sub2.set_ylabel(r'Mass ($\log M_\odot$)')
        # ax_sub2.set_yscale('log')
        ax_sub2.set_xlabel(r'Time (Gyr)')
        ax_sub2.legend(fontsize = 6)

        ax_sub3.plot(np.flip(subh_ages), np.flip(rh_tng/np.sqrt(2)), 'b--', label = r'$R_h$ TNG')
        ax_sub3.set_yscale('log')
        ax_sub3.set_ylabel(r'$R_h$ or $r_{\rm{mx}}$ (kpc)')
        ax_sub3.set_xlabel('Time (Gyr)')
        ax_sub3.legend(fontsize = 6)

        ax_sub4.plot(np.flip(subh_ages), np.flip(vd_tng), 'b--', label = r'$\sigma$ TNG')
        ax_sub4.set_ylabel(r'$\sigma$ or $v_{\rm{mx}}$ (km/s)')
        ax_sub4.set_xlabel('Time (Gyr)')
        ax_sub4.legend(fontsize = 6)

        if show == True: plt.show()





