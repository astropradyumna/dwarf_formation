'''
This program is a second attempt to find the orbits of the subhalo given the energy and the momentum
'''

import requests
import h5py
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import requests
import IPython
import illustris_python as il
import os
from scipy.optimize import fsolve
from scipy.interpolate import UnivariateSpline
from scipy.misc import derivative
from tqdm import tqdm
import time
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import brentq
from scipy.optimize import minimize


baseUrl = 'https://www.tng-project.org/api/TNG50-1/'
headers = {"api-key":"894f4df036abe0cb9d83561e4b1efcf1"}
basePath = '/mainvol/jdopp001/L35n2160TNG_fixed/output'

h = 0.6774
mass_dm = 3.07367708626464e-05 * 1e10/h #This is for TNG50-1
G = 4.5390823753559603e-39 #This is in kpc, Msun and seconds



def find_first_min_max(arr):
    '''
    This function is used to return the first maximum and minimum for an array.
    We use this for the subhalo distance from the center. This gives the first pericenter and apocenter
    '''

    n = len(arr)

    first_min = None
    first_max = None

    for i in range(1, n - 1):
        if first_min is not None and first_max is not None:
            break
        
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1] and first_min is None:
            first_min = arr[i]

        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1] and first_max is None:
            first_max = arr[i]

        

    if first_min == None or first_max == None:
        raise ValueError(f"Cannot find first min or max")

    return first_min, first_max


def get_H(z, h = 0.6774):
    '''
    Calculates the Hubble constant as a function of redshift z in km/s/Mpc
    '''
    # Cosmological model parameters
    hubble_constant = h*100  # Hubble constant in km/s/Mpc
    matter_density = 0.31  # Density of matter in the universe
    dark_energy_density = 0.69  # Density of dark energy in the universe

    # Calculate the Hubble parameter
    hubble_parameter = hubble_constant * np.sqrt(matter_density * (1 + z)**3 + dark_energy_density)

    return hubble_parameter


def get_critical_dens(z):
    '''
    Returns the critical density of the universe at a given redshift in Msun/kpc^3
    '''
    H = get_H(z)*3.24078e-20 #in s^-1
    
    return 3*H**2/(8*np.pi*G) #Msun, kpc



def get_nfw_potential(r, Mvir, c, z):
    '''
    This function returns the potential for an NFW profile in (km/s)^2
    Look at https://ui.adsabs.harvard.edu/abs/2001MNRAS.321..155L/abstract for this formulation


    Args:
    r: in kpc? 
    Mvir: virial mass in Msun?
    '''
    
    rvir = (Mvir / (4 / 3. * np.pi * 200 * get_critical_dens(z))) ** (1/3.)
    s = r / rvir

    Vv = np.sqrt( G * Mvir / rvir )

    def g(c):
        return 1 / ( np.log(1 + c) - c / (1 + c) )

    # Phi_2rvir = Vv ** 2 * -1 * g(c) * np.log(1 + c * 2) / 2 #This is the potential at 2 Rvir. I want to set this to be zero. This does not change anything, checked this already!
    Phi_2rvir = np.zeros(np.shape(s))
    Phi = Vv ** 2 * -1 * g(c) * np.log(1 + c * s) / s - Phi_2rvir 

    return Phi * (3.086e+16)**2 # This is in (km/s)^2


# IPython.embed()



get_potential = get_nfw_potential #This statement is here because you might need the isothermal potential at some point

def get_effective_potential(r, L, mvir, cvir, z):
    '''
    This is the effective potential that we use in classical mechanics
    '''
    r = np.array(r)
    Phi = np.array(get_potential(r, mvir, cvir, z)) #This will be in km/s^2
    # All the following gimmicks because sometimes the input is an array and sometimes it is just a number, but the minimize function requires only scalar output
    if isinstance(Phi, np.ndarray) and isinstance(r, np.ndarray):
        # print(r, Phi)
        if r.size == 1:
            return Phi[0] + L**2 / (2 * r[0]**2) # This is in (km/s)^2
        else: 
            return Phi + L**2 / (2 * r**2)

    else:
        return Phi + L**2 / (2 * r**2)


def get_radroots(subh_pos, subh_vel, mvir, cvir, z):
    '''
    This function returns the pericenter and apocenter 
    for the given potential, energy and angular momentum


    Args:
    subh_pos: The position of the subhalo [x, y, z]
    subh_vel: the velocity of the subhalo [vx, vy, vz]
    BOTH OF THE ABOVE HAVE TO BE PROVIDED WRT CENTER

    mvir: virial mass at the current time 
    cvir: virial concentration at the current time for the host
    z: The redshift (relevant for the potential calculations which involves the critical density)

    Returns:
    b: pericenter
    a: apocenter
    '''
    E = get_potential(np.linalg.norm(subh_pos), mvir, cvir, z) + 0.5*(subh_vel[0]**2  + subh_vel[1]**2 + subh_vel[2]**2) #This the potential without any mass involved!
    L = np.linalg.norm(np.cross(subh_vel, subh_pos))

    rmin = minimize(get_effective_potential, x0 = 100, args = (L, mvir, cvir, z)).x #This is the rad where the energy is minimum
    # print('The radius where the effective potential is minimum is', rmin)
    rpl = np.logspace(1, 4)

    # plt.plot(rpl, get_effective_potential(rpl, L, mvir, cvir, z), label = 'Effective potential')
    # plt.plot(rpl, get_potential(rpl, mvir, cvir, z), label = 'Potential energy')
    # plt.axhline(E, ls = '--', color = 'gray', label = 'Total enegy')
    # plt.xscale('log')
    # # plt.ylim(-1e17, 1e17)
    # plt.title('Effective potential')
    # plt.ylabel(r'$U + \frac{L^2}{2 \mu r^2}$ ($M_\odot (km/s)^2$)')
    # plt.xlabel('Radius (kpc)')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    assert rmin>0, 'The minimum of effective potential has to be greater than zero'

    # b = brentq(lambda r: get_effective_potential(r, L, mvir, cvir, z) - E, 1e-10*rmin, rmin, xtol = 5)
    # a = brentq(lambda r: get_effective_potential(r, L, mvir, cvir, z) - E, rmin, 1e10*rmin, xtol = 5)
    b = brentq(lambda r: (1/r)**2 + 2*(get_potential(r, mvir, cvir, z) - E)/(L)**2, 1e-10*rmin, rmin, xtol = 10)
    a = brentq(lambda r: (1/r)**2 + 2*(get_potential(r, mvir, cvir, z) - E)/(L)**2, rmin, 1e10*rmin, xtol = 10)
    return b, a


'''
This code snippet is to check how NFW potential changes with Mvir at a given radius

mvir_ar = np.logspace(13, 15, 100)
plt.plot(mvir_ar, get_nfw_potential(10, mvir_ar, c=8, z=0), label = '10 kpc')
plt.plot(mvir_ar, get_nfw_potential(100, mvir_ar, c=8, z=0), label = '100 kpc')
plt.plot(mvir_ar, get_nfw_potential(1000, mvir_ar, c=8, z=0), label = '1000 kpc')

plt.xlabel(r'$M_\mathrm{vir}(M_\odot)$')
plt.ylabel(r'$\Phi \,(km/s)^2$')
plt.xscale('log')
plt.legend()
plt.show()
'''





# '''
# The following is to test if there is some change in luck 
# '''


# ages_df = pd.read_csv('ages_tng.csv', comment = '#')

# all_snaps = np.array(ages_df['snapshot'])
# all_redshifts = np.array(ages_df['redshift'])
# all_ages = np.array(ages_df['age(Gyr)'])


# '''
# Getting the position, etc. of the central subhalo
# '''
# central_id_at99 = 0
# central_fields = ['GroupFirstSub', 'SubhaloGrNr', 'SnapNum', 'GroupNsubs', 'SubhaloPos', 'Group_R_Crit200', 'Group_M_Crit200', 'SubhaloVel']
# central_tree = il.sublink.loadTree(basePath, 99, central_id_at99, fields = central_fields, onlyMPB = True)
# central_snaps = central_tree['SnapNum']
# central_redshift = all_redshifts[central_snaps]
# central_x =  central_tree['SubhaloPos'][:, 0]/(1 + central_redshift)/h
# central_y =  central_tree['SubhaloPos'][:, 1]/(1 + central_redshift)/h
# central_z =  central_tree['SubhaloPos'][:, 2]/(1 + central_redshift)/h
# central_r200 = central_tree['Group_R_Crit200']/(1 + central_redshift)/h #This is the virial radius of the group
# ages_rvir = all_ages[central_snaps] #Ages corresponding to the virial radii
# central_grnr = central_tree['SubhaloGrNr']
# central_gr_m200 = central_tree['Group_M_Crit200']*1e10/h #This is the M200 of the central group
# central_vx = central_tree['SubhaloVel'][:, 0] #km/s
# central_vy = central_tree['SubhaloVel'][:, 0]
# central_vz = central_tree['SubhaloVel'][:, 0]


# '''
# Following is the dataset of the entire list of subhalos which infalled after z = 3 and survived
# '''
# survived_df = pd.read_csv('sh_survived_after_z3_tng50_1.csv')

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



# '''
# Following is to get the dataset which has the relevant SubfindIDs
# '''

# id_df = pd.read_csv('errani_checking_dataset.csv', comment = '#')

# snap_if_ar = id_df['snap_at_infall']
# sfid_if_ar = id_df['id_at_infall']
# ms_by_mdm = id_df['ms_by_mdm']






# # pdf_file = "orbits_tng501_single_oca2p.pdf"
# # pdf_pages = PdfPages(pdf_file)

# fields = ['SubhaloGrNr', 'GroupFirstSub', 'SnapNum', 'SubfindID', 'SubhaloPos', 'SubhaloVel', 'SubhaloMass'] #These are the fields that will be taken for the subhalos
# # ke_by_pe_frac = np.zeros(0)
# ix = 13
# tree = il.sublink.loadTree(basePath, snap_if_ar[ix], sfid_if_ar[ix], fields = fields, onlyMDB = True) 
# # tree = il.sublink.loadTree(basePath, tree['SnapNum'][0], tree['SubfindID'][0], fields = fields, onlyMPB = True) 
# subh_snap = tree['SnapNum']
# subh_redshift = all_redshifts[subh_snap]
# subh_x = tree['SubhaloPos'][:, 0]/(1 + subh_redshift)/h
# subh_y = tree['SubhaloPos'][:, 1]/(1 + subh_redshift)/h
# subh_z = tree['SubhaloPos'][:, 2]/(1 + subh_redshift)/h
# subh_vx = tree['SubhaloVel'][:, 0]
# subh_vy = tree['SubhaloVel'][:, 1]
# subh_vz = tree['SubhaloVel'][:, 2]

# common_snaps = np.intersect1d(subh_snap, central_snaps) #This is in acsending order. Descending order after flipping
# common_snaps_des = np.flip(common_snaps) #This will be in descending order to get the indices in central_is and subh_ix
# central_ix = np.where(np.isin(central_snaps, common_snaps))[0] #getting the indices of common snaps in the central suhalo. The indices are in such a way that the snaps are again in descending order.
# subh_ix = np.where(np.isin(subh_snap, common_snaps))[0]
# # print(central_snaps[central_ix] == common_snaps)
# # print(subh_snap)
# # print(common_snaps)


# subh_z_if = all_redshifts[common_snaps[0]] #This is the redshift of infall
# subh_dist = np.sqrt((subh_x[subh_ix] - central_x[central_ix])**2 + (subh_y[subh_ix] - central_y[central_ix])**2 + (subh_z[subh_ix] - central_z[central_ix])**2)
# subh_ages = np.flip(all_ages[common_snaps]) #The ages after flipping will be in descending order

# IPython.embed()

# te_snap = 99
# te_snap_ix = te_snap == common_snaps_des
# te_subh_ix = subh_ix[te_snap_ix]
# te_central_ix = central_ix[te_snap_ix]

# mvir = central_gr_m200[te_central_ix] #this would be the virial mass of the FoF halo infall time 

# subh_vx_if_cen = subh_vx[te_subh_ix] - central_vx[te_central_ix]
# subh_vy_if_cen = subh_vy[te_subh_ix] - central_vy[te_central_ix]
# subh_vz_if_cen = subh_vz[te_subh_ix] - central_vz[te_central_ix] 






# # '''
# # Plot: To check if the orbit is working fine
# # '''
# # rpl = np.logspace(1, 4)
# # fig, ax = plt.subplots()
# # ax.plot(rpl, get_nfw_potential(rpl, 2e14, 8, 0))
# # ax.plot(rpl, get_nfw_potential2(rpl, 2e14, 8, 0))
# # ax.set_xscale('log')
# # plt.tight_layout()
# # plt.show()