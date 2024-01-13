'''
This program is to test the Errani+22 formulation for some selected subhalos
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
from scipy.integrate import quad
from scipy.misc import derivative
from tqdm import tqdm
import time
from matplotlib.backends.backend_pdf import PdfPages

response = requests.get('https://www.tng-project.org/api/TNG50-1/', verify=False) #Set warnings  off

font = {'family' : 'serif',
        'weight' : 'normal',
        'size' : 14}
matplotlib.rc('font', **font)

baseUrl = 'https://www.tng-project.org/api/TNG50-1/'
headers = {"api-key":"894f4df036abe0cb9d83561e4b1efcf1"}
basePath = '/mainvol/jdopp001/L35n2160TNG_fixed/output'
filepath  = '/home/psadh003/tng50/tng_files/' #update this according to the need

h = 0.6774
# G = 4.300917270038e-06 # This is in kpc, (km/s)^2 and seconds 
G = 4.5390823753559603e-39 #This is in kpc, Msun and seconds

mass_dm = 3.07367708626464e-05 * 1e10/h


ages_df = pd.read_csv(filepath + 'ages_tng.csv', comment = '#')

all_snaps = np.array(ages_df['snapshot'])
all_redshifts = np.array(ages_df['redshift'])
all_ages = np.array(ages_df['age(Gyr)'])





'''
Getting the position, etc. of the central subhalo
'''
central_id_at99 = 0
central_fields = ['GroupFirstSub', 'SubhaloGrNr', 'SnapNum', 'GroupNsubs', 'SubhaloPos', 'Group_R_Crit200', 'Group_M_Crit200', 'GroupPos']
central_tree = il.sublink.loadTree(basePath, 99, central_id_at99, fields = central_fields, onlyMPB = True)
central_snaps = central_tree['SnapNum']
central_redshift = all_redshifts[central_snaps]
central_x =  central_tree['SubhaloPos'][:, 0]/(1 + central_redshift)/h
central_y =  central_tree['SubhaloPos'][:, 1]/(1 + central_redshift)/h
central_z =  central_tree['SubhaloPos'][:, 2]/(1 + central_redshift)/h
Rvir = central_tree['Group_R_Crit200']/(1 + central_redshift)/h #This is the virial radius of the group
ages_rvir = all_ages[central_snaps] #Ages corresponding to the virial radii
central_grnr = central_tree['SubhaloGrNr']
central_gr_m200 = central_tree['Group_M_Crit200']*1e10/h #This is the M200 of the central group


'''
The following is to plot the datasets that have been digitized for Rh values
'''
errani_fp = '/home/psadh003/errani23_data/'
rh1by4_df = pd.read_csv(errani_fp + 'rh1by4_errani22.csv') # Rh/rmx0 = 1/4 from digitization
rh1by2_df = pd.read_csv(errani_fp + 'rh1by2_errani22.csv') # Rh/rmx0 = 1/2 from digitization
rh1by8_df = pd.read_csv(errani_fp + 'rh1by8_errani22.csv')
rh1by16_df = pd.read_csv(errani_fp + 'rh1by16_errani22.csv')

l10mmxbymmx0_1by4 = rh1by4_df['l10mmxbymmx0']
l10rbyrmx0_1by4 = rh1by4_df['l10rbyrmx0']

l10rbyrmx0_1by4 = l10rbyrmx0_1by4[np.argsort(l10mmxbymmx0_1by4)] #sorting based on mass for the univariate spline
l10mmxbymmx0_1by4 = l10mmxbymmx0_1by4[np.argsort(l10mmxbymmx0_1by4)]

l10mmxbymmx0_1by2 = rh1by2_df['l10mmxbymmx0']
l10rbyrmx0_1by2 = rh1by2_df['l10rbyrmx0']

l10rbyrmx0_1by2 = l10rbyrmx0_1by2[np.argsort(l10mmxbymmx0_1by2)] #sorting based on mass for the univariate spline
l10mmxbymmx0_1by2 = l10mmxbymmx0_1by2[np.argsort(l10mmxbymmx0_1by2)]

l10mmxbymmx0_1by8 = rh1by8_df['l10mmxbymmx0']
l10rbyrmx0_1by8 = rh1by8_df['l10rbyrmx0']

l10rbyrmx0_1by8 = l10rbyrmx0_1by8[np.argsort(l10mmxbymmx0_1by8)] #sorting based on mass for the univariate spline
l10mmxbymmx0_1by8 = l10mmxbymmx0_1by8[np.argsort(l10mmxbymmx0_1by8)]


l10mmxbymmx0_1by16 = rh1by16_df['l10mmxbymmx0']
l10rbyrmx0_1by16 = rh1by16_df['l10rbyrmx0']

l10rbyrmx0_1by16 = l10rbyrmx0_1by16[np.argsort(l10mmxbymmx0_1by16)] #sorting based on mass for the univariate spline
l10mmxbymmx0_1by16 = l10mmxbymmx0_1by16[np.argsort(l10mmxbymmx0_1by16)]


l10rbyrmx0_1by4_spl = UnivariateSpline(l10mmxbymmx0_1by4, l10rbyrmx0_1by4, k = 3, s = 0.5)
l10rbyrmx0_1by2_spl = UnivariateSpline(l10mmxbymmx0_1by2, l10rbyrmx0_1by2, k = 3, s = 0.5)
l10rbyrmx0_1by8_spl = UnivariateSpline(l10mmxbymmx0_1by8, l10rbyrmx0_1by8, k = 3, s = 0.5)
l10rbyrmx0_1by16_spl = UnivariateSpline(l10mmxbymmx0_1by16, l10rbyrmx0_1by16, k = 3, s = 0.5)

'''
Following is the dataset for the vd values that have been digitized
'''
'''
This part for importing data from the csv files
'''
vd1by4_df = pd.read_csv(errani_fp + 'vd1by4_errani22.csv') # Rh/rmx0 = 1/4 from digitization
vd1by2_df = pd.read_csv(errani_fp + 'vd1by2_errani22.csv') # Rh/rmx0 = 1/2 from digitization
vd1by8_df = pd.read_csv(errani_fp + 'vd1by8_errani22.csv')
vd1by16_df = pd.read_csv(errani_fp + 'vd1by16_errani22.csv')

l10mmxbymmx0_1by4 = vd1by4_df['l10mmxbymmx0']
l10vbyvmx0_1by4 = vd1by4_df['l10vbyvmx0']

l10vbyvmx0_1by4 = l10vbyvmx0_1by4[np.argsort(l10mmxbymmx0_1by4)] #sorting based on mass for the univariate spline
l10mmxbymmx0_1by4 = l10mmxbymmx0_1by4[np.argsort(l10mmxbymmx0_1by4)]

l10mmxbymmx0_1by2 = vd1by2_df['l10mmxbymmx0']
l10vbyvmx0_1by2 = vd1by2_df['l10vbyvmx0']

l10vbyvmx0_1by2 = l10vbyvmx0_1by2[np.argsort(l10mmxbymmx0_1by2)] #sorting based on mass for the univariate spline
l10mmxbymmx0_1by2 = l10mmxbymmx0_1by2[np.argsort(l10mmxbymmx0_1by2)]

l10mmxbymmx0_1by8 = vd1by8_df['l10mmxbymmx0']
l10vbyvmx0_1by8 = vd1by8_df['l10vbyvmx0']

l10vbyvmx0_1by8 = l10vbyvmx0_1by8[np.argsort(l10mmxbymmx0_1by8)] #sorting based on mass for the univariate spline
l10mmxbymmx0_1by8 = l10mmxbymmx0_1by8[np.argsort(l10mmxbymmx0_1by8)]


l10mmxbymmx0_1by16 = vd1by16_df['l10mmxbymmx0']
l10vbyvmx0_1by16 = vd1by16_df['l10vbyvmx0']

l10vbyvmx0_1by16 = l10vbyvmx0_1by16[np.argsort(l10mmxbymmx0_1by16)] #sorting based on mass for the univariate spline
l10mmxbymmx0_1by16 = l10mmxbymmx0_1by16[np.argsort(l10mmxbymmx0_1by16)]


l10vbyvmx0_1by4_spl = UnivariateSpline(l10mmxbymmx0_1by4, l10vbyvmx0_1by4, k = 3, s = 0.5)
l10vbyvmx0_1by2_spl = UnivariateSpline(l10mmxbymmx0_1by2, l10vbyvmx0_1by2, k = 3, s = 0.5)
l10vbyvmx0_1by8_spl = UnivariateSpline(l10mmxbymmx0_1by8, l10vbyvmx0_1by8, k = 3, s = 0.5)
l10vbyvmx0_1by16_spl = UnivariateSpline(l10mmxbymmx0_1by16, l10vbyvmx0_1by16, k = 3, s = 0.5)




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
Following is to get the dataset which has the relevant SubfindIDs
'''

id_df = pd.read_csv(filepath + 'errani_checking_dataset_more1e9msun.csv', comment = '#')

snap_if_ar = id_df['snap_at_infall']
sfid_if_ar = id_df['id_at_infall']
ms_by_mdm = id_df['ms_by_mdm']
# sfid_0_ar = id_df['sfid_0_ar']
# snap_mp_ar = id_df['snap_mp_ar']
# sfid_mp_ar = id_df['sfid_mp_ar']

def get(path, params=None):
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


def get_link(snap, id):
    '''
    Gives you the link for the subhalo
    '''
    return baseUrl+'snapshots/'+str(snap)+'/subhalos/'+str(id)+'/'

def download_data(shsnap_ar, shid_ar):
    '''
    This function downloads the cutout files for the IDs and snaps mentioned in the array
    '''
    shsnap_ar = np.array(shsnap_ar, dtype = int)
    shid_ar = np.array(shid_ar, dtype = int)
    nobj = len(shsnap_ar) #number of objects to be downloaded
    for ix in tqdm(range(nobj)): #This loops over all the systems of interest
        shsnap = shsnap_ar[ix]
        shid = shid_ar[ix]
        snap = get(baseUrl+'snapshots/'+str(shsnap)+'/')
        if not os.path.isfile('cutout_'+str(shid)+'_'+str(shsnap)+'.hdf5'): #if it is not downloaded alraedy
            subh_link = get_link(shsnap, shid)
            subh = get(subh_link, 'r')
            cutout_request = {'gas':'Coordinates,Masses','stars':'Coordinates,Masses','dm':'Coordinates'}
            cutout = get(subh_link+"cutout.hdf5", cutout_request)
            os.rename('cutout_'+str(shid)+'.hdf5', filepath + 'cutout_'+str(shid)+'_'+str(shsnap)+'.hdf5') #Renaming the file to include the snapshot in name
    return None


def get_mass_profiles(shsnap, shid, rl = 'linear', h = 0.6774):
    '''
    Gets you the mass profiles for the ID and snap metioned
    '''
    # snap = get(baseUrl+'snapshots/'+str(shsnap)+'/')
    # z = snap['redshift']
    z = all_redshifts[shsnap]
    filename = filepath + 'cutout_files/cutout_'+str(shid)+'_'+str(shsnap)+'.hdf5'
    if not os.path.exists(filename):
        download_data(shsnap, shid)

    f = h5py.File(filename, 'r') #This is to read the cutout file
    
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
        rad_plot_cont = np.logspace(-3, np.log10(2*max(dm_rad)), 500)

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
    
    if cg == 1: #If there is gas element
        return cg, rad_plot_cont, mass_arr_plot_cont, dm_mass_arr_cont, star_mass_arr_cont, gas_mass_arr_cont
    else: 
        return cg, rad_plot_cont, mass_arr_plot_cont, dm_mass_arr_cont, star_mass_arr_cont, np.zeros(len(rad_plot_cont))


def get_rot_curve(shsnap, shid):
    '''
    Plots the rotation curve 

    Args:
    shsnap: The snapshot of the subhalo
    shid: The SubFind ID of the subhalo

    Returns:
    vmx, rmx, mmx (The velocity, radius and mass inside of the radius where the velocity curve of the DM peaks)
    '''
    cg, rad_plot_cont, mass_arr_plot_cont, dm_mass_arr_cont, star_mass_arr_cont, gas_mass_arr_cont = get_mass_profiles(shsnap, shid)
    

    circ_vel_plot_cont = np.sqrt(G*mass_arr_plot_cont/(rad_plot_cont))*3.086e+16
    dm_vel_plot_cont = np.sqrt(G*dm_mass_arr_cont/(rad_plot_cont))*3.086e+16
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

    ix_max_vel = np.argmax(dm_vel_plot_cont)
    vmx_subh = dm_vel_plot_cont[ix_max_vel]
    rmx_subh = rad_plot_cont[ix_max_vel]
    Mmx_subh = mass_dm*len(dm_rad[dm_rad< rmx_subh]) 

    # fig, ax = plt.subplots(figsize = (5, 5))
    # ax.plot(rad_plot_cont[np.argmax(dm_vel_plot_cont)], max(dm_vel_plot_cont), 'go')

    # ax.plot(rad_plot_cont, circ_vel_plot_cont, 'k', label='Total Mass')
    # ax.plot(rad_plot_cont, dm_vel_plot_cont, 'k--', label='DM')
    # if cg == 1: ax.plot(rad_plot_cont, gas_vel_plot_cont, 'b', label='Gas')
    # ax.plot(rad_plot_cont, star_vel_plot_cont, 'r', label='Stars')
    # ax.set_xlim(0, min(max(dm_rad), 2.5*rmx_subh))
    # ax.set_xlabel(r'Radius $[kpc]$')
    # ax.set_ylabel(r'Circular Velocity $[km/s]$')
    # ax.legend(fontsize=12)
    # ax.set_title('Rotation Curve for subhalo ID = '+str(shid)+' at z = '+str(round(z, 2)))
    # plt.tight_layout()
    # plt.show()

    return vmx_subh, rmx_subh, Mmx_subh


def get_vhrh(shsnap, shid):
    cg, rad_plot_cont, mass_arr_plot_cont, dm_mass_arr_cont, star_mass_arr_cont, gas_mass_arr_cont = get_mass_profiles(shsnap, shid)
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


def get_vmxbyvmx0(rmxbyrmx0, alpha = 0.4, beta = 0.65):
    '''
    This function is Eq. 4 in the paper. Expected tidal track for only DM which is NFW
    '''
    return 2**alpha*rmxbyrmx0**beta*(1 + rmxbyrmx0**2)**(-alpha)


def get_mxbymx0(rmxbyrmx0):
    return (rmxbyrmx0*get_vmxbyvmx0(rmxbyrmx0)**2)

def get_rmxbyrmx0(mxbymx0_reqd):
    # mxbymx0_reqd = np.array(mxbymx0_reqd)
    if isinstance(mxbymx0_reqd, np.ndarray):
        rmxbyrmx0_ar = np.zeros(0)
        for mfrac in mxbymx0_reqd:
            rmxbyrmx0_ar = np.append(rmxbyrmx0_ar, fsolve(lambda x: get_mxbymx0(x) - mfrac, x0 = 0.5))
    else:
        rmxbyrmx0_ar = fsolve(lambda x: get_mxbymx0(x) - mxbymx0_reqd, x0 = 0.5)
    return rmxbyrmx0_ar

def get_L_star(alpha, beta, Es, Mmx_by_Mmx0):
    def dNs_by_dE(E_ar, alpha, beta, Es):
        # dNs_by_dE_ar = np.zeros(0)
        if isinstance(E_ar, float):
            if (E_ar>=0) & (E_ar<1):
                E = E_ar
                return (E**alpha * np.exp(-(E/Es)**beta ))
            else:
                 return 0
        for E in E_ar:
            # print(E)
            if (E >= 0) & (E < 1):
                dNs_by_dE_ar = np.append(dNs_by_dE_ar, E**alpha * np.exp(-(E/Es)**beta ))
            else:
                dNs_by_dE_ar = np.append(dNs_by_dE_ar, 0)
        # dNs_by_dE_ar = dNs_by_dE_ar / (1 + (a * )**b )
        return dNs_by_dE_ar

    def e_mx_t(Mmx_by_Mmx0):
        '''
        This looks alright
        '''
        return 0.77*(Mmx_by_Mmx0)**0.43


    def L_star_integrand(e, alpha, beta, Es , Mmx_by_Mmx0):
        a = 0.85
        b = 12
        I = dNs_by_dE(e, alpha, beta, Es)/(1 + (a * e/e_mx_t(Mmx_by_Mmx0))**b)
        return I 
    L = quad(L_star_integrand, 0, 1, args = (alpha, beta, Es, Mmx_by_Mmx0))
    # print(L)
    return L[0]

def get_LbyL0(mxbymx0, rh0byrmx0):
	LbyL0 = np.zeros(0)
	if rh0byrmx0 == 1/2:
		Es = 0.485
	elif rh0byrmx0 == 1/4:
		Es = 1/3. 
	elif rh0byrmx0 == 1/8:
		Es = 0.21 
	elif rh0byrmx0 == 1/16:
		Es = 0.112  
	if isinstance(mxbymx0, float) or isinstance(mxbymx0, int):
		LbyL0 = np.append(LbyL0, get_L_star(alpha = 3, beta = 3, Es = Es, Mmx_by_Mmx0 = mxbymx0))
	else:
		for mfrac in mxbymx0:
			LbyL0 = np.append(LbyL0, get_L_star(alpha = 3, beta = 3, Es = Es, Mmx_by_Mmx0 = mfrac))
	LbyL0 = LbyL0/get_L_star(alpha = 3, beta = 3, Es = Es, Mmx_by_Mmx0 = 1)
	
	return LbyL0

# print(np.log10(get_LbyL0(1e-2, 1/4)))


def getMstar(Mh, z):
	'''
	This is from Moster 2013 Eq.?
	gives the stellar mass for a given halo
	'''
	M1 = 10**(11.590 + 1.195*(z/(1+z)))
	beta = 1.376 - 0.826*(z/(1+z))
	gamma = 0.608 - 0.329*(z/(1+z))
	N = 0.0351 - 0.0247*(z/(1+z))
	MhbyM1 = Mh/M1
	Mstar = 2*N*Mh
	Mstar *= 1/(MhbyM1**(-beta) + MhbyM1**(gamma))
	return Mstar

def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    
    taken from https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot-with-matplotlib
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="<-", color=color),
        size=size)


def get_m200_if(snap, sfid):
    '''
    This is to obtain the M200 value at infall time from the group catalog 
    '''
    ix = np.where((ssh_sfid == sfid) & (ssh_snap == snap))[0]
    if_snap = ssh_snap1[ix] #this is the infall 1 snapshot of the subhalo
    if_subid = ssh_sfid1[ix] #This is the sfid of the subhalo at infall time 1
    grnr = il.groupcat.loadSingle(basePath, if_snap, subhaloID = if_subid)['SubhaloGrNr']
    m200 = il.groupcat.loadSingle(basePath, if_snap, haloID = grnr)['Group_M_Crit200']*1e10/h
    return m200


def get_H(z = 0, h = 0.6774):
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


def get_critical_dens(z = 0):
    '''
    Returns the critical density of the universe at a given redshift in Msun/kpc^3
    '''
    H = get_H(z)*3.24078e-20 #in s^-1
    
    return 3*H**2/(8*np.pi*G)


def get_converted_nfw_params(m_vir, c_vir, z):
    '''
    Converts virial mass and concentration of an NFW profile to rho_s and r_s parameters.
    
    Args:
        m_vir (float): Virial mass in solar masses.
        c_vir (float): Concentration parameter.
        
    Returns:
        rho_s (float): Scale density in solar masses per cubic kiloparsec.
        r_s (float): Scale radius in kiloparsecs.
    '''
    # Define constants
    # G = 4.302e-6 # Gravitational constant in units of kpc / Msun (km/s)^2
    # Calculate critical density of the Universe
    rho_crit = get_critical_dens(z)
    
    # Calculate virial radius
    r_vir = (3 * m_vir / (4 * np.pi * c_vir**3 * rho_crit))**(1/3)
    
    # Calculate scale radius
    r_s = r_vir / c_vir
    
    # Calculate scale density
    rho_s = m_vir / (4 * np.pi * r_s**3 * (np.log(1 + c_vir) - c_vir / (1 + c_vir)))

    return rho_s, r_s


def get_M(r, M200 = 2e14, c = 8):
    '''
    This returns the mass of the cluster assuming an NFW
    '''
    rho0, rs = get_converted_nfw_params(M200, c, 0)
    x = r/rs
    M = 4*np.pi*rho0*rs**3*(np.log(1 + x) - x/(1 + x))
    return M 


def get_dlnm_by_dlnr(r, M200 = 1e14, c = 8):
    dlnm_by_dlnr = (r/get_M(r))*derivative(get_M, r, dx=1e-6)
    return dlnm_by_dlnr


def get_iso_potential(r, v0, r0):
    '''
    This function returns the isotropic potential mentioned in the Errani series of papers
    '''
    return 3.24078e-17**2*v0**2*np.log(r/r0)





'''
This part is to obtain the mass profile for various redshifts for the cenral halo
'''
# snap_reqd = [25, 33, 50, 67, 99] #This is the list 
# z_reqd = all_redshifts[snap_reqd]
# norm = plt.Normalize(min(z_reqd), max(z_reqd))
# cm = plt.cm.viridis
# sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)

# for six in range(len(snap_reqd)):
#     snap = snap_reqd[six]
#     m200 = central_gr_m200[central_snaps == snap]
#     r200 = Rvir[central_snaps == snap]
#     rpl = np.logspace(-1.5, np.log10(1.5*r200), 500)
#     Mpl = get_M(rpl, m200, 8)
#     rhopl = Mpl/((4./3)*np.pi*rpl**3)
#     plt.plot(rpl, Mpl, color=cm(norm(z_reqd[six])))
#     # plt.plot(rpl, rhopl, color=cm(norm(z_reqd[six])))

# cbar = plt.colorbar(sm)
# cbar.set_label(r'$z$', fontsize = 12)
# plt.xlabel(r'Radius (kpc)')
# plt.ylabel(r'Mass ($M_{\odot}$)')
# plt.title('The mass profiles evolution for progenitor of FoF0', fontsize = 10)
# # plt.title('The density profiles evolution for progenitor of FoF0', fontsize = 10)

# plt.loglog()
# plt.tight_layout()
# plt.show()




'''
This part is to obtain the radius inside which we have 95% of the mass. This has to be compared with the tidal radius (maybe they have to be the same)
'''
# fields = ['SnapNum', 'SubfindID'] 
# r95_ar = np.zeros(0) #This is the array of radii which contains 95% of the mass for the subhalos at redshift 0
# nstar = np.zeros(0)
# ndm = np.zeros(0)
# lg_ms_by_mdm = np.log10(np.array(ms_by_mdm))
# print(lg_ms_by_mdm)
# # plt.figure(figsize = (8, 6))
# for ix in tqdm(range(len(snap_if_ar))): #looping over all the indices which have been given at infall
#     shsnap = snap_if_ar[ix]
#     sfid = sfid_if_ar[ix]
#     sh_tree = il.sublink.loadTree(basePath, shsnap, sfid, fields = fields, onlyMDB = True) #loading the descendant tree
#     this_sh_snaps = sh_tree['SnapNum']
#     this_sh_sfids = sh_tree['SubfindID']
#     cg, rad_plot_cont, mass_arr_plot_cont, dm_mass_arr_cont, star_mass_arr_cont, gas_mass_arr_cont = get_mass_profiles(this_sh_snaps[0], this_sh_sfids[0])
#     # plt.plot(rad_plot_cont, mass_arr_plot_cont)
#     this_sh = il.groupcat.loadSingle(basePath, 99, subhaloID = this_sh_sfids[0])
#     nstar = np.append(nstar, this_sh['SubhaloLenType'][4])
#     ndm = np.append(ndm, this_sh['SubhaloLenType'][1])
#     m95 = 0.95*this_sh['SubhaloMass']*1e10/h #This is 95% of the total mass that is present
#     # m95 = 0.99*this_sh['SubhaloMassType'][4]*1e10/h #This is 95% of the stellar mass that is present
#     r95 = rad_plot_cont[np.searchsorted(mass_arr_plot_cont, m95)] #radius where 95% of the mass is enclosed
#     # r95 = rad_plot_cont[np.searchsorted(star_mass_arr_cont, m95)] #radius where 95% of the mass is enclosed
#     r95_ar = np.append(r95_ar, r95)


# # plt.xlabel('Radius (kpc)')
# # plt.ylabel(r'Mass ($M_\odot$)')
# # plt.loglog()
# # plt.tight_layout()
# # plt.show()


'''
Plot: The following is to plot the no of stellar particles against the number of dark matter particles left at z = 0
'''
# plt.plot(ndm, nstar, 'ko')
# plt.xlabel('Number of DM particles')
# plt.ylabel('Number of stellar particles')
# plt.tight_layout()
# plt.show()


'''
Orbit plotter cell. This cell also take care of finding (and plotting) the pericenter and apocenter
'''
# subh_pc_min = np.zeros(0)
# subh_m200_if = np.zeros(0)
# pdf_file = "orbits_tng501_1e9msun.pdf"
# pdf_pages = PdfPages(pdf_file)

# fields = ['SubhaloGrNr', 'GroupFirstSub', 'SnapNum', 'SubfindID', 'SubhaloPos'] #These are the fields that will be taken for the subhalos
# for ix in range(len(snap_if_ar)): #looping over all the indices which have been given at infall
#     tree = il.sublink.loadTree(basePath, snap_if_ar[ix], sfid_if_ar[ix], fields = fields, onlyMDB = True) #MPB here?
#     tree = il.sublink.loadTree(basePath, tree['SnapNum'][0], tree['SubfindID'][0], fields = fields, onlyMPB = True) 
#     subh_snap = tree['SnapNum']
#     subh_redshift = all_redshifts[subh_snap]
#     subh_x = tree['SubhaloPos'][:, 0]/(1 + subh_redshift)/h
#     subh_y = tree['SubhaloPos'][:, 1]/(1 + subh_redshift)/h
#     subh_z = tree['SubhaloPos'][:, 2]/(1 + subh_redshift)/h

#     common_snaps = np.intersect1d(subh_snap, central_snaps)
#     central_ix = np.where(np.isin(central_snaps, common_snaps)) #getting the indices of common snaps in 
#     subh_ix = np.where(np.isin(subh_snap, common_snaps))
#     # print(central_snaps[central_ix] == common_snaps)
#     # print(subh_snap)
#     # print(common_snaps)


#     subh_dist = np.sqrt((subh_x[subh_ix] - central_x[central_ix])**2 + (subh_y[subh_ix] - central_y[central_ix])**2 + (subh_z[subh_ix] - central_z[central_ix])**2)
#     subh_ages = np.flip(all_ages[common_snaps])

#     # print(subh_dist)

#     # print(subh_ages)

#     subh_pc_min = np.append(subh_pc_min, min(subh_dist)) #This is the closest pericentric passage

#     subh_m200_if = np.append(subh_m200_if, get_m200_if(snap_if_ar[ix], sfid_if_ar[ix]))#This is the M200 of the subhalo at infall


#     # mvir = central_gr_m200[snap_if_ar[ix] == central_snaps] #This is the viriral mass of the host halo at the infall snapshot
#     # cvir = 8 #Assuming so
#     # z = all_redshifts[snap_if_ar[ix]] #this is the redshift of infall
#     # # mvir = 2e14
#     # # cvir=8
#     # rho0, rs = get_converted_nfw_params(mvir, cvir, 0)
#     # rho_crit = get_critical_dens()
#     # rvir = (3 * mvir / (4 * np.pi * cvir**3 * rho_crit))**(1/3)
#     # v0 = np.sqrt(mvir*G/rvir)*3.086e+16 #km/s


#     plt.plot(np.flip(subh_ages), np.flip(subh_dist), lw = 1, color = 'blue', label = r'Orbit')
#     plt.plot(all_ages[central_snaps[central_ix]], Rvir[central_ix], c = 'gray', ls = '--',label = r'$R_{200}$')
#     plt.ylabel('Cluster-centric distance (kpc)')
#     plt.xlabel('Age (Gyr)')
#     plt.tight_layout()
#     plt.legend(fontsize = 10)
#     plt.title('ID at z=0 is '+str(tree['SubfindID'][0]), fontsize = 10)
#     pdf_pages.savefig()
#     plt.close()

# pdf_pages.close()

# # rt = np.zeros(len(subh_pc_min))
# # for kx in range(len(subh_pc_min)):
# #     r = subh_pc_min[kx]
# #     rt[kx] = (subh_m200_if[kx]/((2 - get_dlnm_by_dlnr(r))*get_M(r)))**(1/3) * r

    
# # plt.plot(subh_pc_min, rt, 'ro')
# # plt.ylabel('Tidal radius (kpc)')
# # plt.xlabel('Closest pericentric distance (kpc)')
# # plt.tight_layout()
# # plt.show()




# # plt.show()





'''
This section is to test the orbits wrt to the group position instead of the central position
'''











'''
Plot: The following is to the plot the subhalos r95 radius at z = 0 against the tidal radius
'''
# plt.figure(figsize = (7, 6))
# # norm = plt.Normalize(-1.1, 0.005)
# # plt.scatter(rt, r95_ar, marker = 'o', c = np.log10(np.array(ms_by_mdm)), cmap = 'viridis')
# plt.scatter(rt, r95_ar, marker = 'o', c = 'black')
# # cbar = plt.colorbar()
# # cbar.set_label(r'$\log_{10}(M_\mathrm{star}/M_\mathrm{dm})$')
# plt.plot([0, 1.1*max(r95_ar)], [0, 1.1*max(r95_ar)], 'k--')
# plt.xlabel('Tidal radius (kpc)')
# plt.ylabel(r'$r_\mathrm{95}$ (kpc)')
# plt.tight_layout()
# plt.show()



'''
Plot 1: Creating a figure similar to the Errani+22 Fig. 14. 
This is to check if the tidal track eventually converges on the dashed line, which is Eq. 4 in the paper
'''

# log_rmxbyrmx0_pl = np.linspace(-1.1, 0.5, 100)
# plt.figure(figsize = (8, 6))
# line = plt.plot(log_rmxbyrmx0_pl, np.log10(get_vmxbyvmx0(10**log_rmxbyrmx0_pl)), 'k--', label = 'Subhalo evolution (Errani+21)')[0]
# add_arrow(line, position = min(log_rmxbyrmx0_pl)) #This is to show the direction of tidal track
# '''
# This loop takes care of plotting the cutout files
# '''
# min_frem = np.zeros(0)
# fields = ['SnapNum', 'SubfindID', 'SubhaloLenType'] 
# for ix in tqdm(range(len(snap_if_ar))): #looping over all the indices which have been given at infall
#     # if ix > 5:
#     #     break
#     shsnap = snap_if_ar[ix]
#     sfid = sfid_if_ar[ix]
#     vmx0, rmx0, mmx0 = get_rot_curve(shsnap, sfid) #Assuming these to be the values at the beginning
#     sh_tree = il.sublink.loadTree(basePath, shsnap, sfid, fields = fields, onlyMDB = True) #loading the descendant tree
#     this_sh_snaps = sh_tree['SnapNum']
#     this_sh_sfids = sh_tree['SubfindID']
#     this_sh_lendmstar = sh_tree['SubhaloLenType'][:, 1] + sh_tree['SubhaloLenType'][:, 4] #This is the number of star + DM particles
#     vhar = np.zeros(0) #this will be the array of Vh values for this subhalo (and descendants)
#     rhar = np.zeros(0) #this will be the array of rh values for this subhalo (and descendants)
#     mmx_ar = np.zeros(0) #this will be the values of Mmx for this subhalo (and descendants)

#     vhar_res = np.zeros(0) #This will be the array of Vh values for resolved subhalos (DM + star) particles greater than 3000
#     rhar_res = np.zeros(0) #This will be the array of rh values for resolved subhalos (DM + star) particles greater than 3000
#     mmx_ar_res = np.zeros(0) #This will be the array of mmx values for resolved subhalos (DM + star) particles greater than 3000
    
#     norm = plt.Normalize(-2.5, 0) #this is for the colormap
#     for jx in np.flip(range(len(this_sh_snaps))): #This loops over all the descendants of this subhalo in increasing snapshot way
#         des_snap = int(this_sh_snaps[jx]) #snapshot of the descendant
#         des_sfid = int(this_sh_sfids[jx]) #SubfindID of the descendant
#         des_lenmdstar = int(this_sh_lendmstar[jx]) #This is the length of particles for this decendant
#         if (os.path.isfile('cutout_files/cutout_'+str(des_sfid)+'_'+str(des_snap)+'.hdf5')): #If the file exists, proceed with analysis
#             vh, rh = get_vhrh(des_snap, des_sfid)
#             vmx, rmx, mmx = get_rot_curve(des_snap, des_sfid)
#             if vh * rh != 0 :
#                 vhar = np.append(vhar, vh)
#                 rhar = np.append(rhar, rh)
#                 mmx_ar = np.append(mmx_ar, mmx)
#                 if des_lenmdstar >= 3000:
#                     vhar_res = np.append(vhar_res, vh)
#                     rhar_res = np.append(rhar_res, rh)
#                     mmx_ar_res = np.append(mmx_ar_res, mmx)
#             # print(vh, rh)
#     min_frem = np.append(min_frem, min(np.log10(mmx_ar/mmx0)))
#     # print(shsnap, sfid, np.log10(rhar/rmx0), np.log10(vhar/vmx0))
#     plt.plot(np.log10(rhar/rmx0), np.log10(vhar/vmx0), alpha = 0.4, c = 'darkcyan', ls = '--', lw = 0.5, zorder = 10)
#     plt.scatter(np.log10(rhar/rmx0), np.log10(vhar/vmx0), marker = '$\u25CB$', c=np.log10(mmx_ar/mmx0), cmap='viridis', alpha = 0.8,  norm=norm, facecolors='none', zorder = 20)
#     plt.plot(np.log10(rhar_res/rmx0), np.log10(vhar_res/vmx0), alpha = 1, c = 'darkcyan', lw = 0.5, zorder = 10)
#     plt.scatter(np.log10(rhar_res/rmx0), np.log10(vhar_res/vmx0), marker = 'o', c=np.log10(mmx_ar_res/mmx0), cmap='viridis', alpha = 0.8,  norm=norm, facecolors='none', zorder = 20)

# cbar = plt.colorbar()
# cbar.set_label(r'$\log_{10}(M_\mathrm{mx}/M_\mathrm{mx0})$')
# plt.ylim(top = 0.2, bottom = -0.65)
# plt.xlim(left = -2)
# plt.xlabel(r'$\log_{10} r_\mathrm{h}/r_\mathrm{mx0}$')
# plt.ylabel(r'$\log_{10} V_\mathrm{h}/V_\mathrm{mx0}$')
# plt.tight_layout()
# plt.legend(fontsize = 10)
# plt.show()

# IPython.embed()


'''
Plot 2: The Rh vs Mmx (Appendix figures)
vd against the amount of stripping
and remnant stellar fraction against the amount of stripping
'''
"""
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 16), sharex = True)
log10_mmxbymmx0_pl = np.linspace(-2.5, 0, 100)

ax1 = axes[0]
ax2 = axes[1]
ax3 = axes[2]
# plt.figure(figsize = (10, 6))
# log10_mmxbymmx0_pl = np.linspace(-2.5, 0, 100)
ax1.plot(log10_mmxbymmx0_pl, np.log10(get_rmxbyrmx0(10**(log10_mmxbymmx0_pl))), 'k-', label = 'DM only')
ax1.plot(log10_mmxbymmx0_pl, l10rbyrmx0_1by2_spl(log10_mmxbymmx0_pl), 'r:', alpha = 0.3, label = r'$R_\mathrm{h0}/r_\mathrm{mx0} = 1/2$')
ax1.plot(log10_mmxbymmx0_pl, l10rbyrmx0_1by4_spl(log10_mmxbymmx0_pl), c = 'orange', ls = ':', alpha = 0.3, label = r'$R_\mathrm{h0}/r_\mathrm{mx0} = 1/4$')
ax1.plot(log10_mmxbymmx0_pl, l10rbyrmx0_1by8_spl(log10_mmxbymmx0_pl), c = 'dodgerblue', ls = ':', alpha = 0.3, label = r'$R_\mathrm{h0}/r_\mathrm{mx0} = 1/8$')
ax1.plot(log10_mmxbymmx0_pl, l10rbyrmx0_1by16_spl(log10_mmxbymmx0_pl), c = 'darkblue', ls = ':', alpha = 0.3, label = r'$R_\mathrm{h0}/r_\mathrm{mx0} = 1/16$')


ax2.plot(log10_mmxbymmx0_pl, np.log10(get_vmxbyvmx0(get_rmxbyrmx0(10**(log10_mmxbymmx0_pl)))), 'k-', label = 'DM only')
ax2.plot(log10_mmxbymmx0_pl, l10vbyvmx0_1by2_spl(log10_mmxbymmx0_pl), 'r:', alpha = 0.3, label = r'$R_\mathrm{h0}/r_\mathrm{mx0} = 1/2$')
ax2.plot(log10_mmxbymmx0_pl, l10vbyvmx0_1by4_spl(log10_mmxbymmx0_pl), c = 'orange', ls = ':', alpha = 0.3, label = r'$R_\mathrm{h0}/r_\mathrm{mx0} = 1/4$')
ax2.plot(log10_mmxbymmx0_pl, l10vbyvmx0_1by8_spl(log10_mmxbymmx0_pl), c = 'dodgerblue', ls = ':', alpha = 0.3, label = r'$R_\mathrm{h0}/r_\mathrm{mx0} = 1/8$')
ax2.plot(log10_mmxbymmx0_pl, l10vbyvmx0_1by16_spl(log10_mmxbymmx0_pl), c = 'darkblue', ls = ':', alpha = 0.3, label = r'$R_\mathrm{h0}/r_\mathrm{mx0} = 1/16$')



fields = ['SnapNum', 'SubfindID'] 
ms_by_mdm = np.log10(ms_by_mdm)
norm = plt.Normalize(min(ms_by_mdm), max(ms_by_mdm))
cm = plt.cm.viridis
sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)

for ix in range(len(snap_if_ar)): #looping over all the indices which have been given at infall
    shsnap = snap_if_ar[ix]
    sfid = sfid_if_ar[ix]
    vmx0, rmx0, mmx0 = get_rot_curve(shsnap, sfid) #Assuming these to be the values at the beginning
    sh_tree = il.sublink.loadTree(basePath, shsnap, sfid, fields = fields, onlyMDB = True) #loading the descendant tree
    # mstar0 = il.groupcat.loadSingle(basePath, shsnap, subhaloID = sfid)['SubhaloMassInRadType'][4]*1e10/h

    mstar0 = ssh_max_mstar[np.where((ssh_sfid == sfid) & (ssh_snap == shsnap))] #THIS IS THE MAXIMUM STELLAR MASS
    this_sh_snaps = sh_tree['SnapNum']
    this_sh_sfids = sh_tree['SubfindID']
    vhar = np.zeros(0) #this will be the array of Vh values for this subhalo (and descendants)
    Rhar = np.zeros(0) #this will be the array of rh values for this subhalo (and descendants)
    mmx_ar = np.zeros(0) #this will be the values of Mmx for this subhalo (and descendants)
    vd_ar = np.zeros(0) #This will be the vd array
    mstar_ar = np.zeros(0) #This is the stellar mass array

    # norm = plt.Normalize(-2, 0) #this is for the colormap
    for jx in np.flip(range(len(this_sh_snaps))): #This loops over all the descendants of this subhalo in increasing snapshot way
        des_snap = int(this_sh_snaps[jx]) #snapshot of the descendant
        des_sfid = int(this_sh_sfids[jx]) #SubfindID of the descendant
        if (os.path.isfile('cutout_files/cutout_'+str(des_sfid)+'_'+str(des_snap)+'.hdf5')): #If the file exists, proceed with analysis
            vh, rh = get_vhrh(des_snap, des_sfid)
            vmx, rmx, mmx = get_rot_curve(des_snap, des_sfid)
            des_subh = il.groupcat.loadSingle(basePath, des_snap, subhaloID = des_sfid)
            vd = des_subh['SubhaloVelDisp'] #Thi is the velocity dispersion in km/s
            mstar = des_subh['SubhaloMassInRadType'][4]*1e10/h
            if vh * rh != 0 :
                vhar = np.append(vhar, vh)
                Rhar = np.append(Rhar, rh/np.sqrt(2))
                mmx_ar = np.append(mmx_ar, mmx)
                vd_ar = np.append(vd_ar, vd)
                mstar_ar = np.append(mstar_ar, mstar)

    ax1.plot(np.log10(mmx_ar/mmx0), np.log10(Rhar/rmx0), color=cm(norm(ms_by_mdm[ix])))
    ax2.plot(np.log10(mmx_ar/mmx0), np.log10(vd_ar/vmx0), color=cm(norm(ms_by_mdm[ix])))
    ax3.plot(np.log10(mmx_ar/mmx0), np.log10(mstar_ar/mstar0), color=cm(norm(ms_by_mdm[ix])))

# # norm = plt.Normalize(min(np.log10(mmx_AR)), max(np.log10(mmx_AR))) #this is for the colormap
# # plt.scatter(np.log10(mmx_AR/mmx0_AR), np.log10(Rh_AR/rmx0_AR), marker = 'o', c=np.log10(mmx_ar/mmx0), cmap='viridis', alpha = 0.8,  norm=norm)
#   # print(shsnap, sfid, np.log10(mmx_ar/mmx0))


fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label(r'$\log_{10}(M_\mathrm{star}/M_\mathrm{dm})$ at infall', fontsize = 12)

ax1.tick_params(labelbottom=True)
ax2.tick_params(labelbottom=True)


ax1.set_ylabel(r'$\log_{10} R_\mathrm{h}/r_\mathrm{mx0}$')
ax2.set_ylabel(r'$\log_{10} \sigma_\mathrm{los}/V_\mathrm{mx0}$')
ax3.set_ylabel(r'$\log_{10} M_\mathrm{star}/M_\mathrm{star0}$')

ax3.set_xlabel(r'$\log_{10} M_\mathrm{mx}/M_\mathrm{mx0}$')
ax1.legend(fontsize = 8)
ax2.legend(fontsize = 8)
# plt.tight_layout()
plt.savefig('appendix_images_errani_testing_more1e9msun.pdf')




"""






# IPython.embed()


















#=============================================================================================================
# GARBAGE BELOW
#=============================================================================================================

# gr_snaps = np.arange(0, 100)
# gr_redshift = all_redshifts[gr_snaps]
# gr_x = central_tree['GroupPos'][:, 0]/(1 + gr_redshift)/h
# gr_y = central_tree['GroupPos'][:, 1]/(1 + gr_redshift)/h
# gr_z = central_tree['GroupPos'][:, 2]/(1 + gr_redshift)/h


# # plt.plot(, central_y, label = 'BCG')
# plt.plot(central_snaps, gr_z[central_snaps]/central_z)
# # plt.yscale('log')
# # plt.legend()

# plt.show()

# IPython.embed()

'''
Plot 3: vd against the amount of stripping
'''
# plt.figure(figsize = (10, 6))
# log10_mmxbymmx0_pl = np.linspace(-2.5, 0, 100)
# plt.plot(log10_mmxbymmx0_pl, np.log10(get_vmxbyvmx0(get_rmxbyrmx0(10**(log10_mmxbymmx0_pl)))), 'k-', label = 'DM only')
# plt.plot(log10_mmxbymmx0_pl, l10vbyvmx0_1by2_spl(log10_mmxbymmx0_pl), 'r:', alpha = 0.3, label = r'$R_\mathrm{h0}/r_\mathrm{mx0} = 1/2$')
# plt.plot(log10_mmxbymmx0_pl, l10vbyvmx0_1by4_spl(log10_mmxbymmx0_pl), c = 'orange', ls = ':', alpha = 0.3, label = r'$R_\mathrm{h0}/r_\mathrm{mx0} = 1/4$')
# plt.plot(log10_mmxbymmx0_pl, l10vbyvmx0_1by8_spl(log10_mmxbymmx0_pl), c = 'dodgerblue', ls = ':', alpha = 0.3, label = r'$R_\mathrm{h0}/r_\mathrm{mx0} = 1/8$')
# plt.plot(log10_mmxbymmx0_pl, l10vbyvmx0_1by16_spl(log10_mmxbymmx0_pl), c = 'darkblue', ls = ':', alpha = 0.3, label = r'$R_\mathrm{h0}/r_\mathrm{mx0} = 1/16$')

# fields = ['SnapNum', 'SubfindID'] 
# ms_by_mdm = np.log10(ms_by_mdm)
# norm = plt.Normalize(min(ms_by_mdm), max(ms_by_mdm))
# cm = plt.cm.YlGnBu
# sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)

# for ix in range(len(snap_if_ar)): #looping over all the indices which have been given at infall
#   shsnap = snap_if_ar[ix]
#   sfid = sfid_if_ar[ix]
#   vmx0, rmx0, mmx0 = get_rot_curve(shsnap, sfid) #Assuming these to be the values at the beginning
#   sh_tree = il.sublink.loadTree(basePath, shsnap, sfid, fields = fields, onlyMDB = True) #loading the descendant tree
#   this_sh_snaps = sh_tree['SnapNum']
#   this_sh_sfids = sh_tree['SubfindID']
#   mmx_ar = np.zeros(0) #this will be the values of Mmx for this subhalo (and descendants)
#   vd_ar = np.zeros(0) #this will be the velocity dispersion for this subhalo


#   # norm = plt.Normalize(-2, 0) #this is for the colormap
#   for jx in np.flip(range(len(this_sh_snaps))): #This loops over all the descendants of this subhalo in increasing snapshot way
#       des_snap = int(this_sh_snaps[jx]) #snapshot of the descendant
#       des_sfid = int(this_sh_sfids[jx]) #SubfindID of the descendant
#       if (os.path.isfile('cutout_files/cutout_'+str(des_sfid)+'_'+str(des_snap)+'.hdf5')): #If the file exists, proceed with analysis
#           # vh, rh = get_vhrh(des_snap, des_sfid)
#           vd = il.groupcat.loadSingle(basePath, des_snap, subhaloID = des_sfid)['SubhaloVelDisp'] #Thi is the velocity dispersion in km/s
#           vmx, rmx, mmx = get_rot_curve(des_snap, des_sfid)
#           mmx_ar = np.append(mmx_ar, mmx)
#           vd_ar = np.append(vd_ar, vd)


#   

# # norm = plt.Normalize(min(np.log10(mmx_AR)), max(np.log10(mmx_AR))) #this is for the colormap
# # plt.scatter(np.log10(mmx_AR/mmx0_AR), np.log10(Rh_AR/rmx0_AR), marker = 'o', c=np.log10(mmx_ar/mmx0), cmap='viridis', alpha = 0.8,  norm=norm)
#   # print(shsnap, sfid, np.log10(mmx_ar/mmx0))

# cbar = plt.colorbar(sm)
# # cbar.set_label(r'$\log_{10}(M_\mathrm{mx}/M_\mathrm{mx0})$')
# cbar.set_label(r'$\log_{10}(M_\mathrm{star}/M_\mathrm{dm})$ at infall', fontsize = 12)

# plt.ylabel(r'$\log_{10} \sigma_\mathrm{los}/V_\mathrm{mx0}$')
# plt.xlabel(r'$\log_{10} M_\mathrm{mx}/M_\mathrm{mx0}$')
# plt.legend(fontsize = 10)
# plt.tight_layout()
# plt.show()



'''
Plot 4: Let us now try plotting the remnant stellar mass against the amount of stripping
Errani+22 does not make any direct prediction for this case, though they make a prediction for the luminosity.
'''
# plt.figure(figsize = (10, 6))


# plt.show()






