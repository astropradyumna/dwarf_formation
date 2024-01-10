'''
This program is to test the code for a single subhalo
'''

# import time_evolution_orbits_preamble as p
from orbit_calculator_preamble import *
import pandas as pd
import numpy as np
import illustris_python as il
import IPython
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from galpy.potential import NFWPotential, TimeDependentAmplitudeWrapperPotential
from galpy.orbit import Orbit
from astropy import units as u
from tqdm import tqdm
from matplotlib import gridspec
from scipy.interpolate import UnivariateSpline

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


'''
Modelling the time dependece of virial mass and virial radius of the FoF0
'''

fof_r200_t = UnivariateSpline(np.flip(ages_rvir), np.flip(central_r200)) #kpc
fof_m200_t = UnivariateSpline(np.flip(ages_rvir), np.flip(central_gr_m200)) #Msun

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
Following is to get the dataset which has the relevant SubfindIDs
'''

# id_df = pd.read_csv('errani_checking_dataset.csv', comment = '#')
id_df = pd.read_csv(filepath + 'errani_checking_dataset_more1e9msun.csv', comment = '#')

snap_if_ar = id_df['snap_at_infall']
sfid_if_ar = id_df['id_at_infall']
ms_by_mdm = id_df['ms_by_mdm']


'''
Initializing pdf file for plotting the orbits
'''
pdf_file = "orbits_tng50_teos.pdf"
pdf_pages = PdfPages(pdf_file)


# def get_peri_torb_(snap, sfid):

#     '''
#     This function returns the pericentric distance and the orbital time using ______ 

#     Args:
#     snap and sfid

#     Returns:
#     rperi: pericentric distance in kpc
#     torb: orbital time in Gyr
#     '''
#     return rperi, rapo, torb





def plot_orbit(snap, sfid, ax, merged):
    '''
    This function plots the orbit; FIXME: has to be shortened further after testing different techniques. 

    Args:
    snap and sfid: THIS HAS TO BE THE LAST SURVIVING SNAPSHOT FOR SUBHALOS THAT MERGE
    ax: This is a plotting axis
    merged(boolean): If the subhalo merged, then please give this as True

    Returns None because this is only a plotting routine
    '''
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


    # This is the snapshot at which the input to pericenter estimation codes wll be taken.

    # te_snap = common_snaps[0] #This will be the infall snap
    te_snap = common_snaps[-1] #This will be the last snapshot
    # IPython.embed()

    # IPython.embed()

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
    # print(first_peri_snap)

    # te_snap = snap_r200_if # This will be the time at which the the infaling subhalo crosses the virial radius
    # te_snap = first_peri_snap #This will be the snapshot of the first pericentric passage

    te_snap_z = all_redshifts[te_snap] #This is the refshift at the total energy snapshot
    te_time = all_ages[te_snap]
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


    # Following is method 1: pericenter using the orbit directly
    # b, a = find_first_min_max(np.flip(subh_dist)) #Flipping to have it in increasing age
    # rperi = b


    mvir = central_gr_m200[te_central_ix].item() #this would be the virial mass of the FoF halo infall time 
    # rvir = (mvir / (4 / 3. * np.pi * 200 * get_critical_dens(0))) ** (1/3.)
    # Vv = np.sqrt( G * mvir / rvir ) * 3.086e+16 #km/s? Why is this even here?


    minimax = True #this is just a variable which can later be used to count the number of failures
    # Following is method 1: pericenter using the orbit directly
    try:
        b, a = find_first_min_max(np.flip(subh_dist))#Flipping to have it in increasing age
    except ValueError:
        minimax = False
    if minimax:
        ax.axhline(b, c = 'blue', ls = '--', alpha = 0.5, label = 'peri/apo from min&max')
        ax.axhline(a, c = 'blue', ls = '--', alpha = 0.5) 

    # Following is method 2: Using the analytical method to obtain the 
    # subh_pos_cen = np.array([subh_x_cen, subh_y_cen, subh_z_cen])[:, 0]
    # subh_vel_cen = np.array([subh_vx_cen, subh_vy_cen, subh_vz_cen])[:, 0]
    # try:
    # # Your code that might raise an error
    #     b_anal, a_anal = get_radroots(subh_pos_cen, subh_vel_cen, mvir, 8, te_snap_z)
    #     ax.axhline(b_anal, c = 'green', ls = '-.', alpha = 0.5, label = 'peri/apo from analytical')
    #     ax.axhline(a_anal, c = 'green', ls = '-.', alpha = 0.5)

    # except ValueError:
    #     # Handle the specific error
    #     ax.text(0.1, 0.9, 'Unbound orbit!' , transform=ax.transAxes)
    

    # print(mvir)

    # This is method 3: Using galpy
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
    # ts = np.linspace(0, 13.8 - te_time.item(), 500) #This is only for the remaining time
    ts = np.linspace(0, -13.8 , 500) #This is for 13.8 Gyr before the given snapshot

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

    Torb = subhalo_orbit.Tr(use_physical = True, type = 'spherical')
    rapo = subhalo_orbit.rap(use_physical = True, type= 'spherical')
    rperi = subhalo_orbit.rperi(use_physical = True, type= 'spherical')
    print(f'The subhalo period is {np.round(Torb, 2)}')
    print(f'The apocenter distance is {np.round(rapo, 2)} kpc')
    print(f'The pericentric distance is {np.round(rperi, 2)} kpc')

    t_gp = fig.get_xdata() + te_time
    x_gp = fig.get_ydata()
    y_gp = fig2.get_xdata()
    z_gp = fig2.get_ydata()
    # vx_gp = fig3.get_xdata()
    # vy_gp = fig3.get_ydata()
    # vz_gp = fig4.get_xdata()
    # E_gp = fig4.get_ydata()

    dist_gp = np.sqrt(x_gp**2 + y_gp**2 + z_gp**2)


    # ========== PLOTTING PART ======================== 
    # fig, ax = plt.subplots(figsize = (10, 5))
    ax.plot(np.flip(subh_ages), np.flip(subh_dist), lw = 1, color = 'blue', label = r'Orbit')
    ax.plot(all_ages[central_snaps[central_ixs]], central_r200[central_ixs], c = 'gray', ls = '--',label = r'$R_{200}$', lw = 0.3)
    ax.plot(t_gp, dist_gp, lw = 0.5, alpha = 0.8, label = 'galpy orbit', color = 'r')
    ax.set_ylabel('Cluster-centric distance (kpc)')
    ax.set_xlabel('Age (Gyr)')
    ax.set_xlim(left = 0.9*all_ages[common_snaps[0]], right = 1.1*all_ages[99])
    ax.set_title('ID at z=0 is '+str(tree['SubfindID'][0]), fontsize = 10)
    subh_max_mstar = round(np.log10(max(subh_mstar)), 2)
    ax.text(1.05, 0.25, r'   $\rm{\log_{10}M_\bigstar} = $'+str(subh_max_mstar), transform=ax.transAxes, fontsize = 11)

    # ax.axhline(b, c = 'blue', ls = '--', alpha = 0.5, label = 'peri/apo from min&max')
    # ax.axhline(a, c = 'blue', ls = '--', alpha = 0.5)

    

    ax.axhline(max(dist_gp), c = 'red', ls = ':', alpha = 0.5, label = 'peri/apo from galpy')
    ax.axhline(min(dist_gp), c = 'red', ls = ':', alpha = 0.5)

    # plt.legend()
    # ax.legend(fontsize = 10, loc='upper left', bbox_to_anchor=(1, 1))
    ax.legend(fontsize = 8)

    # ax2.plot(t_gp, z_gp, color = 'red', lw = 0.5)
    # ax2.plot(np.flip(subh_ages), np.flip(subh_z[subh_ixs] - central_z[central_ixs]), color = 'blue')
    # ax2.set_xlim(left = 0.9*all_ages[common_snaps[0]], right = 1.1*all_ages[99])
    # ax2.legend(fontsize = 10)


    # ==============PLOTTING x, y and z separately ====================

    # plt.tight_layout()
    # plt.show()

    


    return minimax 



def plot_orbit_comprehensive(snap, sfid, ax_ar, merged):
    '''
    This function plots the orbit, the velocities in x, y and z, and the total energy; FIXME: has to be shortened further after testing different techniques. 

    Args:
    snap and sfid: THIS HAS TO BE THE LAST SURVIVING SNAPSHOT FOR SUBHALOS THAT MERGE
    ax_ar: This is an array of plotting axes. Please pass 5 axes
    merged(boolean): If the subhalo merged, then please give this as True

    Returns None because this is only a plotting routine
    '''
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


    # This is the snapshot at which the input to pericenter estimation codes wll be taken.

    te_snap = common_snaps[-1] #This will be the infall snap
    # te_snap = common_snaps[-1] #This will be the last snapshot
    # IPython.embed()

    # IPython.embed()

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
    # print(first_peri_snap)

    # te_snap = snap_r200_if # This will be the time at which the the infaling subhalo crosses the virial radius
    # te_snap = first_peri_snap   #This will be the snapshot of the first pericentric passage

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

    minimax = True #this is just a variable which can later be used to count the number of failures
    # Following is method 1: pericenter using the orbit directly
    # try:
    #     b, a = find_first_min_max(np.flip(subh_dist))#Flipping to have it in increasing age
    #     ax.axhline(b, c = 'blue', ls = '--', alpha = 0.5, label = 'peri/apo from min&max')
    #     ax.axhline(a, c = 'blue', ls = '--', alpha = 0.5) 
    # except ValueError:
    #     minimax = False
        


    # rperi = b


    mvir = central_gr_m200[te_central_ix].item() #this would be the virial mass of the FoF halo infall time 
    # rvir = (mvir / (4 / 3. * np.pi * 200 * get_critical_dens(0))) ** (1/3.)
    # Vv = np.sqrt( G * mvir / rvir ) * 3.086e+16 #km/s? Why is this even here?



    # Following is method 2: Using the analytical method to obtain the 
    # subh_pos_cen = np.array([subh_x_cen, subh_y_cen, subh_z_cen])[:, 0]
    # subh_vel_cen = np.array([subh_vx_cen, subh_vy_cen, subh_vz_cen])[:, 0]
    # try:
    # # Your code that might raise an error
    #     b_anal, a_anal = get_radroots(subh_pos_cen, subh_vel_cen, mvir, 8, te_snap_z)
    #     ax.axhline(b_anal, c = 'green', ls = '-.', alpha = 0.5, label = 'peri/apo from analytical')
    #     ax.axhline(a_anal, c = 'green', ls = '-.', alpha = 0.5)

    # except ValueError:
    #     # Handle the specific error
    #     ax.text(0.1, 0.9, 'Unbound orbit!' , transform=ax.transAxes)
    

    # print(mvir)

    # This is method 3: Using galpy


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
    # ts = np.linspace(0, 13.8 - te_time.item(), 500) #This is only for the remaining time
    # ts = np.linspace(0, all_ages[-1] - te_time , 100) #This is for upto 13.8 Gyr after the given snapshot
    ts = np.linspace(0, -12 , 100)

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
    # fig, ax = plt.subplots(figsize = (10, 5))
    ax, ax_sub1, ax_sub2, ax_sub3, ax_sub4 = ax_ar

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

    ax_sub1.plot(t_gp, vx_gp, color = 'red', lw = 0.5)
    ax_sub1.plot(np.flip(subh_ages), np.flip(subh_vx[subh_ixs] - central_vx[central_ixs]), color = 'blue')
    ax_sub1.set_xlim(left = 0.9*all_ages[common_snaps[0]], right = 1.1*all_ages[99])
    ax_sub1.set_xlabel('Age (Gyr)')
    ax_sub1.set_ylabel(r'$v_x$ (km/s)')



    ax_sub2.plot(t_gp, vy_gp, color = 'red', lw = 0.5)
    ax_sub2.plot(np.flip(subh_ages), np.flip(subh_vy[subh_ixs] - central_vy[central_ixs]), color = 'blue')
    ax_sub2.set_xlim(left = 0.9*all_ages[common_snaps[0]], right = 1.1*all_ages[99])
    ax_sub2.set_xlabel('Age (Gyr)')
    ax_sub2.set_ylabel(r'$v_y$ (km/s)')


    ax_sub3.plot(t_gp, vz_gp, color = 'red', lw = 0.5)
    ax_sub3.plot(np.flip(subh_ages), np.flip(subh_vz[subh_ixs] - central_vz[central_ixs]), color = 'blue')
    ax_sub3.set_xlim(left = 0.9*all_ages[common_snaps[0]], right = 1.1*all_ages[99])
    ax_sub3.set_xlabel('Age (Gyr)')
    ax_sub3.set_ylabel(r'$v_z$ (km/s)')


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

    ax_sub4.plot(t_gp, E_gp, color = 'red', alpha = 0.5, label = 'galpy')
    # ax_sub4.axhline(subhalo_orbit.E(use_physical = True), color = 'red', alpha = 0.5, label = 'e2')
    ax_sub4.plot(t_gp, ke_ar_gp, color = 'hotpink', ls = ':')
    ax_sub4.plot(t_gp, pe_ar_gp, color = 'hotpink', ls = '-.')
    ax_sub4.plot(t_gp, pe_ar_gp + ke_ar_gp, color = 'hotpink', ls = '-')

    ax_sub4.plot(np.flip(subh_ages), np.flip(ke_ar), color = 'blue', ls = ':', label = 'KE')
    ax_sub4.plot(np.flip(subh_ages), np.flip(pe_ar), color = 'blue', ls = '-.', label = 'PE')
    ax_sub4.plot(np.flip(subh_ages), np.flip(ke_ar + pe_ar), color = 'blue', ls = '-', label = 'TE')
    ax_sub4.axhline(0, ls = '--', color = 'gray', alpha = 0.2)
    ax_sub4.set_xlim(left = 0.9*all_ages[common_snaps[0]], right = 1.1*all_ages[99])
    ax_sub4.set_xlabel('Age (Gyr)')
    ax_sub4.set_ylabel(r'$E \, \rm{(km/s)^2}$')
    ax_sub4.legend(fontsize = 6)

    # print(f'The different between intial energirs is {(subhalo_orbit.E() - ke_ar_gp[0]) / pe_ar_gp[0] }')
    # ==============PLOTTING x, y and z separately ====================

    # plt.tight_layout()
    # plt.show()

    


    return minimax 




# for ix in tqdm(range(len(snap_if_ar))):
#     '''
#     this loop runs over all the subhalos that are in the subeset file
#     '''
#     # print(ix)
#     fig, ax = plt.subplots(figsize = (10, 5))
#     plot_orbit(snap_if_ar[ix], sfid_if_ar[ix], ax)
#     plt.tight_layout()
#     pdf_pages.savefig()
#     # plt.show()
#     plt.close()

# pdf_pages.close()

# ix = 100
# # fig, [ax, ax2] = plt.subplots(nrows = 2, ncols = 1, figsize = (10, 10), sharex = True)
# fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5)) #This is a single axis
# plot_orbit(snap_if_ar[ix], sfid_if_ar[ix], ax, merged = False)
# # plot_orbit(msh_merger_snap[ix], msh_merger_sfid[ix], ax, merged = True)

# plt.tight_layout()
# plt.show()
# plt.close()

'''
This loop runs over all the subhalos that are in the merger file
'''
# count_no_peri = 0
# for ix in tqdm(range(len(msh_merger_sfid))):
#     fig, ax = plt.subplots(figsize = (10, 5))
#     minimax = plot_orbit(msh_merger_snap[ix], msh_merger_sfid[ix], ax, merged = True) #If this is false, we cannot find the pericenter and apocenter from the orbtit
#     if not minimax:
#         count_no_peri = count_no_peri + 1
#     plt.tight_layout()
#     pdf_pages.savefig()
#     # plt.show()
#     plt.close()

# print(f'Number of subhalos for which we cannot find the peri from simulation are {count_no_peri}')
# pdf_pages.close()


'''
This part is for plotting the velocities and the energy; uses the plot_orbit_comprehensive function
'''

# for ix in tqdm(range(len(sfid_if_ar))):
#     '''
#     this loop runs over all the subhalos that are in the subeset file
#     Plots all the orbits along with velocities
#     '''
#     # print(ix)
#     fig = plt.figure(figsize=(15, 6))
#     gs = gridspec.GridSpec(2, 3, width_ratios=[3, 1.5, 1.5]) #Generating one big image to the left and four small images to the right, from ChatGPT

#     ax_big = plt.subplot(gs[:, 0])
#     ax_sub1 = plt.subplot(gs[0, 1])
#     ax_sub2 = plt.subplot(gs[0, 2])
#     ax_sub3 = plt.subplot(gs[1, 1])
#     ax_sub4 = plt.subplot(gs[1, 2])
#     plot_orbit_comprehensive(snap_if_ar[ix], sfid_if_ar[ix], [ax_big, ax_sub1, ax_sub2, ax_sub3, ax_sub4], merged = False)
#     plt.tight_layout()
#     pdf_pages.savefig()
#     # plt.show()
#     plt.close()

# pdf_pages.close()

# fig = plt.figure(figsize=(15, 6))
# gs = gridspec.GridSpec(2, 3, width_ratios=[3, 1.5, 1.5]) #Generating one big image to the left and four small images to the right, from ChatGPT

# ax_big = plt.subplot(gs[:, 0])
# ax_sub1 = plt.subplot(gs[0, 1])
# ax_sub2 = plt.subplot(gs[0, 2])
# ax_sub3 = plt.subplot(gs[1, 1])
# ax_sub4 = plt.subplot(gs[1, 2])


# ix = 13
# # fig, [ax, ax2] = plt.subplots(nrows = 2, ncols = 1, figsize = (10, 10), sharex = True)
# # fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5)) #This is a single axis
# plot_orbit_comprehensive(snap_if_ar[ix], sfid_if_ar[ix], [ax_big, ax_sub1, ax_sub2, ax_sub3, ax_sub4], merged = False)
# # plot_orbit_comprehensive(msh_merger_snap[ix], msh_merger_sfid[ix], [ax_big, ax_sub1, ax_sub2, ax_sub3, ax_sub4], merged = True)

# plt.tight_layout()
# plt.show()
# plt.close()

# IPython.embed()


'''
This part is to test if we are having the right potential from galpy

mvir = 1e14
z = 0
R = np.logspace(1, 3, 100) #kpc
Z = np.ones(len(R))
potential = NFWPotential(conc=8, mvir=mvir/1e12, wrtcrit = True, overdens = 200 * get_critical_dens(z)/get_critical_dens(0), H = 67.44)
amp = potential._amp
rvir_gp = potential.a * 8 
rvir = (mvir / (4 / 3. * np.pi * 200 * get_critical_dens(z))) ** (1/3.)
gp_potential = potential._evaluate(R=R, z = Z/8)
my_potential = get_nfw_potential(R, mvir, 8, z)

plt.plot(R, amp*gp_potential, label = 'Galpy NFW', color = 'red')
plt.plot(np.sqrt(R**2 + Z**2), my_potential, label = 'NFW', color = 'blue')
plt.axvline(rvir_gp, color = 'red', ls = '--')
plt.axvline(rvir, color = 'blue', ls = '--')
plt.plot()
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.show()

'''

'''
Plotting to check how the virial mass over radius changes so that we can adjust the isothermal parameter accordingly
'''
fig, ax = plt.subplots()
ax.plot(ages_rvir, central_gr_m200 / central_r200)
plt.show()



# Code below is rewritten because it is getting too complicated
'''
-o- Now, let us start writing the main program. 
-o- We have to loop over all the subhalos in the catalog
-o- Then, extract the rmx, Vmx at the infall 
-o- Currently cannot download the subhalo 0 data for the potential
-o- Get the potential at the snapshot of infall
-o- Assume the potential stays constant
-o- For the energy, calculate the 
-o- Today's goal is
'''

# Uncomment all the lines below if you want to restore the previous version 


# # pdf_file = "orbits_tng501_single_teos.pdf"
# # pdf_pages = PdfPages(pdf_file)

# fields = ['SubhaloGrNr', 'GroupFirstSub', 'SnapNum', 'SubfindID', 'SubhaloPos', 'SubhaloVel', 'SubhaloMass'] #These are the fields that will be taken for the subhalos
# # ke_by_pe_frac = np.zeros(0)
# ix = 7
# tree = il.sublink.loadTree(p.basePath, snap_if_ar[ix], sfid_if_ar[ix], fields = fields, onlyMDB = True) 
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
# # print(central_snaps[central_ixs] == common_snaps)
# # print(subh_snap)
# # print(common_snaps)


# subh_z_if = all_redshifts[common_snaps[0]] #This is the redshift of infall
# subh_dist = np.sqrt((subh_x[subh_ixs] - central_x[central_ixs])**2 + (subh_y[subh_ixs] - central_y[central_ixs])**2 + (subh_z[subh_ixs] - central_z[central_ixs])**2)
# subh_ages = np.flip(all_ages[common_snaps]) #The ages after flipping will be in descending order

# # IPython.embed()

# # Following is finding the apo and peri analytically

# # for sx in common_snaps_des:
# # 	if subh_dist[subh_ix[common_snaps_des == sx]] < central_r200[central_snaps == sx]:
# # 		snap_r200_if = sx 

# # te_snap = snap_r200_if #Get the total energy at this snapshot
# # te_snap = 99


# # te_snap = 71

# te_snap = common_snaps[0] #This will be the infall snap
# te_time = all_ages[te_snap]
# te_snap_ix = te_snap == common_snaps_des
# te_subh_ix = subh_ix[te_snap_ix] #Subhalo index for this infall time
# te_central_ix = central_ix[te_snap_ix]

# mvir = central_gr_m200[te_central_ix] #this would be the virial mass of the FoF halo infall time 
# v0, r0 = p.get_iso_params(mvir, 8, 0)

# # ke_ar = np.zeros(0)
# # pe_ar = np.zeros(0)
# # l_ar = np.zeros(0) #angular momentum array

# IPython.embed()

# # for six in range(len(common_snaps)):
# # 	snap = common_snaps_des[six]
# # 	z = all_redshifts[snap]
# # 	te_subh_ix = subh_ix[six] #Subhalo index for this infall time
# # 	te_central_ix = central_ix[six]


# subh_vx_cen = subh_vx[subh_ixs][te_subh_ix] - central_vx[central_ixs][te_central_ix]
# subh_vy_cen = subh_vy[subh_ixs][te_subh_ix] - central_vy[central_ixs][te_central_ix]
# subh_vz_cen = subh_vz[subh_ixs][te_subh_ix] - central_vz[central_ixs][te_central_ix]    


# subh_x_cen = subh_x[subh_ixs][te_subh_ix] - central_x[central_ixs][te_central_ix]
# subh_y_cen = subh_y[subh_ixs][te_subh_ix] - central_y[central_ixs][te_central_ix]
# subh_z_cen = subh_z[subh_ixs][te_subh_ix] - central_z[central_ixs][te_central_ix]


# # 	subh_mass = tree['SubhaloMass'][te_subh_ix]*1e10/h #this is the subhalo mass at infall
# # 	# subh_ke_in = 0.5*subh_mass*((subh_vx[te_subh_ix] - central_vx[te_central_ix])**2 + (subh_vy[te_subh_ix] - central_vy[te_central_ix])**2 + (subh_vz[te_subh_ix] - central_vz[te_central_ix])**2)
# # 	subh_ke_in = 0.5*subh_mass*(subh_vx_if_cen**2+subh_vy_if_cen**2+subh_vz_if_cen**2) # Msun (km/s)^2
# # 	# 3.24078e-17**2


# # 	# v0, r0 = p.get_iso_params(mvir, 8, subh_z_if) 
# # 	# print(v0, r0,common_snaps[0])
# # 	# print('Dist', subh_dist[-1])
# # 	# subh_pe_in = subh_mass*p.get_iso_potential(subh_dist[-1], v0, r0)
# # 	subh_pe_in = subh_mass*p.get_nfw_potential(subh_dist[-1], mvir, 8, z)
# # 	# print(subh_pe_in)
# # 	ke_ar = np.append(ke_ar, subh_ke_in)
# # 	pe_ar = np.append(pe_ar, subh_pe_in)



# # 	subh_te_in = subh_ke_in + subh_pe_in #this is the total energy in units of Msun, kpc and s


# # 	# subh_vx_if_cen = subh_vx[te_subh_ix] - central_vx[te_central_ix]
# # 	# subh_vy_if_cen = subh_vy[te_subh_ix] - central_vy[te_central_ix]
# # 	# subh_vz_if_cen = subh_vz[te_subh_ix] - central_vz[te_central_ix]

# # 	subh_x_if_cen = subh_x[te_subh_ix] - central_x[te_central_ix]
# # 	subh_y_if_cen = subh_y[te_subh_ix] - central_y[te_central_ix]
# # 	subh_z_if_cen = subh_z[te_subh_ix] - central_z[te_central_ix]

# # 	L = np.linalg.norm(subh_mass*np.cross([subh_x_if_cen.item(), subh_y_if_cen.item(), subh_z_if_cen.item()], [subh_vx_if_cen.item(), subh_vy_if_cen.item(), subh_vz_if_cen.item()]))

# # 	l_ar = np.append(l_ar, L)



# # # ===================
# # #  Plotting the energies
# # plt.plot(subh_ages, ke_ar + pe_ar, 'b-', label = 'Total energy')
# # plt.plot(subh_ages, ke_ar, 'b:', label = 'Kinetic energy')
# # plt.plot(subh_ages, pe_ar, 'b--', label = 'Potential energy')
# # plt.axvline(all_ages[snap_r200_if], c = 'gray', ls= '--', label = r'Crosses R_{200}')


# # plt.ylabel(r'Energy ($M_\odot km^2/s^2$)')
# # plt.xlabel('Age (Gyr)')
# # plt.legend(fontsize = 8)
# # plt.tight_layout()
# # plt.show()


# # '''
# # Plotting the angular momentum
# # '''
# # plt.plot(subh_ages, l_ar, label = 'Angular Momentum')
# # plt.legend()
# # plt.xlabel('Age (Gyr)')
# # # plt.ylabel()
# # plt.title('Angular Momentum')
# # plt.tight_layout()
# # plt.yscale('log')
# # plt.show()

# # # ===================

# # te_ar = pe_ar + ke_ar

# # te_ar = te_ar/subh_mass
# # l_ar = l_ar/subh_mass
# # subh_mass = 1


# # b, a = p.get_radroots(te_ar[-1], l_ar[-1], mvir, 8, subh_mass, all_redshifts[snap_r200_if])
# # print(b, a)
# b, a = p.find_first_min_max(np.flip(subh_dist)) #Flipping to have it in increasing age

# rperi = b





# # ========================================
# '''
# This is galpy effort
# '''
# '''
# This is the cell where I am testing the orbit for 13 subhalo directly
# '''

# Mvir = 2e14 #Msun
# rvir = (Mvir / (4 / 3. * np.pi * 200 * p.get_critical_dens(0))) ** (1/3.)
# Vv = np.sqrt( G * Mvir / rvir ) * 3.086e+16

# # Define NFW potential
# potential = NFWPotential(conc=8, mvir=Mvir/1e12, wrtcrit = True)
# # potential = NFWPotential(conc=8, mvir=Mvir/1e12)
# # Set up initial conditions
# x, y, z = subh_x_cen, subh_y_cen, subh_z_cen
# # R, Phi, Z = coords.rect_to_cyl(x, y, z)
# # Convert positions to cylindrical coordinates
# R = np.sqrt(x**2 + y**2)
# Phi = np.arctan2(y, x)
# Z = z


# vx, vy, vz = subh_vx_cen, subh_vy_cen, subh_vz_cen

# vR = (x * vx + y * vy) / R
# vPhi = -(-x * vy + y * vx) / R #Extra minus sign because this is a left handed system (galpy)
# vZ = vz

# initial_conditions = [R * u.kpc, vR * u.kilometer/u.second, vPhi * u.kilometer/u.second, Z * u.kpc, vZ * u.kilometer/u.second, Phi * u.radian ] 
# # initial_conditions = [x, y, z, vx, vy, vz]
# subhalo_orbit = Orbit(initial_conditions)

# # Integrate the orbit
# ts = np.linspace(0, 13.8 - te_time.item(), 100)
# subhalo_orbit.integrate(ts * u.Gyr, potential)

# # Analyze the results

# # velocities = subhalo_orbit.getOrbit()


# # Further analysis as needed for your research



# # xx = subhalo_orbit.x(t = ts, use_physical = True)
# # yy = subhalo_orbit.y(t = ts)
# # zz = subhalo_orbit.z(t = ts)
# fig, = subhalo_orbit.plot(d1 = 't', d2 = 'x')
# plt.close()
# fig2, = subhalo_orbit.plot(d1 = 'y', d2 = 'z')
# plt.close()
# t_gp = fig.get_xdata() + te_time
# x_gp = fig.get_ydata()
# y_gp = fig2.get_xdata()
# z_gp = fig2.get_ydata()

# dist_gp = np.sqrt(x_gp**2 + y_gp**2 + z_gp**2)

# # ========================================


# plt.figure()

# plt.plot(np.flip(subh_ages), np.flip(subh_dist), lw = 1, color = 'blue', label = r'Orbit')
# plt.plot(all_ages[central_snaps[central_ixs]], central_r200[central_ixs], c = 'gray', ls = '--',label = r'$R_{200}$')
# plt.plot(t_gp, dist_gp, lw = 0.5, alpha = 0.3, label = 'galpy orbit', color = 'r')
# plt.ylabel('Cluster-centric distance (kpc)')
# plt.xlabel('Age (Gyr)')
# plt.title('ID at z=0 is '+str(tree['SubfindID'][0]), fontsize = 10)
# plt.axhline(b, c = 'blue', ls = '--', alpha = 0.5, label = 'peri/apo from min&max')
# plt.axhline(a, c = 'blue', ls = '--', alpha = 0.5)

# plt.axhline(max(dist_gp), c = 'red', ls = ':', alpha = 0.5, label = 'peri/apo from galpy')
# plt.axhline(min(dist_gp), c = 'red', ls = ':', alpha = 0.5)

# plt.legend(fontsize = 10)
# plt.tight_layout()
# plt.show()
# # pdf_pages.savefig()
# plt.close()


# IPython.embed()



# tperi = 2*np.pi*rperi/v0 * 3.086e+16 * 3.17098e-17 #in Gyr
# x = b/a
# fecc  = (2*x/(x + 1))**3.2


# # def get_tmx(t):


# # pdf_pages.close()
