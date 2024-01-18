'''
This is an attempt to organize things in this project
main.py combines everything
'''

import numpy as np
import matplotlib.pyplot as plt 
import os 
os.environ["USE_LZMA"] = "0"
import pandas as pd
from errani_plus_tng_subhalo import Subhalo
from tqdm import tqdm
import galpy
import IPython
import illustris_python as il
from matplotlib.backends.backend_pdf import PdfPages
from subhalo_profiles import ExponentialProfile
import warnings

# Suppress the lzma module warning
# warnings.filterwarnings("ignore", category=UserWarning, module="pandas.compat")
warnings.simplefilter(action='ignore', category=FutureWarning)


'''
file import
'''
filepath = '/home/psadh003/tng50/tng_files/'
outpath  = '/home/psadh003/tng50/output_files/'
baseUrl = 'https://www.tng-project.org/api/TNG50-1/'
headers = {"api-key":"894f4df036abe0cb9d83561e4b1efcf1"}
basePath = '/mainvol/jdopp001/L35n2160TNG_fixed/output'

h = 0.6744

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
Following is to get the dataset which has the relevant SubfindIDs
'''

# id_df = pd.read_csv('errani_checking_dataset.csv', comment = '#')
id_df = pd.read_csv(filepath + 'errani_checking_dataset_more1e9msun.csv', comment = '#')

snap_if_ar = id_df['snap_at_infall']
sfid_if_ar = id_df['id_at_infall']
ms_by_mdm = id_df['ms_by_mdm']





'''
=============================================================
Data import ends here
=============================================================
'''

IPython.embed()


'''
Plot 0.1: This is to plot all the half mass radius of stars according to TNG, and the one assumed in the model
'''

fig, ax = plt.subplots(figsize = (5, 5))


for ix in tqdm(range(len(snap_if_ar))):
    '''
    This is to loop over all the surviving subhalos of big dataset with subhalos in between 1e8.5 and 1e9.5 Msun
    ''' 
    subh  = Subhalo(snap = snap_if_ar[ix], sfid = sfid_if_ar[ix])
    # subh.get_infall_properties()

    Rh_tng_max = subh.get_rh(where = 'max')/np.sqrt(2)
    Rh_model = subh.get_rh0byrmx0() * subh.rmx0 #This is the assumption
    if subh.get_rh0byrmx0() == 0.5:
        print('Jai')
    ax.plot(Rh_model, Rh_tng_max, 'ko')
    if ix in [30, 49, 56, 68, 84, 85, 89, 95]: 
        ax.plot(Rh_model, Rh_tng_max, 'ro')
        if ix == 30:ax.plot(Rh_model, Rh_tng_max, 'ro', label = r'Low $f_{\bigstar}$')
    elif ix in [83, 93, 94, 98, 100, 102, 103, 106]:
        ax.plot(Rh_model, Rh_tng_max, 'bo')
        if ix == 83: ax.plot(Rh_model, Rh_tng_max, 'bo', label = r'High $f_{\bigstar}$')

ax.set_xlabel(r'$R_{\rm{h, inf}}$ model (kpc)')
ax.set_ylabel(r'$R_{\rm{h, max\, M_\bigstar}}$ from TNG (kpc)')
ax.set_xlim(0, 8)
ax.set_ylim(0, 8)
x_vals = np.array(ax.get_xlim())    
ax.plot(x_vals, x_vals, 'k--', lw = 0.5)
ax.legend(fontsize = 8)
plt.tight_layout()
plt.show()


'''
Plot 0.2: This is to plot the mass of the ones with no stellar mass remaining
This is to compare with the exponential profile and test if it is a good enough approximation for the stellar profile
'''
pdf_file2 = outpath + "mass profiles_tng50_more1e9msun_tsah.pdf"
pdf_pages2 = PdfPages(pdf_file2)

for ix in tqdm(range(len(snap_if_ar))):
# for ix in tqdm([30, 49, 56, 68, 84, 85, 89, 95]):
    subh  = Subhalo(snap = snap_if_ar[ix], sfid = sfid_if_ar[ix])
    subh.get_mass_profiles(where = 'max', plot = True)
    exp_check = ExponentialProfile(subh.get_mstar(where = 'max', how = 'total'), np.sqrt(2) * subh.get_rh0byrmx0() * subh.rmx0) #This is an instance to check the exponential profile
    left, right = plt.gca().get_xlim()
    rpl = np.logspace(np.log10(left), np.log10(right), 100)
    plt.plot(rpl, exp_check.mass(rpl), label = 'Expoential assumed', color = 'salmon', ls = ':')
    plt.legend()
    pdf_pages2.savefig()
    # plt.show()
    plt.close()

pdf_pages2.close()


'''
Plot 0.3: This is to plot the rotation curve and see why rmx0 values are so less
'''

pdf_file3 = outpath + "rot_curves_tng50_more1e9msun_tsah.pdf"
pdf_pages3 = PdfPages(pdf_file3)

for ix in tqdm([30, 49, 56, 68, 84, 85, 89, 95]):
    subh  = Subhalo(snap = int(snap_if_ar[ix]), sfid = int(sfid_if_ar[ix]))
    print(subh.get_rot_curve(where = int(subh.snap), plot = True))
    # exp_check = ExponentialProfile(subh.get_mstar(where = 'max', how = 'total'), np.sqrt(2) * subh.get_rh0byrmx0() * subh.rmx0) #This is an instance to check the exponential profile
    # left, right = plt.gca().get_xlim()
    # rpl = np.logspace(np.log10(left), np.log10(right), 100)
    # plt.plot(rpl, exp_check.mass(rpl), label = 'Expoential assumed', color = 'salmon', ls = ':')
    # plt.legend()
    pdf_pages3.savefig()
    plt.show()
    plt.close()

pdf_pages3.close()

    


# ======================================
ctr = 0
skipped_ixs = np.zeros(0)
frem_ar = np.zeros(0) #this is the remnant masss fraction from the SAM
frem_tng_ar = np.zeros(0)
subh_m200_ar = np.zeros(0) #This is the virial mass at infall for the subhalos
subh_mstar_99_ar = np.zeros(0) #stellar mass at z = 0 for the subhalos being considered
subh_tinf_ar = np.zeros(0) #This is the array of infall times of the subhalo
subh_fstar_model_ar = np.zeros(0) #This is the array of stellar mass fraction remaining in the model
subh_fstar_tng_ar = np.zeros(0)
subh_fstar_model_from_tngfrem_ar = np.zeros(0) #This is the array of stellar mass fraction that is remaining assuming Mmx/Mmx0 from TNG
subh_mstar_model_from_tngfrem_ar = np.zeros(0) #This is the array of stellar mass that is remaining assuming Mmx/Mmx0 from TNG
subh_mstar_model_ar = np.zeros(0) #this is the stellar mass remaining at the end for the model
subh_mstar_tng_ar = np.zeros(0) #This is the stellar mass remaining at the end for TNG
subh_parlen_99_ar = np.zeros(0) #This is the length of particles at z = 0
subh_fbar_tng_ar = np.zeros(0) #This is the ratio of stars to dark matter in 2Rh for the subhalos at infall
subh_fdm_ar = np.zeros(0) #This is the total remnant dark matter mass to test the Smith+16 paper results


# Both the following indices are for the frem_ar, etc. (there are some subhalos that are missed for having an Unbound Orbit)
ixs_low_mstar = np.zeros(0) #This is the list of indices of the subhalos that have low stellar mass than that in TNG
ixs_high_mstar = np.zeros(0) #This is the list of indices which have high mstar in the model.
ixs_unresol = np.zeros(0, dtype = int)
subh_ixs_low_mstar = np.zeros(0, dtype = int)
'''
Run this loop for doing everything after evolution
'''



for ix in tqdm(range(len(snap_if_ar))):
    '''
    This is to loop over all the surviving subhalos of big dataset with subhalos in between 1e8.5 and 1e9.5 Msun
    '''
    
    # if ix>15: break

    subh  = Subhalo(snap = snap_if_ar[ix], sfid = sfid_if_ar[ix])

    # if ix in [30, 49, 56, 68, 84, 85, 89, 95]: 
    
    # subh.get_infall_properties()
    # t = subh.get_orbit(merged = False, when_te = 'last') #after this, the subhalo has rperi, rapo and torb
    try:
        t = subh.get_orbit(merged = False, when_te = 'last') #after this, the subhalo has rperi, rapo and torb
    except Exception as e:
        # print(e)
        ctr = ctr + 1
        skipped_ixs = np.append(skipped_ixs, ix)
        continue

    subh_fbar_tng_ar = np.append(subh_fbar_tng_ar, subh.get_mstar(where = int(subh.snap), how = '2rh')/ subh.get_mdm(where = int(subh.snap), how = '2rh'))
    subh_fdm_ar = np.append(subh_fdm_ar, subh.get_mdm(where = 99, how = 'total') / subh.get_mdm(where = int(snap_if_ar[ix]), how = 'total'))
    subh_mstar_99 = subh.get_mstar(where=99, how = 'total')
    subh_mstar_99_ar = np.append(subh_mstar_99_ar, subh_mstar_99)
    subh_tinf_ar = np.append(subh_tinf_ar, all_ages[snap_if_ar[ix]])
    fields = ['SubhaloMassInRadType', 'SubhaloGrNr', 'SnapNum', 'GroupNsubs', 'SubhaloPos', 'Group_R_Crit200', 'Group_M_Crit200', 'SubhaloVel']
    subh_m200_ar = np.append(subh_m200_ar, subh.get_m200(where = int(snap_if_ar[ix] - 1))) #Append only when there are no errors
    frem = subh.get_frem(float(all_ages[subh.snap]), t) #FIXME: Some orbits are not unbound as galpy reports
    frem_tng = subh.get_tng_values()[2]/subh.mmx0
    frem_tng_ar = np.append(frem_tng_ar, frem_tng)
    subh_mstar_model_ar = np.append(subh_mstar_model_ar, subh.get_starprops_model(frem = frem)[0])
    subh_fstar_model_ar = np.append(subh_fstar_model_ar, subh_mstar_model_ar[-1]/subh.mstar)

    subh_mstar_model_from_tngfrem_ar = np.append(subh_mstar_model_from_tngfrem_ar, subh.get_starprops_model(frem = frem_tng)[0])
    subh_fstar_model_from_tngfrem_ar = np.append(subh_fstar_model_from_tngfrem_ar, subh_mstar_model_from_tngfrem_ar[-1]/subh.mstar)

    subh_mstar_tng_ar = np.append(subh_mstar_tng_ar, subh_mstar_99)
    subh_fstar_tng_ar = np.append(subh_fstar_tng_ar, subh_mstar_99 / subh.get_mstar(where = 'max', how = 'total'))
    subh_parlen_99_ar = np.append(subh_parlen_99_ar, subh.get_len(where = 99, which = 'dm') + subh.get_len(where = 99, which = 'stars'))
    if subh_parlen_99_ar[-1] < 3000:
        ixs_unresol = np.append(ixs_unresol, int(len(frem_tng_ar) - 1))
    if subh_fstar_model_ar[-1] > 0.8 and subh_fstar_tng_ar[-1] < 0.45 : #Checking why the masses are too low for the Errani
        ixs_high_mstar = np.append(ixs_high_mstar, int(len(frem_tng_ar) - 1))
    elif subh_fstar_model_ar[-1] < 0.1 and subh_fstar_tng_ar[-1] > 0.2:
        subh_ixs_low_mstar = np.append(subh_ixs_low_mstar, ix)
        ixs_low_mstar = np.append(ixs_low_mstar, int(len(frem_tng_ar) - 1))
    frem_ar = np.append(frem_ar, frem)




'''
Plot 1: This is the comparison between  dark matter fractions
'''
ixs_high_mstar = ixs_high_mstar.astype('i')
ixs_low_mstar = ixs_low_mstar.astype('i')
ixs_unresol = ixs_unresol.astype('i')
fig, ax = plt.subplots(figsize = (7, 6))
dummy = np.linspace(0.001, 1, 3)
sc = ax.scatter(frem_ar, frem_tng_ar, c=np.log10(subh_mstar_99_ar), cmap='viridis', marker='o', zorder = 20)
ax.scatter(frem_ar[ixs_low_mstar], frem_tng_ar[ixs_low_mstar], label = r'Low $f_{\bigstar}$', edgecolors = 'red', s = 100, facecolors = 'white', zorder = 10)
ax.scatter(frem_ar[ixs_high_mstar], frem_tng_ar[ixs_high_mstar], label = r'High $f_{\bigstar}$', edgecolors = 'blue', s = 100, facecolors = 'white', zorder = 10)
ax.scatter(frem_ar[ixs_unresol], frem_tng_ar[ixs_unresol], label = r'Unresolved', marker = 'x', s = 100, color = 'black', zorder = 40)

cbar = plt.colorbar(sc)
# cbar.set_label(r'$\log M_{\rm{200, infall}}$')
# cbar.set_label(r'$t_{\rm{inf}}$')
# cbar.set_label(r'$\log f_{\rm{\bigstar, inf}}$')
cbar.set_label(r'$M_{\rm{\bigstar, z=0}}$')
ax.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)
ax.set_xlabel(r'$M_{\rm{mx}}/M_{\rm{mx0}}$ from model')
ax.set_ylabel(r'$M_{\rm{mx}}/M_{\rm{mx0}}$ from TNG')
# ax.set_xlim(-0.1, 1)
# ax.set_ylim(-0.1, 1)
ax.legend(fontsize = 8)
plt.loglog()
plt.tight_layout()
plt.show()

# def plot_fstar():



'''
Plot 2: this is a comparison of the stellar fractions between the model and TNG
'''
fig, ax = plt.subplots(figsize = (7, 6))
dummy = np.linspace(0, 1, 3)
# sc = ax.scatter(subh_fstar_model_ar, subh_fstar_tng_ar, c=subh_tinf_ar, cmap='viridis', marker='o', zorder = 20)
sc = ax.scatter(subh_fstar_model_from_tngfrem_ar, subh_fstar_tng_ar, c=subh_tinf_ar, cmap='viridis', marker='o', zorder = 20)
# sc = ax.scatter(1 - np.exp(-14.20 * subh_fdm_ar), subh_fstar_tng_ar, c=(subh_tinf_ar), cmap='viridis', marker='s', zorder = 20)
cbar = plt.colorbar(sc)
# cbar.set_label(r'$\log M_{\rm{200, infall}}$')
# cbar.set_label(r'$t_{\rm{inf}}$')
# cbar.set_label(r'$ \log(M_{\rm{mx}}/M_{\rm{mx0}})$ from model')
cbar.set_label(r'$t_{\rm{inf}}$')
# cbar.set_label(r'$M_{\rm{\bigstar, z=0}}$')
ax.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)
ax.set_xlabel(r'$M_{\rm{\bigstar, z=0}}/M_{\rm{\bigstar, max}}$ from model')
ax.set_ylabel(r'$M_{\rm{\bigstar, z=0}}/M_{\rm{\bigstar, max}}$ from TNG')
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
ax.set_title(r'Model uses $M_{\rm{mx}}/M_{\rm{mx0}}$ from TNG', color = 'red')
# plt.loglog()
plt.tight_layout()
plt.show()


'''
Plot 2.1: This is a plot of the stellar mass remaining in the TNG as compared to the one in the model
'''
ixs_high_mstar = ixs_high_mstar.astype('i')
ixs_low_mstar = ixs_low_mstar.astype('i')
ixs_unresol = ixs_unresol.astype('i')
fig, ax = plt.subplots(figsize = (7, 6))
# dummy = np.linspace(0, 1, 3)
sc = ax.scatter(subh_mstar_model_ar, subh_mstar_tng_ar, c=subh_tinf_ar, cmap='viridis', marker='o', zorder = 20)
# sc = ax.scatter(subh_mstar_model_from_tngfrem_ar, subh_mstar_tng_ar, c=subh_tinf_ar, cmap='viridis', marker='o', zorder = 20)
cbar = plt.colorbar(sc)
ax.scatter(subh_mstar_model_ar[ixs_low_mstar], subh_mstar_tng_ar[ixs_low_mstar], label = r'Low $f_{\bigstar}$', edgecolors = 'red', s = 100, facecolors = 'white', zorder = 10)
ax.scatter(subh_mstar_model_ar[ixs_high_mstar], subh_mstar_tng_ar[ixs_high_mstar], label = r'High $f_{\bigstar}$', edgecolors = 'blue', s = 100, facecolors = 'white', zorder = 10)
ax.scatter(subh_mstar_model_ar[ixs_unresol], subh_mstar_tng_ar[ixs_unresol], label = r'Unresolved', marker = 'x', s = 100, color = 'black', zorder = 40)

# cbar.set_label(r'$\log M_{\rm{200, infall}}$')
# cbar.set_label(r'$t_{\rm{inf}}$')
# cbar.set_label(r'$ \log(M_{\rm{mx}}/M_{\rm{mx0}})$ from model')
cbar.set_label(r'$t_{\rm{inf}}$')
# cbar.set_label(r'$M_{\rm{\bigstar, z=0}}$')
dummy = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 3) 
ax.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)
ax.set_xlabel(r'$M_{\rm{\bigstar, z=0}}$ from model')
ax.set_ylabel(r'$M_{\rm{\bigstar, z=0}}$ from TNG')
# ax.set_xlim(left = 1e5)
ax.legend(fontsize = 8)
# ax.set_title(r'Model uses $M_{\rm{mx}}/M_{\rm{mx0}}$ from TNG', color = 'red')

# ax.set_ylim(0, 1)
plt.loglog()
plt.tight_layout()
plt.show()



'''
Plot 3: Plot all the orbits for the subhalos considered
'''
pdf_file = outpath + "orbits_tng50_more1e9msun_epts.pdf"
pdf_pages = PdfPages(pdf_file)
subh_ixs_low_mstar = subh_ixs_low_mstar.astype('i')



# for ix in tqdm(range(len(snap_if_ar))):
# for ix in tqdm([30, 49, 56, 68, 84, 85, 89, 95]):
# for ix in tqdm([83, 93, 94, 98, 100, 102, 103, 106]):
for ix in tqdm(subh_ixs_low_mstar):
    '''
    This is to loop over all the surviving subhalos of big dataset with subhalos in between 1e8.5 and 1e9.5 Msun
    ''' 
    subh  = Subhalo(snap = snap_if_ar[ix], sfid = sfid_if_ar[ix])
    try:
        t = subh.get_orbit(merged = False, when_te = 'last') #after this, the subhalo has rperi, rapo and torb
    except Exception as e:
        pass
    subh.plot_orbit_comprehensive(merged = False)
    plt.tight_layout()
    pdf_pages.savefig()
    # plt.show()
    plt.close()   

    
pdf_pages.close()


# '''
# Plot 3: This is Mmx and 
# '''

# for ix in tqdm([30, 49, 56, 68, 84, 85, 89, 95]):
#     subh  = Subhalo(snap = snap_if_ar[ix], sfid = sfid_if_ar[ix])
#     subh.get_mass_profiles(where = 'max', plot = True)
#     exp_check = ExponentialProfile(subh.get_mstar(where = 'max', how = 'total'), np.sqrt(2) * subh.get_rh0byrmx0() * subh.rmx0) #This is an instance to check the exponential profile
#     left, right = plt.gca().get_xlim()
#     rpl = np.logspace(np.log10(left), np.log10(right), 100)
#     plt.plot(rpl, exp_check.mass(rpl), label = 'Expoential assumed', color = 'salmon', ls = ':')
#     plt.legend()
#     pdf_pages2.savefig()
#     plt.show()
#     plt.close()