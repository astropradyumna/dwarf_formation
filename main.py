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
    subh.get_infall_properties()

    Rh_tng_max = subh.get_Rh(where = 'max')/np.sqrt(2)
    Rh_model = subh.get_rh0byrmx0() * subh.rmx0 #This is the assumption
    ax.plot(Rh_model, Rh_tng_max, 'ko')
    if ix in [30, 49, 56, 68, 84, 85, 89, 95]: 
        ax.plot(Rh_model, Rh_tng_max, 'ro')

ax.set_xlabel(r'$R_{\rm{h, inf}}$ model (kpc)')
ax.set_ylabel(r'$R_{\rm{h, max\, M_\bigstar}}$ from TNG (kpc)')
ax.set_xlim(0, 8)
ax.set_ylim(0, 8)
x_vals = np.array(ax.get_xlim())
ax.plot(x_vals, x_vals, 'k--', lw = 0.5)
plt.tight_layout()
plt.show()


'''
Plot 0.2: This is to plot the mass of the ones with no stellar mass remaining
This is to compare with the exponential profile and test if it is a good enough approximation for the stellar profile
'''
pdf_file2 = outpath + "mass profiles_tng50_more1e9msun_tsah.pdf"
pdf_pages2 = PdfPages(pdf_file2)

for ix in tqdm([30, 49, 56, 68, 84, 85, 89, 95]):
    subh  = Subhalo(snap = snap_if_ar[ix], sfid = sfid_if_ar[ix])
    subh.get_mass_profiles(where = 'max', plot = True)
    exp_check = ExponentialProfile(subh.get_mstar(where = 'max', how = 'total'), np.sqrt(2) * subh.get_rh0byrmx0() * subh.rmx0) #This is an instance to check the exponential profile
    left, right = plt.gca().get_xlim()
    rpl = np.logspace(np.log10(left), np.log10(right), 100)
    plt.plot(rpl, exp_check.mass(rpl), label = 'Expoential assumed', color = 'salmon', ls = ':')
    plt.legend()
    pdf_pages2.savefig()
    plt.show()
    plt.close()

pdf_pages2.close()

    


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


'''
Run this loop for doing everything after evolution
'''



for ix in tqdm(range(len(snap_if_ar))):
    '''
    This is to loop over all the surviving subhalos of big dataset with subhalos in between 1e8.5 and 1e9.5 Msun
    '''
    
    # if ix>2: break

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


    subh_mstar_99 = subh.get_mstar(where=99)
    subh_mstar_99_ar = np.append(subh_mstar_99_ar, subh_mstar_99)
    subh_tinf_ar = np.append(subh_tinf_ar, all_ages[snap_if_ar[ix]])
    fields = ['SubhaloMassInRadType', 'SubhaloGrNr', 'SnapNum', 'GroupNsubs', 'SubhaloPos', 'Group_R_Crit200', 'Group_M_Crit200', 'SubhaloVel']
    subh_m200_ar = np.append(subh_m200_ar, subh.get_m200(where = int(snap_if_ar[ix] - 1))) #Append only when there are no errors
    frem = subh.evolve(t, V0 = 800) #FIXME: Some orbits are not unbound as galpy reports
    frem_tng = subh.get_tng_values()[2]/subh.mmx0
    frem_tng_ar = np.append(frem_tng_ar, frem_tng)
    subh_fstar_model_ar = np.append(subh_fstar_model_ar, subh.get_mstar_model(frem = frem)/subh.mstar)
    subh_fstar_tng_ar = np.append(subh_fstar_tng_ar, subh_mstar_99 / subh.get_mstar(where = 'max'))
    # if subh_fstar_model_ar[-1] < 0.1 and subh_fstar_tng_ar[-1] > 0.2: #Checking why the masses are too low for the Errani
    #     print(f'Just the index you are looking for {ix}')
    # print(f'frem from model is {frem:2f} and from the simulation is {frem_tng:.2f}')
    frem_ar = np.append(frem_ar, frem)




'''
Plot 1: This is the comparison between the dark matter fractions
'''
fig, ax = plt.subplots(figsize = (7, 6))
dummy = np.linspace(0, 1, 3)
sc = ax.scatter(frem_ar, frem_tng_ar, c=subh_tinf_ar, cmap='viridis', marker='o', zorder = 20)
cbar = plt.colorbar(sc)
# cbar.set_label(r'$\log M_{\rm{200, infall}}$')
cbar.set_label(r'$t_{\rm{inf}}$')
# cbar.set_label(r'$M_{\rm{\bigstar, z=0}}$')
ax.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)
ax.set_xlabel(r'$M_{\rm{mx}}/M_{\rm{mx0}}$ from model')
ax.set_ylabel(r'$M_{\rm{mx}}/M_{\rm{mx0}}$ from TNG')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()

# def plot_fstar():

'''
Plot 2: this is a comparison of the stellar fractions between the model and TNG
'''
fig, ax = plt.subplots(figsize = (7, 6))
dummy = np.linspace(0, 1, 3)
sc = ax.scatter(subh_fstar_model_ar, subh_fstar_tng_ar, c=np.log10(frem_ar), cmap='viridis', marker='o', zorder = 20)
cbar = plt.colorbar(sc)
# cbar.set_label(r'$\log M_{\rm{200, infall}}$')
# cbar.set_label(r'$t_{\rm{inf}}$')
cbar.set_label(r'$ \log(M_{\rm{mx}}/M_{\rm{mx0}})$ from model')
# cbar.set_label(r'$M_{\rm{\bigstar, z=0}}$')
ax.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)
ax.set_xlabel(r'$M_{\rm{\bigstar, z=0}}/M_{\rm{\bigstar, inf}}$ from model')
ax.set_ylabel(r'$M_{\rm{\bigstar, z=0}}/M_{\rm{\bigstar, max}}$ from TNG')
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()


'''
Plot 3: Plot all the orbits for the subhalos considered
'''
pdf_file = outpath + "orbits_tng50_more1e9msun_epts.pdf"
pdf_pages = PdfPages(pdf_file)


# for ix in tqdm(range(len(snap_if_ar))):
for ix in tqdm([30, 49, 56, 68, 84, 85, 89, 95]):
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
    # pdf_pages.savefig()
    plt.show()
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