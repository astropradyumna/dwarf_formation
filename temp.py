'''
This is a temporary program to analyze all the surviving subhalos

Currently working to obtain the best fit functional forms for Mstar-Vmax relations
'''


import numpy as np
import matplotlib.pyplot as plt 
import os 
os.environ["USE_LZMA"] = "0"
import pandas as pd
from tng_subhalo_and_halo import TNG_Subhalo
from tqdm import tqdm
import galpy
import IPython
import illustris_python as il
from matplotlib.backends.backend_pdf import PdfPages
from subhalo_profiles import ExponentialProfile, NFWProfile
from scipy.stats import median_abs_deviation
import warnings
# import emcee
from scipy.optimize import minimize, curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from populating_stars import *

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

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
survived_df = pd.read_csv(filepath + 'sh_survived_after_z3_tng50_1_everything.csv') #This does not have 100 particle restriction as well
# survived_df = pd.read_csv(filepath + 'sh_survived_after_z3_tng50_1_nomstar.csv')

ssh_sfid = survived_df['SubfindID'] #is this at infall?
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
Following is the import of the subhalos that get merged
'''


merged_df = pd.read_csv(filepath + 'sh_merged_after_z3_tng50_1_everything.csv')

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
Following is the data of the files that we already ran. We need rh and mstar value to obtain
'''
outpath  = '/home/psadh003/tng50/output_files/'
rvir_fof0 = 811.66/0.6744

dfs = pd.read_csv(outpath + 'surviving_evolved_fof0.csv', delimiter = ',')
dfs = dfs[dfs['dist_f_ar']<rvir_fof0]
dfs1 = dfs
dfs = dfs[(dfs['mstar_f_ar']>1e1) & (dfs['mstar_max_ar']<1e10)]
# dfs = dfs[(dfs['vd_f_ar']>1e1) & (dfs['mstar_max_ar']<1e10)]



sdist_f_ar = dfs['dist_f_ar']
spos_f_ar = dfs['pos_f_ar']

svd_f_ar_tng = dfs['vd_f_ar_tng']
srh_f_ar_tng = dfs['rh_f_ar_tng']
smstar_f_ar_tng = dfs['mstar_f_ar_tng']
smmx_f_ar_tng = dfs['mmx_f_ar_tng']
srmx_f_ar_tng = dfs['rmx_f_ar_tng']
svmx_f_ar_tng = dfs['vmx_f_ar_tng']

stinf_ar = dfs['tinf_ar']
storb_ar = dfs['torb_ar']
srapo_ar = dfs['rapo_ar']
srperi_ar = dfs['rperi_ar']

svd_f_ar = dfs['vd_f_ar']
srh_f_ar = dfs['rh_f_ar']
smstar_f_ar = dfs['mstar_f_ar']

svd_max_ar = dfs['vd_max_ar']
srh_max_ar = dfs['rh_max_ar']
smstar_max_ar = dfs['mstar_max_ar']

ssnap_if_ar = dfs['snap_if_ar']
ssf_id_if_ar = dfs['sfid_if_ar']

smmx_f_ar = dfs['mmx_f_ar']
srmx_f_ar = dfs['rmx_f_ar']
svmx_f_ar = dfs['vmx_f_ar']

smmx_if_ar = dfs['mmx_if_ar']
srmx_if_ar = dfs['rmx_if_ar']
svmx_if_ar = dfs['vmx_if_ar']




dfm = pd.read_csv(outpath + 'merged_evolved_fof0_wmbp.csv', delimiter = ',')
dfm = dfm[dfm['dist_f_ar']<rvir_fof0]
dfm1 = dfm 
dfm = dfm[(dfm['mstar_f_ar']>1e1) & (dfm['mstar_max_ar']<1e10)]

mdist_f_ar = dfm['dist_f_ar']
mmbpid_ar = dfm['mbpid_ar']

mtinf_ar = dfm['tinf_ar']
mtorb_ar = dfm['torb_ar']
mrapo_ar = dfm['rapo_ar']
mrperi_ar = dfm['rperi_ar']

mvd_f_ar = dfm['vd_f_ar']
mrh_f_ar = dfm['rh_f_ar']
mmstar_f_ar = dfm['mstar_f_ar']

mvd_max_ar = dfm['vd_max_ar']
mrh_max_ar = dfm['rh_max_ar']
mmstar_max_ar = dfm['mstar_max_ar']

msnap_if_ar = dfm['snap_if_ar']
msfid_if_ar = dfm['sfid_if_ar']

mmmx_f_ar = dfm['mmx_f_ar']
mrmx_f_ar = dfm['rmx_f_ar']
mvmx_f_ar = dfm['vmx_f_ar']

mmmx_if_ar = dfm['mmx_if_ar']
mrmx_if_ar = dfm['rmx_if_ar']
mvmx_if_ar = dfm['vmx_if_ar']









# ==============================================================
IPython.embed()



mdm_tot_tng = np.zeros(0) #This is the total DM present
mvir_tng = np.zeros(0) #This is from the previous snapshot
mtot_tng = np.zeros(0) #This is the total mass of the subhalo
vmax_tng = np.zeros(0) #this is the vmax
rh_tng = np.zeros(0)

for ix in tqdm(range(len(ssh_snap))): #This would run over all the subhalos surviving till z = 0

    # if ix > 5:
    #     break
    subh = TNG_Subhalo(snap = int(ssh_snap[ix]), sfid = int(ssh_sfid[ix]), last_snap=int(99))
    mvir_tng = np.append(mvir_tng, subh.get_m200(where = subh.snap - 1))
    mdm_tot_tng = np.append(mdm_tot_tng, subh.get_mdm(where = subh.snap))
    mtot_tng = np.append(mtot_tng, subh.get_mtot(where = subh.snap))
    vmax_tng = np.append(vmax_tng, subh.get_vmax(where = subh.snap))
    rh_tng = np.append(rh_tng, subh.get_rh(where = subh.snap)*3/4)






mmdm_tot_tng = np.zeros(0) #This is the total DM present
mmvir_tng = np.zeros(0) #This is from the previous snapshot
mmtot_tng = np.zeros(0) #This is the total mass of the subhalo
mvmax_tng = np.zeros(0) #this is the vmax
mrh_tng = np.zeros(0)

for ix in tqdm(range(len(msh_snap))):
    '''
    This is to loop over all the merging subhalos of big dataset with subhalos in between 1e8.5 and 1e9.5 Msun
    '''
    if msh_mstar[ix] > 1e10: continue #takes lot of time to get compiled

    subh  = TNG_Subhalo(snap = int(msh_snap[ix]), sfid = int(msh_sfid[ix]), last_snap = int(msh_merger_snap[ix]))
    mmvir_tng = np.append(mmvir_tng, subh.get_m200(where = subh.snap - 1))
    mmdm_tot_tng = np.append(mmdm_tot_tng, subh.get_mdm(where = subh.snap))
    mmtot_tng = np.append(mmtot_tng, subh.get_mtot(where = subh.snap))
    this_vmax = subh.get_vmax(where = subh.snap)
    if len(this_vmax) == 0: #Some of them have no vmax, accounting for that
        mvmax_tng = np.append(mvmax_tng, 0)
    else:
        mvmax_tng = np.append(mvmax_tng, this_vmax)
        
    mrh_tng = np.append(mrh_tng, subh.get_rh(where = subh.snap)*3/4)




'''
Plot 1: This is to plot Mstar vs halo mass
'''
def get_moster_shm(mtot):
    '''
    This is to plot the Moster+13 relation
    '''
    M = mtot
    M1 = 10 ** 11.59
    N = 0.0351  
    beta = 1.376 
    gamma = 0.608
    mstar = 2 * M * N * ((M/M1)**(-beta) + (M/M1)**(gamma))**-1

    return mstar

ssh_mstar[ssh_mstar == 0] = 1e3

fig, ax = plt.subplots(figsize = (5.5, 5))
mtotpl = np.logspace(7.5, 13, 100)
ax.plot(mtotpl, get_moster_shm(mtotpl), 'k--', label = 'Moster+13 at z = 0')
ax.scatter(mtot_tng, ssh_mstar, color = 'dodgerblue', marker = 'o', s = 1, alpha = 0.1, label = 'TNG at infall')
ax.set_ylabel(r'$M_\bigstar(M_\odot)$')
ax.set_xlabel(r'$M_{\rm{tot}}(M_\odot)$')
ax.axhline(5e6, ls = ':', color = 'gray', alpha = 0.4)
ax.legend(fontsize = 8)
plt.loglog()
plt.tight_layout()

plt.show()



'''
Plot 1.1: very similar to the above plot Mstar vs Mdm just to make sure nothing insane is going on
'''
h = 0.6774
mass_dm = 3.07367708626464e-05 *  1e10/h

def get_moster_shm(mtot):
    '''
    This is to plot the Moster+13 relation
    '''
    M = mtot
    M1 = 10 ** 11.59
    N = 0.0351  
    beta = 1.376 
    gamma = 0.608
    mstar = 2 * M * N * ((M/M1)**(-beta) + (M/M1)**(gamma))**-1

    return mstar

ssh_mstar[ssh_mstar == 0] = 1e3

fig, ax = plt.subplots(figsize = (5.5, 5))
mtotpl = np.logspace(7.5, 13, 100)
ax.plot(mtotpl, get_moster_shm(mtotpl), 'k--', label = 'Moster+13 at z = 0')
ax.scatter(mdm_tot_tng, ssh_mstar, color = 'dodgerblue', marker = 'o', s = 1, alpha = 0.1, label = 'TNG at infall')
ax.set_ylabel(r'$M_\bigstar(M_\odot)$')
ax.set_xlabel(r'$M_{\rm{DM}}(M_\odot)$')
ax.axhline(5e6, ls = ':', color = 'gray', alpha = 0.4)
ax.axvline(100*mass_dm, ls = ':', color = 'gray', alpha = 0.4)
ax.legend(fontsize = 8)
plt.loglog()
plt.tight_layout()

plt.show()


'''
Plot 2: This is to plot Mstar-Vmax
'''

def get_mstar_co(lvmax, alpha = 3.36, mu = -2.4, M0 = 3e8):
    eta = 10**lvmax / 50
    mstar = eta**alpha *np.exp(-eta**mu) * M0
    return np.log10(mstar)


def get_mstar_pl(lvmaxar, m1, m2, b):
    '''
    This is the power law model from Santos-Santos 2022
    '''
    lmstarar = np.zeros(0)
    for lvmax in lvmaxar:
        if lvmax >= np.log10(87):
            lmstar = m1 * lvmax + b
        elif lvmax < np.log10(87):
            lmstar = m2 * lvmax + (m1 - m2)*np.log10(87) + b
        lmstarar = np.append(lmstarar, lmstar)
    return lmstarar


def get_scatter(lvmaxar, sigma0 = 0.24, kappa = -1.26, V0 = 88.6):
    '''
    This function returns the scatter for both power law and the cutoff models
    '''
    vmaxar = 10**lvmaxar 
    sigma_ar = np.zeros(0)
    for vmax in vmaxar:
        if vmax > 57:
            sigma = sigma0
        elif vmax <= 57:
            sigma = kappa * np.log10(vmax/V0)
        sigma_ar = np.append(sigma_ar, sigma)
    return sigma_ar


vmax_tng_cut = vmax_tng[ssh_mstar > 1e3]
ssh_mstar_cut = ssh_mstar[ssh_mstar > 1e3]  #currently only considering the subhalos which have stellar mass to calibrate the relation

log_vmax = np.log10(vmax_tng_cut)
log_mstar = np.log10(ssh_mstar_cut)

bins = np.array([vmax_tng_cut.min(), 30, 40,60, 80, 100,175, 250, vmax_tng_cut.max()+1]) #choosing bins manually
num_bins = len(bins)
# Digitize x into logarithmic bins
bin_indices = np.digitize(vmax_tng_cut, bins)
median_mstar_per_bin = np.zeros(0)
mean_vmax_per_bin = np.zeros(0)
mad_mstar_per_bin = np.zeros(0)
for i in range(1, num_bins):
    # print(i, log_vmax[bin_indices == i])
    median_mstar_per_bin = np.append(median_mstar_per_bin, np.median(log_mstar[bin_indices == i]))
    mad_mstar_per_bin = np.append(mad_mstar_per_bin, median_abs_deviation(log_mstar[bin_indices == i]))
    mean_vmax_per_bin = np.append(mean_vmax_per_bin, np.mean(log_vmax[bin_indices == i]))


# def log_likelihood_pl(theta, x, y, yerr):
#     '''
#     Stolen from https://emcee.readthedocs.io/en/stable/tutorials/line/
#     '''
#     m, b, log_f = theta
#     model = get_mstar_pl(x, m, b)
#     sigma2 = yerr**2 + model**2 * np.exp(2 * log_f) #this is probably sigma^2
#     return -1 * -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))



# def log_likelihood_co(theta, x, y, yerr):
#     '''
#     Stolen from https://emcee.readthedocs.io/en/stable/tutorials/line/
#     '''
#     alpha, mu, M0, log_f = theta
#     model = get_mstar_co(x,alpha, mu, M0)
#     sigma2 = yerr**2 + model**2 * np.exp(2 * log_f) #this is probably sigma^2
#     return -1 * -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))





np.random.seed(42)
# nll = lambda *args: -log_likelihood(*args)

popt = curve_fit(get_mstar_pl, mean_vmax_per_bin, median_mstar_per_bin, sigma = mad_mstar_per_bin, p0 = [3, 4.5, 0])[0]
m1_ml, m2_ml, b_ml = popt

popt = curve_fit(get_mstar_co, mean_vmax_per_bin, median_mstar_per_bin, sigma = mad_mstar_per_bin, p0 = [3.36, -2.4, 3e8])[0]
alpha, mu, M0 = popt


# popt = curve_fit(get_mstar_pl, log_vmax, log_mstar)[0]
# m_ml2, b_ml2 = popt
# popt = curve_fit(get_mstar_co, log_vmax, log_mstar, p0 = [3.36, -2.4, 3e8])[0]
# alpha2, mu2, M02 = popt


# initial = np.array([4.5, 0.8, np.log(0.1)]) + 0 * np.random.randn(3)
# soln = minimize(log_likelihood_pl, initial, args=(mean_vmax_per_bin, median_mstar_per_bin, mad_mstar_per_bin))
# m_ml, b_ml, log_f_ml = soln.x


# initial = np.array([3.36, -2.4, 3e8, np.log(0.1)]) + 0.1 * np.random.randn(4)
# soln = minimize(log_likelihood_co, initial, args=(mean_vmax_per_bin, median_mstar_per_bin, mad_mstar_per_bin))
# m_ml, b_ml, log_f_ml = soln.x







fig, ax = plt.subplots(figsize = (5.5, 5))
vmxpl = np.logspace(0.5, 3)
lvmxpl = np.log10(vmxpl)


ax.plot(np.log10(vmxpl), get_mstar_co(np.log10(vmxpl), alpha, mu, M0), 'g--', label = 'cut-off', lw = 1.2)
ax.fill_between(np.log10(vmxpl), get_mstar_co(lvmxpl, alpha, mu, M0) - get_scatter(lvmxpl), get_mstar_co(lvmxpl, alpha, mu, M0) + get_scatter(lvmxpl), 'g', alpha = 0.3)
ax.plot(np.log10(vmxpl), get_mstar_pl(np.log10(vmxpl), m1_ml, m2_ml, b_ml), 'r-.', label = 'power law', lw = 1.2)
ax.fill_between(np.log10(vmxpl), get_mstar_pl(lvmxpl, m1_ml, m2_ml, b_ml) - get_scatter(lvmxpl), get_mstar_pl(lvmxpl, m1_ml, m2_ml, b_ml) + get_scatter(lvmxpl), 'r', alpha = 0.3)
ax.scatter(log_vmax, log_mstar, color = 'dodgerblue', marker = 'o', s = 1, alpha = 0.1, label = 'TNG at infall')
ax.errorbar(mean_vmax_per_bin, median_mstar_per_bin, yerr = mad_mstar_per_bin, color = 'darkblue', marker = 'o', label = 'median values', ls = '')
ax.set_ylabel(r'$\log M_\bigstar(M_\odot)$')
ax.set_xlabel(r'$\log V_{\rm{max}}(M_\odot)$')
ax.axhline(np.log10(5e6), ls = ':', color = 'gray', alpha = 0.4)
ax.set_ylim(bottom = 2, top = 12)
ax.legend(fontsize = 8)
plt.tight_layout()
plt.show()

# globals().update(locals())






'''
Plot 3: This is to fit the LG dwarfs to a straight line
'''
filepath = '/home/psadh003/dwarf_data/'
filein = filepath + 'LGdata_withErrors.dat'  
data = np.loadtxt(filein,usecols=[1,2,6,7,8])

#pdb.set_trace()
mstr = data[:,0]
rh_proj = data[:,1]   #half-light
## ---- take ellipticity into account ##
ell = data[:,4]
rh_proj *= np.sqrt(1.-ell)
## ---------
# rh = rh_proj * 4./3.   #still light, but 3D
rh = rh_proj #plotting the projected radius
err_mstr = data[:,2]
err_rh = data[:,3]
err_rh *= 4./3.

#pdb.set_trace()
aux = (mstr>0) & (rh > 0)
mstr = mstr[aux]
rh = rh[aux]
err_mstr = err_mstr[aux]
err_rh = err_rh[aux]


def get_lrh(lmstar_ar, m1, m2, b):
    '''
    This function returns the log rh for a given Mstar
    '''
    lrh_ar = np.zeros(0)
    for lmstar in lmstar_ar:
        if lmstar > 6.5:
            lrh = m1 * lmstar + b 
        elif lmstar <= 6.5:
            lrh = m2 * lmstar + (m1 - m2) * 6.5 + b
        lrh_ar = np.append(lrh_ar, lrh)
    return lrh_ar


cond = (srh_f_ar_tng > 0) & (smstar_max_ar > 0)
lrh_data = np.concatenate([np.log10(rh), np.log10(srh_f_ar_tng[cond])])
lmstar_data = np.concatenate([np.log10(mstr), np.log10(smstar_max_ar[cond])])


# noise_std = 0.75
# gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=9)
# gaussian_process.fit(lmstar_data.reshape(-1, 1), lrh_data.reshape(-1, 1))


bins = np.array([lmstar_data.min(), 3, 4, 5, 6, 7.5, 8, 9, lmstar_data.max()+1]) #choosing bins manually
num_bins = len(bins)
# Digitize x into logarithmic bins
bin_indices = np.digitize(lmstar_data, bins)
median_rh_per_bin = np.zeros(0)
mean_mstar_per_bin = np.zeros(0)
mad_rh_per_bin = np.zeros(0)
for i in range(1, num_bins):
    # print(i, log_vmax[bin_indices == i])
    median_rh_per_bin = np.append(median_rh_per_bin, np.median(lrh_data[bin_indices == i]))
    mad_rh_per_bin = np.append(mad_rh_per_bin, median_abs_deviation(lrh_data[bin_indices == i]))
    mean_mstar_per_bin = np.append(mean_mstar_per_bin, np.mean(lmstar_data[bin_indices == i]))



popt = curve_fit(get_lrh, lmstar_data, lrh_data, p0 = [0, 0, 1])[0]
m1_rm, m2_rm, b_rm = popt

sigma_rm = np.std(lrh_data - get_lrh(lmstar_data, m1_rm, m2_rm, b_rm))

lmstar_pl = np.linspace(1.5, 10)

mean_prediction, std_prediction = gaussian_process.predict(lmstar_pl.reshape(-1, 1), return_std=True)


fig, ax = plt.subplots(figsize = (8, 6))
ax.plot(np.log10(mstr), np.log10(rh),marker='o',ms=4, mfc='white', mec = 'black',ls='none', alpha = 0.2, label = 'LG')
ax.plot(np.log10(smstar_max_ar), np.log10(srh_f_ar_tng), marker = 's', mfc = 'white', mec = 'darkgreen', ls = 'none', alpha = 0.2,  label = 'surviving at infall')
ax.plot(lmstar_pl, get_lrh(lmstar_pl, m1_rm, m2_rm, b_rm), color = 'black', ls = '--', label = 'Best fit power law')
ax.errorbar(mean_mstar_per_bin, median_rh_per_bin, yerr = mad_rh_per_bin, color = 'black', marker = 'o', label = 'median values', ls = '', mew = 3)

ax.fill_between(lmstar_pl, get_lrh(lmstar_pl, m1_rm, m2_rm, b_rm) - sigma_rm,get_lrh(lmstar_pl, m1_rm, m2_rm, b_rm) + sigma_rm, color = 'gray', alpha = 0.2)

# plt.plot(lmstar_pl, mean_prediction, color = 'tab:orange', ls = ':', label="using GPR")
# plt.fill_between(
#     lmstar_pl.ravel(),
#     mean_prediction[:, 0] - 1.96 * std_prediction,
#     mean_prediction[:, 0] + 1.96 * std_prediction,
#     color="tab:orange",
#     alpha=0.1,
# )
# ax.set_xlim(left = 1e1)
# ax.set_ylim(bottom = 10)
ax.set_xlabel(r'$\log M_{\bigstar}\,\rm{(M_\odot)}$')
ax.set_ylabel(r'$\log R_{\rm{h}}$ (kpc)')
ax.legend(fontsize = 8)
# plt.loglog()
plt.tight_layout()
plt.show()


'''
Plot 3.1: same as above but we now are combining LG and Virgo
'''
filepath = '/home/psadh003/dwarf_data/'
filein = filepath + 'LGdata_withErrors.dat'  
data = np.loadtxt(filein,usecols=[1,2,6,7,8])

#pdb.set_trace()
mstr = data[:,0]
rh_proj = data[:,1]   #half-light
## ---- take ellipticity into account ##
ell = data[:,4]
rh_proj *= np.sqrt(1.-ell)
## ---------
# rh = rh_proj * 4./3.   #still light, but 3D
rh = rh_proj #plotting the projected radius
err_mstr = data[:,2]
err_rh = data[:,3]
err_rh *= 4./3.

#pdb.set_trace()
aux = (mstr>0) & (rh > 0)
mstr = mstr[aux]
rh = rh[aux]
err_mstr = err_mstr[aux]
err_rh = err_rh[aux]


den_df = pd.read_csv(filepath + 'Virgo_nucleated_dEs.txt', header=0, delimiter=' ')
den_re = den_df['Re(pc)']/1e3 #This is now in kpc
den_mg = den_df['Mag_g']

def get_ms_from_mag(x):
    '''
    This function is to obtain the stellar mass from magnitude in  Msun
    '''
    return  10**(-0.439 * x + 1.549)


def get_lrh(lmstar_ar, m1, m2, b):
    '''
    This function returns the log rh for a given Mstar
    '''
    lrh_ar = np.zeros(0)
    for lmstar in lmstar_ar:
        if lmstar > 6.5:
            lrh = m1 * lmstar + b 
        elif lmstar <= 6.5:
            lrh = m2 * lmstar + (m1 - m2) * 6.5 + b
        lrh_ar = np.append(lrh_ar, lrh)
    return lrh_ar


# cond = (srh_f_ar_tng > 0) & (smstar_max_ar > 0)
lrh_data = np.concatenate([np.log10(rh), np.log10(den_re)])
lmstar_data = np.concatenate([np.log10(mstr), np.log10(get_ms_from_mag(den_mg))])


# noise_std = 0.75
# gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=9)
# gaussian_process.fit(lmstar_data.reshape(-1, 1), lrh_data.reshape(-1, 1))


bins = np.array([lmstar_data.min(), 3, 4, 5, 6, 7, 8, 9, lmstar_data.max()+1]) #choosing bins manually
num_bins = len(bins)
# Digitize x into logarithmic bins
bin_indices = np.digitize(lmstar_data, bins)
median_rh_per_bin = np.zeros(0)
mean_mstar_per_bin = np.zeros(0)
mad_rh_per_bin = np.zeros(0)
for i in range(1, num_bins):
    # print(i, log_vmax[bin_indices == i])
    median_rh_per_bin = np.append(median_rh_per_bin, np.median(lrh_data[bin_indices == i]))
    mad_rh_per_bin = np.append(mad_rh_per_bin, median_abs_deviation(lrh_data[bin_indices == i]))
    mean_mstar_per_bin = np.append(mean_mstar_per_bin, np.mean(lmstar_data[bin_indices == i]))



popt = curve_fit(get_lrh, lmstar_data, lrh_data, p0 = [0, 0, 1])[0]
m1_rm, m2_rm, b_rm = popt

sigma_rm = np.std(lrh_data - get_lrh(lmstar_data, m1_rm, m2_rm, b_rm))

lmstar_pl = np.linspace(1.5, 10)



fig, ax = plt.subplots(figsize = (8, 6))
# ax.plot(np.log10(smstar_max_ar), np.log10(srh_f_ar_tng), marker = 's', mfc = 'white', mec = 'darkgreen', ls = 'none', alpha = 0.2,  label = 'surviving at infall')
ax.plot(np.log10(mstr), np.log10(rh),marker='o',ms=4, mfc='white', mec = 'black',ls='none', alpha = 0.2, label = 'LG')
ax.plot(np.log10(get_ms_from_mag(den_mg)), np.log10(den_re), marker = 'o', color = 'darkkhaki', label = 'Nucleated galaxies', 
            lw = 0, ms = 4, alpha = 0.2)
ax.plot(lmstar_pl, get_lrh(lmstar_pl, m1_rm, m2_rm, b_rm), color = 'black', ls = '--', label = 'Best fit power law')
ax.errorbar(mean_mstar_per_bin, median_rh_per_bin, yerr = mad_rh_per_bin, color = 'black', marker = 'o', label = 'median values', ls = '', mew = 3)

ax.fill_between(lmstar_pl, get_lrh(lmstar_pl, m1_rm, m2_rm, b_rm) - sigma_rm,get_lrh(lmstar_pl, m1_rm, m2_rm, b_rm) + sigma_rm, color = 'gray', alpha = 0.2)
ax.set_xlabel(r'$\log M_{\bigstar}\,\rm{(M_\odot)}$')
ax.set_ylabel(r'$\log R_{\rm{h}}$ (kpc)')
ax.legend(fontsize = 8)
# plt.loglog()
plt.tight_layout()
plt.show()






'''
Plot 4: This is to obtain the stellar mass of the subhalos for a given vmax from the simulation
'''

# get_(vmax_tng)
cond = (ssh_mstar < 5e6)
vmax_tng_ur_cut = vmax_tng[cond] #These would be the subhalos which are considered unresolved

mstar_tng_ur_cut = get_mstar_pl_wsc(np.log10(vmax_tng_ur_cut))
mstar_tng_ur_cut_co = get_mstar_co_wsc(np.log10(vmax_tng_ur_cut))

vmax_tng_cut = vmax_tng[ssh_mstar > 1e3]
ssh_mstar_cut = ssh_mstar[ssh_mstar > 1e3]  #currently only considering the subhalos which have stellar mass to calibrate the relation

log_vmax = np.log10(vmax_tng_cut)
log_mstar = np.log10(ssh_mstar_cut)

fig, ax = plt.subplots(figsize = (5.5, 5))
vmxpl = np.logspace(0.5, 3)
lvmxpl = np.log10(vmxpl)


# ax.plot(np.log10(vmxpl), get_mstar_co(np.log10(vmxpl), alpha, mu, M0), 'g--', label = 'cut-off', lw = 1.2)
# ax.fill_between(np.log10(vmxpl), get_mstar_co(lvmxpl, alpha, mu, M0) - get_scatter(lvmxpl), get_mstar_co(lvmxpl, alpha, mu, M0) + get_scatter(lvmxpl), 'g', alpha = 0.3)
# ax.plot(np.log10(vmxpl), get_mstar_pl(np.log10(vmxpl), m1_ml, m2_ml, b_ml), 'r-.', label = 'power law', lw = 1.2)
# ax.fill_between(np.log10(vmxpl), get_mstar_pl(lvmxpl, m1_ml, m2_ml, b_ml) - get_scatter(lvmxpl), get_mstar_pl(lvmxpl, m1_ml, m2_ml, b_ml) + get_scatter(lvmxpl), 'r', alpha = 0.3)
ax.scatter(log_vmax, log_mstar, color = 'dodgerblue', marker = 'o', s = 1, alpha = 0.1, label = 'TNG at infall')
ax.scatter(np.log10(vmax_tng_ur_cut), np.log10(mstar_tng_ur_cut),  color = 'red', marker = 'o', s = 1, alpha = 0.05, label = 'power-law model')
ax.scatter(np.log10(vmax_tng_ur_cut), np.log10(mstar_tng_ur_cut_co),  color = 'green', marker = 'o', s = 1, alpha = 0.05, label = 'power-law model')
# ax.errorbar(mean_vmax_per_bin, median_mstar_per_bin, yerr = mad_mstar_per_bin, color = 'darkblue', marker = 'o', label = 'median values', ls = '')
ax.set_ylabel(r'$\log M_\bigstar(M_\odot)$')
ax.set_xlabel(r'$\log V_{\rm{max}}(M_\odot)$')
ax.axhline(np.log10(5e6), ls = ':', color = 'gray', alpha = 0.4)
ax.axhline(np.log10(1e2), ls = ':', color = 'gray', alpha = 0.4)
# ax.set_ylim(bottom = 2, top = 12)
ax.legend(fontsize = 8)
plt.tight_layout()
plt.show()




'''
Plot 5: This is to obtain the stellar mass of the subhalos for a given vmax from the simulation
MERGED SUBHALOS
'''

# get_(vmax_tng)
cond = (msh_mstar < 5e6)
vmax_tng_ur_cut = mvmax_tng[cond] #These would be the subhalos which are considered unresolved

mstar_tng_ur_cut = get_mstar_pl_wsc(np.log10(vmax_tng_ur_cut))
mstar_tng_ur_cut_co = get_mstar_co_wsc(np.log10(vmax_tng_ur_cut))

vmax_tng_cut = mvmax_tng[ssh_mstar > 1e3]
msh_mstar_cut = msh_mstar[ssh_mstar > 1e3]  #currently only considering the subhalos which have stellar mass to calibrate the relation

log_vmax = np.log10(vmax_tng_cut)
log_mstar = np.log10(msh_mstar_cut)

fig, ax = plt.subplots(figsize = (5.5, 5))
vmxpl = np.logspace(0.5, 3)
lvmxpl = np.log10(vmxpl)


# ax.plot(np.log10(vmxpl), get_mstar_co(np.log10(vmxpl), alpha, mu, M0), 'g--', label = 'cut-off', lw = 1.2)
# ax.fill_between(np.log10(vmxpl), get_mstar_co(lvmxpl, alpha, mu, M0) - get_scatter(lvmxpl), get_mstar_co(lvmxpl, alpha, mu, M0) + get_scatter(lvmxpl), 'g', alpha = 0.3)
# ax.plot(np.log10(vmxpl), get_mstar_pl(np.log10(vmxpl), m1_ml, m2_ml, b_ml), 'r-.', label = 'power law', lw = 1.2)
# ax.fill_between(np.log10(vmxpl), get_mstar_pl(lvmxpl, m1_ml, m2_ml, b_ml) - get_scatter(lvmxpl), get_mstar_pl(lvmxpl, m1_ml, m2_ml, b_ml) + get_scatter(lvmxpl), 'r', alpha = 0.3)
ax.scatter(log_vmax, log_mstar, color = 'dodgerblue', marker = 'o', s = 1, alpha = 0.1, label = 'TNG at infall')
ax.scatter(np.log10(vmax_tng_ur_cut), np.log10(mstar_tng_ur_cut),  color = 'red', marker = 'o', s = 1, alpha = 0.05, label = 'power-law model')
ax.scatter(np.log10(vmax_tng_ur_cut), np.log10(mstar_tng_ur_cut_co),  color = 'green', marker = 'o', s = 1, alpha = 0.05, label = 'power-law model')
# ax.errorbar(mean_vmax_per_bin, median_mstar_per_bin, yerr = mad_mstar_per_bin, color = 'darkblue', marker = 'o', label = 'median values', ls = '')
ax.set_ylabel(r'$\log M_\bigstar(M_\odot)$')
ax.set_xlabel(r'$\log V_{\rm{max}}(M_\odot)$')
ax.axhline(np.log10(5e6), ls = ':', color = 'gray', alpha = 0.4)
ax.axhline(np.log10(1e2), ls = ':', color = 'gray', alpha = 0.4)
# ax.set_ylim(bottom = 2, top = 12)
ax.legend(fontsize = 8)
plt.tight_layout()
plt.show()