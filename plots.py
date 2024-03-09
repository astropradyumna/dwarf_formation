import pandas as pd 
import matplotlib.pyplot as plt
import IPython
import numpy as np 
from dwarf_plotting import plot_lg_virgo, plot_lg_vd
import matplotlib
from populating_stars import *

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size' : 14}

plt.figure()
plt.close()
matplotlib.rc('font', **font)



outpath  = '/rhome/psadh003/bigdata/tng50/output_files/'
rvir_fof0 = 811.66/0.6744

dfs = pd.read_csv(outpath + 'surviving_evolved_fof0_everything.csv', delimiter = ',')
# dfs = pd.read_csv(outpath + 'surviving_evolved_fof0.csv', delimiter = ',')
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




dfm = pd.read_csv(outpath + 'merged_evolved_fof0_everything_wmbp.csv', delimiter = ',')
# dfm = pd.read_csv(outpath + 'merged_evolved_fof0_wmbp.csv', delimiter = ',')
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



IPython.embed()


'''
Plot 1: Plot M* vs Mvir
'''
def get_moster_shm(mmx):
    '''
    This is to plot the Moster+13 relation
    '''
    M = mmx/0.2
    M1 = 10 ** 11.59
    N = 0.0351  
    beta = 1.376 
    gamma = 0.608
    mstar = 2 * M * N * ((M/M1)**(-beta) + (M/M1)**(gamma))**-1

    return mstar


fig, ax = plt.subplots(figsize = (5.5, 5))
ax.scatter(smmx_f_ar, smstar_f_ar, alpha = 0.1, s = 6, color = 'darkgreen', label = 'surviving - model')
ax.scatter(mmmx_f_ar, mmstar_f_ar, alpha = 0.1, color = 'purple', s = 6, label = 'merged - model')
# ax.set_xlim(left = 1e4)
# ax.set_ylim(bottom  = 1e1)

mmxpl = np.logspace(2, 12, 100)
ax.plot(mmxpl, get_moster_shm(mmxpl), 'k--', label = 'Moster+13')
ax.plot(mmxpl, 10**get_mstar_co(np.log10(get_Mmx_from_Vmax(mmxpl))), 'k-.', label = 'cutoff')
ax.set_xlabel(r'$M_{\rm{mx}}\,\rm{(M_\odot)}$ ')
ax.set_ylabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
ax.set_title('z=0 from the model')
ax.legend(fontsize = 8)
plt.tight_layout()
plt.loglog()
plt.show()


'''
Plot 2: Subhalo mass functtion for the subhalos of the given file
'''

mstarpl = np.logspace(1, 11, 100)
Nm_ar = np.zeros(0) #shmf for merged subhalos
Ns_ar = np.zeros(0) #shmf for surviving subhalos 
Ntng_ar = np.zeros(0)
for (ix, ms) in enumerate(mstarpl):
    Nm_ar = np.append(Nm_ar, len(mmstar_f_ar[mmstar_f_ar > ms]))
    Ns_ar = np.append(Ns_ar, len(smstar_f_ar[smstar_f_ar > ms]))
    Ntng_ar = np.append(Ntng_ar, len(smstar_f_ar_tng[smstar_f_ar_tng > ms]))

fig, ax = plt.subplots(figsize = (5.5, 5))
ax.plot(mstarpl, Ns_ar, color = 'darkgreen', label = 'Model Surviving', alpha = 0.5)
ax.plot(mstarpl, Ns_ar + Nm_ar, color = 'purple', label = 'Model (all)', alpha = 0.5)
ax.plot(mstarpl, Ntng_ar, color = 'blue', label = 'TNG surviving', alpha = 0.5)
ax.legend(fontsize = 8)
ax.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
ax.set_ylabel(r'$N(>M_{\bigstar})$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(left = 1e1)
ax.set_title('z=0 from the model')

plt.tight_layout()
plt.show()



'''
Plot 3: Distance from the center vs M*
'''
fig, ax = plt.subplots(figsize = (5.5, 5))

ax.scatter(mmstar_f_ar, mdist_f_ar, color = 'purple', alpha = 0.1, s = 2, label = 'Merged')
ax.scatter(smstar_f_ar, sdist_f_ar, color = 'darkgreen', alpha = 0.1, s = 2, label = 'Surviving')
ax.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
ax.set_ylabel('Distance from center (kpc)')
ax.set_xlim(left = 1e1)
ax.set_ylim(bottom = 1e1)
ax.legend(fontsize = 8)
plt.loglog()
plt.tight_layout()
plt.show()



'''
Plot 4: N(<r)
'''
rpl = np.logspace(1, 3.2, 100)
Nm_ar = np.zeros(0) #shmf for merged subhalos
N_all_ar = np.zeros(0) #This is for all the subhalos inside virial radius
Ns_ar = np.zeros(0) #shmf for surviving subhalos 
Ntng_ar = np.zeros(0)
Ndm_ar = np.array([7.26007000e+05, 7.83449000e+05, 8.45384000e+05, 9.11628000e+05,
       9.83445000e+05, 1.06100800e+06, 1.14504500e+06, 1.23594500e+06,
       1.33408800e+06, 1.44044200e+06, 1.55612800e+06, 1.68105300e+06,
       1.81763000e+06, 1.96687600e+06, 2.12957200e+06, 2.30760600e+06,
       2.50243900e+06, 2.71673200e+06, 2.95097000e+06, 3.20765600e+06,
       3.48689500e+06, 3.79080200e+06, 4.12353100e+06, 4.48425900e+06,
       4.87492200e+06, 5.29924500e+06, 5.75880500e+06, 6.26397000e+06,
       6.80811500e+06, 7.38816300e+06, 8.01474000e+06, 8.68916500e+06,
       9.41651700e+06, 1.01945610e+07, 1.10287510e+07, 1.19245400e+07,
       1.28905830e+07, 1.39462060e+07, 1.50543080e+07, 1.62356140e+07,
       1.75035590e+07, 1.88439190e+07, 2.02752510e+07, 2.18098240e+07,
       2.34512650e+07, 2.52187030e+07, 2.71145670e+07, 2.91489910e+07,
       3.13481660e+07, 3.36580630e+07, 3.61007390e+07, 3.87241510e+07,
       4.15363730e+07, 4.45857010e+07, 4.78467400e+07, 5.12138770e+07,
       5.47311770e+07, 5.85538100e+07, 6.25258340e+07, 6.66550980e+07,
       7.08852110e+07, 7.52341050e+07, 7.98082850e+07, 8.45806860e+07,
       8.96575090e+07, 9.49362490e+07, 1.00327940e+08, 1.05793610e+08,
       1.11529568e+08, 1.17471516e+08, 1.23859419e+08, 1.30638982e+08,
       1.37935332e+08, 1.45766809e+08, 1.54319611e+08, 1.62605783e+08,
       1.70610087e+08, 1.78560606e+08, 1.86549103e+08, 1.95348981e+08,
       2.04170309e+08, 2.13528088e+08, 2.23188491e+08, 2.33025443e+08,
       2.42886564e+08, 2.52311457e+08, 2.60924861e+08, 2.69535566e+08,
       2.79584957e+08, 2.90861239e+08, 3.01104547e+08, 3.10675529e+08,
       3.20867704e+08, 3.32224243e+08, 3.43057890e+08, 3.53522251e+08,
       3.66029020e+08, 3.76169617e+08, 3.84308554e+08, 3.91530595e+08])
Ndm_ar = Ndm_ar/Ndm_ar[-1]

for (ix, ms) in enumerate(rpl): #ms is still radius
    Nm_ar = np.append(Nm_ar, len(mmstar_f_ar[mdist_f_ar < ms]))
    Ns_ar = np.append(Ns_ar, len(smstar_f_ar[sdist_f_ar < ms]))
    N_all_ar = np.append(N_all_ar, len(dfs1[dfs1['dist_f_ar']<ms]) + len(dfm1[dfm1['dist_f_ar']<ms]))
    # Ntng_ar = np.append()

Ndm_ar = Ndm_ar * (Nm_ar[-1] + Ns_ar[-1])
fig, ax = plt.subplots(figsize = (5.5, 5))
ax.plot(rpl, Ns_ar, color = 'blue', label = 'Surviving')
ax.plot(rpl, Ns_ar + Nm_ar, color = 'purple', label = r'Model ($>10\,\rm{M_\odot}$)')
ax.plot(rpl, N_all_ar, color = 'purple', ls = '--', lw = 0.5, label = r'Model (all)')
ax.plot(rpl, Ndm_ar, color = 'black', ls = '--', label = 'DM in TNG', alpha = 0.5)
ax.legend(fontsize = 8)
ax.set_xlabel('Distance from center (kpc)')
ax.set_ylabel(r'$N(<r)$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('z=0 from the model')

plt.tight_layout()
plt.show()



'''
Plot 5: Rh vs Mstar
'''
fig, ax = plt.subplots(figsize = (10, 6.5))
mspl_log = np.linspace(1, 11, 100)
plot_lg_virgo(ax)
ax.scatter(smstar_f_ar, srh_f_ar * 1e3, marker = 's', color = 'darkgreen', alpha = 0.15, s = 10, label = 'Survived', zorder = 200, edgecolor = 'black', linewidth = 0.7)
ax.scatter(mmstar_f_ar, mrh_f_ar * 1e3, marker = 's', color = 'purple', alpha = 0.15, s = 10, label = 'Merged', zorder = 200, edgecolor = 'black', linewidth = 0.7)
ax.plot(10**mspl_log, 1e3 * 10**get_lrh(mspl_log), color = 'k')
ax.set_xlim(left = 1e1)
ax.set_ylim(bottom = 10)
ax.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
ax.set_ylabel(r'$R_{\rm{h}}$ (pc)')
ax.set_title('At z=0')
ax.legend(fontsize = 8)
plt.loglog()
plt.tight_layout()
plt.show()


'''
Plot 5.1: Rh vs Mstar at infall
'''
mspl_log = np.linspace(1, 11, 100)
fig, ax = plt.subplots(figsize = (10, 6.5))
plot_lg_virgo(ax)
ax.scatter(smstar_max_ar, srh_max_ar * 1e3, marker = 's', color = 'darkgreen', alpha = 0.15, s = 10, label = 'Survived', zorder = 200, edgecolor = 'black', linewidth = 0.7)
ax.scatter(mmstar_max_ar, mrh_max_ar * 1e3, marker = 's', color = 'purple', alpha = 0.15, s = 10, label = 'Merged', zorder = 200, edgecolor = 'black', linewidth = 0.7)
ax.plot(10**mspl_log, 1e3 * 10**get_lrh(mspl_log), color = 'k')
ax.set_xlim(left = 1e1)
ax.set_ylim(bottom = 10)
ax.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
ax.set_ylabel(r'$R_{\rm{h}}$ (pc)')
ax.legend(fontsize = 8)
ax.set_title('At infall')
plt.loglog()
plt.tight_layout()
plt.show()


'''
PLot 6: Mstar - sigma relation
'''
fig, ax = plt.subplots(figsize = (6, 6))
# plot_lg_virgo(ax)

mvd_f_ar[mvd_f_ar < 0] = 0
svd_f_ar[svd_f_ar < 0] = 0
ax.scatter(mmstar_f_ar , mvd_f_ar, marker = 's', color = 'purple', alpha = 0.3, s = 15, label = 'Merged (model)', zorder = 0, edgecolor = 'black', linewidth = 0.5)
ax.scatter(smstar_f_ar , svd_f_ar, marker = 's', color = 'darkgreen', alpha = 0.3, s = 15, label = 'Survived (model)', zorder = 0, edgecolor = 'black', linewidth = 0.5)
plot_lg_vd(ax)
# ax.set_xlim(left = 1e1)
ax.set_ylim(bottom = 1)
ax.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
ax.set_ylabel(r'$\sigma$ (km/s)')
ax.legend(fontsize = 8)
dummy = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 3) 
# ax.plot( 10**(4 * np.log10(dummy) + 3.1), dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)
# ax.set_xscale('log')
plt.loglog()
plt.tight_layout()
plt.show()


'''
Plot 7: This is to compare the Rh values at the beginning and at the end 
'''
fig, ax = plt.subplots(figsize = (6, 5))
# col_ar = np.log10(smmx_f_ar/smmx_if_ar)
# col_label = r'$\log (M_{\rm{mmx, z = 0}}/M_{\rm{mmx, inf}})$'
col_ar = np.log10(smstar_f_ar)
col_label = r'$\log (M_{\bigstar, z = 0}/M_\odot)$'
# plot_lg_virgo(ax)

# mvd_f_ar[mvd_f_ar < 0] = 0
# svd_f_ar[svd_f_ar < 0] = 0

# ax.scatter(mmstar_f_ar , mvd_f_ar, marker = 's', color = 'purple', alpha = 0.6, s = 15, label = 'Merged', zorder = 200, edgecolor = 'black', linewidth = 0.5)
sc = ax.scatter(srh_max_ar*1e3 , srh_f_ar*1e3, marker = 'o', c=col_ar, cmap='viridis', alpha = 0.6, s = 15, label = 'Survived', zorder = 200, edgecolor = 'black', linewidth = 0.5)
# sc = ax.scatter(srh_max_ar*1e3 , srh_f_ar_tng*1e3, marker = 's', c=col_ar, cmap='viridis', alpha = 0.6, s = 15, label = 'Survived', zorder = 200, edgecolor = 'black', linewidth = 0.5)
cbar = plt.colorbar(sc, ax = ax)
cbar.set_label(col_label)
# ax.set_xlim(left = 1e1)
# ax.set_ylim(bottom = 1)
ax.set_xlabel(r'$R_{\rm{h}}$ at infall (pc)')
ax.set_ylabel(r'$R_{\rm{h}}$ at z = 0 (pc)')
ax.legend(fontsize = 8)
dummy = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 3) 
ax.plot( dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)
# ax.set_xscale('log')
plt.loglog()
plt.tight_layout()
plt.show()



'''
Plot 7.1: This is to compare the values of Rh/rmx0 from the originav values vs that assumed in the simulation
'''

def get_rh0byrmx0(Rh0, rmx0):
    '''
    This is a function to calculate the initial rh0burmx0 for the subhalo
    '''
    values = [1/2, 1/4, 1/8, 1/16]
    closest_value = min(values, key=lambda x: abs(np.log(x) - np.log(Rh0/rmx0)))
    return closest_value


fig, ax = plt.subplots(figsize = (5, 5))

for ix in tqdm(range(len(ssh_snap))):
    '''
    This is to loop over all the surviving subhalos of big dataset with subhalos in between 1e8.5 and 1e9.5 Msun
    ''' 
    subh  = Subhalo(snap = ssh_snap[ix], sfid = ssh_sfid[ix], last_snap = 99 )
    # subh.get_infall_properties()
    try:
        Rh_tng_max = subh.get_rh(where = 'max')/np.sqrt(2)
        Rh_model = subh.get_rh0byrmx0() * subh.rmx0 #This is the assumption
    except:
        continue
    # if subh.get_rh0byrmx0() == 0.5:
    #     print('Jai')
    # ax.plot(Rh_model/subh.rmx0, Rh_tng_max/subh.rmx0, 'ko', ms = 3.5)
    ax.plot(Rh_model, Rh_tng_max, 'ko', ms = 3.5)
    # if ix in [30, 49, 56, 68, 84, 85, 89, 95]: 
    #     ax.plot(Rh_model, Rh_tng_max, 'ro')
    #     if ix == 30:ax.plot(Rh_model, Rh_tng_max, 'ro', label = r'Low $f_{\bigstar}$')
    # elif ix in [83, 93, 94, 98, 100, 102, 103, 106]:
    #     ax.plot(Rh_model, Rh_tng_max, 'bo')
    #     if ix == 83: ax.plot(Rh_model, Rh_tng_max, 'bo', label = r'High $f_{\bigstar}$')

ax.set_xlabel(r'$R_{\rm{h, inf}}/r_{\rm{mx0}}$ model (kpc)')
ax.set_ylabel(r'$R_{\rm{h, max\, M_\bigstar}}/r_{\rm{mx0}}$ from TNG (kpc)')
# ax.set_xlim(0, 8)
# ax.set_ylim(0, 8)
x_vals = np.array(ax.get_xlim())    
ax.plot(x_vals, x_vals, 'k--', lw = 0.5)
ax.legend(fontsize = 8)
plt.loglog()
plt.tight_layout()
plt.show()




'''
Plot 7.2: This is to plot the rmx0 against the Mmx0 to check if there are some weird outliers
'''
fig, ax = plt.subplots(figsize = (7, 7))

ax.scatter(smstar_max_ar, srh_max_ar/srmx_if_ar, marker = 'o', color = 'darkgreen', s = 10, alpha = 0.2, label = 'surviving')
ax.scatter(mmstar_max_ar, mrh_max_ar/mrmx_if_ar, marker = 'o', color = 'purple', s = 10, alpha = 0.2, label = 'merged')

rh0_by_rmx0_ar = np.concatenate([np.array(srh_max_ar/srmx_if_ar), np.array(mrh_max_ar/mrmx_if_ar)])
mstar_ar = np.concatenate([np.array(smstar_max_ar), np.array(mmstar_max_ar)])
ixs = np.where(rh0_by_rmx0_ar<1/32)[0]

rh0_by_rmx0_ar = rh0_by_rmx0_ar[ixs]
mstar_ar = mstar_ar[ixs]

new_rh0_by_rmx0 = np.logspace(np.log10(np.quantile(rh0_by_rmx0_ar, 0.05)), np.log10(np.quantile(rh0_by_rmx0_ar, 0.95)), 3)

for ix, frac in enumerate(new_rh0_by_rmx0):
    if ix == 0: 
        ax.axhline(frac, ls = '--', c = 'blue', label = r'required $R_{h0}/r_{\rm{mx0}}$')
    else:
        ax.axhline(frac, ls = '--', c = 'blue')

ax.axhline(0.5, ls = '--', c = 'gray', label = r'Errani+22 $R_{h0}/r_{\rm{mx0}}$')
ax.axhline(0.25, ls = '--', c = 'gray')
ax.axhline(0.125, ls = '--', c = 'gray')
ax.axhline(0.0625, ls = '--', c = 'gray')
ax.set_xlabel(r'$M_\bigstar(M_\odot)$')
ax.set_ylabel(r'$R_{h0}/r_{\rm{mx0}}$')
ax.legend(fontsize = 8)
plt.loglog()
plt.tight_layout()
plt.show()



# =========================
# These are the comparison plots for surviving subhalos
# =========================

'''
One master plot to compare all six quantities predicted by the model
'''
fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (12, 6))
((ax1, ax2, ax3), (ax4, ax5, ax6)) = axs 
alpha = 0.8
msize = 2
# col_ar = stinf_ar
# col_label = r'$t_{\rm{inf}}$'
col_ar = np.log10(smstar_max_ar)
col_label = r'$\log M_{\rm{\bigstar, max}}$'

sc = ax1.scatter(smmx_f_ar, smmx_f_ar_tng, c=col_ar, cmap='viridis', alpha = alpha, s = msize, marker='o', zorder = 20)

cbar = plt.colorbar(sc, ax = ax1)
cbar.set_label(col_label)

dummy = np.linspace(ax1.get_xlim()[0], ax1.get_xlim()[1], 3) 
ax1.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)

ax1.set_xlabel(r'$M_{\rm{mx}}$ from model')
ax1.set_ylabel(r'$M_{\rm{mx}}$ from TNG')
ax1.set_xscale('log')
ax1.set_yscale('log')





# =======================

sc = ax4.scatter(smstar_f_ar, smstar_f_ar_tng, c=col_ar, cmap='viridis', alpha = alpha, s = msize, marker='o', zorder = 20)

cbar = plt.colorbar(sc, ax = ax4)
cbar.set_label(col_label)

dummy = np.linspace(ax4.get_xlim()[0], ax4.get_xlim()[1], 3) 
ax4.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)

ax4.set_xlabel(r'$M_{\rm{\bigstar}}$ from model')
ax4.set_ylabel(r'$M_{\rm{\bigstar}}$ from TNG')
ax4.set_xscale('log')
ax4.set_yscale('log')

ax4.set_xlim(1e2, 1e10)
ax4.set_ylim(1e2, 1e10)


# =======================

sc = ax2.scatter(srmx_f_ar, srmx_f_ar_tng, c=col_ar, cmap='viridis', alpha = alpha, s = msize, marker='o', zorder = 20)

cbar = plt.colorbar(sc, ax = ax2)
cbar.set_label(col_label)

dummy = np.linspace(ax2.get_xlim()[0], ax2.get_xlim()[1], 3) 
ax2.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)

ax2.set_xlabel(r'$r_{\rm{mx}}$ from model')
ax2.set_ylabel(r'$r_{\rm{mx}}$ from TNG')
ax2.set_xscale('log')
ax2.set_yscale('log')

# ==========================


sc = ax3.scatter(svmx_f_ar, svmx_f_ar_tng, c=col_ar, cmap='viridis', alpha = alpha, s = msize, marker='o', zorder = 20)

cbar = plt.colorbar(sc, ax = ax3)
cbar.set_label(col_label)

dummy = np.linspace(ax3.get_xlim()[0], ax3.get_xlim()[1], 3) 
ax3.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)

ax3.set_xlabel(r'$v_{\rm{mx}}$ from model')
ax3.set_ylabel(r'$v_{\rm{mx}}$ from TNG')
ax3.set_xscale('log')
ax3.set_yscale('log')



# ============================

sc = ax5.scatter(srh_f_ar, srh_f_ar_tng, c=col_ar, cmap='viridis', alpha = alpha, s = msize, marker='o', zorder = 20)

cbar = plt.colorbar(sc, ax = ax5)
cbar.set_label(col_label)

dummy = np.linspace(ax5.get_xlim()[0], ax5.get_xlim()[1], 3) 
ax5.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)

ax5.set_xlabel(r'$R_{\rm{h}}$ from model')
ax5.set_ylabel(r'$R_{\rm{h}}$ from TNG')
# ax5.set_xscale('log')
# ax5.set_yscale('log')


# ===========================

sc = ax6.scatter(svd_f_ar, svd_f_ar_tng, c=col_ar, cmap='viridis', alpha = alpha, s = msize, marker='o', zorder = 20)

cbar = plt.colorbar(sc, ax = ax6)
cbar.set_label(col_label)

dummy = np.linspace(ax6.get_xlim()[0], ax6.get_xlim()[1], 3) 

ax6.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)

ax6.set_xlabel(r'$\sigma$ from model')
ax6.set_ylabel(r'$\sigma$ from TNG')
# ax6.set_xscale('log')
# ax6.set_yscale('log')



# plt.loglog()
plt.tight_layout()
plt.savefig(outpath+'surviving_fof0_comparison.pdf')
plt.show()



'''
Same as above but now fractions for a more relevant comparison
'''
fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (12, 6))
((ax1, ax2, ax3), (ax4, ax5, ax6)) = axs 
alpha = 0.8
msize = 2
# col_ar = stinf_ar
# col_label = r'$t_{\rm{inf}}$'
col_ar = np.log10(smstar_max_ar)
col_label = r'$\log M_{\rm{\bigstar, max}}$'

sc = ax1.scatter(smmx_f_ar/smmx_if_ar, smmx_f_ar_tng/smmx_if_ar, c=col_ar, cmap='viridis', alpha = alpha, s = msize, marker='o', zorder = 20)

cbar = plt.colorbar(sc, ax = ax1)
cbar.set_label(col_label)

dummy = np.linspace(ax1.get_xlim()[0], ax1.get_xlim()[1], 3) 
ax1.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)

ax1.set_xlabel(r'$M_{\rm{mx}}/M_{\rm{mx0}}$ from model')
ax1.set_ylabel(r'$M_{\rm{mx}}/M_{\rm{mx0}}$ from TNG')
ax1.set_xscale('log')
ax1.set_yscale('log')
min_axes = min(ax1.get_xlim()[0], ax1.get_ylim()[0])
ax1.set_xlim(left = min_axes, right = 1)
ax1.set_ylim(bottom = min_axes, top = 1)





# =======================

sc = ax4.scatter(smstar_f_ar/smstar_max_ar, smstar_f_ar_tng/smstar_max_ar, c=col_ar, cmap='viridis', alpha = alpha, s = msize, marker='o', zorder = 20)

cbar = plt.colorbar(sc, ax = ax4)
cbar.set_label(col_label)

dummy = np.linspace(ax4.get_xlim()[0], ax4.get_xlim()[1], 3) 
ax4.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)

ax4.set_xlabel(r'$M_{\rm{\bigstar}}/M_{\rm{\bigstar 0}}$ from model')
ax4.set_ylabel(r'$M_{\rm{\bigstar}}/M_{\rm{\bigstar 0}}$ from TNG')
ax4.set_xscale('log')
ax4.set_yscale('log')
min_axes = min(ax4.get_xlim()[0], ax4.get_ylim()[0])
ax4.set_xlim(left = min_axes, right = 1)
ax4.set_ylim(bottom = min_axes, top = 1)
# ax4.set_xlim(1e2, 1e10)
# ax4.set_ylim(1e2, 1e10)


# =======================

sc = ax2.scatter(srmx_f_ar/srmx_if_ar, srmx_f_ar_tng/srmx_if_ar, c=col_ar, cmap='viridis', alpha = alpha, s = msize, marker='o', zorder = 20)

cbar = plt.colorbar(sc, ax = ax2)
cbar.set_label(col_label)

dummy = np.linspace(ax2.get_xlim()[0], ax2.get_xlim()[1], 3) 
ax2.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)

ax2.set_xlabel(r'$r_{\rm{mx}}/r_{\rm{mx0}}$ from model')
ax2.set_ylabel(r'$r_{\rm{mx}}/r_{\rm{mx0}}$ from TNG')
ax2.set_xscale('log')
ax2.set_yscale('log')
min_axes = min(ax2.get_xlim()[0], ax2.get_ylim()[0])
ax2.set_xlim(left = min_axes, right = 1)
ax2.set_ylim(bottom = min_axes, top = 1)

# ==========================


sc = ax3.scatter(svmx_f_ar/svmx_if_ar, svmx_f_ar_tng/svmx_if_ar, c=col_ar, cmap='viridis', alpha = alpha, s = msize, marker='o', zorder = 20)

cbar = plt.colorbar(sc, ax = ax3)
cbar.set_label(col_label)

dummy = np.linspace(ax3.get_xlim()[0], ax3.get_xlim()[1], 3) 
ax3.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)

ax3.set_xlim(left = 2e-2)
ax3.set_ylim(bottom = 1e-1)
ax3.set_xlabel(r'$v_{\rm{mx}}/v_{\rm{mx0}}$ from model')
ax3.set_ylabel(r'$v_{\rm{mx}}/v_{\rm{mx0}}$ from TNG')
# ax3.set_xscale('log')
# ax3.set_yscale('log')
min_axes = min(ax3.get_xlim()[0], ax3.get_ylim()[0])
ax3.set_xlim(left = min_axes, right = 1)
ax3.set_ylim(bottom = min_axes, top = 1)


# ============================

sc = ax5.scatter(srh_f_ar/srh_max_ar, srh_f_ar_tng/srh_max_ar, c=col_ar, cmap='viridis', alpha = alpha, s = msize, marker='o', zorder = 20)

cbar = plt.colorbar(sc, ax = ax5)
cbar.set_label(col_label)

dummy = np.linspace(ax5.get_xlim()[0], ax5.get_xlim()[1], 3) 
ax5.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)

ax5.set_xlabel(r'$R_{\rm{h}}/R_{\rm{h0}}$ from model')
ax5.set_ylabel(r'$R_{\rm{h}}/R_{\rm{h0}}$ from TNG')
# ax5.set_xscale('log')
# ax5.set_yscale('log')
min_axes = min(ax5.get_xlim()[0], ax5.get_ylim()[0])
ax5.set_xlim(left = min_axes)
ax5.set_ylim(bottom = min_axes)


# ===========================

sc = ax6.scatter(svd_f_ar/svd_max_ar, svd_f_ar_tng/svd_max_ar, c=col_ar, cmap='viridis', alpha = alpha, s = msize, marker='o', zorder = 20)

cbar = plt.colorbar(sc, ax = ax6)
cbar.set_label(col_label)

dummy = np.linspace(ax6.get_xlim()[0], ax6.get_xlim()[1], 3) 

ax6.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)

ax6.set_xlabel(r'$\sigma/\sigma_0$ from model')
ax6.set_ylabel(r'$\sigma/\sigma_0$ from TNG')
min_axes = min(ax6.get_xlim()[0], ax6.get_ylim()[0])
ax6.set_xlim(left = min_axes, right = 1)
ax6.set_ylim(bottom = min_axes, top = 1)
# ax6.set_xscale('log')
# ax6.set_yscale('log')



# plt.loglog()
plt.tight_layout()
plt.savefig(outpath+'surviving_fof0_comparison.pdf')
plt.show()

