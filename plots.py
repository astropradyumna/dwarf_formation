import pandas as pd 
import matplotlib.pyplot as plt
import IPython
import numpy as np 
from dwarf_plotting import plot_lg_virgo



outpath  = '/home/psadh003/tng50/output_files/'
rvir_fof0 = 811.66/0.6744

dfs = pd.read_csv(outpath + 'surviving_evolved_fof0.csv', delimiter = ',')
dfs = dfs[dfs['dist_f_ar']<rvir_fof0]

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




dfm = pd.read_csv(outpath + 'merged_evolved_fof0.csv', delimiter = ',')
dfm = dfm[dfm['dist_f_ar']<rvir_fof0]

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
ax.scatter(mmmx_f_ar, mmstar_f_ar, alpha = 0.3, color = 'purple', s = 2, label = 'Merged')
ax.scatter(smmx_f_ar, smstar_f_ar, alpha = 0.3, s = 2, color = 'darkgreen', label = 'Surviving')
ax.set_xlim(left = 1e4)
ax.set_ylim(bottom  = 1e2)

mmxpl = np.logspace(2, 12, 100)
ax.plot(mmxpl, get_moster_shm(mmxpl), 'k--', label = 'Moster+13')
ax.set_xlabel(r'$M_{\rm{mx}}\,\rm{(M_\odot)}$ ')
ax.set_ylabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
ax.set_title('z=0 from the model')
ax.legend(fontsize = 8)
plt.loglog()
plt.show()


'''
Plot 2: Subhalo mass functtion for the subhalos of the given file
'''

mstarpl = np.logspace(2, 11, 100)
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
ax.set_xlim(left = 1e2)
ax.set_title('z=0 from the model')


plt.show()



'''
Plot 3: Distance from the center vs M*
'''
fig, ax = plt.subplots(figsize = (5.5, 5))

ax.scatter(mmstar_f_ar, mdist_f_ar, color = 'purple', alpha = 0.5, s = 2, label = 'Merged')
ax.scatter(smstar_f_ar, sdist_f_ar, color = 'darkgreen', alpha = 0.5, s = 2, label = 'Surviving')
ax.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
ax.set_ylabel('Distance from center (kpc)')
ax.set_xlim(left = 1e2)
ax.set_ylim(bottom = 1e1)

plt.loglog()
plt.tight_layout()
plt.show()



'''
Plot 4: N(<r)
'''
rpl = np.logspace(1, 3.2, 100)
Nm_ar = np.zeros(0) #shmf for merged subhalos
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

for (ix, ms) in enumerate(rpl):
    Nm_ar = np.append(Nm_ar, len(mmstar_f_ar[mdist_f_ar < ms]))
    Ns_ar = np.append(Ns_ar, len(smstar_f_ar[sdist_f_ar < ms]))
    # Ntng_ar = np.append()

Ndm_ar = Ndm_ar * (Nm_ar[-1] + Ns_ar[-1])
fig, ax = plt.subplots(figsize = (5.5, 5))
ax.plot(rpl, Ns_ar, color = 'darkgreen', label = 'Surviving')
ax.plot(rpl, Ns_ar + Nm_ar, color = 'purple', label = 'Model (all)')
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
plot_lg_virgo(ax)
ax.scatter(mmstar_f_ar, mrh_f_ar * 1e3, marker = 's', color = 'purple', alpha = 0.7, s = 25, label = 'Merged', zorder = 200)
ax.scatter(smstar_f_ar, srh_f_ar * 1e3, marker = 's', color = 'darkgreen', alpha = 0.7, s = 25, label = 'Survived', zorder = 200)
ax.set_xlim(left = 1e2)
ax.set_ylim(bottom = 10)
ax.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
ax.set_ylabel(r'$r_{\rm{h}}$ (pc)')
ax.legend(fontsize = 8)
plt.loglog()
plt.tight_layout()
plt.show()




