import pandas as pd 
import matplotlib.pyplot as plt
import IPython
import numpy as np 



outpath  = '/home/psadh003/tng50/output_files/'

dfs = pd.read_csv(outpath + 'surviving_evolved_fof0.csv', delimiter = ',')
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
fig, ax = plt.subplots()
ax.scatter(mmmx_f_ar, mmstar_f_ar, alpha = 0.3, color = 'purple', s = 2)
ax.scatter(smmx_f_ar, smstar_f_ar, alpha = 0.3, s = 2, color = 'darkgreen')
ax.set_xlim(left = 1e2)
ax.set_ylim(bottom  = 1e2)
plt.loglog()
plt.show()

'''
Plot 2: Subhalo mass functtion for the subhalos of the given file
'''



