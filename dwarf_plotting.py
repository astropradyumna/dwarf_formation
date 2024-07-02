'''
This is code to plot the dwarfs in Virgo and LG
- can plot velocity dispersion, Rh and stellar mass
'''

import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 

filepath = '/rhome/psadh003/bigdata/dwarf_data/'

'''
This cell is for data import of Kaixiang data
'''
ucd_df = pd.read_csv(filepath + 'Virgo_UCDs.txt', header=0, delimiter=' ')
ucd_re = ucd_df['Re(pc)']
ucd_mg = ucd_df['Mag_g']

eucd_df = pd.read_csv(filepath + 'Virgo_eUCDs.txt', header=0, delimiter=' ')
eucd_re = eucd_df['Re(pc)']
eucd_mg = eucd_df['Mag_g']

gc_df = pd.read_csv(filepath + 'Virgo_GCs.txt', header=0, delimiter=' ')
gc_re = gc_df['Re(pc)']
gc_mg = gc_df['Mag_g']

nsc_df = pd.read_csv(filepath + 'Virgo_NSCs.txt', header=0, delimiter=' ')
nsc_re = nsc_df['Re(pc)']
nsc_mg = nsc_df['Mag_g']

den_df = pd.read_csv(filepath + 'Virgo_nucleated_dEs.txt', header=0, delimiter=' ')
den_re = den_df['Re(pc)']
den_mg = den_df['Mag_g']

sden_df = pd.read_csv(filepath + 'Virgo_strongly-nucleated_dEs.txt', header=0, delimiter=' ')
sden_re = sden_df['Re(pc)']
sden_mg = sden_df['Mag_g']

udg_df = pd.read_csv(filepath + 'Virgo_UDGs.txt', header=0, delimiter=' ')
udg_re = udg_df['Re(pc)']
udg_mg = udg_df['Mag_g']


'''
This cell is to import the LG data sent by Laura for LG dwarfs
'''
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

'''
These are LG GCs
'''
filein=filepath + 'Laura_list_from_RaphaErrani.dat'
data=np.loadtxt(filein,usecols=[2,4])
rh_gc = data[:,0]
# rh_gc /= 1000.
mstr_gc = data[:,1]  #this is actually luminosity, assuming M/L=1 is mass

# mstr_gc
# plt.loglog()

def get_mag_from_ms(x):
    '''
    This function is to obtain the magnitude from stellar mass
    '''
    return (-2.278 * x + 3.524)

def get_ms_from_mag(x):
    '''
    This function is to obtain the stellar mass from magnitude in log Msun
    '''
    return  10**(-0.439 * x + 1.549)


'''
This cell is to generate the plots
'''
# plt.figure(figsize  = (11, 6))
# fig, ax = plt.subplots(figsize = (11, 7))

def plot_lg_virgo(ax, alpha = 0.8, mec = None):
    '''
    this function plots all the Virgo and LG data
    Use this to avoid clutter in the main code
    '''
    ax.plot(get_ms_from_mag(den_mg), den_re, marker = 'o', color = 'silver', label = 'Nucleated galaxies', 
            lw = 0, ms = 4, alpha = alpha, mec = mec)
    ax.plot(get_ms_from_mag(sden_mg), sden_re, marker = 'D', color = 'orangered',  label = 'Strongly nucleated dE,Ns', 
            lw = 0, ms = 5, alpha = alpha, mec = mec)
    ax.plot(get_ms_from_mag(gc_mg), gc_re, marker = 'o', color = 'darkgray', label = 'GCs', 
            lw = 0, ms = 1.5, alpha = alpha)
    ax.plot(get_ms_from_mag(nsc_mg), nsc_re, marker = 'o', color = 'maroon',  label = 'NSCs', 
            lw = 0, ms = 4, alpha = alpha, mec = mec)
    ax.plot(get_ms_from_mag(eucd_mg), eucd_re, marker = 'D', color = 'deeppink',  label = 'UCD,Es', 
            lw = 0, ms = 5, alpha = alpha, mec = mec)
    ax.plot(get_ms_from_mag(ucd_mg), ucd_re, marker = 'o', color = 'orange', label = 'UCDs', 
            lw = 0, ms = 5, alpha = alpha, mec = mec)
    ax.plot(get_ms_from_mag(udg_mg), udg_re, marker = '^', color = 'skyblue',  label = 'UDGs', 
            lw = 0, ms = 8, alpha = alpha, mec = mec)
    '''
    Following is LG data
    '''
    ax.plot(mstr_gc,rh_gc,marker = '.',markersize=3.5,alpha=alpha,color='gray', lw = 0, label = 'LG GCs')
    # ax.errorbar(get_mag_from_ms(np.log10(mstr)), rh*1e3, yerr=err_rh, xerr=err_mstr,marker='o',ms=8, mfc='white', mec = 'black', mew=1,ecolor='darkgray',ls='none',zorder=-32, alpha = 0.8, label = 'LG')ax.errorbar(get_mag_from_ms(np.log10(mstr)), rh*1e3, yerr=err_rh, xerr=err_mstr,marker='o',ms=8, mfc='white', mec = 'black', mew=1,ecolor='darkgray',ls='none',zorder=-32, alpha = 0.8, label = 'LG')
    ax.plot(mstr, rh*1e3,marker='o',ms=4, mfc='white', mec = 'black',ls='none', alpha = alpha, label = 'LG')



def plot_lg_virgo_some(ax, zorder = 300, alpha = 0.1, mec = 'black'):
    '''
    this function plots all the Virgo and LG data
    We will only be using the some of the data in the main paper to avoid clutter
    '''
    ax.plot(np.log10(get_ms_from_mag(den_mg)), np.log10(den_re), marker = 'D', mfc='white', mec = mec, label = 'Nucleated galaxies', 
            lw = 0, ms = 4, alpha = alpha, zorder = zorder)
#     ax.plot(get_ms_from_mag(sden_mg), sden_re, marker = 'D', color = 'orangered',  label = 'Strongly nucleated dE,Ns', 
#             lw = 0, ms = 5, alpha = alpha, zorder = zorder)
#     ax.plot(get_ms_from_mag(gc_mg), gc_re, marker = 'o', color = 'darkgray', label = 'GCs', 
#             lw = 0, ms = 1.5, alpha = alpha, zorder = zorder)
#     ax.plot(get_ms_from_mag(nsc_mg), nsc_re, marker = 'o', color = 'maroon',  label = 'NSCs', 
#             lw = 0, ms = 4, alpha = alpha, zorder = zorder)
#     ax.plot(get_ms_from_mag(eucd_mg), eucd_re, marker = 'D', color = 'deeppink',  label = 'UCD,Es', 
#             lw = 0, ms = 5, alpha = alpha, zorder = zorder)
#     ax.plot(get_ms_from_mag(ucd_mg), ucd_re, marker = 'o', color = 'orange', label = 'UCDs', 
#             lw = 0, ms = 5, alpha = alpha, zorder = zorder)
#     ax.plot(get_ms_from_mag(udg_mg), udg_re, marker = '^', color = 'skyblue',  label = 'UDGs', 
#             lw = 0, ms = 8, alpha = alpha, zorder = zorder)
    '''
    Following is LG data
    '''
#     ax.plot(mstr_gc,rh_gc,marker = '.',markersize=3.5,alpha=alpha,color='gray', lw = 0, label = 'LG GCs', zorder = zorder)
    # ax.errorbar(get_mag_from_ms(np.log10(mstr)), rh*1e3, yerr=err_rh, xerr=err_mstr,marker='o',ms=8, mfc='white', mec = 'black', mew=1,ecolor='darkgray',ls='none',zorder=-32, alpha = 0.8, label = 'LG')ax.errorbar(get_mag_from_ms(np.log10(mstr)), rh*1e3, yerr=err_rh, xerr=err_mstr,marker='o',ms=8, mfc='white', mec = 'black', mew=1,ecolor='darkgray',ls='none',zorder=-32, alpha = 0.8, label = 'LG')
    ax.plot(np.log10(mstr), np.log10(rh*1e3),marker='o',ms=4, mfc='white', mec = mec,ls='none', alpha = alpha, label = 'LG', zorder = zorder)


# The following lines involve data import from LG data for velocity dispersion

filein = filepath + 'LGdata_withErrors.dat'  
data = np.loadtxt(filein,usecols=[1,3])

#pdb.set_trace()
mstr1 = data[:,0]
vd = data[:,1] #This is the velocity dispersion
non_nan_indices = np.where(~np.isnan(vd))[0]

mstr_cut = mstr1[non_nan_indices]
vd_cut = vd[non_nan_indices]

def plot_lg_vd(ax):
    ax.plot(mstr_cut, vd_cut,marker='o',ms=4, mfc='white', mec = 'black',ls='none', alpha = alpha, label = 'LG')
    
