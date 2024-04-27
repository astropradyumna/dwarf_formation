import pandas as pd 
import matplotlib.pyplot as plt
import IPython
import numpy as np 
from dwarf_plotting import plot_lg_virgo, plot_lg_vd
import matplotlib
import illustris_python as il
from populating_stars import *
import sys
import ast
import os
from plot_sachi import plot_sachi
from scipy.optimize import fsolve

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size' : 14}

plt.figure()
plt.close()
matplotlib.rc('font', **font)


def convert_to_float(value):
    try:
        blah = ast.literal_eval(value)
        if isinstance(blah, list):
            blah2 = float(blah[0])     
        else:
            blah2 = float(blah)        
        return blah2
    except (ValueError, SyntaxError):
        return value 



basePath = '/rhome/psadh003/bigdata/L35n2160TNG_fixed/output'
outpath  = '/rhome/psadh003/bigdata/tng50/output_files/'
plotpath = '/bigdata/saleslab/psadh003/tng50/output_plots/'

porc = str(sys.argv[2]) #This would tell if we have to make plots for cutoff case or the powerlaw case
if porc != 'p' and porc != 'c':
    raise ValueError('Check value for porc!')

fof_no = int(sys.argv[1])
fof_str = 'fof' + str(fof_no)

this_fof = il.groupcat.loadSingle(basePath, 99, haloID = fof_no)
central_sfid_99 = this_fof['GroupFirstSub']
rvir_fof = this_fof['Group_R_Crit200']/0.6744

this_fof_plotppath = '/bigdata/saleslab/psadh003/tng50/output_plots/' + fof_str + '/'
if not os.path.exists(this_fof_plotppath): #If the directory does not exist, then just create it!
    os.makedirs(this_fof_plotppath)


dfs = pd.read_csv(outpath + fof_str + '_surviving_evolved_everything.csv', delimiter = ',', low_memory=False)
dfs = dfs.applymap(convert_to_float)


dfs = dfs[dfs['dist_f_ar']<rvir_fof]

dfs1 = dfs
if porc == 'p':
    # print(len(dfs[dfs['mstar_f_ar_tng'].values>5e6]))
    columns_to_max = ['mstar_max_ar', 'mstar_max_pl_ar']
    max_values = dfs[columns_to_max].max(axis=1)
    dfs = dfs[(max_values > 1e3)]

    columns_to_max = ['mstar_f_ar_tng', 'mstar_f_pl_ar']
    max_values = dfs[columns_to_max].max(axis=1)
    cutoff = 10

    dfs = dfs[(max_values > cutoff)]
    

    sdist_f_ar = dfs['dist_f_ar'].values
    spos_f_ar = dfs['pos_f_ar'].values

    svd_f_ar_tng = dfs['vd_f_ar_tng'].values
    srh_f_ar_tng = dfs['rh_f_ar_tng'].values
    smstar_f_ar_tng = dfs['mstar_f_ar_tng'].values
    smmx_f_ar_tng = dfs['mmx_f_ar_tng'].values
    srmx_f_ar_tng = dfs['rmx_f_ar_tng'].values
    svmx_f_ar_tng = dfs['vmx_f_ar_tng'].values

    stinf_ar = dfs['tinf_ar'].values
    storb_ar = dfs['torb_ar'].values
    srapo_ar = dfs['rapo_ar'].values
    srperi_ar = dfs['rperi_ar'].values

    svd_f_ar = dfs['vd_f_ar'].values
    srh_f_ar = dfs['rh_f_ar'].values
    smstar_f_ar = dfs['mstar_f_ar'].values

    svd_max_ar = dfs['vd_max_ar'].values
    srh_max_ar = dfs['rh_max_ar'].values
    smstar_max_ar = dfs['mstar_max_ar'].values

    ssnap_if_ar = dfs['snap_if_ar'].values
    ssf_id_if_ar = dfs['sfid_if_ar'].values

    smmx_f_ar = dfs['mmx_f_ar'].values
    srmx_f_ar = dfs['rmx_f_ar'].values
    svmx_f_ar = dfs['vmx_f_ar'].values

    smmx_if_ar = dfs['mmx_if_ar'].values
    srmx_if_ar = dfs['rmx_if_ar'].values
    svmx_if_ar = dfs['vmx_if_ar'].values

    svd_f_pl_ar = dfs['vd_f_pl_ar'].values
    srh_f_pl_ar = dfs['rh_f_pl_ar'].values
    smstar_f_pl_ar = dfs['mstar_f_pl_ar'].values
    svd_f_co_ar = dfs['vd_f_co_ar'].values
    srh_f_co_ar = dfs['rh_f_co_ar'].values
    smstar_f_co_ar = dfs['mstar_f_co_ar'].values
    srh_max_pl_ar = dfs['rh_max_pl_ar'].values
    srh_max_co_ar = dfs['rh_max_co_ar'].values
    smstar_max_pl_ar = dfs['mstar_max_pl_ar'].values
    smstar_max_co_ar = dfs['mstar_max_co_ar'].values
elif porc == 'c':
    columns_to_max = ['mstar_max_ar', 'mstar_max_co_ar']
    max_values = dfs[columns_to_max].max(axis=1)
    dfs = dfs[(max_values > 1e3)]

    columns_to_max = ['mstar_f_ar_tng', 'mstar_f_co_ar']

    # Compute the maximum along the specified axis (axis=1 for row-wise)
    max_values = dfs[columns_to_max].max(axis=1)

    cutoff = 10
    # print(max_values > cutoff)

    dfs = dfs[(max_values > cutoff)]

    sdist_f_ar = dfs['dist_f_ar'].values
    spos_f_ar = dfs['pos_f_ar'].values

    svd_f_ar_tng = dfs['vd_f_ar_tng'].values
    srh_f_ar_tng = dfs['rh_f_ar_tng'].values
    smstar_f_ar_tng = dfs['mstar_f_ar_tng'].values
    smmx_f_ar_tng = dfs['mmx_f_ar_tng'].values
    srmx_f_ar_tng = dfs['rmx_f_ar_tng'].values
    svmx_f_ar_tng = dfs['vmx_f_ar_tng'].values

    stinf_ar = dfs['tinf_ar'].values
    storb_ar = dfs['torb_ar'].values
    srapo_ar = dfs['rapo_ar'].values
    srperi_ar = dfs['rperi_ar'].values

    svd_f_ar = dfs['vd_f_ar'].values
    srh_f_ar = dfs['rh_f_ar'].values
    smstar_f_ar = dfs['mstar_f_ar'].values

    svd_max_ar = dfs['vd_max_ar'].values
    srh_max_ar = dfs['rh_max_ar'].values
    smstar_max_ar = dfs['mstar_max_ar'].values

    ssnap_if_ar = dfs['snap_if_ar'].values
    ssf_id_if_ar = dfs['sfid_if_ar'].values

    smmx_f_ar = dfs['mmx_f_ar'].values
    srmx_f_ar = dfs['rmx_f_ar'].values
    svmx_f_ar = dfs['vmx_f_ar'].values

    smmx_if_ar = dfs['mmx_if_ar'].values
    srmx_if_ar = dfs['rmx_if_ar'].values
    svmx_if_ar = dfs['vmx_if_ar'].values

    svd_f_pl_ar = dfs['vd_f_pl_ar'].values
    srh_f_pl_ar = dfs['rh_f_pl_ar'].values
    smstar_f_pl_ar = dfs['mstar_f_pl_ar'].values
    svd_f_co_ar = dfs['vd_f_co_ar'].values
    srh_f_co_ar = dfs['rh_f_co_ar'].values
    smstar_f_co_ar = dfs['mstar_f_co_ar'].values
    srh_max_pl_ar = dfs['rh_max_pl_ar'].values
    srh_max_co_ar = dfs['rh_max_co_ar'].values
    smstar_max_pl_ar = dfs['mstar_max_pl_ar'].values
    smstar_max_co_ar = dfs['mstar_max_co_ar'].values


dfm = pd.read_csv(outpath + fof_str + '_merged_evolved_wmbp_everything.csv', delimiter = ',')
dfm = dfm.applymap(convert_to_float)
dfm = dfm[dfm['dist_f_ar']<rvir_fof]
dfm1 = dfm 

if porc == 'p':
    columns_to_max = ['mstar_max_ar', 'mstar_max_pl_ar']
    max_values = dfm[columns_to_max].max(axis=1)
    dfm = dfm[(max_values > 1e3)]

    columns_to_max = ['mstar_f_ar', 'mstar_f_pl_ar']

    # Compute the maximum along the specified axis (axis=1 for row-wise)
    max_values = dfm[columns_to_max].max(axis=1)

    # Sample cutoff value
    cutoff = 10

    # print(len(dfm))
    dfm = dfm[max_values > cutoff]
    # print(len(dfm))
    mdist_f_ar = dfm['dist_f_ar'].values
    mmbpid_ar = dfm['mbpid_ar'].values

    mtinf_ar = dfm['tinf_ar'].values
    mtorb_ar = dfm['torb_ar'].values
    mrapo_ar = dfm['rapo_ar'].values
    mrperi_ar = dfm['rperi_ar'].values

    mvd_f_ar = dfm['vd_f_ar'].values
    mrh_f_ar = dfm['rh_f_ar'].values
    mmstar_f_ar = dfm['mstar_f_ar'].values

    mvd_max_ar = dfm['vd_max_ar'].values
    mrh_max_ar = dfm['rh_max_ar'].values
    mmstar_max_ar = dfm['mstar_max_ar'].values

    msnap_if_ar = dfm['snap_if_ar'].values
    msfid_if_ar = dfm['sfid_if_ar'].values

    mmmx_f_ar = dfm['mmx_f_ar'].values
    mrmx_f_ar = dfm['rmx_f_ar'].values
    mvmx_f_ar = dfm['vmx_f_ar'].values

    mmmx_if_ar = dfm['mmx_if_ar'].values
    mrmx_if_ar = dfm['rmx_if_ar'].values
    mvmx_if_ar = dfm['vmx_if_ar'].values

    mvd_f_pl_ar = dfm['vd_f_pl_ar'].values
    mrh_f_pl_ar = dfm['rh_f_pl_ar'].values
    mmstar_f_pl_ar = dfm['mstar_f_pl_ar'].values
    mvd_f_co_ar = dfm['vd_f_co_ar'].values
    mrh_f_co_ar = dfm['rh_f_co_ar'].values
    mmstar_f_co_ar = dfm['mstar_f_co_ar'].values
    mrh_max_pl_ar = dfm['rh_max_pl_ar'].values
    mrh_max_co_ar = dfm['rh_max_co_ar'].values
    mmstar_max_pl_ar = dfm['mstar_max_pl_ar'].values
    mmstar_max_co_ar = dfm['mstar_max_co_ar'].values
elif porc == 'c':
    columns_to_max = ['mstar_max_ar', 'mstar_max_co_ar']
    max_values = dfm[columns_to_max].max(axis=1)
    dfm = dfm[(max_values > 1e3)]


    columns_to_max = ['mstar_f_ar', 'mstar_f_co_ar']

    # Compute the maximum along the specified axis (axis=1 for row-wise)
    max_values = dfm[columns_to_max].max(axis=1)

    # Sample cutoff value
    cutoff = 10

    # print(len(dfm))
    dfm = dfm[max_values > cutoff]
    # print(len(dfm))
    mdist_f_ar = dfm['dist_f_ar'].values
    mmbpid_ar = dfm['mbpid_ar'].values

    mtinf_ar = dfm['tinf_ar'].values
    mtorb_ar = dfm['torb_ar'].values
    mrapo_ar = dfm['rapo_ar'].values
    mrperi_ar = dfm['rperi_ar'].values

    mvd_f_ar = dfm['vd_f_ar'].values
    mrh_f_ar = dfm['rh_f_ar'].values
    mmstar_f_ar = dfm['mstar_f_ar'].values

    mvd_max_ar = dfm['vd_max_ar'].values
    mrh_max_ar = dfm['rh_max_ar'].values
    mmstar_max_ar = dfm['mstar_max_ar'].values

    msnap_if_ar = dfm['snap_if_ar'].values
    msfid_if_ar = dfm['sfid_if_ar'].values

    mmmx_f_ar = dfm['mmx_f_ar'].values
    mrmx_f_ar = dfm['rmx_f_ar'].values
    mvmx_f_ar = dfm['vmx_f_ar'].values

    mmmx_if_ar = dfm['mmx_if_ar'].values
    mrmx_if_ar = dfm['rmx_if_ar'].values
    mvmx_if_ar = dfm['vmx_if_ar'].values

    mvd_f_pl_ar = dfm['vd_f_pl_ar'].values
    mrh_f_pl_ar = dfm['rh_f_pl_ar'].values
    mmstar_f_pl_ar = dfm['mstar_f_pl_ar'].values
    mvd_f_co_ar = dfm['vd_f_co_ar'].values
    mrh_f_co_ar = dfm['rh_f_co_ar'].values
    mmstar_f_co_ar = dfm['mstar_f_co_ar'].values
    mrh_max_pl_ar = dfm['rh_max_pl_ar'].values
    mrh_max_co_ar = dfm['rh_max_co_ar'].values
    mmstar_max_pl_ar = dfm['mstar_max_pl_ar'].values
    mmstar_max_co_ar = dfm['mstar_max_co_ar'].values




#Please place all you cuts above this line


def get_smstar(porc, iorf):
    '''
    This function is to return the stellar mass at either the infall or the final time
    Also acccounts for the power law or the cutoff way of obtaining the stellar mass
    '''
    smstar_all = np.zeros(len(smmx_f_ar)) #This is the array where we will be combining everything
    res_ixs = np.where(smstar_max_ar >= 5e6)[0] #These are the resolved indices
    unres_ixs = np.where(smstar_max_ar < 5e6)[0] #These are the unresolved indices
    if iorf == 'i':
        smstar_all[res_ixs] = smstar_max_ar[res_ixs]
        if porc == 'p':
            smstar_all[unres_ixs] = smstar_max_pl_ar[unres_ixs]
        elif porc == 'c':
            smstar_all[unres_ixs] = smstar_max_co_ar[unres_ixs]
    elif iorf == 'f':
        smstar_all[res_ixs] = smstar_f_ar[res_ixs]
        if porc == 'p':
            smstar_all[unres_ixs] = smstar_f_pl_ar[unres_ixs]
        elif porc == 'c':
            smstar_all[unres_ixs] = smstar_f_co_ar[unres_ixs]
    return smstar_all


def get_srh(porc, iorf):
    '''
    Same as above, but now for the half light radius
    '''
    rh_all = np.zeros(len(smmx_f_ar)) #This is the array where we will be combining everything
    res_ixs = np.where(smstar_max_ar >= 5e6)[0] #These are the resolved indices
    unres_ixs = np.where(smstar_max_ar < 5e6)[0] #These are the unresolved indices
    if iorf == 'i':
        rh_all[res_ixs] = srh_max_ar[res_ixs]
        if porc == 'p':
            rh_all[unres_ixs] = srh_max_pl_ar[unres_ixs]
        elif porc == 'c':
            rh_all[unres_ixs] = srh_max_co_ar[unres_ixs]
    elif iorf == 'f':
        rh_all[res_ixs] = srh_f_ar[res_ixs]
        if porc == 'p':
            rh_all[unres_ixs] = srh_f_pl_ar[unres_ixs]
        elif porc == 'c':
            rh_all[unres_ixs] = srh_f_co_ar[unres_ixs]
    return rh_all
     

def get_svd(porc, iorf):
    '''
    Same as above, but now for the half light radius
    '''
    vd_all = np.zeros(len(smmx_f_ar)) #This is the array where we will be combining everything
    res_ixs = np.where(smstar_max_ar >= 5e6)[0] #These are the resolved indices
    unres_ixs = np.where(smstar_max_ar < 5e6)[0] #These are the unresolved indices
    if iorf == 'i':
        vd_all[res_ixs] = svd_max_ar[res_ixs]
        if porc == 'p':
            vd_all[unres_ixs] = svd_max_ar[unres_ixs]
        elif porc == 'c':
            vd_all[unres_ixs] = svd_max_ar[unres_ixs]
    elif iorf == 'f':
        vd_all[res_ixs] = svd_f_ar[res_ixs]
        if porc == 'p':
            vd_all[unres_ixs] = svd_f_pl_ar[unres_ixs]
        elif porc == 'c':
            vd_all[unres_ixs] = svd_f_co_ar[unres_ixs]
    return vd_all




def get_mmstar(porc, iorf):
    '''
    This function is to return the stellar mass at either the infall or the final time
    Also acccounts for the power law or the cutoff way of obtaining the stellar mass

    For merged subhalos
    '''
    mmstar_all = np.zeros(len(mmmx_f_ar)) #This is the array where we will be combining everything
    res_ixs = np.where(mmstar_max_ar >= 5e6)[0] #These are the resolved indices
    unres_ixs = np.where(mmstar_max_ar < 5e6)[0] #These are the unresolved indices
    if iorf == 'i':
        mmstar_all[res_ixs] = mmstar_max_ar[res_ixs]
        if porc == 'p':
            mmstar_all[unres_ixs] = mmstar_max_pl_ar[unres_ixs]
        elif porc == 'c':
            mmstar_all[unres_ixs] = mmstar_max_co_ar[unres_ixs]
    elif iorf == 'f':
        mmstar_all[res_ixs] = mmstar_f_ar[res_ixs]
        if porc == 'p':
            mmstar_all[unres_ixs] = mmstar_f_pl_ar[unres_ixs]
        elif porc == 'c':
            mmstar_all[unres_ixs] = mmstar_f_co_ar[unres_ixs]
    return mmstar_all


def get_mrh(porc, iorf):
    '''
    Same as above, but now for the half light radius
    '''
    rh_all = np.zeros(len(mmmx_f_ar)) #This is the array where we will be combining everything
    res_ixs = np.where(mmstar_max_ar >= 5e6)[0] #These are the resolved indices
    unres_ixs = np.where(mmstar_max_ar < 5e6)[0] #These are the unresolved indices
    if iorf == 'i':
        rh_all[res_ixs] = mrh_max_ar[res_ixs]
        if porc == 'p':
            rh_all[unres_ixs] = mrh_max_pl_ar[unres_ixs]
        elif porc == 'c':
            rh_all[unres_ixs] = mrh_max_co_ar[unres_ixs]
    elif iorf == 'f':
        rh_all[res_ixs] = mrh_f_ar[res_ixs]
        if porc == 'p':
            rh_all[unres_ixs] = mrh_f_pl_ar[unres_ixs]
        elif porc == 'c':
            rh_all[unres_ixs] = mrh_f_co_ar[unres_ixs]
    return rh_all
     

def get_mvd(porc, iorf):
    '''
    Same as above, but now for the half light radius
    '''
    vd_all = np.zeros(len(mmmx_f_ar)) #This is the array where we will be combining everything
    res_ixs = np.where(mmstar_max_ar >= 5e6)[0] #These are the resolved indices
    unres_ixs = np.where(mmstar_max_ar < 5e6)[0] #These are the unresolved indices
    if iorf == 'i':
        vd_all[res_ixs] = mvd_max_ar[res_ixs]
        if porc == 'p':
            vd_all[unres_ixs] = mvd_max_ar[unres_ixs]
        elif porc == 'c':
            vd_all[unres_ixs] = mvd_max_ar[unres_ixs]
    elif iorf == 'f':
        vd_all[res_ixs] = mvd_f_ar[res_ixs]
        if porc == 'p':
            vd_all[unres_ixs] = mvd_f_pl_ar[unres_ixs]
        elif porc == 'c':
            vd_all[unres_ixs] = mvd_f_co_ar[unres_ixs]
    return vd_all




# PLOTS BEGIN HERE


def plot_mstar_vs_mvir(porc):
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


    fig, ax = plt.subplots(figsize = (6, 5))
    if porc == 'p':
        ax.scatter(smmx_f_ar, get_smstar(iorf='f', porc= 'p'), alpha = 0.1, s = 6, color = 'darkgreen', label = 'surviving - model')
        ax.scatter(mmmx_f_ar, get_mmstar(iorf='f', porc= 'p'), alpha = 0.1, color = 'purple', s = 6, label = 'merged - model')
        ax.set_title('z=0 from the model - Power law')
    elif porc == 'c':
        
        ax.scatter(smmx_f_ar, get_smstar(iorf='f', porc= 'c'), alpha = 0.1, s = 6, color = 'darkgreen', label = 'surviving - model')
        ax.scatter(mmmx_f_ar, get_mmstar(iorf='f', porc= 'c'), alpha = 0.1, color = 'purple', s = 6, label = 'merged - model')
        ax.set_title('z=0 from the model - Cutoff')
    # ax.set_xlim(left = 1e2)
    ax.set_ylim(bottom  = 1)

    mmxpl = np.logspace(2, 12, 100)
    ax.plot(mmxpl, get_moster_shm(mmxpl), 'k--', label = 'Moster+13')
    # ax.plot(mmxpl, 10**get_mstar_co(np.log10(get_Mmx_from_Vmax(mmxpl))), 'k-.', label = 'cutoff')
    ax.set_xlabel(r'$M_{\rm{mx}}\,\rm{(M_\odot)}$ ')
    ax.set_ylabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
    
    ax.legend(fontsize = 8)
    plt.loglog()
    plt.tight_layout()
    
    if porc == 'p': plt.savefig(this_fof_plotppath + 'pl_SMHM_relation.png')
    elif porc == 'c': plt.savefig(this_fof_plotppath + 'co_SMHM_relation.png')
    plt.close()



plot_mstar_vs_mvir(porc = porc) #Do not change this line



def plot_subh_mf(porc):
    '''
    This function is to plot the subhalos mass function
    '''
    mstarpl = np.logspace(1, 11, 100)
    Nm_ar = np.zeros(0) #shmf for merged subhalos
    Ns_ar = np.zeros(0) #shmf for surviving subhalos 
    Ntng_ar = np.zeros(0)
    mmstar_ar = get_mmstar(iorf = 'f', porc = porc)
    smstar_ar = get_smstar(iorf = 'f', porc = porc)
    for (ix, ms) in enumerate(mstarpl):
        Nm_ar = np.append(Nm_ar, len(mmstar_f_ar[mmstar_ar > ms]))
        Ns_ar = np.append(Ns_ar, len(smstar_f_ar[smstar_ar > ms]))
        Ntng_ar = np.append(Ntng_ar, len(smstar_f_ar_tng[smstar_f_ar_tng > ms]))

    fig, ax = plt.subplots(figsize = (5.5, 5))
    ax.plot(mstarpl, Ns_ar, color = 'darkgreen', label = 'Model Surviving', alpha = 0.5)
    ax.plot(mstarpl, Ns_ar + Nm_ar, color = 'purple', label = 'Model (all)', alpha = 0.5)
    # plot_sachi(ax)
    ax.plot(mstarpl, Ntng_ar, color = 'blue', label = 'TNG surviving', alpha = 0.5)
    ax.legend(fontsize = 8)
    ax.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
    ax.set_ylabel(r'$N(>M_{\bigstar})$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(left = 1e1)
    

    if porc == 'p':
        ax.set_title('z=0 from the model - power law')
        plt.tight_layout()
        plt.savefig(this_fof_plotppath + 'pl_mass_function.png')
    elif porc == 'c':
        ax.set_title('z=0 from the model - cutoff')
        plt.tight_layout()
        plt.savefig(this_fof_plotppath + 'co_mass_function.png')

    plt.close()



plot_subh_mf(porc = porc)



'''
Plot 3: Distance from the center vs M*
'''
def plot_dist_vs_mstar(porc, alpha = 0.15):
    fig, ax = plt.subplots(figsize = (5.5, 5))
    mmstar_ar = get_mmstar(iorf = 'f', porc = porc)
    smstar_ar = get_smstar(iorf = 'f', porc = porc)
    ax.scatter(mmstar_ar, mdist_f_ar, color = 'purple', alpha = alpha, s = 2, label = 'Merged')
    ax.scatter(smstar_ar, sdist_f_ar, color = 'darkgreen', alpha = alpha, s = 2, label = 'Surviving')
    ax.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
    ax.set_ylabel('Distance from center (kpc)')
    ax.set_xlim(left = 1e1)
    ax.set_ylim(bottom = 1e1)
    ax.legend(fontsize = 8)
    plt.loglog()
    plt.tight_layout()
    if porc == 'p':
        ax.set_title('power law')
        plt.tight_layout()
        plt.savefig(this_fof_plotppath + 'pl_dist_vs_mstar.png')
    elif porc == 'c':
        ax.set_title('cutoff')
        plt.tight_layout()
        plt.savefig(this_fof_plotppath + 'co_dist_vs_mstar.png')

    plt.close()


plot_dist_vs_mstar(porc)




def plot_cumulative_rad_dist(porc):
    '''
    Cumulative radial distribution
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
    # ax.plot(rpl, Ndm_ar, color = 'black', ls = '--', label = 'DM in TNG', alpha = 0.5)
    ax.legend(fontsize = 8)
    ax.set_xlabel('Distance from center (kpc)')
    ax.set_ylabel(r'$N(<r)$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    if porc == 'p':
        ax.set_title('z=0 from the model - power law')
        plt.tight_layout()
        plt.savefig(this_fof_plotppath + 'pl_cumulative_rad_dist.png')
    elif porc == 'c':
        ax.set_title('z=0 from the model - cutoff')
        plt.tight_layout()
        plt.savefig(this_fof_plotppath + 'co_cumulative_rad_dist.png')
    plt.close()


plot_cumulative_rad_dist(porc)


def plot_rh_vs_mstar(porc, iorf):
    '''
    Plot 5: Rh vs Mstar
    '''
    def get_line_of_constant_surfb(mstar, S):
        '''
        This is to plot the line of constant surface brightness
        mstar: Stellar mass
        S: Surface brightness

        Returns:
        R: Half light radius in pc
        '''
        logR = -(1/5.)*(4.83+21.57-2.5*np.log10(mstar)+2.5*np.log10(np.pi)-S)
        return 10**logR

    def inverse_get_line_of_constant_surfb(R, S):
        '''
        Inverse function of get_line_of_constant_surfb
        R: Half light radius in pc
        S: Surface brightness

        Returns:
        mstar: Stellar mass
        '''
        def equation(mstar):
            return R - get_line_of_constant_surfb(mstar, S)

        mstar_guess = 1e3  # Initial guess for mstar
        mstar_solution = fsolve(equation, mstar_guess)
        return mstar_solution[0]

    fig, ax = plt.subplots(figsize = (10, 6.5))
    mspl_log = np.linspace(1, 11, 100)
    plot_lg_virgo(ax)
    mmstar_ar = get_mmstar(iorf = iorf, porc = porc)
    smstar_ar = get_smstar(iorf = iorf, porc = porc)

    mrh_ar = get_mrh(iorf = iorf, porc = porc)
    srh_ar = get_srh(iorf = iorf, porc = porc)

    ax.scatter(smstar_ar, np.array(srh_ar) * 1e3, marker = 's', color = 'darkgreen', alpha = 0.45, s = 10, label = 'Survived', zorder = 200, edgecolor = 'black', linewidth = 0.7)
    ax.scatter(mmstar_ar, np.array(mrh_ar) * 1e3, marker = 's', color = 'purple', alpha = 0.45, s = 10, label = 'Merged', zorder = 200, edgecolor = 'black', linewidth = 0.7)
    top_data = ax.get_ylim()[1]
    right_data = ax.get_xlim()[1]
    ax.plot(10**mspl_log, 1e3 * 10**get_lrh(mspl_log), color = 'k')
    ax.set_xlim(left = 1e1, right = right_data)
    ax.set_ylim(bottom = 10, top = top_data)
    ax.plot(10**mspl_log, get_line_of_constant_surfb(10**mspl_log, 25), ls = '--', color = 'gray')
    ax.annotate(r'25 mag arcsec$^{-2}$', xy = (inverse_get_line_of_constant_surfb(2e3, 25), 2e3), xytext = (inverse_get_line_of_constant_surfb(2e3, 25), 1.05* 2e3), 
            rotation=46, color='gray', fontsize=10, rotation_mode='anchor')
    ax.plot(10**mspl_log, get_line_of_constant_surfb(10**mspl_log, 28), ls = '--', color = 'gray')
    ax.annotate(r'28 mag arcsec$^{-2}$', xy = (inverse_get_line_of_constant_surfb(2e3, 28), 2e3), xytext = (inverse_get_line_of_constant_surfb(2e3, 28), 1.05* 2e3), 
            rotation=46, color='gray', fontsize=10, rotation_mode='anchor')
    ax.plot(10**mspl_log, get_line_of_constant_surfb(10**mspl_log, 35), ls = '--', color = 'gray')
    ax.annotate(r'35 mag arcsec$^{-2}$', xy = (inverse_get_line_of_constant_surfb(2e3, 35), 2e3), xytext = (inverse_get_line_of_constant_surfb(2e3, 35), 1.05* 2e3), 
            rotation=46, color='gray', fontsize=10, rotation_mode='anchor')
    

    # # Example usage:
    # R = 100  # Half light radius in pc
    # S = 25  # Surface brightness
    # mstar_inverse = inverse_get_line_of_constant_surfb(R, S)
    # print(mstar_inverse)
    ax.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
    ax.set_ylabel(r'$R_{\rm{h}}$ (pc)')
    ax.legend(fontsize = 8)
    plt.loglog()
    
    if iorf == 'f':
        ax.set_title('At z=0')
        plt.tight_layout()
        if porc == 'p':
            plt.savefig(this_fof_plotppath + 'pl_z0_rh_vs_mstar.png')
        elif porc == 'c':
            plt.savefig(this_fof_plotppath + 'co_z0_rh_vs_mstar.png')
    elif iorf == 'i':
        ax.set_title('At infall')
        plt.tight_layout()
        if porc == 'p':
            plt.savefig(this_fof_plotppath + 'pl_inf_rh_vs_mstar.png')
        elif porc == 'c':
            plt.savefig(this_fof_plotppath + 'co_inf_rh_vs_mstar.png')
    
    
    plt.close()

plot_rh_vs_mstar(porc = porc, iorf = 'i')
plot_rh_vs_mstar(porc = porc, iorf = 'f')





def mstar_vs_vd(porc, iorf = 'f'):
    '''
    PLot 6: Mstar - sigma relation
    '''
    fig, ax = plt.subplots(figsize = (6, 6))
    # plot_lg_virgo(ax)
    mmstar_ar = get_mmstar(iorf = iorf, porc = porc)
    smstar_ar = get_smstar(iorf = iorf, porc = porc)

    mvd_ar = get_mvd(iorf = iorf, porc = porc)
    svd_ar = get_svd(iorf = iorf, porc = porc)

    mvd_ar[mvd_ar < 0] = 0
    svd_ar[svd_ar < 0] = 0
    ax.scatter(mmstar_ar , mvd_ar, marker = 's', color = 'purple', alpha = 0.3, s = 15, label = 'Merged (model)', zorder = 0, edgecolor = 'black', linewidth = 0.5)
    ax.scatter(smstar_ar , svd_ar, marker = 's', color = 'darkgreen', alpha = 0.3, s = 15, label = 'Survived (model)', zorder = 0, edgecolor = 'black', linewidth = 0.5)
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
    if porc == 'p':
        plt.savefig(this_fof_plotppath + 'pl_z0_mstar_vs_vd.png')
    elif porc == 'c':
        plt.savefig(this_fof_plotppath + 'co_z0_mstar_vs_vd.png')
    plt.close()


mstar_vs_vd(porc = porc)










def plot_test_rh(porc = porc, iorf = 'i'):
    '''
    Plot 7.2: This is to plot the rmx0 against the Mmx0 to check if there are some weird outliers
    '''
    fig, ax = plt.subplots(figsize = (7, 7))

    mmstar_ar = get_mmstar(iorf = iorf, porc = porc)
    smstar_ar = get_smstar(iorf = iorf, porc = porc)

    mrh_ar = get_mrh(iorf = iorf, porc = porc)
    srh_ar = get_srh(iorf = iorf, porc = porc)

    ax.scatter(smstar_ar, srh_ar/srmx_if_ar, marker = 'o', color = 'darkgreen', s = 10, alpha = 0.2, label = 'surviving')
    ax.scatter(mmstar_ar, mrh_ar/mrmx_if_ar, marker = 'o', color = 'purple', s = 10, alpha = 0.2, label = 'merged')

    rh0_by_rmx0_ar = np.concatenate([np.array(srh_ar/srmx_if_ar), np.array(mrh_ar/mrmx_if_ar)])
    mstar_ar = np.concatenate([np.array(smstar_ar), np.array(mmstar_ar)])
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
    if porc == 'p':
        plt.savefig(this_fof_plotppath + 'pl_rh_by_rmx0.png')
    elif porc == 'c':
        plt.savefig(this_fof_plotppath + 'co_rh_by_rmx0.png')
    plt.close()


plot_test_rh(porc = porc)





def plot_surv_comparison():
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
    # plt.savefig(outpath+'surviving_fof0_comparison.pdf')
    plt.savefig(this_fof_plotppath + 'surviving_fof0_comparison.png')
    plt.close()
    # plt.show()




def plot_difference_comparison():
    fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (12, 6))
    ((ax1, ax3, ax11), (ax2, ax4, ax22)) = axs 
    alpha = 0.8
    msize = 2
    # col_ar = stinf_ar
    # col_label = r'$t_{\rm{inf}}$'
    col_ar = np.log10(smstar_max_ar)
    col_label = r'$\log M_{\rm{\bigstar, max}}$'

    sc = ax1.scatter(smmx_if_ar - smmx_f_ar, smmx_if_ar - smmx_f_ar_tng, c=col_ar, cmap='viridis', alpha = alpha, s = msize, marker='o', zorder = 20)

    cbar = plt.colorbar(sc, ax = ax1)
    cbar.set_label(col_label)

    dummy = np.linspace(ax1.get_xlim()[0], ax1.get_xlim()[1], 3) 
    ax1.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)

    ax1.set_xlabel(r'$\Delta M_{\rm{mx}}$ from model')
    ax1.set_ylabel(r'$\Delta M_{\rm{mx}}$ from TNG')
    ax1.set_xscale('log')
    ax1.set_yscale('log')





    # =======================

    sc = ax2.scatter(smstar_max_ar - smstar_f_ar, smstar_max_ar - smstar_f_ar_tng, c=col_ar, cmap='viridis', alpha = alpha, s = msize, marker='o', zorder = 20)

    cbar = plt.colorbar(sc, ax = ax2)
    cbar.set_label(col_label)

    dummy = np.linspace(ax2.get_xlim()[0], ax2.get_xlim()[1], 3) 
    ax2.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)

    ax2.set_xlabel(r'$\Delta M_{\rm{\bigstar}}$ from model')
    ax2.set_ylabel(r'$\Delta M_{\rm{\bigstar}}$ from TNG')
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    # ax2.set_xlim(1e2, 1e10)
    # ax2.set_ylim(1e2, 1e10)


    # =======================


    sc = ax3.scatter((smmx_if_ar - smmx_f_ar)/smmx_if_ar, (smmx_if_ar - smmx_f_ar_tng)/smmx_if_ar, c=col_ar, cmap='viridis', alpha = alpha, s = msize, marker='o', zorder = 20)

    cbar = plt.colorbar(sc, ax = ax3)
    cbar.set_label(col_label)

    dummy = np.linspace(ax3.get_xlim()[0], ax3.get_xlim()[1], 3) 
    ax3.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)

    ax3.set_xlabel(r'$\Delta M_{\rm{mx}}/M_{\rm{mx0}}$ from model')
    ax3.set_ylabel(r'$\Delta M_{\rm{mx}}/M_{\rm{mx0}}$ from TNG')
    ax3.set_xscale('log')
    ax3.set_yscale('log')





    # =======================

    sc = ax4.scatter((smstar_max_ar - smstar_f_ar)/smstar_max_ar, (smstar_max_ar - smstar_f_ar_tng)/smstar_max_ar, c=col_ar, cmap='viridis', alpha = alpha, s = msize, marker='o', zorder = 20)

    cbar = plt.colorbar(sc, ax = ax4)
    cbar.set_label(col_label)

    dummy = np.linspace(ax4.get_xlim()[0], ax4.get_xlim()[1], 3) 
    ax4.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)

    ax4.set_xlabel(r'$\Delta M_{\rm{\bigstar}}/M_{\rm{\bigstar}}$ from model')
    ax4.set_ylabel(r'$\Delta M_{\rm{\bigstar}}/M_{\rm{\bigstar}}$ from TNG')
    ax4.set_xscale('log')
    ax4.set_yscale('log')

    # ax4.set_xlim(1e2, 1e10)
    # ax4.set_ylim(1e2, 1e10)

    sc = ax11.scatter(smmx_f_ar, smmx_f_ar_tng, c=col_ar, cmap='viridis', alpha = alpha, s = msize, marker='o', zorder = 20)

    cbar = plt.colorbar(sc, ax = ax11)
    cbar.set_label(col_label)

    dummy = np.linspace(ax11.get_xlim()[0], ax11.get_xlim()[1], 3) 
    ax11.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)

    ax11.set_xlabel(r'$ M_{\rm{mx}}$ from model')
    ax11.set_ylabel(r'$ M_{\rm{mx}}$ from TNG')
    ax11.set_xscale('log')
    ax11.set_yscale('log')

    # =======================

    sc = ax22.scatter(smstar_f_ar, smstar_f_ar_tng, c=col_ar, cmap='viridis', alpha = alpha, s = msize, marker='o', zorder = 20)

    cbar = plt.colorbar(sc, ax = ax22)
    cbar.set_label(col_label)

    dummy = np.linspace(ax22.get_xlim()[0], ax22.get_xlim()[1], 3) 
    ax22.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)

    ax22.set_xlabel(r'$ M_{\rm{\bigstar}}$ from model')
    ax22.set_ylabel(r'$ M_{\rm{\bigstar}}$ from TNG')
    ax22.set_xscale('log')
    ax22.set_yscale('log')




    # plt.loglog()
    plt.tight_layout()
    plt.savefig(this_fof_plotppath+'surviving_fof0_diff_comparison.png')
    plt.close()
    # plt.show()


    return None

# plot_difference_comparison()



def plot_fractions_comparison():
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
    plt.savefig(outpath+'surviving_fof0_frac_comparison.png')
    plt.close()
    # plt.show()
    return None

# plot_fractions_comparison()


