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
ferrarese_data_path = '/bigdata/saleslab/psadh003/misc_files/Ferrarese_virgo_smf.csv'

'''
This part is to plot the NGVS data of virgo core set of satellites
'''
fdata = pd.read_csv(ferrarese_data_path, delimiter = ',')
fmstar = fdata['mstar']
fngal = fdata['ngal']
fngal_cum = np.cumsum(fngal)

fof_no = int(sys.argv[1])
fof_str = 'fof' + str(fof_no)

this_fof = il.groupcat.loadSingle(basePath, 99, haloID = fof_no)
central_sfid_99 = this_fof['GroupFirstSub']
rvir_fof = this_fof['Group_R_Crit200']/0.6744

this_fof_plotppath = '/bigdata/saleslab/psadh003/tng50/final_plots/' + fof_str + '/'
if not os.path.exists(this_fof_plotppath): #If the directory does not exist, then just create it!
    os.makedirs(this_fof_plotppath)

'''
In the following lines, 
(i) ps stands for power law values of surviving subhalos
(ii) pm stands for power law values of merged subhalos
(iii) cs stands for cutoff values of surviving subhalos
(iv) cm stands for cutoff values of merged subhalos
'''

mcutoff = 1e2


if True: #This is a section for getting power law values of survived subhalos
    pdfs = pd.read_csv(outpath + fof_str + '_surviving_evolved_everything.csv', delimiter = ',', low_memory=False)
    pdfs = pdfs.applymap(convert_to_float)


    pdfs = pdfs[pdfs['dist_f_ar']<rvir_fof]

    pdfs1 = pdfs
    # if porc == 'p':
        # print(len(pdfs[pdfs['mstar_f_ar_tng'].values>5e6]))
    columns_to_max = ['mstar_max_ar', 'mstar_max_pl_ar']
    max_values = pdfs[columns_to_max].max(axis=1)
    pdfs = pdfs[(max_values > mcutoff)]

    columns_to_max = ['mstar_f_ar_tng', 'mstar_f_pl_ar']
    max_values = pdfs[columns_to_max].max(axis=1)
    cutoff = 10

    pdfs = pdfs[(max_values > cutoff)]


    psdist_f_ar = pdfs['dist_f_ar'].values
    pspos_f_ar = pdfs['pos_f_ar'].values

    psvd_f_ar_tng = pdfs['vd_f_ar_tng'].values
    psrh_f_ar_tng = pdfs['rh_f_ar_tng'].values
    psmstar_f_ar_tng = pdfs['mstar_f_ar_tng'].values
    psmmx_f_ar_tng = pdfs['mmx_f_ar_tng'].values
    psrmx_f_ar_tng = pdfs['rmx_f_ar_tng'].values
    psvmx_f_ar_tng = pdfs['vmx_f_ar_tng'].values

    pstinf_ar = pdfs['tinf_ar'].values
    pstorb_ar = pdfs['torb_ar'].values
    psrapo_ar = pdfs['rapo_ar'].values
    psrperi_ar = pdfs['rperi_ar'].values

    psvd_f_ar = pdfs['vd_f_ar'].values
    psrh_f_ar = pdfs['rh_f_ar'].values
    psmstar_f_ar = pdfs['mstar_f_ar'].values

    psvd_max_ar = pdfs['vd_max_ar'].values
    psrh_max_ar = pdfs['rh_max_ar'].values
    psmstar_max_ar = pdfs['mstar_max_ar'].values

    pssnap_if_ar = pdfs['snap_if_ar'].values
    pssf_id_if_ar = pdfs['sfid_if_ar'].values

    psmmx_f_ar = pdfs['mmx_f_ar'].values
    psrmx_f_ar = pdfs['rmx_f_ar'].values
    psvmx_f_ar = pdfs['vmx_f_ar'].values

    psmmx_if_ar = pdfs['mmx_if_ar'].values
    psrmx_if_ar = pdfs['rmx_if_ar'].values
    psvmx_if_ar = pdfs['vmx_if_ar'].values

    psvd_f_pl_ar = pdfs['vd_f_pl_ar'].values
    psrh_f_pl_ar = pdfs['rh_f_pl_ar'].values
    psmstar_f_pl_ar = pdfs['mstar_f_pl_ar'].values
    psvd_f_co_ar = pdfs['vd_f_co_ar'].values
    psrh_f_co_ar = pdfs['rh_f_co_ar'].values
    psmstar_f_co_ar = pdfs['mstar_f_co_ar'].values
    psrh_max_pl_ar = pdfs['rh_max_pl_ar'].values
    psrh_max_co_ar = pdfs['rh_max_co_ar'].values
    psmstar_max_pl_ar = pdfs['mstar_max_pl_ar'].values
    psmstar_max_co_ar = pdfs['mstar_max_co_ar'].values

    psmstar_all = np.zeros(len(psmmx_f_ar))
    psrh_all = np.zeros(len(psmmx_f_ar))
    psvd_all = np.zeros(len(psmmx_f_ar))
    res_ixs = np.where(psmstar_max_ar >= 5e6)[0] #These are the resolved indices
    unres_ixs = np.where(psmstar_max_ar < 5e6)[0] #These are the unresolved indices
    psmstar_all[res_ixs] = psmstar_f_ar[res_ixs]
    psmstar_all[unres_ixs] = psmstar_f_pl_ar[unres_ixs]
    psrh_all[res_ixs] = psrh_f_ar[res_ixs]
    psrh_all[unres_ixs] = psrh_f_pl_ar[unres_ixs]
    psvd_all[res_ixs] = psvd_f_ar[res_ixs]
    psvd_all[unres_ixs] = psvd_f_pl_ar[unres_ixs]





if True: #This is a section for getting power law values of merged subhalos
    pdfm = pd.read_csv(outpath + fof_str + '_merged_evolved_wmbp_everything.csv', delimiter = ',')
    pdfm = pdfm.applymap(convert_to_float)
    pdfm = pdfm[pdfm['dist_f_ar']<rvir_fof]
    pdfm1 = pdfm 

    columns_to_max = ['mstar_max_ar', 'mstar_max_pl_ar']
    max_values = pdfm[columns_to_max].max(axis=1)
    pdfm = pdfm[(max_values > mcutoff)]

    columns_to_max = ['mstar_f_ar', 'mstar_f_pl_ar']

    # Compute the maximum along the specified axis (axis=1 for row-wise)
    max_values = pdfm[columns_to_max].max(axis=1)

    # Sample cutoff value
    cutoff = 10

    # print(len(pdfm))
    pdfm = pdfm[max_values > cutoff]
    # print(len(pdfm))
    pmdist_f_ar = pdfm['dist_f_ar'].values
    pmmbpid_ar = pdfm['mbpid_ar'].values

    pmtinf_ar = pdfm['tinf_ar'].values
    pmtorb_ar = pdfm['torb_ar'].values
    pmrapo_ar = pdfm['rapo_ar'].values
    pmrperi_ar = pdfm['rperi_ar'].values

    pmvd_f_ar = pdfm['vd_f_ar'].values
    pmrh_f_ar = pdfm['rh_f_ar'].values
    pmmstar_f_ar = pdfm['mstar_f_ar'].values

    pmvd_max_ar = pdfm['vd_max_ar'].values
    pmrh_max_ar = pdfm['rh_max_ar'].values
    pmmstar_max_ar = pdfm['mstar_max_ar'].values

    pmsnap_if_ar = pdfm['snap_if_ar'].values
    pmsfid_if_ar = pdfm['sfid_if_ar'].values

    pmmmx_f_ar = pdfm['mmx_f_ar'].values
    pmrmx_f_ar = pdfm['rmx_f_ar'].values
    pmvmx_f_ar = pdfm['vmx_f_ar'].values

    pmmmx_if_ar = pdfm['mmx_if_ar'].values
    pmrmx_if_ar = pdfm['rmx_if_ar'].values
    pmvmx_if_ar = pdfm['vmx_if_ar'].values

    pmvd_f_pl_ar = pdfm['vd_f_pl_ar'].values
    pmrh_f_pl_ar = pdfm['rh_f_pl_ar'].values
    pmmstar_f_pl_ar = pdfm['mstar_f_pl_ar'].values
    pmvd_f_co_ar = pdfm['vd_f_co_ar'].values
    pmrh_f_co_ar = pdfm['rh_f_co_ar'].values
    pmmstar_f_co_ar = pdfm['mstar_f_co_ar'].values
    pmrh_max_pl_ar = pdfm['rh_max_pl_ar'].values
    pmrh_max_co_ar = pdfm['rh_max_co_ar'].values
    pmmstar_max_pl_ar = pdfm['mstar_max_pl_ar'].values
    pmmstar_max_co_ar = pdfm['mstar_max_co_ar'].values


    pmmstar_all = np.zeros(len(pmmmx_f_ar))
    pmrh_all = np.zeros(len(pmmmx_f_ar))
    pmvd_all = np.zeros(len(pmmmx_f_ar))
    res_ixs = np.where(pmmstar_max_ar >= 5e6)[0] #These are the resolved indices
    unres_ixs = np.where(pmmstar_max_ar < 5e6)[0] #These are the unresolved indices
    pmmstar_all[res_ixs] = pmmstar_f_ar[res_ixs]
    pmmstar_all[unres_ixs] = pmmstar_f_pl_ar[unres_ixs]
    pmrh_all[res_ixs] = pmrh_f_ar[res_ixs]
    pmrh_all[unres_ixs] = pmrh_f_pl_ar[unres_ixs]
    pmvd_all[res_ixs] = pmvd_f_ar[res_ixs]
    pmvd_all[unres_ixs] = pmvd_f_pl_ar[unres_ixs]




#Once we have the power law data for surviving and merged subhalos, we should obtain their respective stellar masses etc checking if they are resolved




if True: #This is a section for getting cutoff values of survived subhalos
    cdfs = pd.read_csv(outpath + fof_str + '_surviving_evolved_everything.csv', delimiter = ',', low_memory=False)
    cdfs = cdfs.applymap(convert_to_float)


    cdfs = cdfs[cdfs['dist_f_ar']<rvir_fof]

    cdfs1 = cdfs

    # elif porc == 'c':
    columns_to_max = ['mstar_max_ar', 'mstar_max_co_ar']
    max_values = cdfs[columns_to_max].max(axis=1)
    cdfs = cdfs[(max_values > mcutoff)]

    columns_to_max = ['mstar_f_ar_tng', 'mstar_f_co_ar']

    # Compute the maximum along the specified axis (axis=1 for row-wise)
    max_values = cdfs[columns_to_max].max(axis=1)

    cutoff = 10
    # print(max_values > cutoff)

    cdfs = cdfs[(max_values > cutoff)]

    csdist_f_ar = cdfs['dist_f_ar'].values
    cspos_f_ar = cdfs['pos_f_ar'].values

    csvd_f_ar_tng = cdfs['vd_f_ar_tng'].values
    csrh_f_ar_tng = cdfs['rh_f_ar_tng'].values
    csmstar_f_ar_tng = cdfs['mstar_f_ar_tng'].values
    csmmx_f_ar_tng = cdfs['mmx_f_ar_tng'].values
    csrmx_f_ar_tng = cdfs['rmx_f_ar_tng'].values
    csvmx_f_ar_tng = cdfs['vmx_f_ar_tng'].values

    cstinf_ar = cdfs['tinf_ar'].values
    cstorb_ar = cdfs['torb_ar'].values
    csrapo_ar = cdfs['rapo_ar'].values
    csrperi_ar = cdfs['rperi_ar'].values

    csvd_f_ar = cdfs['vd_f_ar'].values
    csrh_f_ar = cdfs['rh_f_ar'].values
    csmstar_f_ar = cdfs['mstar_f_ar'].values

    csvd_max_ar = cdfs['vd_max_ar'].values
    csrh_max_ar = cdfs['rh_max_ar'].values
    csmstar_max_ar = cdfs['mstar_max_ar'].values

    cssnap_if_ar = cdfs['snap_if_ar'].values
    cssf_id_if_ar = cdfs['sfid_if_ar'].values

    csmmx_f_ar = cdfs['mmx_f_ar'].values
    csrmx_f_ar = cdfs['rmx_f_ar'].values
    csvmx_f_ar = cdfs['vmx_f_ar'].values

    csmmx_if_ar = cdfs['mmx_if_ar'].values
    csrmx_if_ar = cdfs['rmx_if_ar'].values
    csvmx_if_ar = cdfs['vmx_if_ar'].values

    csvd_f_pl_ar = cdfs['vd_f_pl_ar'].values
    csrh_f_pl_ar = cdfs['rh_f_pl_ar'].values
    csmstar_f_pl_ar = cdfs['mstar_f_pl_ar'].values
    csvd_f_co_ar = cdfs['vd_f_co_ar'].values
    csrh_f_co_ar = cdfs['rh_f_co_ar'].values
    csmstar_f_co_ar = cdfs['mstar_f_co_ar'].values
    csrh_max_pl_ar = cdfs['rh_max_pl_ar'].values
    csrh_max_co_ar = cdfs['rh_max_co_ar'].values
    csmstar_max_pl_ar = cdfs['mstar_max_pl_ar'].values
    csmstar_max_co_ar = cdfs['mstar_max_co_ar'].values

    csmstar_all = np.zeros(len(csmmx_f_ar))
    csrh_all = np.zeros(len(csmmx_f_ar))
    csvd_all = np.zeros(len(csmmx_f_ar))
    res_ixs = np.where(csmstar_max_ar >= 5e6)[0] #These are the resolved indices
    unres_ixs = np.where(csmstar_max_ar < 5e6)[0] #These are the unresolved indices
    csmstar_all[res_ixs] = csmstar_f_ar[res_ixs]
    csmstar_all[unres_ixs] = csmstar_f_pl_ar[unres_ixs]
    csrh_all[res_ixs] = csrh_f_ar[res_ixs]
    csrh_all[unres_ixs] = csrh_f_pl_ar[unres_ixs]
    csvd_all[res_ixs] = csvd_f_ar[res_ixs]
    csvd_all[unres_ixs] = csvd_f_pl_ar[unres_ixs]


if True: #This is a section for getting cutoff values of merged subhalos
    cdfm = pd.read_csv(outpath + fof_str + '_merged_evolved_wmbp_everything.csv', delimiter = ',')
    cdfm = cdfm.applymap(convert_to_float)
    cdfm = cdfm[cdfm['dist_f_ar']<rvir_fof]
    cdfm1 = cdfm 

    columns_to_max = ['mstar_max_ar', 'mstar_max_co_ar']
    max_values = cdfm[columns_to_max].max(axis=1)
    cdfm = cdfm[(max_values > mcutoff)]

    columns_to_max = ['mstar_f_ar', 'mstar_f_co_ar']

    # Compute the maximum along the specified axis (axis=1 for row-wise)
    max_values = cdfm[columns_to_max].max(axis=1)

    # Sample cutoff value
    cutoff = 10
    cdfm = cdfm[max_values > cutoff]
    # print(len(cdfm))
    cmdist_f_ar = cdfm['dist_f_ar'].values
    cmmbpid_ar = cdfm['mbpid_ar'].values

    cmtinf_ar = cdfm['tinf_ar'].values
    cmtorb_ar = cdfm['torb_ar'].values
    cmrapo_ar = cdfm['rapo_ar'].values
    cmrperi_ar = cdfm['rperi_ar'].values

    cmvd_f_ar = cdfm['vd_f_ar'].values
    cmrh_f_ar = cdfm['rh_f_ar'].values
    cmmstar_f_ar = cdfm['mstar_f_ar'].values

    cmvd_max_ar = cdfm['vd_max_ar'].values
    cmrh_max_ar = cdfm['rh_max_ar'].values
    cmmstar_max_ar = cdfm['mstar_max_ar'].values

    cmsnap_if_ar = cdfm['snap_if_ar'].values
    cmsfid_if_ar = cdfm['sfid_if_ar'].values

    cmmmx_f_ar = cdfm['mmx_f_ar'].values
    cmrmx_f_ar = cdfm['rmx_f_ar'].values
    cmvmx_f_ar = cdfm['vmx_f_ar'].values

    cmmmx_if_ar = cdfm['mmx_if_ar'].values
    cmrmx_if_ar = cdfm['rmx_if_ar'].values
    cmvmx_if_ar = cdfm['vmx_if_ar'].values

    cmvd_f_pl_ar = cdfm['vd_f_pl_ar'].values
    cmrh_f_pl_ar = cdfm['rh_f_pl_ar'].values
    cmmstar_f_pl_ar = cdfm['mstar_f_pl_ar'].values
    cmvd_f_co_ar = cdfm['vd_f_co_ar'].values
    cmrh_f_co_ar = cdfm['rh_f_co_ar'].values
    cmmstar_f_co_ar = cdfm['mstar_f_co_ar'].values
    cmrh_max_pl_ar = cdfm['rh_max_pl_ar'].values
    cmrh_max_co_ar = cdfm['rh_max_co_ar'].values
    cmmstar_max_pl_ar = cdfm['mstar_max_pl_ar'].values
    cmmstar_max_co_ar = cdfm['mstar_max_co_ar'].values

    cmmstar_all = np.zeros(len(cmmmx_f_ar))
    cmrh_all = np.zeros(len(cmmmx_f_ar))
    cmvd_all = np.zeros(len(cmmmx_f_ar))
    res_ixs = np.where(cmmstar_max_ar >= 5e6)[0] #These are the resolved indices
    unres_ixs = np.where(cmmstar_max_ar < 5e6)[0] #These are the unresolved indices
    cmmstar_all[res_ixs] = cmmstar_f_ar[res_ixs]
    cmmstar_all[unres_ixs] = cmmstar_f_pl_ar[unres_ixs]
    cmrh_all[res_ixs] = cmrh_f_ar[res_ixs]
    cmrh_all[unres_ixs] = cmrh_f_pl_ar[unres_ixs]
    cmvd_all[res_ixs] = cmvd_f_ar[res_ixs]
    cmvd_all[unres_ixs] = cmvd_f_pl_ar[unres_ixs]



def plot_subh_mf():
    '''
    This function is to plot the subhalos mass function
    '''
    mstarpl = np.logspace(1, 11, 100)
    pNm_ar = np.zeros(0) #shmf for merged subhalos
    pNs_ar = np.zeros(0) #shmf for surviving subhalos 
    cNm_ar = np.zeros(0) #shmf for merged subhalos
    cNs_ar = np.zeros(0) #shmf for surviving subhalos 
    Ntng_ar = np.zeros(0)
    for (ix, ms) in enumerate(mstarpl):
        pNm_ar = np.append(pNm_ar, len(pmmstar_f_ar[pmmstar_all > ms]))
        pNs_ar = np.append(pNs_ar, len(psmstar_f_ar[psmstar_all > ms]))
        cNm_ar = np.append(cNm_ar, len(cmmstar_f_ar[cmmstar_all > ms]))
        cNs_ar = np.append(cNs_ar, len(csmstar_f_ar[csmstar_all > ms]))
        Ntng_ar = np.append(Ntng_ar, len(psmstar_f_ar_tng[psmstar_f_ar_tng > ms]))

    fig, ax = plt.subplots(figsize = (6, 6.25))
    
    # ax.plot(mstarpl, Ns_ar, color = 'darkgreen', label = 'Model Surviving', alpha = 0.5)
    ax.plot(mstarpl, cNs_ar + cNm_ar, color = 'darkgreen', label = 'Model cutoff', alpha = 0.5)
    ax.plot(mstarpl, pNs_ar + pNm_ar, color = 'red', label = 'Model power law', alpha = 0.5)
    # plot_sachi(ax)
    ax.plot(mstarpl, Ntng_ar, color = 'blue', label = 'TNG surviving', alpha = 0.5)
    ax.legend(fontsize = 8)
    ax.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
    ax.set_ylabel(r'$N(>M_{\bigstar})$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(left = 1e1)
    ax.tick_params(axis='y', which = 'both', left=True, right=True, direction = 'in')
    ax.tick_params(axis='x', which = 'both', direction = 'in')
    
    ax.set_title('FoF'+str(fof_no))
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'mass_function.png')


    plt.close()

plot_subh_mf()



def plot_subh_mf_core():
    '''
    This is to plot the satellite mass function and ompare it with the Virgo core data from ferrarese 2016 paper
    '''
    rcore = 0.2 * rvir_fof #This is assumed to be the core radius temporarily
    mstarpl = np.logspace(1, 11, 100)
    pNm_ar = np.zeros(0) #shmf for merged subhalos
    pNs_ar = np.zeros(0) #shmf for surviving subhalos 
    cNm_ar = np.zeros(0) #shmf for merged subhalos
    cNs_ar = np.zeros(0) #shmf for surviving subhalos 
    Ntng_ar = np.zeros(0)
    for (ix, ms) in enumerate(mstarpl):
        pNm_ar = np.append(pNm_ar, len(pmmstar_f_ar[(pmmstar_all > ms) & (pmdist_f_ar < rcore)]))
        pNs_ar = np.append(pNs_ar, len(psmstar_f_ar[(psmstar_all > ms) & (psdist_f_ar < rcore)]))
        cNm_ar = np.append(cNm_ar, len(cmmstar_f_ar[(cmmstar_all > ms) & (cmdist_f_ar < rcore)]))
        cNs_ar = np.append(cNs_ar, len(csmstar_f_ar[(csmstar_all > ms) & (csdist_f_ar < rcore)]))
        Ntng_ar = np.append(Ntng_ar, len(psmstar_f_ar_tng[(psmstar_f_ar_tng > ms) & (psdist_f_ar < rcore)]))

    fig, ax = plt.subplots(figsize = (6, 6.25))
    
    # ax.plot(mstarpl, Ns_ar, color = 'darkgreen', label = 'Model Surviving', alpha = 0.5)
    ax.plot(10**fmstar, fngal_cum, color = 'black', marker = 'o', label = 'Ferrarese+16 Virgo core')

    ax.plot(mstarpl, cNs_ar + cNm_ar, color = 'darkgreen', label = 'Model cutoff', alpha = 0.5)
    ax.plot(mstarpl, pNs_ar + pNm_ar, color = 'red', label = 'Model power law', alpha = 0.5)
    # plot_sachi(ax)
    ax.plot(mstarpl, Ntng_ar, color = 'blue', label = 'TNG surviving', alpha = 0.5)
    ax.legend(fontsize = 8)
    ax.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
    ax.set_ylabel(r'$N(>M_{\bigstar})$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(left = 1e1)
    ax.tick_params(axis='y', which = 'both', left=True, right=True, direction = 'in')
    ax.tick_params(axis='x', which = 'both', direction = 'in')
    
    ax.set_title('FoF'+str(fof_no)+ f' within core {rcore:.0f} kpc')
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'mass_function_core.png')


    plt.close()

    return

plot_subh_mf_core()

def plot_rh_vs_mstar():
    '''
    Rh vs Mstar
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

    fig, (ax, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6.15))
    mspl_log = np.linspace(1, 13, 100)
    plot_lg_virgo(ax)
    plot_lg_virgo(ax2)
    

    ax.scatter(csmstar_all, np.array(csrh_all) * 1e3, marker = 's', fc = 'darkgreen', alpha = 0.3, s = 10, label = 'Survived', zorder = 200, edgecolor = 'darkgreen', linewidth = 0.7)
    ax.scatter(cmmstar_all, np.array(cmrh_all) * 1e3, marker = 's', fc = 'purple', alpha = 0.3, s = 10, label = 'Merged', zorder = 200, edgecolor = 'purple', linewidth = 0.7)
    top_data = ax.get_ylim()[1]
    right_data = ax.get_xlim()[1]
    ax.plot(10**mspl_log, 1e3 * 10**get_lrh(mspl_log), color = 'k')
    ax.set_xlim(left = 1e1, right = right_data)
    ax.set_ylim(bottom = 10, top = top_data)

    ax2.scatter(psmstar_all, np.array(psrh_all) * 1e3, marker = 's', fc = 'darkgreen', alpha = 0.3, s = 10, label = 'Survived', zorder = 200, edgecolor = 'darkgreen', linewidth = 0.7)
    ax2.scatter(pmmstar_all, np.array(pmrh_all) * 1e3, marker = 's', fc = 'purple', alpha = 0.3, s = 10, label = 'Merged', zorder = 200, edgecolor = 'purple', linewidth = 0.7)
    top_data2 = ax2.get_ylim()[1]
    right_data = ax2.get_xlim()[1]
    ax2.plot(10**mspl_log, 1e3 * 10**get_lrh(mspl_log), color = 'k')
    ax2.set_xlim(left = 1e1, right = right_data)
    ax2.set_ylim(bottom = 10, top = top_data2)


    if True: #This section is for lines of constant surface brightness
        angle = 59
        yval = 0.21*top_data
        ax.plot(10**mspl_log, get_line_of_constant_surfb(10**mspl_log, 25), ls = '--', color = 'gray')
        ax.annotate(r'25 mag arcsec$^{-2}$', xy = (inverse_get_line_of_constant_surfb(yval, 25), yval), xytext = (inverse_get_line_of_constant_surfb(yval, 25), 1.05* yval), 
                rotation=angle, color='gray', fontsize=10, rotation_mode='anchor')
        ax.plot(10**mspl_log, get_line_of_constant_surfb(10**mspl_log, 28), ls = '--', color = 'gray')
        ax.annotate(r'28 mag arcsec$^{-2}$', xy = (inverse_get_line_of_constant_surfb(yval, 28), yval), xytext = (inverse_get_line_of_constant_surfb(yval, 28), 1.05* yval), 
                rotation=angle, color='gray', fontsize=10, rotation_mode='anchor')
        ax.plot(10**mspl_log, get_line_of_constant_surfb(10**mspl_log, 35), ls = '--', color = 'gray')
        ax.annotate(r'35 mag arcsec$^{-2}$', xy = (inverse_get_line_of_constant_surfb(yval, 35), yval), xytext = (inverse_get_line_of_constant_surfb(yval, 35), 1.05* yval), 
                rotation=angle, color='gray', fontsize=10, rotation_mode='anchor')
        
        yval = 0.21*top_data2
        ax2.plot(10**mspl_log, get_line_of_constant_surfb(10**mspl_log, 25), ls = '--', color = 'gray')
        ax2.annotate(r'25 mag arcsec$^{-2}$', xy = (inverse_get_line_of_constant_surfb(yval, 25), yval), xytext = (inverse_get_line_of_constant_surfb(yval, 25), 1.05* yval), 
                rotation=angle, color='gray', fontsize=10, rotation_mode='anchor')
        ax2.plot(10**mspl_log, get_line_of_constant_surfb(10**mspl_log, 28), ls = '--', color = 'gray')
        ax2.annotate(r'28 mag arcsec$^{-2}$', xy = (inverse_get_line_of_constant_surfb(yval, 28), yval), xytext = (inverse_get_line_of_constant_surfb(yval, 28), 1.05* yval), 
                rotation=angle, color='gray', fontsize=10, rotation_mode='anchor')
        ax2.plot(10**mspl_log, get_line_of_constant_surfb(10**mspl_log, 35), ls = '--', color = 'gray')
        ax2.annotate(r'35 mag arcsec$^{-2}$', xy = (inverse_get_line_of_constant_surfb(yval, 35), yval), xytext = (inverse_get_line_of_constant_surfb(yval, 35), 1.05* yval), 
                rotation=angle, color='gray', fontsize=10, rotation_mode='anchor')
    

    # # Example usage:
    # R = 100  # Half light radius in pc
    # S = 25  # Surface brightness
    # mstar_inverse = inverse_get_line_of_constant_surfb(R, S)
    # print(mstar_inverse)
    ax.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
    ax.set_ylabel(r'$R_{\rm{h}}$ (pc)')
    ax.legend(fontsize = 8)
    ax.tick_params(axis='both', which = 'both', left=True, right=True, bottom = True, top = True, direction = 'in')
    ax.text(0.01, 0.99, 'Cutoff', ha = 'left', va = 'top', transform=ax.transAxes)

    ax2.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
    ax2.set_ylabel(r'$R_{\rm{h}}$ (pc)')
    ax2.legend(fontsize = 8)
    ax2.tick_params(axis='both', which = 'both', left=True, right=True, bottom = True, top = True, direction = 'in')
    ax2.text(0.01, 0.99, 'Power law', ha = 'left', va = 'top', transform=ax2.transAxes)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    fig.suptitle('FoF'+str(fof_no), fontsize = 14)
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'rh_vs_mstar.png')
    plt.close()

plot_rh_vs_mstar()





def plot_cumulative_rad_dist():
    '''
    Cumulative radial distribution
    '''
    rpl = np.logspace(1, 3.2, 100)
    pNm_ar = np.zeros(0) #shmf for merged subhalos
    pNs_ar = np.zeros(0) #shmf for surviving subhalos 
    cNm_ar = np.zeros(0) #shmf for merged subhalos
    cNs_ar = np.zeros(0) #shmf for surviving subhalos

    N_all_ar = np.zeros(0) #This is for all the subhalos inside virial radius
    Ntng_ar = np.zeros(0)

    rpl = np.logspace(1, 3.2, 100) #r = m
    if fof_no == 0:
        Ndm_ar =[721148, 778322, 839804, 905710, 976898, 1054137, 1137501, 1227676, 1325214, 1430877, 1545758, 1669730, 1805234, 1953330, 2114837, 2291374, 2484727, 2697632, 2929798, 3184435, 3461447, 3763454, 4093457, 4451911, 4839639, 5261067, 5717234, 6218030, 6759960, 7335930, 7958128, 8628601, 9351223, 10124871, 10954252, 11843815, 12803150, 13852694, 14955612, 16130128, 17390800, 18724037, 20147057, 21672913, 23303629, 25059909, 26945144, 28965678, 31153056, 33453105, 35883715, 38490363, 41284274, 44311781, 47556207, 50917974, 54416144, 58216056, 62174887, 66291888, 70515423, 74848372, 79404305, 84157527, 89209147, 94471028, 99857920, 105317116, 111028328, 116939549, 123289439, 130030026, 137275327, 145069112, 153548719, 161907244, 169905804, 177886839, 185830162, 194592334, 203383663, 212695914, 222350306, 232159462, 242040951, 251514511, 260202731, 268769952, 278511000, 289919552, 300273807, 309815832, 319892274, 331241231, 342176869, 352467656, 365061316, 375410309, 383675808, 390894527]
    elif fof_no == 1:
        Ndm_ar = [786023, 841839, 902134, 966380, 1035837, 1110113, 1189602, 1274186, 1364366, 1460423, 1562817, 1672316, 1788483, 1913425, 2046885, 2190148, 2343775, 2511901, 2690528, 2878121, 3078467, 3294791, 3528395, 3780942, 4053436, 4350232, 4685298, 5050150, 5416132, 5796879, 6199598, 6623522, 7072307, 7546453, 8050591, 8585909, 9153650, 9756524, 10397233, 11074877, 11795747, 12560524, 13376455, 14255740, 15217979, 16289952, 17349673, 18430107, 19555220, 20756630, 22010268, 23314416, 24673283, 26107659, 27631468, 29312494, 31108073, 32886889, 34739969, 36734573, 38844189, 41049934, 43382025, 45812785, 48395953, 51173751, 54141628, 57270029, 60739449, 64360174, 68074798, 71803090, 75817335, 80041042, 84373691, 89111485, 94087743, 99601105, 104804935, 109994769, 115260983, 121202017, 127487242, 133053679, 138547841, 144456102, 151341815, 158789550, 164815392, 169912646, 174986366, 179518933, 183635753, 187442091, 191210434, 194512610, 197453028, 199783883, 201489573, 202770641]
    elif fof_no == 2:
        Ndm_ar = [506233, 543093, 582765, 625491, 671876, 722219, 777025, 836818, 901297, 970418, 1045227, 1126104, 1213949, 1309191, 1412584, 1524440, 1646551, 1777799, 1919728, 2072632, 2238248, 2417271, 2609800, 2817277, 3040510, 3281139, 3539926, 3817121, 4114546, 4432843, 4774753, 5139467, 5529198, 5945510, 6389364, 6861022, 7362273, 7896148, 8464347, 9073994, 9720932, 10401647, 11120919, 11885151, 12702569, 13567823, 14490623, 15490151, 16538923, 17659839, 18860064, 20139640, 21539654, 23004408, 24521251, 26118857, 27830425, 29690401, 31594495, 33583623, 35826373, 38084587, 40355477, 42603051, 44893225, 47240038, 49634641, 52131364, 54785123, 57566214, 60516069, 63630213, 66872280, 70032466, 73281595, 76559042, 79927775, 83271713, 86831349, 90335048, 93936908, 97529383, 101157411, 104732133, 108474590, 112510285, 116254382, 119352925, 122309945, 125330731, 128093855, 130971259, 133082588, 134592107, 135760830, 136423222, 136980286, 137712615, 138772681, 139954151]
    elif fof_no == 26:
        Ndm_ar = [239953, 256253, 273552, 291791, 310704, 330711, 351525, 373490, 396658, 420925, 446686, 473709, 502118, 531535, 562707, 595171, 629226, 665034, 702308, 741202, 782045, 826185, 874120, 922950, 972819, 1024629, 1079471, 1137098, 1197356, 1259641, 1325336, 1395293, 1469335, 1547957, 1631980, 1722746, 1819338, 1921095, 2029784, 2143472, 2265772, 2396648, 2537363, 2688239, 2850114, 3025771, 3208892, 3404213, 3615374, 3850242, 4111836, 4395373, 4714886, 5082421, 5526928, 6036303, 6478957, 6902310, 7312423, 7719669, 8117097, 8520474, 8909400, 9317755, 9735826, 10192305, 10711855, 11194842, 11657268, 12199848, 12661054, 13031014, 13357369, 13658016, 13947750, 14219049, 14425996, 14651231, 14949001, 15489930, 16353425, 16772844, 17034437, 17219395, 17366872, 17506581, 17736083, 17870634, 18008980, 18075210, 18098054, 18099020, 18099020, 18099020, 18099020, 18099020, 18099020, 18099020, 18099020, 18099020]

    Ndm_ar = np.array(Ndm_ar)/Ndm_ar[-1]
    mplcutoff = 10
    for (ix, rs) in enumerate(rpl): #ms is still radius
        pNm_ar = np.append(pNm_ar, len(pmmstar_f_ar[(pmmstar_all > mplcutoff) & (pmdist_f_ar < rs)]))
        pNs_ar = np.append(pNs_ar, len(psmstar_f_ar[(psmstar_all > mplcutoff) & (psdist_f_ar < rs)]))
        cNm_ar = np.append(cNm_ar, len(cmmstar_f_ar[(cmmstar_all > mplcutoff) & (cmdist_f_ar < rs)]))
        cNs_ar = np.append(cNs_ar, len(csmstar_f_ar[(csmstar_all > mplcutoff) & (csdist_f_ar < rs)]))
        Ntng_ar = np.append(Ntng_ar, len(psmstar_f_ar_tng[(psmstar_f_ar_tng > mplcutoff) & (psdist_f_ar < rs)]))
        # N_all_ar = np.append(N_all_ar, len(dfs1[dfs1['dist_f_ar']<rs]) + len(dfm1[dfm1['dist_f_ar']<rs]))
        # Ntng_ar = np.append()

    Ndm_ar = Ndm_ar * (cNm_ar[-1] + cNs_ar[-1]) #Normalizing the particles thing to cutoff model total temporarily
    fig, ax = plt.subplots(figsize = (6, 6.25))
    # ax.plot(rpl, Ns_ar, color = 'blue', label = 'Surviving')
    ax.plot(rpl, Ntng_ar, color = 'blue', label = r'TNG')
    ax.plot(rpl, pNs_ar + pNm_ar, color = 'red', label = r'power law ($>10\,\rm{M_\odot}$)')
    ax.plot(rpl, cNs_ar + cNm_ar, color = 'darkgreen', label = r'cutoff ($>10\,\rm{M_\odot}$)')
    # ax.plot(rpl, N_all_ar, color = 'purple', ls = '--', lw = 0.5, label = r'Model (all)')
    ax.plot(rpl, Ndm_ar, color = 'black', ls = '--', label = 'DM in TNG', alpha = 0.5)
    ax.legend(fontsize = 8)
    ax.set_xlabel('Distance from center (kpc)')
    ax.set_ylabel(r'$N(<r)$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # if porc == 'p':
    #     ax.set_title('z=0 from the model - power law')
    #     plt.tight_layout()
    #     plt.savefig(this_fof_plotppath + 'pl_cumulative_rad_dist.png')
    # elif porc == 'c':
    ax.set_title('FoF'+str(fof_no), fontsize = 14)
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'cumulative_rad_dist.png')
    plt.close()


plot_cumulative_rad_dist()

rpl = np.logspace(1, 3.2, 100) #r = m




