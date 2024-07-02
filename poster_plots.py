import pandas as pd 
import matplotlib.pyplot as plt
import IPython
import numpy as np 
from dwarf_plotting import plot_lg_virgo, plot_lg_vd, plot_lg_virgo_some
from Pradyumna_plot import plot_tng_subhalos #This is to plot the subhalos from TNG
import matplotlib
import illustris_python as il
from populating_stars import *
from subhalo_profiles import NFWProfile
import sys
import ast
import os
from colossus.cosmology import cosmology
from colossus.halo import concentration
from plot_sachi import plot_sachi
from scipy.optimize import fsolve
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.collections as mc
from testing_errani import get_rot_curve, get_rmxbyrmx0, get_vmxbyvmx0, get_mxbymx0, get_LbyL0, l10rbyrmx0_1by4_spl,l10rbyrmx0_1by2_spl, l10rbyrmx0_1by8_spl, l10rbyrmx0_1by16_spl, l10vbyvmx0_1by2_spl, l10vbyvmx0_1by4_spl, l10vbyvmx0_1by8_spl, l10vbyvmx0_1by16_spl, l10rbyrmx0_1by66_spl, l10rbyrmx0_1by250_spl, l10rbyrmx0_1by1000_spl, l10vbyvmx0_1by66_spl, l10vbyvmx0_1by250_spl, l10vbyvmx0_1by1000_spl
import matplotlib.ticker as ticker

cosmology.setCosmology('planck18')

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size' : 14}

plt.figure()
plt.close()
matplotlib.rc('font', **font)


def convert_to_float(value):
    try:
        if isinstance(value, float) or isinstance(value, int):
            return value
        blah = ast.literal_eval(value)
        if isinstance(blah, list):
            if len(blah) == 1:
                blah2 = float(blah[0])   
            elif len(blah) == 3:
                blah2 = np.array([float(blah[0]), float(blah[1]), float(blah[2])])
        else:
            blah2 = float(blah)        
        return blah2
    except (ValueError, SyntaxError):
        # print(f"Error converting {value}") #Looks like only inf values are not being converted, which is good
        return value 


def get_med_values(arr, cuts):
    '''
    This is going to take one big array of length cuts*whatever, breaks it into cuts parts and returns medians of the corresponding elements
    '''
    return np.median(np.reshape(arr, (cuts, -1)), axis = 0)

def get_quantiles(arr, cuts):
    '''
    This is going to take one big array of length cuts*whatever, breaks it into cuts parts and returns medians of the corresponding elements
    '''
    return np.quantile(np.reshape(arr, (cuts, -1)), axis = 0, q = [0.05, 0.95])


basePath = '/rhome/psadh003/bigdata/L35n2160TNG_fixed/output'
outpath  = '/rhome/psadh003/bigdata/tng50/output_files/'
plotpath = '/bigdata/saleslab/psadh003/tng50/output_plots/'
misc_path = '/bigdata/saleslab/psadh003/misc_files/'
ferrarese_data_path = '/bigdata/saleslab/psadh003/misc_files/Ferrarese_virgo_smf.csv'
venhola_data_path = '/bigdata/saleslab/psadh003/misc_files/Venhola_fornax_smf.csv'

'''
This part is to plot the NGVS data of virgo core set of satellites
'''
fdata = pd.read_csv(ferrarese_data_path, delimiter = ',')
fmstar = fdata['mstar']
fngal = fdata['ngal']
fngal_cum = np.cumsum(fngal)

'''
This part is to plot the Fornax mass function (everything inside virial radius)
'''
vdata = pd.read_csv(venhola_data_path, delimiter = ',')
vmstar = vdata['mstar']
vngal = vdata['ngal']
vngal_cum = np.cumsum(vngal)



fof_no = 210
fof_str = 'fof210'

this_fof0 = il.groupcat.loadSingle(basePath, 99, haloID = 0)
central_sfid_99_0 = this_fof0['GroupFirstSub']
rvir_fof0 = this_fof0['Group_R_Crit200']/0.6744
mvir_fof0 = this_fof0['Group_M_Crit200']*1e10/0.6744

this_fof1 = il.groupcat.loadSingle(basePath, 99, haloID = 1)
central_sfid_99_1 = this_fof1['GroupFirstSub']
rvir_fof1 = this_fof1['Group_R_Crit200']/0.6744
mvir_fof1 = this_fof1['Group_M_Crit200']*1e10/0.6744

this_fof2 = il.groupcat.loadSingle(basePath, 99, haloID = 2)
central_sfid_99_2 = this_fof2['GroupFirstSub']
rvir_fof2 = this_fof2['Group_R_Crit200']/0.6744
mvir_fof2 = this_fof2['Group_M_Crit200']*1e10/0.6744

this_fof_plotppath = '/bigdata/saleslab/psadh003/tng50/final_plots/poster_plots/'
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


    # subset1 = pdfs[(pdfs['dist_f_ar']<rvir_fof0) & (pdfs['fof'] == 0)]
    # subset2 = pdfs[(pdfs['dist_f_ar']<rvir_fof1) & (pdfs['fof'] == 1)]
    # subset3 = pdfs[(pdfs['dist_f_ar']<rvir_fof2) & (pdfs['fof'] == 2)]

    # pdfs = pd.concat([subset1, subset2, subset3], axis=0)
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
    # pdfs = pdfs[pdfs[]]


    psdist_f_ar = pdfs['dist_f_ar'].values
    pspos_f_ar = np.stack(np.array(pdfs['pos_f_ar'].values))
    psvel_f_ar = np.stack(np.array(pdfs['vel_f_ar'].values))
    # print(pspos_f_ar)
    # print(pspos_f_ar.shape)
    psxydist_f_ar = np.sqrt(np.sum(np.square(pspos_f_ar[:, :2]), axis = 1))
    psyzdist_f_ar = np.sqrt(np.sum(np.square(pspos_f_ar[:, 1:]), axis = 1))
    psxzdist_f_ar = np.sqrt(np.sum(np.square(pspos_f_ar[:, [0, 2]]), axis = 1))


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
    psfof = pdfs['fof'].values
    psmstar_if_ar = pdfs['mstar_if_ar'].values
    psvmax_if_ar = pdfs['vmax_if_ar'].values

    psmstar_all = np.zeros(len(psmmx_f_ar))
    psrh_all = np.zeros(len(psmmx_f_ar))
    psvd_all = np.zeros(len(psmmx_f_ar))
    res_ixs = np.where(psmstar_max_ar >= 5e6)[0] #These are the resolved indices
    unres_ixs = np.where(psmstar_max_ar < 5e6)[0] #These are the unresolved indices

    psmstar_max_all = np.zeros(len(psmmx_f_ar))
    psrh_max_all = np.zeros(len(psmmx_f_ar))

    psmstar_max_all[res_ixs] = psmstar_max_ar[res_ixs]
    psmstar_max_all[unres_ixs] = psmstar_max_pl_ar[unres_ixs]

    psrh_max_all[res_ixs] = psrh_max_ar[res_ixs]
    psrh_max_all[unres_ixs] = psrh_max_pl_ar[unres_ixs]

    psmstar_all[res_ixs] = psmstar_f_ar[res_ixs]
    psmstar_all[unres_ixs] = psmstar_f_pl_ar[unres_ixs]
    psrh_all[res_ixs] = psrh_f_ar[res_ixs]
    psrh_all[unres_ixs] = psrh_f_pl_ar[unres_ixs]
    psvd_all[res_ixs] = psvd_f_ar[res_ixs]
    psvd_all[unres_ixs] = psvd_f_pl_ar[unres_ixs]

    pssigma_all = 4.83 +21.57 -2.5 * np.log10(psmstar_all / (np.pi * (psrh_all*1e3) ** 2)) #This will be in mag/arcsec^2





if True: #This is a section for getting power law values of merged subhalos
    pdfm = pd.read_csv(outpath + fof_str + '_merged_evolved_wmbp_everything.csv', delimiter = ',')
    # print(pdfm.head(20))
    # print(f'Length before is: {len(pdfm)}')
    # pdfm = pdfm[pdfm['dist_f_ar'] != '']
    # print(f'Length after is: {len(pdfm)}')

    pdfm = pdfm.dropna(subset = ['dist_f_ar'])
    # print(pdfm['pos_f_ar'])
    pdfm = pdfm.applymap(convert_to_float)
    # pdfm = pdfm[pdfm['dist_f_ar']<rvir_fof]

    # subset1 = pdfm[(pdfm['dist_f_ar']<rvir_fof0) & (pdfm['fof'] == 0)]
    # subset2 = pdfm[(pdfm['dist_f_ar']<rvir_fof1) & (pdfm['fof'] == 1)]
    # subset3 = pdfm[(pdfm['dist_f_ar']<rvir_fof2) & (pdfm['fof'] == 2)]

    # pdfm = pd.concat([subset1, subset2, subset3], axis=0)
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
    pmpos_ar = np.stack(np.array(pdfm['pos_f_ar'].values))
    pmvel_f_ar = np.stack(np.array(pdfm['vel_f_ar'].values))
    # pmpos_ar
    # print(pmpos_ar)
    pmxydist_f_ar = np.sqrt(np.sum(np.square(pmpos_ar[:, :2]), axis = 1))
    pmyzdist_f_ar = np.sqrt(np.sum(np.square(pmpos_ar[:, 1:]), axis = 1))
    pmxzdist_f_ar = np.sqrt(np.sum(np.square(pmpos_ar[:, [0, 2]]), axis = 1))
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
    pmfof = pdfm['fof'].values

    pmmstar_max_all = np.zeros(len(pmmmx_f_ar))
    pmrh_max_all = np.zeros(len(pmmmx_f_ar))

    pmmstar_all = np.zeros(len(pmmmx_f_ar))
    pmrh_all = np.zeros(len(pmmmx_f_ar))
    pmvd_all = np.zeros(len(pmmmx_f_ar))
    res_ixs = np.where(pmmstar_max_ar >= 5e6)[0] #These are the resolved indices
    unres_ixs = np.where(pmmstar_max_ar < 5e6)[0] #These are the unresolved indices
    pmmstar_all[res_ixs] = pmmstar_f_ar[res_ixs]
    pmmstar_all[unres_ixs] = pmmstar_f_pl_ar[unres_ixs]

    pmmstar_max_all[res_ixs] = pmmstar_max_ar[res_ixs]
    pmmstar_max_all[unres_ixs] = pmmstar_max_pl_ar[unres_ixs]


    pmrh_max_all[res_ixs] = pmrh_max_ar[res_ixs]
    pmrh_max_all[unres_ixs] = pmrh_max_pl_ar[unres_ixs]



    pmrh_all[res_ixs] = pmrh_f_ar[res_ixs]
    pmrh_all[unres_ixs] = pmrh_f_pl_ar[unres_ixs]
    pmvd_all[res_ixs] = pmvd_f_ar[res_ixs]
    pmvd_all[unres_ixs] = pmvd_f_pl_ar[unres_ixs]

    pmsigma_all = 4.83 +21.57 -2.5 * np.log10(pmmstar_all / (np.pi * (pmrh_all*1e3) ** 2)) #This will be in mag/arcsec^2




#Once we have the power law data for surviving and merged subhalos, we should obtain their respective stellar masses etc checking if they are resolved




if True: #This is a section for getting cutoff values of survived subhalos
    cdfs = pd.read_csv(outpath + fof_str + '_surviving_evolved_everything.csv', delimiter = ',', low_memory=False)
    cdfs = cdfs.applymap(convert_to_float)


    # cdfs = cdfs[cdfs['dist_f_ar']<rvir_fof]
    # subset1 = cdfs[(cdfs['dist_f_ar']<rvir_fof0) & (cdfs['fof'] == 0)]
    # subset2 = cdfs[(cdfs['dist_f_ar']<rvir_fof1) & (cdfs['fof'] == 1)]
    # subset3 = cdfs[(cdfs['dist_f_ar']<rvir_fof2) & (cdfs['fof'] == 2)]

    # cdfs = pd.concat([subset1, subset2, subset3], axis=0)
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
    cspos_f_ar = np.stack(np.array(cdfs['pos_f_ar'].values))
    csvel_f_ar = np.stack(np.array(cdfs['vel_f_ar'].values))

    csxydist_f_ar = np.sqrt(np.sum(np.square(cspos_f_ar[:, :2]), axis = 1))
    csyzdist_f_ar = np.sqrt(np.sum(np.square(cspos_f_ar[:, 1:]), axis = 1))
    csxzdist_f_ar = np.sqrt(np.sum(np.square(cspos_f_ar[:, [0, 2]]), axis = 1))

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
    csfof = cdfs['fof'].values

    csmstar_all = np.zeros(len(csmmx_f_ar))
    csrh_all = np.zeros(len(csmmx_f_ar))
    csvd_all = np.zeros(len(csmmx_f_ar))
    res_ixs = np.where(csmstar_max_ar >= 5e6)[0] #These are the resolved indices
    unres_ixs = np.where(csmstar_max_ar < 5e6)[0] #These are the unresolved indices

    csmstar_max_all = np.zeros(len(csmmx_f_ar))
    csrh_max_all = np.zeros(len(csmmx_f_ar))

    csmstar_max_all[res_ixs] = csmstar_max_ar[res_ixs]
    csmstar_max_all[unres_ixs] = csmstar_max_co_ar[unres_ixs]

    csrh_max_all[res_ixs] = csrh_max_ar[res_ixs]
    csrh_max_all[unres_ixs] = csrh_max_co_ar[unres_ixs]

    csmstar_all[res_ixs] = csmstar_f_ar[res_ixs]
    csmstar_all[unres_ixs] = csmstar_f_co_ar[unres_ixs]
    csrh_all[res_ixs] = csrh_f_ar[res_ixs]
    csrh_all[unres_ixs] = csrh_f_co_ar[unres_ixs]
    csvd_all[res_ixs] = csvd_f_ar[res_ixs]
    csvd_all[unres_ixs] = csvd_f_co_ar[unres_ixs]

    cssigma_all = 4.83 +21.57 -2.5 * np.log10(csmstar_all / (np.pi * (csrh_all*1e3) ** 2)) #This will be in mag/arcsec^2


if True: #This is a section for getting cutoff values of merged subhalos
    cdfm = pd.read_csv(outpath + fof_str + '_merged_evolved_wmbp_everything.csv', delimiter = ',')
    cdfm = cdfm.dropna(subset = ['dist_f_ar'])
    cdfm = cdfm.applymap(convert_to_float)


    # cdfm = cdfm[cdfm['dist_f_ar']<rvir_fof]
    
    # subset1 = cdfm[(cdfm['dist_f_ar']<rvir_fof0) & (cdfm['fof'] == 0)]
    # subset2 = cdfm[(cdfm['dist_f_ar']<rvir_fof1) & (cdfm['fof'] == 1)]
    # subset3 = cdfm[(cdfm['dist_f_ar']<rvir_fof2) & (cdfm['fof'] == 2)]

    # cdfm = pd.concat([subset1, subset2, subset3], axis=0)

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
    cmpos_ar = np.stack(np.array(cdfm['pos_f_ar'].values))
    cmvel_f_ar = np.stack(np.array(cdfm['vel_f_ar'].values))


    cmxydist_f_ar = np.sqrt(np.sum(np.square(cmpos_ar[:, :2]), axis = 1))
    cmyzdist_f_ar = np.sqrt(np.sum(np.square(cmpos_ar[:, 1:]), axis = 1))
    cmxzdist_f_ar = np.sqrt(np.sum(np.square(cmpos_ar[:, [0, 2]]), axis = 1))


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
    cmfof = cdfm['fof'].values

    cmmstar_max_all = np.zeros(len(cmmmx_f_ar))
    cmrh_max_all = np.zeros(len(cmmmx_f_ar))

    cmmstar_all = np.zeros(len(cmmmx_f_ar))
    cmrh_all = np.zeros(len(cmmmx_f_ar))
    cmvd_all = np.zeros(len(cmmmx_f_ar))
    res_ixs = np.where(cmmstar_max_ar >= 5e6)[0] #These are the resolved indices
    unres_ixs = np.where(cmmstar_max_ar < 5e6)[0] #These are the unresolved indices
    cmmstar_all[res_ixs] = cmmstar_f_ar[res_ixs]
    cmmstar_all[unres_ixs] = cmmstar_f_co_ar[unres_ixs]

    cmmstar_max_all[res_ixs] = cmmstar_max_ar[res_ixs]
    cmmstar_max_all[unres_ixs] = cmmstar_max_co_ar[unres_ixs]

    cmrh_max_all[res_ixs] = cmrh_max_ar[res_ixs]
    cmrh_max_all[unres_ixs] = cmrh_max_co_ar[unres_ixs]

    cmrh_all[res_ixs] = cmrh_f_ar[res_ixs]
    cmrh_all[unres_ixs] = cmrh_f_co_ar[unres_ixs]
    cmvd_all[res_ixs] = cmvd_f_ar[res_ixs]
    cmvd_all[unres_ixs] = cmvd_f_co_ar[unres_ixs]

    cmsigma_all = 4.83 +21.57 -2.5 * np.log10(cmmstar_all / (np.pi * (cmrh_all*1e3) ** 2)) #This will be in mag/arcsec^2



# fof_no = int(sys.argv[1])
# fof_str = 'fof' + str(fof_no)

# IPython.embed()

# print('Data import is a success!')
label_font = 24
legend_size = 14



def niusha_plot(rproj = 300, zmax = 50):
    '''
    This is to make the most good looking plot for the paper
    Planning to plot the distribution of the FoF0 as given by TNG based on Niusha's code

    We are making an xy projection plot

    To that, we add points of given position from the power law and cutoff models
    '''
    r_vir_cen = rvir_fof0
    plt.rcParams['axes.facecolor'] = 'black'
    fig, ax = plt.subplots(1,3, figsize = (13, 13/3.), sharex = True, sharey = True) #This line has been moved to final_plots.py
    plt.subplots_adjust(wspace = 0.02)
    for i,a in enumerate(ax.flat): #We are just formatting the ticks here
        a.tick_params(length = 8, width = 2, direction = 'inout')
        a.xaxis.tick_bottom()
        a.yaxis.tick_left()
    

    cat_sub = il.groupcat.loadSubhalos(basePath, 99, fields = ['SubhaloGrNr','SubhaloMassType', 'SubhaloMassInRadType', 'SubhaloMassInHalfRadType','SubhaloHalfmassRadType', 'SubhaloPos'])

    pos_sub = cat_sub['SubhaloPos']
    r_h_sub = cat_sub['SubhaloHalfmassRadType'][:,4] * 1/0.6774
    grnr = cat_sub['SubhaloGrNr']
    mstr_sub = cat_sub['SubhaloMassType'][:,4] * 1e10/0.6774
    mdm_sub = cat_sub['SubhaloMassType'][:,1] * 1e10/0.6774

    r_h_cen = r_h_sub[central_sfid_99_0]
    mdm_cen = mdm_sub[central_sfid_99_0]

    # mstr_sub = cat_sub['SubhaloMassType'][:,4] * 1e10/0.6774

    sfid = np.arange(len(grnr))

    outpath  = '/rhome/psadh003/bigdata/tng50/output_files/'
    result_satellite = np.load(outpath + 'fof0_plot.npy')

    x_pos_satellite = result_satellite[:,0]
    y_pos_satellite = result_satellite[:,1]
    z_pos_satellite = result_satellite[:,2]
    mass_satellite = result_satellite[:,3]

    print(f'x positions: {len(x_pos_satellite)}')
    print(f'x positions: {x_pos_satellite}')

    r_satellite = np.sqrt((x_pos_satellite**2)+(y_pos_satellite**2)+(z_pos_satellite**2))

    # aux = r_satellite < r_vir_cen
    # Testing different definitions of radii
    aux = (r_satellite > 2* r_h_cen) & (np.sqrt((x_pos_satellite**2)+(y_pos_satellite**2)) < rproj) & (np.absolute(z_pos_satellite) < zmax)

    x_pos_satellite = x_pos_satellite[aux]
    y_pos_satellite = y_pos_satellite[aux]
    z_pos_satellite = z_pos_satellite[aux]
    mass_satellite = mass_satellite[aux]

    print(len(x_pos_satellite))
    print('max min mstar pp:', mass_satellite.max(), mass_satellite.min())
    mass_bin_sat, xedges_sat, yedges_sat, binnumber = stats.binned_statistic_2d(x_pos_satellite, y_pos_satellite, mass_satellite, statistic='sum', bins=300) #We are adding the mass of all the satellites in a given bin


    print('shape of mass_bin_sat in pp: ', mass_bin_sat.shape)
    # XX_sat, YY_sat = np.meshgrid(xedges_sat, yedges_sat) #This maybe just generates the cordinates for the plot

    max_mass = mass_bin_sat.max()/10
    min_mass = 50000 * 8/5

    extent = xedges_sat[0], xedges_sat[-1], yedges_sat[0], yedges_sat[-1]

    for i in range(3):
        im3 = ax[i].imshow(mass_bin_sat.T, cmap = 'inferno', norm = 'log', vmin = 1e4, vmax = max_mass, interpolation='nearest', extent = extent, origin='lower', zorder = 100)

        # divider = make_axes_locatable(ax[i])
        # cax = divider.append_axes('top', size='3%', pad=0.05)
        # cb = fig.colorbar(im3, cax=cax, orientation="horizontal")
        # cb.set_label(r'$\Sigma_{\rm*, \, sat}$ [M$_{\odot}$/kpc$^{2}$]', fontsize = 17)
        # cb.ax.xaxis.set_label_position('top')
        # cb.ax.xaxis.set_ticks_position('top')

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im3, cax=cbar_ax)
    cbar.set_label(r'$\Sigma_{\rm*, \, sat}$ [M$_{\odot}$/kpc$^{2}$]', fontsize = label_font)
    #Following lines are for drawing circles, brilliant!

    angle = np.linspace( 0 , 2 * np.pi , 150 )
    x_vir = rproj * np.cos( angle )
    y_vir = rproj * np.sin( angle )

    x_r_h_str = 2 * r_h_cen * np.cos( angle )
    y_r_h_str = 2 * r_h_cen * np.sin( angle )

    for i in range (3):
        if rproj == rvir_fof0:
            ax[i].plot(x_vir, y_vir, color = 'orange', lw = 2, label = '$r_{vir}$')
            # ax[i].plot(x_r_h_str, y_r_h_str, color = 'magenta', lw = 2, label = r'$2 \times r_{half \,\, mass_{*}}$')
        else:
            ax[i].plot(x_vir, y_vir, color = 'orange', lw = 2, label = '$r = $' + str(rproj) + ' kpc')
        ax[i].plot(x_r_h_str, y_r_h_str, color = 'magenta', lw = 2, label = r'$2 \times r_{half \,\, mass_{*}}$')

        ax[i].set_aspect(1)

        ax[i].set_xlim([-rproj, rproj])
        ax[i].set_ylim([-rproj, rproj])

        ax[i].xaxis.set_tick_params(labelsize=12)
        ax[i].yaxis.set_tick_params(labelsize=12)


    ax[0].set_xlabel('kpc', fontsize = label_font)
    ax[0].set_ylabel('kpc', fontsize = label_font)
    ax[1].set_xlabel('kpc', fontsize = label_font)
    ax[2].set_xlabel('kpc', fontsize = label_font)

    ax[1].set_xticks([-200, -100, 0, 100, 200])


    ps_cond = (np.sqrt(pspos_f_ar[:,0]**2 + pspos_f_ar[:,1]**2) < rproj) & (psfof == 0) & (np.absolute(pspos_f_ar[:,2]) < zmax)  &  (np.sqrt(pspos_f_ar[:,0]**2 + pspos_f_ar[:,1]**2 + pspos_f_ar[:,2]**2) > 2*0)
    # 
    ps_x = pspos_f_ar[:, 0][ps_cond]
    ps_y = pspos_f_ar[:, 1][ps_cond]
    ps_mstar = psmstar_all[ps_cond]
    ps_rh = psrh_all[ps_cond]

    pm_cond = (np.sqrt(pmpos_ar[:,0]**2 + pmpos_ar[:,1]**2) < rproj) & (pmfof == 0)  & (np.absolute(pmpos_ar[:,2]) < zmax)   &  (np.sqrt(pmpos_ar[:,0]**2 + pmpos_ar[:,1]**2 + pmpos_ar[:,2]**2) > 2*0)
    #
    pm_x = pmpos_ar[:, 0][pm_cond]
    pm_y = pmpos_ar[:, 1][pm_cond]
    pm_mstar = pmmstar_all[pm_cond]
    pm_rh = pmrh_all[pm_cond]

    pl_x = np.append(ps_x, pm_x)
    pl_y = np.append(ps_y, pm_y)
    pl_mstar = np.append(ps_mstar, pm_mstar)
    pl_rh = np.append(ps_rh, pm_rh)

    cs_cond = (np.sqrt(cspos_f_ar[:, 0]**2 + cspos_f_ar[:, 1]**2) < rproj) & (csfof == 0) & (np.absolute(cspos_f_ar[:,2]) < zmax) & (np.sqrt(cspos_f_ar[:, 0]**2 + cspos_f_ar[:, 1]**2 + cspos_f_ar[:, 2]**2) > 2*0)
    cs_x = cspos_f_ar[:, 0][cs_cond]
    cs_y = cspos_f_ar[:, 1][cs_cond]
    cs_mstar = csmstar_all[cs_cond]
    cs_rh = csrh_all[cs_cond]

    cm_cond = (np.sqrt(cmpos_ar[:, 0]**2 + cmpos_ar[:, 1]**2) < rproj) & (cmfof == 0) & (np.absolute(cmpos_ar[:,2]) < zmax) & (np.sqrt(cmpos_ar[:, 0]**2 + cmpos_ar[:, 1]**2 + cmpos_ar[:, 2]**2) > 2*0)
    cm_x = cmpos_ar[:, 0][cm_cond]
    cm_y = cmpos_ar[:, 1][cm_cond]
    cm_mstar = cmmstar_all[cm_cond]
    cm_rh = cmrh_all[cm_cond]

    cl_x = np.append(cs_x, cm_x)
    cl_y = np.append(cs_y, cm_y)
    cl_mstar = np.append(cs_mstar, cm_mstar)
    cl_rh = np.append(csrh_all, cmrh_all)
    

    print('len pl_x', len(pl_x))
    print('max min mstar fp', min(pl_mstar), max(pl_mstar))
    #This is the 2d histogram for power law model
    mass_bin_sat, xedges_sat, yedges_sat, binnumber = stats.binned_statistic_2d(pl_x, pl_y, pl_mstar, statistic='sum', bins=500) #We are adding the mass of all the satellites in a given bin

    print('mass_bin_sat size in fp: ', mass_bin_sat.shape)
    # XX_sat, YY_sat = np.meshgrid(xedges_sat, yedges_sat) #This maybe just generates the cordinates for the plot

    max_mass = mass_bin_sat.max()/10
    min_mass = 50000 * 8/5 

    print('min and max of mass bins: ', min_mass, max_mass)

    extent = xedges_sat[0], xedges_sat[-1], yedges_sat[0], yedges_sat[-1]
    # im3 = ax[1].imshow(mass_bin_sat.T, color = 'white', norm = 'log', vmin = 1e4, vmax = max_mass, interpolation='nearest', extent = extent, origin='lower')
    # im3 = ax[1].imshow(mass_bin_sat.T, cmap = 'inferno', norm = 'log', vmin = 1e4, vmax = max_mass, interpolation='nearest', extent = extent, origin='lower')

    # divider = make_axes_locatable(ax[1])
    # cax = divider.append_axes('top', size='3%', pad=0.05)
    # cb = fig.colorbar(im3, cax=cax, orientation="horizontal")
    # cb.set_label(r'$\Sigma_{\rm*, \,{power law}}$ [M$_{\odot}$/kpc$^{2}$]', fontsize = 17)
    # cb.ax.xaxis.set_label_position('top')
    # cb.ax.xaxis.set_ticks_position('top')


    #This is the 2d histogram for cutoff model
    mass_bin_cen, xedges_cen, yedges_cen, binnumber = stats.binned_statistic_2d(cl_x, cl_y, cl_mstar, statistic='sum', bins=500) #We are adding the mass of all the satellites in a given bin

    # XX_cen, YY_cen = np.meshgrid(xedges_cen, yedges_cen) #This maybe just generates the cordinates for the plot

    max_mass = mass_bin_cen.max()/10
    min_mass = 50000 * 8/5

    print('min and max of mass bins: ', min_mass, max_mass)

    extent = xedges_cen[0], xedges_cen[-1], yedges_cen[0], yedges_cen[-1]

    collection2 = mc.CircleCollection(20*cs_rh, offsets=np.column_stack((cs_x, cs_y)), transOffset=ax[2].transData, zorder = 200, 
                                      facecolors='none',  edgecolors='white', linewidths = 0.5)
    ax[2].add_collection(collection2)

    collection = mc.CircleCollection(20*pl_rh, offsets=np.column_stack((pl_x, pl_y)), transOffset=ax[1].transData, zorder = 200, 
                                      facecolors='none',  edgecolors='white', linewidths = 0.5)
    ax[1].add_collection(collection) 

    # ax[0].text(0.05, 0.95, 'TNG50-1', color = 'white', transform=ax[0].transAxes, fontsize = label_font)  
    # ax[1].text(0.05, 0.95, 'Power law', color = 'white', transform=ax[1].transAxes, fontsize = label_font)
    # ax[2].text(0.05, 0.95, 'Cutoff', color = 'white', transform=ax[2].transAxes, fontsize = label_font) 

    ax[0].set_title('TNG50-1', color = 'black', transform=ax[0].transAxes, fontsize = label_font)
    ax[1].set_title('Power law', color = 'black', transform=ax[1].transAxes, fontsize = label_font)
    ax[2].set_title('Cutoff', color = 'black', transform=ax[2].transAxes, fontsize = label_font)



    fig.savefig(this_fof_plotppath +'niusha_plot.png', bbox_inches='tight', dpi = 3000)
    return None

niusha_plot()
plt.rcParams['axes.facecolor'] = 'white'





def plot_radial_density_dist_3panel():
    '''
    This is to plot the radial density of subhalos in 3 panels for FoF0 in three mass ranges
    '''
    fig, axs = plt.subplots(1, 3, figsize = (21, 7))
    Ndm_ar =[721148, 778322, 839804, 905710, 976898, 1054137, 1137501, 1227676, 1325214, 1430877, 1545758, 1669730, 1805234, 1953330, 2114837, 2291374, 2484727, 2697632, 2929798, 3184435, 3461447, 3763454, 4093457, 4451911, 4839639, 5261067, 5717234, 6218030, 6759960, 7335930, 7958128, 8628601, 9351223, 10124871, 10954252, 11843815, 12803150, 13852694, 14955612, 16130128, 17390800, 18724037, 20147057, 21672913, 23303629, 25059909, 26945144, 28965678, 31153056, 33453105, 35883715, 38490363, 41284274, 44311781, 47556207, 50917974, 54416144, 58216056, 62174887, 66291888, 70515423, 74848372, 79404305, 84157527, 89209147, 94471028, 99857920, 105317116, 111028328, 116939549, 123289439, 130030026, 137275327, 145069112, 153548719, 161907244, 169905804, 177886839, 185830162, 194592334, 203383663, 212695914, 222350306, 232159462, 242040951, 251514511, 260202731, 268769952, 278511000, 289919552, 300273807, 309815832, 319892274, 331241231, 342176869, 352467656, 365061316, 375410309, 383675808, 390894527]
    Mstar_ar = [1215764100000.0, 1254275500000.0, 1293225400000.0, 1332839300000.0, 1373199400000.0, 1414346300000.0, 1456465800000.0, 1499649100000.0, 1543340700000.0, 1587188200000.0, 1631277400000.0, 1676285400000.0, 1723836500000.0, 1770465500000.0, 1818711300000.0, 1868407000000.0, 1921189400000.0, 1975527500000.0, 2031582000000.0, 2091765200000.0, 2152079200000.0, 2214667500000.0, 2279144400000.0, 2346879800000.0, 2415670000000.0, 2483960200000.0, 2560993300000.0, 2639518500000.0, 2717090000000.0, 2789425800000.0, 2865190000000.0, 2938990200000.0, 3014576300000.0, 3088882300000.0, 3166544000000.0, 3234522200000.0, 3303603000000.0, 3403499700000.0, 3475743700000.0, 3541706200000.0, 3611283200000.0, 3671716800000.0, 3740037800000.0, 3803736700000.0, 3863171600000.0, 3924207300000.0, 3984920700000.0, 4047178600000.0, 4116152300000.0, 4179430200000.0, 4235346300000.0, 4302655000000.0, 4361331100000.0, 4432361700000.0, 4508296400000.0, 4587731300000.0, 4649099700000.0, 4724291500000.0, 4775264400000.0, 4826232000000.0, 4874225000000.0, 4915582000000.0, 4958145500000.0, 4994681500000.0, 5043397300000.0, 5096712000000.0, 5136737400000.0, 5176501500000.0, 5222220400000.0, 5250391500000.0, 5282301000000.0, 5315951000000.0, 5351944000000.0, 5401151000000.0, 5537075300000.0, 5626221000000.0, 5672295500000.0, 5750817000000.0, 5774166000000.0, 5900201400000.0, 5929926000000.0, 5960066700000.0, 5986782300000.0, 6014626000000.0, 6088623000000.0, 6145640500000.0, 6163691000000.0, 6178112000000.0, 6282066000000.0, 6651228000000.0, 6749657000000.0, 6765084700000.0, 6809034700000.0, 6949706000000.0, 6986958000000.0, 7002995600000.0, 7282621400000.0, 7327393000000.0, 7357744000000.0, 7428319000000.0]

    rpl = np.logspace(1, 3.2, 100)
    rho_dm = Ndm_ar = np.array(Ndm_ar)*4.5e5 #This would be the density in Msun/kpc^3
        # Ndm_ar = Ndm_ar/Ndm_ar[-1]
    rho_star = Mstar_ar = np.array(Mstar_ar) #This would be the density in Msun/kpc^3
    rho_dm = Ndm_ar/((4/3.) * np.pi * rpl**3)
    rho_star = Mstar_ar/((4/3.) * np.pi * rpl**3)  
    Nstar_and_dm = (rho_dm + rho_star) / (rho_dm[-1] + rho_star[-1])

    Ndm_ar = Ndm_ar / (rho_dm[-1] + rho_star[-1]) /((4/3.) * np.pi * rpl**3)
    Mstar_ar = Mstar_ar / (rho_dm[-1] + rho_star[-1]) /((4/3.) * np.pi * rpl**3)

    

    for jx in range(3):
        pNm_ar = np.zeros(0) #shmf for merged subhalos
        pNs_ar = np.zeros(0) #shmf for surviving subhalos 
        cNm_ar = np.zeros(0) #shmf for merged subhalos
        cNs_ar = np.zeros(0) #shmf for surviving subhalos

        N_all_ar = np.zeros(0) #This is for all the subhalos inside virial radius
        Ntng_ar = np.zeros(0)
              


        if jx == 0:
            mplcutoff = 10**8.5
            mmaxcutoff = 10**15
            axs[jx].set_title(r'$M_{\star} > 10^{8.5}\,\rm{M_\odot}$', fontsize = label_font)
        elif jx == 1:
            mplcutoff = 10**5
            mmaxcutoff = 10**8.5
            axs[jx].set_title(r'$10^{5} < M_{\star} < 10^{8.5}\,\rm{M_\odot}$', fontsize = label_font)
        elif jx ==2:
            mplcutoff = 10**1
            mmaxcutoff = 10**5
            axs[jx].set_title(r'$M_{\star} < 10^{5}\,\rm{M_\odot}$', fontsize = label_font)
        rpl2 = np.logspace(1, 3.2, 20)
        for (ix, rs) in enumerate(rpl2): #rs is still radius, has nothing to do with rs of NFW profile
            pNm_ar = np.append(pNm_ar, len(pmmstar_f_ar[(pmmstar_all > mplcutoff) & (pmmstar_all < mmaxcutoff) & (pmdist_f_ar < rs) & (pmfof == 0)]))
            pNs_ar = np.append(pNs_ar, len(psmstar_f_ar[(psmstar_all > mplcutoff) & (psmstar_all < mmaxcutoff) & (psdist_f_ar < rs) & (psfof == 0)]))
            cNm_ar = np.append(cNm_ar, len(cmmstar_f_ar[(cmmstar_all > mplcutoff) & (cmmstar_all < mmaxcutoff) & (cmdist_f_ar < rs) & (cmfof == 0)]))
            cNs_ar = np.append(cNs_ar, len(csmstar_f_ar[(csmstar_all > mplcutoff) & (csmstar_all < mmaxcutoff) & (csdist_f_ar < rs) & (csfof == 0)]))
            Ntng_ar = np.append(Ntng_ar, len(psmstar_f_ar_tng[(psmstar_f_ar_tng > mplcutoff) & (psmstar_f_ar_tng < mmaxcutoff)  & (psdist_f_ar < rs)]))

        

        
        Ntng_ar = Ntng_ar/Ntng_ar[-1] * (rpl2[-1]**3 / rpl2**3)
        pN_ar = (pNs_ar + pNm_ar) / (pNs_ar[-1] + pNm_ar[-1]) *  (rpl2[-1]**3 / rpl2**3)
        cN_ar = (cNs_ar + cNm_ar) / (cNs_ar[-1] + cNm_ar[-1]) *  (rpl2[-1]**3 / rpl2**3)
        

        axs[jx].plot(rpl2, Ntng_ar, 'bo-', label = r'TNG')
        axs[jx].plot(rpl2, pN_ar, color = 'red', label = r'Power law', lw  = 2.5)
        axs[jx].plot(rpl2, cN_ar, color = 'darkgreen', label = r'Cutoff', lw = 2.5)

        axs[jx].plot(rpl, Ndm_ar, color = 'black', ls = '--', label = 'DM in TNG', alpha = 0.5)
        axs[jx].plot(rpl, Mstar_ar, color = 'black', ls = ':', label = 'Stars in TNG', alpha = 0.5)
        axs[jx].plot(rpl, Nstar_and_dm, color = 'black', ls = '-', label = 'Stars and DM in TNG', alpha = 0.5)
        if jx == 0:
            axs[jx].legend(fontsize = legend_size)
            axs[jx].set_ylabel(r'Number density (<r) $[\rm{kpc^{-3}}]$', fontsize = label_font)
        axs[jx].set_xlabel('Cluster-centric distance (kpc)', fontsize = label_font)
        axs[jx].set_ylim(bottom = 0.5)
        
        axs[jx].set_xscale('log')
        axs[jx].set_yscale('log')
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'radial_density_dist_3panel.png', dpi = 1800)

    return

plot_radial_density_dist_3panel()






def plot_abundance_matching():
    '''
    This plot will have a dual panel for Abundance matching relation and Size-stellar mass relation that we use
    '''
    fig, ax = plt.subplots(1, 1, figsize = (7.7, 7.7))

    psmstar_if_ar[psmstar_if_ar == 0] = 1e3
    lvmax_pl = np.linspace(0.1, np.log10(600), 100)
    ax.scatter(np.log10(psvmax_if_ar), np.log10(psmstar_if_ar), color = 'gray', marker = 'o', s = 1, alpha = 0.1, label = 'TNG at infall')
    ax.plot(lvmax_pl, get_mstar_pl(lvmax_pl), color = 'red', label = 'Power law', lw = 2)
    ax.fill_between(lvmax_pl, get_mstar_pl(lvmax_pl) - get_scatter(lvmax_pl), get_mstar_pl(lvmax_pl) + get_scatter(lvmax_pl), color = 'red', alpha = 0.1)
    ax.plot(lvmax_pl, get_mstar_co(lvmax_pl), color = 'darkgreen', label = 'Cutoff', ls = '--', lw = 2)
    ax.fill_between(lvmax_pl, get_mstar_co(lvmax_pl) - get_scatter(lvmax_pl), get_mstar_co(lvmax_pl) + get_scatter(lvmax_pl), color = 'darkgreen', alpha = 0.1)
    ax.set_ylabel(r'$\log M_{\rm{star}}(M_\odot)$', fontsize = label_font)
    ax.set_xlabel(r'$\log V_{\rm{max}}(\rm{km/s})$', fontsize = label_font)
    ax.axhline(np.log10(5e6), ls = ':', color = 'gray', alpha = 0.4)
    ax.text(0.31, 5.8, 'Unresolved in size\nand mass', fontsize = legend_size)
    ax.axhline(np.log10(1e8), ls = ':', color = 'gray', alpha = 0.4)
    ax.text(0.31, 7, 'Unresolved in size,\nresolved in mass', fontsize = legend_size)
    ax.text(0.31, 9, 'Resolved in size and mass', fontsize = legend_size)

    ax.tick_params(axis='y', which = 'both', left=True, right=True, direction = 'in')
    ax.tick_params(axis='x', which = 'both', direction = 'in', top = True)
    ax.set_ylim(bottom = np.log10(7e1), top = 12)
    ax.set_xlim(left = 0.25)
    ax.legend(fontsize = legend_size, loc = 'lower right') 

    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'abundance_matching.png', dpi = 720)

    return None

plot_abundance_matching()





def tidal_tracks():
    '''
    This is to plot the tidal tracks of the satellites based on Errani 22 or the plots given by Rapha
    '''
    
    fig, (ax, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize = (7.7,7.7), sharex = True)
    fpl = np.linspace(-6.4, 0, 100)

        
    def get_l10rhbyrmx0(fpl_ar, Rh0byrmx0):
        l10rhbyrmx0_ar = np.zeros(0)

        for fpl in fpl_ar:
            if (fpl > -5) and (Rh0byrmx0 == 1/66 or Rh0byrmx0 == 1/250):
                if Rh0byrmx0 == 1/66:
                    l10rhbyrmx0_ar = np.append(l10rhbyrmx0_ar, l10rbyrmx0_1by66_spl(fpl))
                elif Rh0byrmx0 == 1/250:
                    l10rhbyrmx0_ar = np.append(l10rhbyrmx0_ar, l10rbyrmx0_1by250_spl(fpl))
            elif (fpl > -2.5) and (Rh0byrmx0 in [1/2, 1/4, 1/8, 1/16]):
                if Rh0byrmx0 == 1/2:
                    l10rhbyrmx0_ar = np.append(l10rhbyrmx0_ar, l10rbyrmx0_1by2_spl(fpl))
                elif Rh0byrmx0 == 1/4:
                    l10rhbyrmx0_ar = np.append(l10rhbyrmx0_ar, l10rbyrmx0_1by4_spl(fpl))
                elif Rh0byrmx0 == 1/8:
                    l10rhbyrmx0_ar = np.append(l10rhbyrmx0_ar, l10rbyrmx0_1by8_spl(fpl))
                elif Rh0byrmx0 == 1/16:
                    l10rhbyrmx0_ar = np.append(l10rhbyrmx0_ar, l10rbyrmx0_1by16_spl(fpl))
            elif (fpl > -6.4) and (Rh0byrmx0 == 1/1000):
                    l10rhbyrmx0_ar = np.append(l10rhbyrmx0_ar, l10rbyrmx0_1by1000_spl(fpl))
            else:
                l10rhbyrmx0_ar = np.append(l10rhbyrmx0_ar, np.log10(get_rmxbyrmx0(10**fpl)))
        return l10rhbyrmx0_ar

        # return l10rhbyrmx0
    ax.plot(fpl, get_l10rhbyrmx0(fpl, 1/2), c = 'r', label = r'$R_{\rm{h0}}/r_{\rm{mx0}} = 1/2$')
    ax.plot(fpl, get_l10rhbyrmx0(fpl, 1/4), c = 'orange', label = r'$R_{\rm{h0}}/r_{\rm{mx0}} = 1/4$')
    ax.plot(fpl, get_l10rhbyrmx0(fpl, 1/8), c = 'skyblue', label = r'$R_{\rm{h0}}/r_{\rm{mx0}} = 1/8$')
    ax.plot(fpl, get_l10rhbyrmx0(fpl, 1/16), c = 'darkblue', label = r'$R_{\rm{h0}}/r_{\rm{mx0}} = 1/16$')
    ax.plot(fpl, get_l10rhbyrmx0(fpl, 1/66), c = 'purple', ls = ':')
    ax.plot(fpl, get_l10rhbyrmx0(fpl, 1/250), c = 'limegreen', ls = ':')
    ax.plot(fpl, get_l10rhbyrmx0(fpl, 1/1000), c = 'darkgreen', ls = ':')

    ax.plot(fpl, np.log10(get_rmxbyrmx0(10**fpl)), c = 'black', ls = '--')
    ax.annotate(r'$r_{\rm{mx}}$', xy = (fpl[-20], np.log10(get_rmxbyrmx0(10**fpl))[-20]), xytext = (fpl[-20], 0.8* np.log10(get_rmxbyrmx0(10**fpl))[-20]), 
                rotation=30, color='black', fontsize=legend_size, rotation_mode='anchor')

    # ax.set_xlabel(r'$\log M_{\rm{mx}}/M_{\rm{mx0}}$', fontsize = 14)
    ax.set_ylabel(r'$\log R_{\rm{h}}/r_{\rm{mx0}}$', fontsize = label_font)
    ax.legend(fontsize = legend_size)
    ax.tick_params(axis='y', which = 'both', left=True, right=True, direction = 'in')
    ax.tick_params(axis='x', which = 'both', direction = 'in')

    # print('Why are you like this!', get_LbyL0(fpl, 1/2))
    ax2.plot(fpl, np.log10(get_LbyL0(10 ** fpl, 1/2)), c = 'r')
    ax2.plot(fpl, np.log10(get_LbyL0(10 ** fpl, 1/4)), c = 'orange')
    ax2.plot(fpl, np.log10(get_LbyL0(10 ** fpl, 1/8)), c = 'skyblue')
    ax2.plot(fpl, np.log10(get_LbyL0(10 ** fpl, 1/16)), c = 'darkblue')
    ax2.plot(fpl, np.log10(get_LbyL0(10 ** fpl, 1/66)), c = 'purple', label = r'$R_{\rm{h0}}/r_{\rm{mx0}} = 1/66$', ls = ':')
    ax2.plot(fpl, np.log10(get_LbyL0(10 ** fpl, 1/250)), c = 'limegreen', label = r'$R_{\rm{h0}}/r_{\rm{mx0}} = 1/250$', ls = ':')
    ax2.plot(fpl, np.log10(get_LbyL0(10 ** fpl, 1/1000)), c = 'darkgreen', label = r'$R_{\rm{h0}}/r_{\rm{mx0}} = 1/1000$', ls = ':')

    ax2.set_xlabel(r'$\log M_{\rm{mx}}/M_{\rm{mx0}}$', fontsize = label_font)
    ax2.set_ylabel(r'$\log M_{\rm{star}}/M_{\rm{star,0}}$', fontsize = label_font)
    ax2.set_ylim(bottom = -3)
    ax2.legend(fontsize = legend_size)
    ax2.tick_params(axis='y', which = 'both', left=True, right=True, direction = 'in')
    ax2.tick_params(axis='x', which = 'both', direction = 'in')


    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'tidal_tracks.png', dpi = 720)

    return None

tidal_tracks()





def plot_mass_fn_completeness():
    '''
    This is to plot the mass function with different surface brightness limits along with completeness in the bottom panel
    '''
    fig, (ax, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize = (7, 7), gridspec_kw={'height_ratios':[2, 1]}, sharex=True)
    mstarpl = np.logspace(2, 11, 100)

    sigma_pl_ar = [35, 28, 24] #This is the surface brightness cuts that we would be using
    ls_ar = ['-', '--', '-.']
    for jx in range(len(sigma_pl_ar)): #+2 because we also need to plot without any surface brightness limits
        pNm_ar = np.zeros(0) #shmf for merged subhalos
        pNs_ar = np.zeros(0) #shmf for surviving subhalos 
        cNm_ar = np.zeros(0) #shmf for merged subhalos
        cNs_ar = np.zeros(0) #shmf for surviving subhalos 
        Ntng_ar = np.zeros(0)
        for (ix, ms) in enumerate(mstarpl):
            pNm_ar = np.append(pNm_ar, len(pmmstar_f_ar[(pmmstar_all > ms) & (pmfof == 0) & (pmsigma_all < sigma_pl_ar[jx])]))
            pNs_ar = np.append(pNs_ar, len(psmstar_f_ar[(psmstar_all > ms) &  (psfof == 0) & (pssigma_all < sigma_pl_ar[jx])]))
            cNm_ar = np.append(cNm_ar, len(cmmstar_f_ar[(cmmstar_all > ms) & (cmfof == 0) & (cmsigma_all < sigma_pl_ar[jx])]))
            cNs_ar = np.append(cNs_ar, len(csmstar_f_ar[(csmstar_all > ms) & (csfof == 0) & (cssigma_all < sigma_pl_ar[jx])]))
            # Ntng_ar = np.append(Ntng_ar, len(psmstar_f_ar_tng[psmstar_f_ar_tng > ms]))

        # fig, ax = plt.subplots(figsize = (6, 6.25))
        
        # ax.plot(mstarpl, Ns_ar, color = 'darkgreen', label = 'Model Surviving', alpha = 0.5)
        if jx == -1: #NULL
            ax.plot(mstarpl, cNs_ar + cNm_ar, color = 'darkgreen', ls = ls_ar[jx], label = r'Cutoff all', alpha = 0.5)
            ax.plot(mstarpl, pNs_ar + pNm_ar, color = 'red', ls = ls_ar[jx], label = r'Power all', alpha = 0.5)
        else:
            ax.plot(mstarpl, cNs_ar + cNm_ar, color = 'darkgreen', ls = ls_ar[jx], label = r'Cutoff $\Sigma < $'+str(sigma_pl_ar[jx]), alpha = 0.5, lw = 2.2)
            ax.plot(mstarpl, pNs_ar + pNm_ar, color = 'red', ls = ls_ar[jx], label = r'Power law $\Sigma < $'+str(sigma_pl_ar[jx]), alpha = 0.5, lw = 2.2)
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.tick_params(axis='both', which = 'both', left=True, right=True, bottom = True, top = True, direction = 'in')
    ax.legend(fontsize = legend_size, loc = 'upper right')
    ax.set_ylabel(r'$N(>M_{\bigstar})$', fontsize = label_font)
    ax2.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$', fontsize = label_font)


    for ix in range(len(sigma_pl_ar)):
        ccompleteness_ar = np.zeros(0)
        pcompleteness_ar = np.zeros(0)
        cmstar_pl_ar = np.zeros(0)
        pmstar_pl_ar = np.zeros(0)
        for jx in range(len(mstarpl) - 1):
            try:
                ccompleteness_ar = np.append(ccompleteness_ar, (len(cmmstar_all[(cmsigma_all < sigma_pl_ar[ix]) & (mstarpl[jx] < cmmstar_all) & (cmmstar_all < mstarpl[jx+1])]) + len(csmstar_all[(cssigma_all < sigma_pl_ar[ix]) & (mstarpl[jx] < csmstar_all) & (csmstar_all < mstarpl[jx+1])]))/(len(cmmstar_all[(mstarpl[jx] < cmmstar_all) & (cmmstar_all < mstarpl[jx+1])]) + len(csmstar_all[(mstarpl[jx] < csmstar_all) & (csmstar_all < mstarpl[jx+1])])))
                cmstar_pl_ar =np.append(cmstar_pl_ar, mstarpl[jx])
            except Exception as e:
                pass
            try:
                pcompleteness_ar = np.append(pcompleteness_ar, (len(pmmstar_all[(pmsigma_all < sigma_pl_ar[ix]) & (mstarpl[jx] < pmmstar_all) & (pmmstar_all < mstarpl[jx+1])]) + len(psmstar_all[(pssigma_all < sigma_pl_ar[ix]) & (mstarpl[jx] < psmstar_all) & (psmstar_all < mstarpl[jx+1])]))/(len(pmmstar_all[(mstarpl[jx] < pmmstar_all) & (pmmstar_all < mstarpl[jx+1])]) + len(psmstar_all[(mstarpl[jx] < psmstar_all) & (psmstar_all < mstarpl[jx+1])])))
                pmstar_pl_ar =np.append(pmstar_pl_ar, mstarpl[jx])
            except Exception as e:
                pass
        ax2.plot(cmstar_pl_ar, ccompleteness_ar, color = 'darkgreen', ls = ls_ar[ix], label = r'Cutoff $\Sigma < $'+str(sigma_pl_ar[ix]))
        ax2.plot(pmstar_pl_ar, pcompleteness_ar, color = 'red', ls = ls_ar[ix], label = r'Power law $\Sigma < $'+str(sigma_pl_ar[ix]))
    
    ax2.set_ylim(bottom = -0.1, top = 1.2)
    ax2.tick_params(axis='both', which = 'both', left=True, right=True, bottom = True, top = True, direction = 'in')
    ax2.set_ylabel('Completeness', fontsize = label_font)

    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'mass_function_completeness.png', dpi = 720)
    plt.close()
    return None

plot_mass_fn_completeness()




def plot_subh_mf_core():
    '''
    This is to plot the satellite mass function and ompare it with the Virgo core data from ferrarese 2016 paper
    '''
    # rcore = 0.2 * rvir_fof #This is assumed to be the core radius temporarily
    rcore_virgo = 309 #This is assumed to be the core radius temporarily
    

    mstarpl = np.logspace(2, 11, 100)


    fig, ax = plt.subplots(figsize = (7, 7))
    pNm_ar = np.zeros(0) #shmf for merged subhalos
    pNs_ar = np.zeros(0) #shmf for surviving subhalos 
    cNm_ar = np.zeros(0) #shmf for merged subhalos
    cNs_ar = np.zeros(0) #shmf for surviving subhalos
    for this_fof in [0, 1, 2]:
        for projection in [0, 1, 2]:
            
            if projection == 0:
                pm_proj_dist = pmxydist_f_ar
                ps_proj_dist = psxydist_f_ar
                cm_proj_dist = cmxydist_f_ar
                cs_proj_dist = csxydist_f_ar
            elif projection == 1:
                pm_proj_dist = pmyzdist_f_ar
                ps_proj_dist = psyzdist_f_ar
                cm_proj_dist = cmyzdist_f_ar
                cs_proj_dist = csyzdist_f_ar
            elif projection == 2:
                pm_proj_dist = pmxzdist_f_ar
                ps_proj_dist = psxzdist_f_ar
                cm_proj_dist = cmxzdist_f_ar
                cs_proj_dist = csxzdist_f_ar

            for (ix, ms) in enumerate(mstarpl):
                pNm_ar = np.append(pNm_ar, len(pmmstar_f_ar[(pmmstar_all > ms) & (pmfof == this_fof) & (pm_proj_dist <= rcore_virgo)]))
                pNs_ar = np.append(pNs_ar, len(psmstar_f_ar[(psmstar_all > ms) &  (psfof == this_fof) & (ps_proj_dist <= rcore_virgo) ]))
                cNm_ar = np.append(cNm_ar, len(cmmstar_f_ar[(cmmstar_all > ms) & (cmfof == this_fof) & (cm_proj_dist <= rcore_virgo) ]))
                cNs_ar = np.append(cNs_ar, len(csmstar_f_ar[(csmstar_all > ms) & (csfof == this_fof) & (cs_proj_dist <= rcore_virgo) ]))
    
    ax.plot(mstarpl, get_med_values(cNs_ar + cNm_ar, 9), color = 'darkgreen', label = 'Cutoff', alpha = 0.5, ls = '-')
    ax.plot(mstarpl, get_med_values(pNs_ar + pNm_ar, 9), color = 'red', label = 'Power', alpha = 0.5, ls = '-')
    ax.fill_between(mstarpl, get_quantiles(cNs_ar + cNm_ar, 9)[0], get_quantiles(cNs_ar + cNm_ar, 9)[1], color = 'darkgreen', alpha = 0.3)
    ax.fill_between(mstarpl, get_quantiles(pNs_ar + pNm_ar, 9)[0], get_quantiles(pNs_ar + pNm_ar, 9)[1], color = 'red', alpha = 0.3)
    ax.plot(10**fmstar, fngal_cum, color = 'black', marker = 'o', label = 'Ferrarese+16 Virgo core')
    ax.legend(fontsize = legend_size)
    ax.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$', fontsize = label_font)
    ax.set_ylabel(r'$N(>M_{\bigstar})$', fontsize = label_font)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(left = 1e2)
    ax.tick_params(axis='y', which = 'both', left=True, right=True, direction = 'in')
    ax.tick_params(axis='x', which = 'both', direction = 'in')
    
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'virgo_mass_function_proj.png', dpi = 720)

    return

plot_subh_mf_core()




def plot_rh_vs_mstar_hist2d(fof_no, alpha_points = 0.1, size_points = 7.5):
    '''
    This is to plot the 2d histogram of size-mass relation
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

    fig, (ax, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 7), sharey = True)
    mspl_log = np.linspace(1, 13, 100)
    # plot_lg_virgo(ax)
    # plot_lg_virgo(ax2)
    


    if fof_no == 210:
        csix = np.where((csfof == 0) | (csfof == 1 )| (csfof == 2))[0]
        cmix = np.where((cmfof == 0) |( cmfof == 1) | (cmfof == 2))[0]
        psix = np.where((psfof == 0) |( psfof == 1) | (psfof == 2))[0]
        pmix = np.where((pmfof == 0) | (pmfof == 1) | (pmfof == 2))[0]
    else:
        csix = np.where(csfof == fof_no)[0]
        cmix = np.where(cmfof == fof_no)[0]
        psix = np.where(psfof == fof_no)[0]
        pmix = np.where(pmfof == fof_no)[0]


    # ax.scatter(csmstar_all[csix], np.array(csrh_all[csix]) * 1e3, marker = 's', fc = 'darkgreen', alpha = alpha_points, s = size_points, label = 'Survived', zorder = 200, edgecolor = 'darkgreen', linewidth = 0.7)
    # ax.scatter(cmmstar_all[cmix], np.array(cmrh_all[cmix]) * 1e3, marker = 's', fc = 'purple', alpha = alpha_points, s = size_points, label = 'Merged', zorder = 200, edgecolor = 'purple', linewidth = 0.7)
    
    cmstar = np.append(csmstar_all[csix], cmmstar_all[cmix]) #msun, THis is all the subhalos in the cutoff model
    crh = np.append(csrh_all[csix], cmrh_all[cmix]) * 1e3 #kpc, This is rh of all the subhalos in the cutoff model
    cmstar = cmstar[crh > 0]
    crh = crh[crh > 0]


    pmstar = np.append(psmstar_all[psix], pmmstar_all[pmix])
    prh = np.append(psrh_all[psix], pmrh_all[pmix]) * 1e3
    pmstar = pmstar[prh > 0]
    prh = prh[prh > 0]


    y_space = np.logspace(np.log10(min(crh)), np.log10(max(crh)), 100)
    x_space = np.logspace(np.log10(min(cmstar)), np.log10(max(cmstar)), 100)


    y_space1 = np.logspace(np.log10(min(prh)), np.log10(max(prh)), 100)
    x_space1 = np.logspace(np.log10(min(pmstar)), np.log10(max(pmstar)), 100)

    hist1, _, _ = np.histogram2d(cmstar, crh, bins=(x_space, y_space))
    hist2, _, _ = np.histogram2d(pmstar, prh, bins=(x_space1, y_space1))

    vmin = min(hist1.min(), hist2.min())
    vmax = max(hist1.max(), hist2.max())

    print(vmin, vmax)

    ax.scatter(cmstar, crh, marker = 's', fc = 'darkblue', alpha = alpha_points, s = size_points, zorder = 0, edgecolor = 'darkblue', linewidth = 0.7)
    ax.hist2d(cmstar, crh, bins = (x_space, y_space), cmin = 9, norm = 'log', zorder = 100, vmin = 9, vmax = vmax)
    ax.plot(10**mspl_log, 1e3 * 10**get_lrh(mspl_log), color = 'k', zorder = 200)
    top_data = ax.get_ylim()[1]
    right_data = ax.get_xlim()[1]
    ax.set_xlim(left = 1e1, right = right_data)
    ax.set_ylim(bottom = 10, top = top_data)

    # ax2.scatter(psmstar_all[psix], np.array(psrh_all[psix]) * 1e3, marker = 's', fc = 'darkgreen', alpha = alpha_points, s = size_points, label = 'Survived', zorder = 200, edgecolor = 'darkgreen', linewidth = 0.7)
    # ax2.scatter(pmmstar_all[pmix], np.array(pmrh_all[pmix]) * 1e3, marker = 's', fc = 'purple', alpha = alpha_points, s = size_points, label = 'Merged', zorder = 200, edgecolor = 'purple', linewidth = 0.7)
    
    
    
    ax2.scatter(pmstar, prh, marker = 's', fc = 'darkblue', alpha = alpha_points, s = size_points, zorder = 0, edgecolor = 'darkblue', linewidth = 0.7)
    
    ax2.hist2d(pmstar, prh, bins = (x_space1, y_space1), cmin = 9, norm = 'linear', zorder = 100, vmin = 9, vmax = vmax)
    ax2.plot(10**mspl_log, 1e3 * 10**get_lrh(mspl_log), color = 'k', zorder = 200)
    top_data2 = ax2.get_ylim()[1]
    right_data2 = ax2.get_xlim()[1]
    ax2.set_xlim(left = 1e1, right = right_data2)
    ax2.set_ylim(bottom = 10, top = top_data2)


    if True: #This section is for lines of constant surface brightness
        angle = 62
        yval = 0.21*top_data
        ax.plot(10**mspl_log, get_line_of_constant_surfb(10**mspl_log, 24), ls = '--', color = 'gray')
        ax.annotate(r'24 mag arcsec$^{-2}$', xy = (inverse_get_line_of_constant_surfb(yval, 24), yval), xytext = (inverse_get_line_of_constant_surfb(yval, 24), 1.05* yval), 
                rotation=angle, color='gray', fontsize=10, rotation_mode='anchor')
        ax.plot(10**mspl_log, get_line_of_constant_surfb(10**mspl_log, 28), ls = '--', color = 'gray')
        ax.annotate(r'28 mag arcsec$^{-2}$', xy = (inverse_get_line_of_constant_surfb(yval, 28), yval), xytext = (inverse_get_line_of_constant_surfb(yval, 28), 1.05* yval), 
                rotation=angle, color='gray', fontsize=10, rotation_mode='anchor')
        ax.plot(10**mspl_log, get_line_of_constant_surfb(10**mspl_log, 35), ls = '--', color = 'gray')
        ax.annotate(r'35 mag arcsec$^{-2}$', xy = (inverse_get_line_of_constant_surfb(yval, 35), yval), xytext = (inverse_get_line_of_constant_surfb(yval, 35), 1.05* yval), 
                rotation=angle, color='gray', fontsize=10, rotation_mode='anchor')
        
        yval = 0.21*top_data2
        ax2.plot(10**mspl_log, get_line_of_constant_surfb(10**mspl_log, 24), ls = '--', color = 'gray')
        ax2.annotate(r'24 mag arcsec$^{-2}$', xy = (inverse_get_line_of_constant_surfb(yval, 24), yval), xytext = (inverse_get_line_of_constant_surfb(yval, 24), 1.05* yval), 
                rotation=angle, color='gray', fontsize=10, rotation_mode='anchor')
        ax2.plot(10**mspl_log, get_line_of_constant_surfb(10**mspl_log, 28), ls = '--', color = 'gray')
        ax2.annotate(r'28 mag arcsec$^{-2}$', xy = (inverse_get_line_of_constant_surfb(yval, 28), yval), xytext = (inverse_get_line_of_constant_surfb(yval, 28), 1.05* yval), 
                rotation=angle, color='gray', fontsize=10, rotation_mode='anchor')
        ax2.plot(10**mspl_log, get_line_of_constant_surfb(10**mspl_log, 35), ls = '--', color = 'gray')
        ax2.annotate(r'35 mag arcsec$^{-2}$', xy = (inverse_get_line_of_constant_surfb(yval, 35), yval), xytext = (inverse_get_line_of_constant_surfb(yval, 35), 1.05* yval), 
                rotation=angle, color='gray', fontsize=10, rotation_mode='anchor')
    

    # plot_lg_virgo_some(ax, alpha = 0.2)
    # plot_lg_virgo_some(ax2, alpha = 0.2)

    # # Example usage:
    # R = 100  # Half light radius in pc
    # S = 25  # Surface brightness
    # mstar_inverse = inverse_get_line_of_constant_surfb(R, S)
    # print(mstar_inverse)
    ax.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$', fontsize = label_font)
    ax.set_ylabel(r'$R_{\rm{h}}$ (pc)', fontsize = label_font)
    ax.text(0.01, 0.99, 'Cutoff', ha = 'left', va = 'top', transform=ax.transAxes, fontsize = legend_size)
    # ax.legend(fontsize = 8, loc = 'lower right')
    ax.tick_params(axis='both', which = 'both', left=True, right=True, bottom = True, top = True, direction = 'in')

    ax2.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$', fontsize = label_font)
    # ax2.set_ylabel(r'$R_{\rm{h}}$ (pc)', fontsize = label_font)
    ax2.text(0.01, 0.99, 'Power law', ha = 'left', va = 'top', transform=ax2.transAxes, fontsize = legend_size)
    # ax2.legend(fontsize = 8, loc = 'lower right')
    ax2.tick_params(axis='both', which = 'both', left=True, right=True, bottom = True, top = True, direction = 'in')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    # fig.suptitle('FoF'+str(fof_no), fontsize = 14)
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'rh_vs_mstar_hist.png', dpi = 720)
    plt.close()
    return None

plot_rh_vs_mstar_hist2d(210)





sys.exit()


def plot_fig1():
    '''
    This plot will have a dual panel for Abundance matching relation and Size-stellar mass relation that we use
    '''
    fig, (ax, ax2) = plt.subplots(1, 2, figsize = (12, 6))

    psmstar_if_ar[psmstar_if_ar == 0] = 1e3
    lvmax_pl = np.linspace(0.1, np.log10(600), 100)
    # ax.scatter(psvmax_if_ar, psmstar_if_ar, color = 'dodgerblue', marker = 'o', s = 1, alpha = 0.1, label = 'TNG at infall')
    # ax.plot(10**lvmax_pl, 10**get_mstar_pl(lvmax_pl), color = 'red', label = 'Power law')
    # ax.fill_between(10**lvmax_pl, 10**get_mstar_pl(lvmax_pl) - 2.303*10**get_mstar_pl(lvmax_pl)*get_scatter(lvmax_pl), 10**get_mstar_pl(lvmax_pl) + 2.303*10**get_mstar_pl(lvmax_pl)*get_scatter(lvmax_pl), color = 'red', alpha = 0.3)
    # ax.plot(10**lvmax_pl, 10**get_mstar_co(lvmax_pl), color = 'darkgreen', label = 'Cutoff')
    # ax.fill_between(10**lvmax_pl, 10**get_mstar_co(lvmax_pl) - 2.303*10**get_mstar_co(lvmax_pl) * get_scatter(lvmax_pl), 10**get_mstar_co(lvmax_pl) + 2.303*10**get_mstar_co(lvmax_pl) * get_scatter(lvmax_pl), color = 'darkgreen', alpha = 0.3)
    ax.scatter(np.log10(psvmax_if_ar), np.log10(psmstar_if_ar), color = 'gray', marker = 'o', s = 1, alpha = 0.1, label = 'TNG at infall')
    ax.plot(lvmax_pl, get_mstar_pl(lvmax_pl), color = 'red', label = 'Power law', lw = 2)
    ax.fill_between(lvmax_pl, get_mstar_pl(lvmax_pl) - get_scatter(lvmax_pl), get_mstar_pl(lvmax_pl) + get_scatter(lvmax_pl), color = 'red', alpha = 0.1)
    ax.plot(lvmax_pl, get_mstar_co(lvmax_pl), color = 'darkgreen', label = 'Cutoff', ls = '--', lw = 2)
    ax.fill_between(lvmax_pl, get_mstar_co(lvmax_pl) - get_scatter(lvmax_pl), get_mstar_co(lvmax_pl) + get_scatter(lvmax_pl), color = 'darkgreen', alpha = 0.1)
    ax.set_ylabel(r'$\log M_{\rm{star}}(M_\odot)$', fontsize = label_font)
    ax.set_xlabel(r'$\log V_{\rm{max}}(\rm{km/s})$', fontsize = label_font)
    ax.axhline(np.log10(5e6), ls = ':', color = 'gray', alpha = 0.4)
    ax.text(0.31, 6, 'Unresolved in size\nand mass', fontsize = 8)
    ax.axhline(np.log10(1e8), ls = ':', color = 'gray', alpha = 0.4)
    ax.text(0.31, 7.3, 'Unresolved in size,\nresolved in mass', fontsize = 8)
    ax.text(0.31, 9, 'Resolved in size and mass', fontsize = 8)

    ax.tick_params(axis='y', which = 'both', left=True, right=True, direction = 'in')
    ax.tick_params(axis='x', which = 'both', direction = 'in', top = True)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_ylim(bottom = np.log10(7e1), top = 12)
    ax.set_xlim(left = 0.25)
    ax.legend(fontsize = 8, loc = 'lower right') 
    
    #We will be plotting all the subhalos with at least 1e8 Msun from TNG at infall in second panel
    # ax2.plot()
    df_1e6 = pd.read_csv(misc_path + '1e6_tidal_tracks.csv')
    lmstar_1e6 = np.log10(df_1e6['mstar'].values)
    lrh_1e6 = np.log10(df_1e6['rh'].values)
    df_5e9 = pd.read_csv(misc_path + '5e9_tidal_tracks.csv')
    lmstar_5e9 = np.log10(df_5e9['mstar'].values)
    lrh_5e9 = np.log10(df_5e9['rh'].values)

    ax2.plot(lmstar_1e6, lrh_1e6, color = 'green', zorder = 1000, lw = 2, ls = '--', label = 'TTK1 - 1e6')
    ax2.plot(lmstar_5e9, lrh_5e9, color = 'red', zorder = 1000, lw = 2, ls = '-.', label = 'TTK2 - 5e9')

    mspl_log = np.linspace(1, 13, 100)
    ax2.scatter(np.log10(psmstar_max_ar[psmstar_max_ar >= 1e8]), np.log10(1e3 * psrh_max_ar[psmstar_max_ar >= 1e8]), color = 'gray', marker = 'o', s = 2.5, alpha = 0.1, label = 'TNG at infall')
    plot_lg_virgo_some(ax2, alpha = 0.2, mec = 'goldenrod')
    ax2.plot(mspl_log, np.log10(1e3 * 10**get_lrh(mspl_log)), color = 'k', zorder = 200, lw = 1)
    ax2.fill_between(mspl_log, np.log10(1e3 * 10**get_lrh(mspl_log)) - 0.2, np.log10(1e3 * 10**get_lrh(mspl_log)) + 0.2, color = 'k', alpha = 0.1)
    ax2.set_ylabel(r'$\log R_{\rm{h}}(\rm{pc})$', fontsize = label_font)
    ax2.set_xlabel(r'$\log M_{\rm{star}}(M_\odot)$', fontsize = label_font)
    ax2.legend(fontsize = 8)


    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'fig1.png')

    return None

plot_fig1()


def plot_fig2():
    '''
    Figure 2 would be the stellar segregation, justifying the segregations that we have
    '''
    fig, ax = plt.subplots(figsize = (6, 6))
    ax.scatter(np.log10(psmstar_max_all), np.log10(psrh_max_all/psrmx_if_ar), color = 'gray', marker = 'o', s = 2, alpha = 0.1, label = 'TNG at infall')
    ax.scatter(np.log10(pmmstar_max_all), np.log10(pmrh_max_all/pmrmx_if_ar), color = 'gray', marker = 'o', s = 2, alpha = 0.1)
    ax.axhline(np.log10(0.5), ls = '-', color = 'gray', alpha = 0.4, lw = 1)
    ax.text(11.9, np.log10(0.5) + 0.05, '1/2', fontsize = 8, ha = 'right')
    ax.axhline(np.log10(0.25), ls = '-', color = 'gray', alpha = 0.4, lw = 1)
    ax.text(11.9, np.log10(0.25) + 0.05, '1/4', fontsize = 8, ha = 'right')
    ax.axhline(np.log10(0.125), ls = '-', color = 'gray', alpha = 0.4, lw = 1)
    ax.text(11.9, np.log10(0.125) + 0.05, '1/8', fontsize = 8, ha = 'right')
    ax.axhline(np.log10(0.0625), ls = '-', color = 'gray', alpha = 0.4, lw = 1)
    ax.text(11.9, np.log10(0.0625) + 0.05, '1/16', fontsize = 8, ha = 'right')
    ax.axhline(np.log10(1/66), ls = ':', color = 'gray', alpha = 0.4, lw = 1)
    ax.text(11.9, np.log10(1/66) + 0.05, '1/66', fontsize = 8, ha = 'right')
    ax.axhline(np.log10(1/250), ls = ':', color = 'gray', alpha = 0.4, lw = 1)
    ax.text(11.9, np.log10(1/250) + 0.05, '1/250', fontsize = 8, ha = 'right')
    ax.axhline(np.log10(1/1000), ls = ':', color = 'gray', alpha = 0.4, lw = 1)
    ax.text(11.9, np.log10(1/1000) + 0.05, '1/1000', fontsize = 8, ha = 'right')

    ax.set_xlabel(r'$\log M_{\rm{star}}(M_\odot)$', fontsize = label_font)
    ax.set_ylabel(r'$\log (R_{\rm{h0}}/r_{\rm{mx0}})$', fontsize = label_font)
    # ax.axhline()
    ax.set_xlim(left = 2, right = 12)
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'fig2.png')
    return None

plot_fig2()



def plot_frac():
    '''
    This is to plot the fraction of subhalos that survived in TNG over number that got artificially merged
    '''
    fig, ax = plt.subplots(figsize = (6, 6))
    mstar_bins = np.linspace(2, 12, 10) #These are the bins into which we will bin the subhalos
    mstar_pl = np.zeros(0) #This will be the midpoint of bin edges defined above
    nsurv_bin = np.zeros(0)
    nmerg_bin = np.zeros(0)
    for i in range(len(mstar_bins) - 1):
        mstar_pl = np.append(mstar_pl, (mstar_bins[i] + mstar_bins[i + 1]) / 2)
        nsurv_bin = np.append(nsurv_bin, len(psmstar_all[(psmstar_all < 10**mstar_bins[i + 1]) & (psmstar_all >= 10**mstar_bins[i])]))
        nmerg_bin = np.append(nmerg_bin, len(pmmstar_all[(pmmstar_all < 10**mstar_bins[i + 1]) & (pmmstar_all >= 10**mstar_bins[i])]))

    ax.plot(mstar_pl, nmerg_bin / (nsurv_bin + nmerg_bin), color = 'gray')
    # ax.plot(mstar_pl, nmerg_bin / nsurv_bin , color = 'gray')
    ax.set_xlabel(r'$\log M_{\rm{star}}(M_\odot)$', fontsize = label_font)
    # ax.set_ylabel(r'$f_{\rm{type2}}/f_{\rm{type1}}$', fontsize = label_font)
    ax.set_ylabel(r'$f_{\rm{type2}}/(f_{\rm{type1}} + f_{\rm{type1}})$', fontsize = label_font)
    ax.set_yscale('log')
    ax.set_ylim(bottom = 0.04, top = 1)
    ax.set_yticks([0.05, 0.1, 0.5, 1])
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'frac_type1_type2.png')
    
    return None

plot_frac()


def plot_cumulative_rad_dist(fof_no =  0):
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

    # Ndm_ar = Ndm_ar * (cNm_ar[-1] + cNs_ar[-1]) #Normalizing the particles thing to cutoff model total temporarily
    fig, ax = plt.subplots(figsize = (6, 6.25))
    # ax.plot(rpl, Ns_ar, color = 'blue', label = 'Surviving')
    ax.plot(rpl, Ntng_ar/Ntng_ar[-1], color = 'blue', label = r'TNG')
    ax.plot(rpl, (pNs_ar + pNm_ar)/(pNs_ar[-1] + pNm_ar[-1]), color = 'red', label = r'power law ($>10\,\rm{M_\odot}$)')
    ax.plot(rpl, (cNs_ar + cNm_ar)/(cNs_ar[-1] + cNm_ar[-1]), color = 'darkgreen', label = r'cutoff ($>10\,\rm{M_\odot}$)')
    # ax.plot(rpl, N_all_ar, color = 'purple', ls = '--', lw = 0.5, label = r'Model (all)')
    ax.plot(rpl, Ndm_ar/Ndm_ar[-1], color = 'black', ls = '--', label = 'DM in TNG', alpha = 0.5)
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


plot_cumulative_rad_dist(0)



def plot_proj_full():
    '''
    This function is to plot the subhalos mass function
    We will be plotting with Fornax here
    '''
    mstarpl = np.logspace(1, 11, 100)

    rvir_fornax = 700 #kpc
    
    fig, ax = plt.subplots(figsize = (6, 6.25))
    ls_ar = ['-.', '--', '-']
    pNm_ar = np.zeros(0) #shmf for merged subhalos
    pNs_ar = np.zeros(0) #shmf for surviving subhalos 
    cNm_ar = np.zeros(0) #shmf for merged subhalos
    cNs_ar = np.zeros(0) #shmf for surviving subhalos
    mstar_match = 1e7 #This is the stellar mass where we try to match the fornax and the model
    N_fitting = np.zeros(0)
    mvir_ar = np.array([mvir_fof0, mvir_fof0, mvir_fof0, mvir_fof1, mvir_fof1, mvir_fof1, mvir_fof2, mvir_fof2, mvir_fof2])

    for this_fof in [0, 1, 2]:
        for projection in [0, 1, 2]:
            
            if projection == 0:
                pm_proj_dist = pmxydist_f_ar
                ps_proj_dist = psxydist_f_ar
                cm_proj_dist = cmxydist_f_ar
                cs_proj_dist = csxydist_f_ar
            elif projection == 1:
                pm_proj_dist = pmyzdist_f_ar
                ps_proj_dist = psyzdist_f_ar
                cm_proj_dist = cmyzdist_f_ar
                cs_proj_dist = csyzdist_f_ar
            elif projection == 2:
                pm_proj_dist = pmxzdist_f_ar
                ps_proj_dist = psxzdist_f_ar
                cm_proj_dist = cmxzdist_f_ar
                cs_proj_dist = csxzdist_f_ar

            N_fitting = np.append(N_fitting, len(psmstar_f_ar[(psmstar_all > mstar_match) & (psfof == this_fof) & (ps_proj_dist <= rvir_fornax)]))


            for (ix, ms) in enumerate(mstarpl):
                pNm_ar = np.append(pNm_ar, len(pmmstar_f_ar[(pmmstar_all > ms) & (pmfof == this_fof) & (pm_proj_dist <= rvir_fornax)]))
                pNs_ar = np.append(pNs_ar, len(psmstar_f_ar[(psmstar_all > ms) &  (psfof == this_fof) & (ps_proj_dist <= rvir_fornax) ]))
                cNm_ar = np.append(cNm_ar, len(cmmstar_f_ar[(cmmstar_all > ms) & (cmfof == this_fof) & (cm_proj_dist <= rvir_fornax) ]))
                cNs_ar = np.append(cNs_ar, len(csmstar_f_ar[(csmstar_all > ms) & (csfof == this_fof) & (cs_proj_dist <= rvir_fornax) ]))
    
    #We now have to fit a straight line to log N_fitting vs log mvir_ar
    m_bf, b_bf = np.polyfit(np.log10(mvir_ar), np.log10(N_fitting), 1) #This returns slope and intercept
    Nforn_1e7 = 217.37 #Number of galaxies above 1e7 Msun in Fornax
    lmvir_fornax_calc = ((np.log10(Nforn_1e7) - b_bf)/m_bf) #These are the number of galaxies above 1e7 Msun in Fornax
    print(f'Virgo virial mass is (log Msun): {lmvir_fornax_calc}')

    diffN = np.log10(np.median(N_fitting)) - np.log10(Nforn_1e7)
    print(f'median: {np.median(N_fitting)} and diffN: {diffN}')
    # dif
    # fN = 0



    ax.plot(mstarpl, 10**(np.log10(get_med_values(cNs_ar + cNm_ar, 9)) - diffN), color = 'darkgreen', label = 'Cutoff', alpha = 0.5, ls = '-')
    ax.plot(mstarpl, 10**(np.log10(get_med_values(pNs_ar + pNm_ar, 9)) - diffN), color = 'red', label = 'Power', alpha = 0.5, ls = '-')
    # ax.fill_between(mstarpl, get_quantiles(cNs_ar + cNm_ar, 9)[0] - diffN, get_quantiles(cNs_ar + cNm_ar, 9)[1] - diffN, color = 'darkgreen', alpha = 0.3)
    # ax.fill_between(mstarpl, get_quantiles(pNs_ar + pNm_ar, 9)[0] - diffN, get_quantiles(pNs_ar + pNm_ar, 9)[1] - diffN, color = 'red', alpha = 0.3)
    ax.plot(10**vmstar, vngal_cum, color = 'black', marker = '^', label = 'Venhola+19 Fornax')
    # for jx in range(len(sigma_pl_ar) + 1): #+2 because we also need to plot without any surface brightness limits
    # pNm_ar = np.zeros(0) #shmf for merged subhalos
    # pNs_ar = np.zeros(0) #shmf for surviving subhalos 
    # cNm_ar = np.zeros(0) #shmf for merged subhalos
    # cNs_ar = np.zeros(0) #shmf for surviving subhalos 
    # Ntng_ar = np.zeros(0)


    # for (ix, ms) in enumerate(mstarpl):
    #     pNm_ar = np.append(pNm_ar, len(pmmstar_f_ar[(pmmstar_all > ms) & (pmfof == 0) & (pmxydist_f_ar < rvir_fornax)]))
    #     pNs_ar = np.append(pNs_ar, len(psmstar_f_ar[(psmstar_all > ms) &  (psfof == 0) & (psxydist_f_ar < rvir_fornax)]))
    #     cNm_ar = np.append(cNm_ar, len(cmmstar_f_ar[(cmmstar_all > ms) & (cmfof == 0) & (cmxydist_f_ar < rvir_fornax)]))
    #     cNs_ar = np.append(cNs_ar, len(csmstar_f_ar[(csmstar_all > ms) & (csfof == 0) & (csxydist_f_ar < rvir_fornax)]))
    #     # Ntng_ar = np.append(Ntng_ar, len(psmstar_f_ar_tng[psmstar_f_ar_tng > ms]))

    # fig, ax = plt.subplots(figsize = (6, 6.25))
    
    # # ax.plot(mstarpl, Ns_ar, color = 'darkgreen', label = 'Model Surviving', alpha = 0.5)
    # ax.plot(mstarpl, cNs_ar + cNm_ar, color = 'darkgreen', label = 'Cutoff', alpha = 0.5)
    # ax.plot(mstarpl, pNs_ar + pNm_ar, color = 'red', label = 'Power law', alpha = 0.5)
    # plot_sachi(ax)
    # ax.plot(mstarpl, Ntng_ar, color = 'blue', label = 'TNG surviving', alpha = 0.5)
    ax.legend(fontsize = 8)
    ax.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
    ax.set_ylabel(r'$N(>M_{\bigstar})$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(left = 1e1)
    ax.tick_params(axis='y', which = 'both', left=True, right=True, direction = 'in')
    ax.tick_params(axis='x', which = 'both', direction = 'in')
    
    # ax.set_title('FoF'+str(fof_no))
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'fornax_mass_function_proj.png')


    plt.close()

plot_proj_full()



def plot_tidal_tracks_sim():
    '''
    Here, we will be plotting tidal track plots from the simulation

    This is again to validate the model using the simulation
    Panel 1: Vmx/Vmx0 vs rmx/rmx0
    Panel 2: Mstar/Mstar0 vs Mmx/Mmx0
    Panel 3: rh/rh0 vs Mmx/Mmx0
    '''
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize = (18, 6))
    # cond = psmstar_all > 1e8
    ndm_mx_ar = psmmx_f_ar_tng/4.5e5 #These are the number of DM particles inside rmx for the subhalo
    # print('Number of dark matter particles inside rmx', np.median(ndm_mx_ar), np.quantile(ndm_mx_ar, 0.16), np.quantile(ndm_mx_ar, 0.84))
    cond = (ndm_mx_ar > 3e3) & (psmstar_f_ar_tng > 5e6) #FIXME: This needs to be checked again
    ax.scatter(psrmx_f_ar_tng[cond]/psrmx_if_ar[cond], psvmx_f_ar_tng[cond]/psvmx_if_ar[cond], color = 'gray', marker = 'o', s = 8, alpha = 0.5, label = 'TNG')
    ax.set_ylabel(r'$V_{\rm{mx}}/V_{\rm{mx0}}$', fontsize = label_font)
    ax.set_xlabel(r'$r_{\rm{mx}}/r_{\rm{mx0}}$', fontsize = label_font)
    rmxbyrmx0_pl = np.logspace(-2, 0, 100)
    ax.plot(rmxbyrmx0_pl, get_vmxbyvmx0(rmxbyrmx0_pl), color = 'black', label = 'Tidal track')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_yticks([0.4, 0.6, 1])
    ax.set_yticklabels([0.4, 0.6, 1])
    # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    ax.set_xlim(left = 0.05, right = 5)
    ax.set_ylim(bottom = 0.4, top = 1.25)
    ax.legend(fontsize = 8)
    
    ax2.scatter(psmmx_f_ar_tng[cond]/psmmx_if_ar[cond], psmstar_f_ar_tng[cond]/psmstar_max_ar[cond], color = 'gray', marker = 'o', s = 2, alpha = 0.1, label = 'TNG')
    fpl = np.logspace(-2, 0, 100)
    ax2.set_xlabel(r'$M_{\rm{mx}}/M_{\rm{mx0}}$', fontsize = label_font)
    ax2.set_ylabel(r'$M_{\rm{star}}/M_{\rm{star0}}$', fontsize = label_font)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend(fontsize = 8)

    ax3.scatter(psmmx_f_ar_tng[cond]/psmmx_if_ar[cond], psrh_f_ar_tng[cond]/psrh_max_ar[cond], color = 'gray', marker = 'o', s = 2, alpha = 0.1, label = 'TNG')
    ax3.set_xlabel(r'$M_{\rm{mx}}/M_{\rm{mx0}}$', fontsize = label_font)
    ax3.set_ylabel(r'$R_{\rm{h}}/R_{\rm{h0}}$', fontsize = label_font)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.legend(fontsize = 8)


    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'tidal_tracks_sim.png')
    return 

plot_tidal_tracks_sim()






sys.exit() #Just making plots required for the paper


def plot_abundance_relation():
    '''
    THis is to plot the abundance relation with overlay of the best fit that we have
    '''
    #Let's first check if we have the required fields. We can run some routine otherwise to get required fields
    fig, ax = plt.subplots(figsize = (6, 6))
    #FIXME: psmstar_max_all has to be replaced with psmstar_if_ar after 
    psmstar_if_ar[psmstar_if_ar == 0] = 1e3
    lvmax_pl = np.linspace(0.1, np.log10(600), 100)
    # ax.scatter(psvmax_if_ar, psmstar_if_ar, color = 'dodgerblue', marker = 'o', s = 1, alpha = 0.1, label = 'TNG at infall')
    # ax.plot(10**lvmax_pl, 10**get_mstar_pl(lvmax_pl), color = 'red', label = 'Power law')
    # ax.fill_between(10**lvmax_pl, 10**get_mstar_pl(lvmax_pl) - 2.303*10**get_mstar_pl(lvmax_pl)*get_scatter(lvmax_pl), 10**get_mstar_pl(lvmax_pl) + 2.303*10**get_mstar_pl(lvmax_pl)*get_scatter(lvmax_pl), color = 'red', alpha = 0.3)
    # ax.plot(10**lvmax_pl, 10**get_mstar_co(lvmax_pl), color = 'darkgreen', label = 'Cutoff')
    # ax.fill_between(10**lvmax_pl, 10**get_mstar_co(lvmax_pl) - 2.303*10**get_mstar_co(lvmax_pl) * get_scatter(lvmax_pl), 10**get_mstar_co(lvmax_pl) + 2.303*10**get_mstar_co(lvmax_pl) * get_scatter(lvmax_pl), color = 'darkgreen', alpha = 0.3)
    ax.scatter(np.log10(psvmax_if_ar), np.log10(psmstar_if_ar), color = 'gray', marker = 'o', s = 1, alpha = 0.1, label = 'TNG at infall')
    ax.plot(lvmax_pl, get_mstar_pl(lvmax_pl), color = 'red', label = 'Power law', lw = 2)
    ax.fill_between(lvmax_pl, get_mstar_pl(lvmax_pl) - get_scatter(lvmax_pl), get_mstar_pl(lvmax_pl) + get_scatter(lvmax_pl), color = 'red', alpha = 0.1)
    ax.plot(lvmax_pl, get_mstar_co(lvmax_pl), color = 'darkgreen', label = 'Cutoff', ls = '--', lw = 2)
    ax.fill_between(lvmax_pl, get_mstar_co(lvmax_pl) - get_scatter(lvmax_pl), get_mstar_co(lvmax_pl) + get_scatter(lvmax_pl), color = 'darkgreen', alpha = 0.1)
    ax.set_ylabel(r'$\log M_\bigstar(M_\odot)$', fontsize = label_font)
    ax.set_xlabel(r'$\log V_{\rm{max}}(\rm{km/s})$', fontsize = label_font)
    ax.axhline(np.log10(5e6), ls = ':', color = 'gray', alpha = 0.4)
    ax.text(0.31, 6, 'Unresolved in size\nand mass', fontsize = 8)
    ax.axhline(np.log10(1e8), ls = ':', color = 'gray', alpha = 0.4)
    ax.text(0.31, 7.3, 'Unresolved in size,\nresolved in mass', fontsize = 8)
    ax.text(0.31, 9, 'Resolved in size and mass', fontsize = 8)

    ax.tick_params(axis='y', which = 'both', left=True, right=True, direction = 'in')
    ax.tick_params(axis='x', which = 'both', direction = 'in', top = True)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_ylim(bottom = np.log10(7e1), top = 12)
    ax.set_xlim(left = 0.25)
    ax.legend(fontsize = 8, loc = 'lower right') 
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'abundance_relation.png')
    
    return None


plot_abundance_relation()

def plot_subh_mf_3d():
    '''
    This  will be the first of figures in the paper for results section
    '''
    mstarpl = np.logspace(1, 11, 100)
    fig, ax = plt.subplots(figsize = (6, 6.25))
    ls_ar = ['-.', '--', '-']
    

    Ntng_ar = np.zeros(0) #We will be plotting the averge value for all FoFs combined

    for this_fof in [0, 1, 2]:
        pNm_ar = np.zeros(0) #shmf for merged subhalos
        pNs_ar = np.zeros(0) #shmf for surviving subhalos 
        cNm_ar = np.zeros(0) #shmf for merged subhalos
        cNs_ar = np.zeros(0) #shmf for surviving subhalos 
        
        if this_fof == 0:
            this_rvir = rvir_fof0
        elif this_fof == 1:
            this_rvir = rvir_fof1
        elif this_fof == 2:
            this_rvir = rvir_fof2

        for (ix, ms) in enumerate(mstarpl):

            pNm_ar = np.append(pNm_ar, len(pmmstar_f_ar[(pmmstar_all > ms) & (pmfof == this_fof) & (pmdist_f_ar < this_rvir)]))
            pNs_ar = np.append(pNs_ar, len(psmstar_f_ar[(psmstar_all > ms) &  (psfof == this_fof) & (psdist_f_ar < this_rvir)]))
            cNm_ar = np.append(cNm_ar, len(cmmstar_f_ar[(cmmstar_all > ms) & (cmfof == this_fof) & (cmdist_f_ar < this_rvir)]))
            cNs_ar = np.append(cNs_ar, len(csmstar_f_ar[(csmstar_all > ms) & (csfof == this_fof) & (csdist_f_ar < this_rvir)]))
            Ntng_ar = np.append(Ntng_ar, len(psmstar_f_ar_tng[(psmstar_f_ar_tng > ms) & (psfof == this_fof) & (psdist_f_ar < this_rvir)]))
    
        ax.plot(mstarpl, cNs_ar + cNm_ar, color = 'darkgreen', label = 'Cutoff FoF = ' + str(this_fof), alpha = 0.5, ls = ls_ar[this_fof])
        ax.plot(mstarpl, pNs_ar + pNm_ar, color = 'red', label = 'Power law FoF = ' + str(this_fof), alpha = 0.5, ls = ls_ar[this_fof])

    NTNGpl = (Ntng_ar[:int(len(Ntng_ar)/3)] + Ntng_ar[int(len(Ntng_ar)/3) : int(2 * len(Ntng_ar)/3)] + Ntng_ar[int(2 * len(Ntng_ar)/3) :])/3.
    ax.plot(mstarpl[::4], NTNGpl[::4], color = 'black', marker = '^', label = 'TNG average for FoFs')
    ax.legend(fontsize = 8)
    ax.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
    ax.set_ylabel(r'$N(>M_{\bigstar})$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(left = 1e2)
    ax.set_ylim(bottom = 3)
    ax.tick_params(axis='y', which = 'both', left=True, right=True, direction = 'in')
    ax.tick_params(axis='x', which = 'both', direction = 'in')
    
    # ax.set_title('FoF'+str(fof_no))
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + '3d_mass_fn.png')

    return None

plot_subh_mf_3d()




def plot_subh_mf_diff_regions():
    '''
    This is to plot te subhalo mass function for different cuts in final positions of the subhalos
    '''
    mstarpl = np.logspace(1, 11, 100)
    fig, ax = plt.subplots(figsize = (6, 6.25))
    ls_ar = ['-.', '--', '-']


    Ntng_ar = np.zeros(0) #We will be plotting the averge value for all FoFs combined

    for rvir_frac in [0.2, 0.4, 0.6, 0.8, 1]:
        pNm_ar = np.zeros(0) #shmf for merged subhalos
        pNs_ar = np.zeros(0) #shmf for surviving subhalos 
        cNm_ar = np.zeros(0) #shmf for merged subhalos
        cNs_ar = np.zeros(0) #shmf for surviving subhalos 
        for this_fof in [0, 1, 2]:
            if this_fof == 0:
                this_rvir = rvir_fof0
            elif this_fof == 1:
                this_rvir = rvir_fof1
            elif this_fof == 2:
                this_rvir = rvir_fof2

            for (ix, ms) in enumerate(mstarpl):

                pNm_ar = np.append(pNm_ar, len(pmmstar_f_ar[(pmmstar_all > ms) & (pmfof == this_fof) & (pmdist_f_ar < rvir_frac * this_rvir)]))
                pNs_ar = np.append(pNs_ar, len(psmstar_f_ar[(psmstar_all > ms) &  (psfof == this_fof) & (psdist_f_ar < rvir_frac * this_rvir)]))
                # cNm_ar = np.append(cNm_ar, len(cmmstar_f_ar[(cmmstar_all > ms) & (cmfof == this_fof) & (cmdist_f_ar < this_rvir)]))
                # cNs_ar = np.append(cNs_ar, len(csmstar_f_ar[(csmstar_all > ms) & (csfof == this_fof) & (csdist_f_ar < this_rvir)]))
                # Ntng_ar = np.append(Ntng_ar, len(psmstar_f_ar_tng[(psmstar_f_ar_tng > ms) & (psfof == this_fof) & (psdist_f_ar < this_rvir)]))
        
            # ax.plot(mstarpl, cNs_ar + cNm_ar, color = 'darkgreen', label = 'Cutoff FoF = ' + str(this_fof), alpha = 0.5, ls = ls_ar[this_fof])
        pN_ar = pNs_ar + pNm_ar
        ax.plot(mstarpl, (pN_ar[:int(len(pN_ar)/3)] + pN_ar[int(len(pN_ar)/3): 2* int(len(pN_ar)/3)] + pN_ar[2* int(len(pN_ar)/3):]) / 3., color = 'red', label = f'< {rvir_frac} rvir', alpha = rvir_frac)

    # NTNGpl = (Ntng_ar[:int(len(Ntng_ar)/3)] + Ntng_ar[int(len(Ntng_ar)/3) : int(2 * len(Ntng_ar)/3)] + Ntng_ar[int(2 * len(Ntng_ar)/3) :])/3.
    # ax.plot(mstarpl, NTNGpl, color = 'black', marker = '^', label = 'TNG')
    ax.legend(fontsize = 8)
    ax.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
    ax.set_ylabel(r'$N(>M_{\bigstar})$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(left = 1e2)
    ax.set_ylim(bottom = 3)
    ax.tick_params(axis='y', which = 'both', left=True, right=True, direction = 'in')
    ax.tick_params(axis='x', which = 'both', direction = 'in')
    
    # ax.set_title('FoF'+str(fof_no))
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'subh_mf_diff_regions.png')

    return None


plot_subh_mf_diff_regions()







def plot_subh_mf_core():
    '''
    This is to plot the satellite mass function and ompare it with the Virgo core data from ferrarese 2016 paper
    '''
    # rcore = 0.2 * rvir_fof #This is assumed to be the core radius temporarily
    rcore_virgo = 309 #This is assumed to be the core radius temporarily
    

    mstarpl = np.logspace(1, 11, 100)

    rvir_fornax = 700 #kpc

    fig, ax = plt.subplots(figsize = (6, 6.25))
    ls_ar = ['-.', '--', '-']
    pNm_ar = np.zeros(0) #shmf for merged subhalos
    pNs_ar = np.zeros(0) #shmf for surviving subhalos 
    cNm_ar = np.zeros(0) #shmf for merged subhalos
    cNs_ar = np.zeros(0) #shmf for surviving subhalos
    for this_fof in [0, 1, 2]:
        for projection in [0, 1, 2]:
            
            if projection == 0:
                pm_proj_dist = pmxydist_f_ar
                ps_proj_dist = psxydist_f_ar
                cm_proj_dist = cmxydist_f_ar
                cs_proj_dist = csxydist_f_ar
            elif projection == 1:
                pm_proj_dist = pmyzdist_f_ar
                ps_proj_dist = psyzdist_f_ar
                cm_proj_dist = cmyzdist_f_ar
                cs_proj_dist = csyzdist_f_ar
            elif projection == 2:
                pm_proj_dist = pmxzdist_f_ar
                ps_proj_dist = psxzdist_f_ar
                cm_proj_dist = cmxzdist_f_ar
                cs_proj_dist = csxzdist_f_ar

            for (ix, ms) in enumerate(mstarpl):
                pNm_ar = np.append(pNm_ar, len(pmmstar_f_ar[(pmmstar_all > ms) & (pmfof == this_fof) & (pm_proj_dist <= rcore_virgo)]))
                pNs_ar = np.append(pNs_ar, len(psmstar_f_ar[(psmstar_all > ms) &  (psfof == this_fof) & (ps_proj_dist <= rcore_virgo) ]))
                cNm_ar = np.append(cNm_ar, len(cmmstar_f_ar[(cmmstar_all > ms) & (cmfof == this_fof) & (cm_proj_dist <= rcore_virgo) ]))
                cNs_ar = np.append(cNs_ar, len(csmstar_f_ar[(csmstar_all > ms) & (csfof == this_fof) & (cs_proj_dist <= rcore_virgo) ]))
    
    ax.plot(mstarpl, get_med_values(cNs_ar + cNm_ar, 9), color = 'darkgreen', label = 'Cutoff', alpha = 0.5, ls = '-')
    ax.plot(mstarpl, get_med_values(pNs_ar + pNm_ar, 9), color = 'red', label = 'Power', alpha = 0.5, ls = '-')
    ax.fill_between(mstarpl, get_quantiles(cNs_ar + cNm_ar, 9)[0], get_quantiles(cNs_ar + cNm_ar, 9)[1], color = 'darkgreen', alpha = 0.3)
    ax.fill_between(mstarpl, get_quantiles(pNs_ar + pNm_ar, 9)[0], get_quantiles(pNs_ar + pNm_ar, 9)[1], color = 'red', alpha = 0.3)
    ax.plot(10**fmstar, fngal_cum, color = 'black', marker = 'o', label = 'Ferrarese+16 Virgo core')
    # for jx in range(len(sigma_pl_ar) + 1): #+2 because we also need to plot without any surface brightness limits
    # pNm_ar = np.zeros(0) #shmf for merged subhalos
    # pNs_ar = np.zeros(0) #shmf for surviving subhalos 
    # cNm_ar = np.zeros(0) #shmf for merged subhalos
    # cNs_ar = np.zeros(0) #shmf for surviving subhalos 
    # Ntng_ar = np.zeros(0)


    # for (ix, ms) in enumerate(mstarpl):
    #     pNm_ar = np.append(pNm_ar, len(pmmstar_f_ar[(pmmstar_all > ms) & (pmfof == 0) & (pmxydist_f_ar < rvir_fornax)]))
    #     pNs_ar = np.append(pNs_ar, len(psmstar_f_ar[(psmstar_all > ms) &  (psfof == 0) & (psxydist_f_ar < rvir_fornax)]))
    #     cNm_ar = np.append(cNm_ar, len(cmmstar_f_ar[(cmmstar_all > ms) & (cmfof == 0) & (cmxydist_f_ar < rvir_fornax)]))
    #     cNs_ar = np.append(cNs_ar, len(csmstar_f_ar[(csmstar_all > ms) & (csfof == 0) & (csxydist_f_ar < rvir_fornax)]))
    #     # Ntng_ar = np.append(Ntng_ar, len(psmstar_f_ar_tng[psmstar_f_ar_tng > ms]))

    # fig, ax = plt.subplots(figsize = (6, 6.25))
    
    # # ax.plot(mstarpl, Ns_ar, color = 'darkgreen', label = 'Model Surviving', alpha = 0.5)
    # ax.plot(mstarpl, cNs_ar + cNm_ar, color = 'darkgreen', label = 'Cutoff', alpha = 0.5)
    # ax.plot(mstarpl, pNs_ar + pNm_ar, color = 'red', label = 'Power law', alpha = 0.5)
    # plot_sachi(ax)
    # ax.plot(mstarpl, Ntng_ar, color = 'blue', label = 'TNG surviving', alpha = 0.5)
    ax.legend(fontsize = 8)
    ax.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
    ax.set_ylabel(r'$N(>M_{\bigstar})$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(left = 1e1)
    ax.tick_params(axis='y', which = 'both', left=True, right=True, direction = 'in')
    ax.tick_params(axis='x', which = 'both', direction = 'in')
    
    # ax.set_title('FoF'+str(fof_no))
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'virgo_mass_function_proj.png')


    # plt.close()
    # mstarpl = np.logspace(1, 11, 100)

    
    # Ntng_ar = np.zeros(0)

    # fig, ax = plt.subplots(figsize = (6, 6.25))
    # ls_ar = ['-.', '--', '-']
    # for this_fof in [0, 1, 2]:
    #     pNm_ar = np.zeros(0) #shmf for merged subhalos
    #     pNs_ar = np.zeros(0) #shmf for surviving subhalos 
    #     cNm_ar = np.zeros(0) #shmf for merged subhalos
    #     cNs_ar = np.zeros(0) #shmf for surviving subhalos 
    #     for (ix, ms) in enumerate(mstarpl):
    #         pNm_ar = np.append(pNm_ar, len(pmmstar_f_ar[(pmmstar_all > ms) & (pmfof == this_fof) & (pmxydist_f_ar <= rcore_virgo)]))
    #         pNs_ar = np.append(pNs_ar, len(psmstar_f_ar[(psmstar_all > ms) &  (psfof == this_fof) & (psxydist_f_ar <= rcore_virgo)]))
    #         cNm_ar = np.append(cNm_ar, len(cmmstar_f_ar[(cmmstar_all > ms) & (cmfof == this_fof) & (cmxydist_f_ar <= rcore_virgo)]))
    #         cNs_ar = np.append(cNs_ar, len(csmstar_f_ar[(csmstar_all > ms) & (csfof == this_fof) & (csxydist_f_ar <= rcore_virgo)]))
    
    #     ax.plot(mstarpl, cNs_ar + cNm_ar, color = 'darkgreen', label = 'Cutoff FoF = ' + str(this_fof), alpha = 0.5, ls = ls_ar[this_fof])
    #     ax.plot(mstarpl, pNs_ar + pNm_ar, color = 'red', label = 'Power law FoF = ' + str(this_fof), alpha = 0.5, ls = ls_ar[this_fof])

    # # ax.plot(mstarpl, Ns_ar, color = 'darkgreen', label = 'Model Surviving', alpha = 0.5)
    # ax.plot(10**fmstar, fngal_cum, color = 'black', marker = 'o', label = 'Ferrarese+16 Virgo core')
    # # plot_sachi(ax)
    # # ax.plot(mstarpl, Ntng_ar, color = 'blue', label = 'TNG surviving', alpha = 0.5)
    # ax.legend(fontsize = 8)
    # ax.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
    # ax.set_ylabel(r'$N(>M_{\bigstar})$')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_xlim(left = 1e1)
    # ax.tick_params(axis='y', which = 'both', left=True, right=True, direction = 'in')
    # ax.tick_params(axis='x', which = 'both', direction = 'in')
    
    # # ax.set_title('FoF'+str(fof_no)+ f' within core {rcore_virgo:.0f} kpc')
    # plt.tight_layout()
    # plt.savefig(this_fof_plotppath + 'mass_function_core.png')


    # plt.close()

    return

plot_subh_mf_core()





def plot_proj_combined():
    '''
    This is to have all the plots in a single panel
    '''
    mstarpl = np.logspace(1, 11, 100)

    rvir_fornax = 700 #kpc
    rcore_virgo = 309 #This is assumed to be the core radius temporarily

    fig, ax = plt.subplots(figsize = (6, 6.25))
    ls_ar = ['-.', '--', '-']
    pNm_ar_virgo = np.zeros(0) #shmf for merged subhalos
    pNs_ar_virgo = np.zeros(0) #shmf for surviving subhalos 
    cNm_ar_virgo = np.zeros(0) #shmf for merged subhalos
    cNs_ar_virgo = np.zeros(0) #shmf for surviving subhalos

    pNm_ar_fnx = np.zeros(0) #shmf for merged subhalos
    pNs_ar_fnx = np.zeros(0) #shmf for surviving subhalos 
    cNm_ar_fnx = np.zeros(0) #shmf for merged subhalos
    cNs_ar_fnx = np.zeros(0) #shmf for surviving subhalos



    for this_fof in [0, 1, 2]:
        for projection in [0, 1, 2]:
            
            if projection == 0:
                pm_proj_dist = pmxydist_f_ar
                ps_proj_dist = psxydist_f_ar
                cm_proj_dist = cmxydist_f_ar
                cs_proj_dist = csxydist_f_ar
            elif projection == 1:
                pm_proj_dist = pmyzdist_f_ar
                ps_proj_dist = psyzdist_f_ar
                cm_proj_dist = cmyzdist_f_ar
                cs_proj_dist = csyzdist_f_ar
            elif projection == 2:
                pm_proj_dist = pmxzdist_f_ar
                ps_proj_dist = psxzdist_f_ar
                cm_proj_dist = cmxzdist_f_ar
                cs_proj_dist = csxzdist_f_ar

            for (ix, ms) in enumerate(mstarpl):
                pNm_ar_virgo = np.append(pNm_ar_virgo, len(pmmstar_f_ar[(pmmstar_all > ms) & (pmfof == this_fof) & (pm_proj_dist <= rcore_virgo)]))
                pNs_ar_virgo = np.append(pNs_ar_virgo, len(psmstar_f_ar[(psmstar_all > ms) &  (psfof == this_fof) & (ps_proj_dist <= rcore_virgo) ]))
                cNm_ar_virgo = np.append(cNm_ar_virgo, len(cmmstar_f_ar[(cmmstar_all > ms) & (cmfof == this_fof) & (cm_proj_dist <= rcore_virgo) ]))
                cNs_ar_virgo = np.append(cNs_ar_virgo, len(csmstar_f_ar[(csmstar_all > ms) & (csfof == this_fof) & (cs_proj_dist <= rcore_virgo) ]))

                pNm_ar_fnx = np.append(pNm_ar_fnx, len(pmmstar_f_ar[(pmmstar_all > ms) & (pmfof == this_fof) & (pm_proj_dist <= rvir_fornax)]))
                pNs_ar_fnx = np.append(pNs_ar_fnx, len(psmstar_f_ar[(psmstar_all > ms) &  (psfof == this_fof) & (ps_proj_dist <= rvir_fornax)]))
                cNm_ar_fnx = np.append(cNm_ar_fnx, len(cmmstar_f_ar[(cmmstar_all > ms) & (cmfof == this_fof) & (cm_proj_dist <= rvir_fornax)]))
                cNs_ar_fnx = np.append(cNs_ar_fnx, len(csmstar_f_ar[(csmstar_all > ms) & (csfof == this_fof) & (cs_proj_dist <= rvir_fornax)]))
    
    ax.plot(mstarpl, get_med_values(cNm_ar_virgo + cNs_ar_virgo, 9), color = 'darkgreen', label = 'Cutoff (< 309 kpc)', alpha = 0.5, ls = '-')
    ax.plot(mstarpl, get_med_values(pNs_ar_virgo + pNm_ar_virgo, 9), color = 'red', label = 'Power law (< 309 kpc)', alpha = 0.5, ls = '-')

    ax.plot(mstarpl, get_med_values(cNm_ar_fnx + cNs_ar_fnx, 9), color = 'darkgreen', label = 'Cutoff (< 700 kpc)', alpha = 0.5, ls = '--')
    ax.plot(mstarpl, get_med_values(pNs_ar_fnx + pNm_ar_fnx, 9), color = 'red', label = 'Power law (< 700 kpc)', alpha = 0.5, ls = '--')

    ax.fill_between(mstarpl, get_quantiles(cNm_ar_virgo + cNs_ar_virgo, 9)[0], get_quantiles(cNm_ar_virgo + cNs_ar_virgo, 9)[1], color = 'darkgreen', alpha = 0.3)
    ax.fill_between(mstarpl, get_quantiles(pNs_ar_virgo + pNm_ar_virgo, 9)[0], get_quantiles(pNs_ar_virgo + pNm_ar_virgo, 9)[1], color = 'red', alpha = 0.3)
    ax.plot(10**vmstar, vngal_cum, color = 'black', marker = '^', ls= '--', label = 'Venhola+19 Fornax')
    ax.plot(10**fmstar, fngal_cum, color = 'black', marker = 'o', label = 'Ferrarese+16 Virgo core')



    # for jx in range(len(sigma_pl_ar) + 1): #+2 because we also need to plot without any surface brightness limits
    # pNm_ar = np.zeros(0) #shmf for merged subhalos
    # pNs_ar = np.zeros(0) #shmf for surviving subhalos 
    # cNm_ar = np.zeros(0) #shmf for merged subhalos
    # cNs_ar = np.zeros(0) #shmf for surviving subhalos 
    # Ntng_ar = np.zeros(0)


    # for (ix, ms) in enumerate(mstarpl):
    #     pNm_ar = np.append(pNm_ar, len(pmmstar_f_ar[(pmmstar_all > ms) & (pmfof == 0) & (pmxydist_f_ar < rvir_fornax)]))
    #     pNs_ar = np.append(pNs_ar, len(psmstar_f_ar[(psmstar_all > ms) &  (psfof == 0) & (psxydist_f_ar < rvir_fornax)]))
    #     cNm_ar = np.append(cNm_ar, len(cmmstar_f_ar[(cmmstar_all > ms) & (cmfof == 0) & (cmxydist_f_ar < rvir_fornax)]))
    #     cNs_ar = np.append(cNs_ar, len(csmstar_f_ar[(csmstar_all > ms) & (csfof == 0) & (csxydist_f_ar < rvir_fornax)]))
    #     # Ntng_ar = np.append(Ntng_ar, len(psmstar_f_ar_tng[psmstar_f_ar_tng > ms]))

    # fig, ax = plt.subplots(figsize = (6, 6.25))
    
    # # ax.plot(mstarpl, Ns_ar, color = 'darkgreen', label = 'Model Surviving', alpha = 0.5)
    # ax.plot(mstarpl, cNs_ar + cNm_ar, color = 'darkgreen', label = 'Cutoff', alpha = 0.5)
    # ax.plot(mstarpl, pNs_ar + pNm_ar, color = 'red', label = 'Power law', alpha = 0.5)
    # plot_sachi(ax)
    # ax.plot(mstarpl, Ntng_ar, color = 'blue', label = 'TNG surviving', alpha = 0.5)
    ax.legend(fontsize = 8)
    ax.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
    ax.set_ylabel(r'$N(>M_{\bigstar})$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(left = 1e1)
    ax.tick_params(axis='y', which = 'both', left=True, right=True, direction = 'in')
    ax.tick_params(axis='x', which = 'both', direction = 'in')
    
    # ax.set_title('FoF'+str(fof_no))
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'mass_function_proj.png')


    plt.close()
    return None


plot_proj_combined()





def plot_proj_everything():
    '''
    This is to combine observations and also mass function at different radii
    '''
    mstarpl = np.logspace(1, 11, 100)

    rvir_fornax = 700 #kpc
    rcore_virgo = 309 #This is assumed to be the core radius temporarily

    fig, ax = plt.subplots(figsize = (6, 6.25))
    ls_ar = ['-.', '--', '-']

    pNm_ar_150 = np.zeros(0) #shmf for merged subhalos in 150 kpc
    pNs_ar_150 = np.zeros(0) #shmf for surviving subhalos in 150 kpc
    cNm_ar_150 = np.zeros(0) #shmf for merged subhalos in 150 kpc
    cNs_ar_150 = np.zeros(0) #shmf for surviving subhalos in 150 kpc

    pNm_ar_virgo = np.zeros(0) #shmf for merged subhalos
    pNs_ar_virgo = np.zeros(0) #shmf for surviving subhalos 
    cNm_ar_virgo = np.zeros(0) #shmf for merged subhalos
    cNs_ar_virgo = np.zeros(0) #shmf for surviving subhalos

    pNm_ar_fnx = np.zeros(0) #shmf for merged subhalos
    pNs_ar_fnx = np.zeros(0) #shmf for surviving subhalos 
    cNm_ar_fnx = np.zeros(0) #shmf for merged subhalos
    cNs_ar_fnx = np.zeros(0) #shmf for surviving subhalos



    for this_fof in [0, 1, 2]: #For each FoF
        if this_fof == 0:
            rvir = rvir_fof0
        elif this_fof == 1:
            rvir = rvir_fof1
        elif this_fof == 2:
            rvir = rvir_fof2
        for projection in [0, 1, 2]: #for xy, yz and zx
            
            if projection == 0:
                pm_proj_dist = pmxydist_f_ar
                ps_proj_dist = psxydist_f_ar
                cm_proj_dist = cmxydist_f_ar
                cs_proj_dist = csxydist_f_ar
            elif projection == 1:
                pm_proj_dist = pmyzdist_f_ar
                ps_proj_dist = psyzdist_f_ar
                cm_proj_dist = cmyzdist_f_ar
                cs_proj_dist = csyzdist_f_ar
            elif projection == 2:
                pm_proj_dist = pmxzdist_f_ar
                ps_proj_dist = psxzdist_f_ar
                cm_proj_dist = cmxzdist_f_ar
                cs_proj_dist = csxzdist_f_ar

            for (ix, ms) in enumerate(mstarpl):
                frac1 = 0.2
                pNm_ar_virgo = np.append(pNm_ar_virgo, len(pmmstar_f_ar[(pmmstar_all > ms) & (pmfof == this_fof) & (pm_proj_dist <= frac1 * rvir)]))
                pNs_ar_virgo = np.append(pNs_ar_virgo, len(psmstar_f_ar[(psmstar_all > ms) &  (psfof == this_fof) & (ps_proj_dist <= frac1 * rvir) ]))
                cNm_ar_virgo = np.append(cNm_ar_virgo, len(cmmstar_f_ar[(cmmstar_all > ms) & (cmfof == this_fof) & (cm_proj_dist <= frac1 * rvir) ]))
                cNs_ar_virgo = np.append(cNs_ar_virgo, len(csmstar_f_ar[(csmstar_all > ms) & (csfof == this_fof) & (cs_proj_dist <= frac1 * rvir) ]))

                frac2 = 0.7
                pNm_ar_fnx = np.append(pNm_ar_fnx, len(pmmstar_f_ar[(pmmstar_all > ms) & (pmfof == this_fof) & (pm_proj_dist <= frac2 * rvir)]))
                pNs_ar_fnx = np.append(pNs_ar_fnx, len(psmstar_f_ar[(psmstar_all > ms) &  (psfof == this_fof) & (ps_proj_dist <= frac2 * rvir)]))
                cNm_ar_fnx = np.append(cNm_ar_fnx, len(cmmstar_f_ar[(cmmstar_all > ms) & (cmfof == this_fof) & (cm_proj_dist <= frac2 * rvir)]))
                cNs_ar_fnx = np.append(cNs_ar_fnx, len(csmstar_f_ar[(csmstar_all > ms) & (csfof == this_fof) & (cs_proj_dist <= frac2 * rvir)]))

                frac3 = 1
                pNm_ar_150 = np.append(pNm_ar_150, len(pmmstar_f_ar[(pmmstar_all > ms) & (pmfof == this_fof) & (pm_proj_dist <= frac3 * rvir)]))
                pNs_ar_150 = np.append(pNs_ar_150, len(psmstar_f_ar[(psmstar_all > ms) &  (psfof == this_fof) & (ps_proj_dist <= frac3 * rvir) ]))
                cNm_ar_150 = np.append(cNm_ar_150, len(cmmstar_f_ar[(cmmstar_all > ms) & (cmfof == this_fof) & (cm_proj_dist <= frac3 * rvir) ]))
                cNs_ar_150 = np.append(cNs_ar_150, len(csmstar_f_ar[(csmstar_all > ms) & (csfof == this_fof) & (cs_proj_dist <= frac3 * rvir) ]))
    
    ax.plot(mstarpl, get_med_values(cNm_ar_virgo + cNs_ar_virgo, 9), color = 'darkgreen', label = 'Cutoff (0.2 rvir)', alpha = 0.5, ls = '-')
    ax.plot(mstarpl, get_med_values(pNs_ar_virgo + pNm_ar_virgo, 9), color = 'red', label = 'Power law (0.2 rvir)', alpha = 0.5, ls = '-')

    ax.plot(mstarpl, get_med_values(cNm_ar_fnx + cNs_ar_fnx, 9), color = 'darkgreen', label = 'Cutoff (0.7 rvir)', alpha = 0.5, ls = '--')
    ax.plot(mstarpl, get_med_values(pNs_ar_fnx + pNm_ar_fnx, 9), color = 'red', label = 'Power law (0.7 rvir)', alpha = 0.5, ls = '--')

    ax.plot(mstarpl, get_med_values(cNm_ar_150 + cNs_ar_150, 9), color = 'darkgreen', label = 'Cutoff (rvir)', alpha = 0.5, ls = ':')
    ax.plot(mstarpl, get_med_values(pNs_ar_150 + pNm_ar_150, 9), color = 'red', label = 'Power law (rvir)', alpha = 0.5, ls = ':')

    ax.fill_between(mstarpl, get_quantiles(cNm_ar_virgo + cNs_ar_virgo, 9)[0], get_quantiles(cNm_ar_virgo + cNs_ar_virgo, 9)[1], color = 'darkgreen', alpha = 0.3)
    ax.fill_between(mstarpl, get_quantiles(pNs_ar_virgo + pNm_ar_virgo, 9)[0], get_quantiles(pNs_ar_virgo + pNm_ar_virgo, 9)[1], color = 'red', alpha = 0.3)
    ax.plot(10**vmstar, vngal_cum, color = 'black', marker = '^', ls= '--', label = 'Venhola+19 Fornax')
    ax.plot(10**fmstar, fngal_cum, color = 'black', marker = 'o', label = 'Ferrarese+16 Virgo core')



    # for jx in range(len(sigma_pl_ar) + 1): #+2 because we also need to plot without any surface brightness limits
    # pNm_ar = np.zeros(0) #shmf for merged subhalos
    # pNs_ar = np.zeros(0) #shmf for surviving subhalos 
    # cNm_ar = np.zeros(0) #shmf for merged subhalos
    # cNs_ar = np.zeros(0) #shmf for surviving subhalos 
    # Ntng_ar = np.zeros(0)


    # for (ix, ms) in enumerate(mstarpl):
    #     pNm_ar = np.append(pNm_ar, len(pmmstar_f_ar[(pmmstar_all > ms) & (pmfof == 0) & (pmxydist_f_ar < rvir_fornax)]))
    #     pNs_ar = np.append(pNs_ar, len(psmstar_f_ar[(psmstar_all > ms) &  (psfof == 0) & (psxydist_f_ar < rvir_fornax)]))
    #     cNm_ar = np.append(cNm_ar, len(cmmstar_f_ar[(cmmstar_all > ms) & (cmfof == 0) & (cmxydist_f_ar < rvir_fornax)]))
    #     cNs_ar = np.append(cNs_ar, len(csmstar_f_ar[(csmstar_all > ms) & (csfof == 0) & (csxydist_f_ar < rvir_fornax)]))
    #     # Ntng_ar = np.append(Ntng_ar, len(psmstar_f_ar_tng[psmstar_f_ar_tng > ms]))

    # fig, ax = plt.subplots(figsize = (6, 6.25))
    
    # # ax.plot(mstarpl, Ns_ar, color = 'darkgreen', label = 'Model Surviving', alpha = 0.5)
    # ax.plot(mstarpl, cNs_ar + cNm_ar, color = 'darkgreen', label = 'Cutoff', alpha = 0.5)
    # ax.plot(mstarpl, pNs_ar + pNm_ar, color = 'red', label = 'Power law', alpha = 0.5)
    # plot_sachi(ax)
    # ax.plot(mstarpl, Ntng_ar, color = 'blue', label = 'TNG surviving', alpha = 0.5)
    ax.legend(fontsize = 8)
    ax.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
    ax.set_ylabel(r'$N(>M_{\bigstar})$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(left = 1e1)
    ax.tick_params(axis='y', which = 'both', left=True, right=True, direction = 'in')
    ax.tick_params(axis='x', which = 'both', direction = 'in')
    
    # ax.set_title('FoF'+str(fof_no))
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'mass_function_all_in_one.png')


    plt.close()
    return None


plot_proj_everything()


def plot_mass_fn_completeness():
    '''
    This is to plot the mass function with different surface brightness limits along with completeness in the bottom panel
    '''
    fig, (ax, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize = (6, 6.25), gridspec_kw={'height_ratios':[2, 1]}, sharex=True)
    mstarpl = np.logspace(1, 11, 100)

    sigma_pl_ar = [24, 28, 35] #This is the surface brightness cuts that we would be using
    ls_ar = ['-.', '--', '-']
    for jx in range(len(sigma_pl_ar)): #+2 because we also need to plot without any surface brightness limits
        pNm_ar = np.zeros(0) #shmf for merged subhalos
        pNs_ar = np.zeros(0) #shmf for surviving subhalos 
        cNm_ar = np.zeros(0) #shmf for merged subhalos
        cNs_ar = np.zeros(0) #shmf for surviving subhalos 
        Ntng_ar = np.zeros(0)
        for (ix, ms) in enumerate(mstarpl):
            pNm_ar = np.append(pNm_ar, len(pmmstar_f_ar[(pmmstar_all > ms) & (pmfof == 0) & (pmsigma_all < sigma_pl_ar[jx])]))
            pNs_ar = np.append(pNs_ar, len(psmstar_f_ar[(psmstar_all > ms) &  (psfof == 0) & (pssigma_all < sigma_pl_ar[jx])]))
            cNm_ar = np.append(cNm_ar, len(cmmstar_f_ar[(cmmstar_all > ms) & (cmfof == 0) & (cmsigma_all < sigma_pl_ar[jx])]))
            cNs_ar = np.append(cNs_ar, len(csmstar_f_ar[(csmstar_all > ms) & (csfof == 0) & (cssigma_all < sigma_pl_ar[jx])]))
            # Ntng_ar = np.append(Ntng_ar, len(psmstar_f_ar_tng[psmstar_f_ar_tng > ms]))

        # fig, ax = plt.subplots(figsize = (6, 6.25))
        
        # ax.plot(mstarpl, Ns_ar, color = 'darkgreen', label = 'Model Surviving', alpha = 0.5)
        if jx == -1: #NULL
            ax.plot(mstarpl, cNs_ar + cNm_ar, color = 'darkgreen', ls = ls_ar[jx], label = r'Cutoff all', alpha = 0.5)
            ax.plot(mstarpl, pNs_ar + pNm_ar, color = 'red', ls = ls_ar[jx], label = r'Power all', alpha = 0.5)
        else:
            ax.plot(mstarpl, cNs_ar + cNm_ar, color = 'darkgreen', ls = ls_ar[jx], label = r'Cutoff $\Sigma < $'+str(sigma_pl_ar[jx]), alpha = 0.5)
            ax.plot(mstarpl, pNs_ar + pNm_ar, color = 'red', ls = ls_ar[jx], label = r'Power law $\Sigma < $'+str(sigma_pl_ar[jx]), alpha = 0.5)
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.tick_params(axis='both', which = 'both', left=True, right=True, bottom = True, top = True, direction = 'in')
    ax.legend(fontsize = 8)
    ax.set_ylabel(r'$N(>M_{\bigstar})$')
    ax2.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')


    for ix in range(len(sigma_pl_ar)):
        ccompleteness_ar = np.zeros(0)
        pcompleteness_ar = np.zeros(0)
        cmstar_pl_ar = np.zeros(0)
        pmstar_pl_ar = np.zeros(0)
        for jx in range(len(mstarpl) - 1):
            try:
                ccompleteness_ar = np.append(ccompleteness_ar, (len(cmmstar_all[(cmsigma_all < sigma_pl_ar[ix]) & (mstarpl[jx] < cmmstar_all) & (cmmstar_all < mstarpl[jx+1])]) + len(csmstar_all[(cssigma_all < sigma_pl_ar[ix]) & (mstarpl[jx] < csmstar_all) & (csmstar_all < mstarpl[jx+1])]))/(len(cmmstar_all[(mstarpl[jx] < cmmstar_all) & (cmmstar_all < mstarpl[jx+1])]) + len(csmstar_all[(mstarpl[jx] < csmstar_all) & (csmstar_all < mstarpl[jx+1])])))
                cmstar_pl_ar =np.append(cmstar_pl_ar, mstarpl[jx])
            except Exception as e:
                pass
            try:
                pcompleteness_ar = np.append(pcompleteness_ar, (len(pmmstar_all[(pmsigma_all < sigma_pl_ar[ix]) & (mstarpl[jx] < pmmstar_all) & (pmmstar_all < mstarpl[jx+1])]) + len(psmstar_all[(pssigma_all < sigma_pl_ar[ix]) & (mstarpl[jx] < psmstar_all) & (psmstar_all < mstarpl[jx+1])]))/(len(pmmstar_all[(mstarpl[jx] < pmmstar_all) & (pmmstar_all < mstarpl[jx+1])]) + len(psmstar_all[(mstarpl[jx] < psmstar_all) & (psmstar_all < mstarpl[jx+1])])))
                pmstar_pl_ar =np.append(pmstar_pl_ar, mstarpl[jx])
            except Exception as e:
                pass
        ax2.plot(cmstar_pl_ar, ccompleteness_ar, color = 'darkgreen', ls = ls_ar[ix], label = r'Cutoff $\Sigma < $'+str(sigma_pl_ar[ix]))
        ax2.plot(pmstar_pl_ar, pcompleteness_ar, color = 'red', ls = ls_ar[ix], label = r'Power law $\Sigma < $'+str(sigma_pl_ar[ix]))
    
    ax2.set_ylim(bottom = -0.1, top = 1.2)
    ax2.tick_params(axis='both', which = 'both', left=True, right=True, bottom = True, top = True, direction = 'in')
    ax2.set_ylabel('Completeness')
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'mass_function_completeness.png')
    plt.close()
    return None

plot_mass_fn_completeness()


def just_dwarfs():
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 6.15))
    plot_lg_virgo(ax, alpha = 0.8, mec = 'black')
    ax.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$')
    ax.set_ylabel(r'$R_{\rm{h}}$ (pc)')
    ax.legend(fontsize = 8)
    ax.tick_params(axis='both', which = 'both', left=True, right=True, bottom = True, top = True, direction = 'in')
    # ax.text(0.01, 0.99, 'Cutoff', ha = 'left', va = 'top', transform=ax.transAxes)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'dwarfs.png')

    return None

just_dwarfs()


def plot_rh_vs_mstar(fof_no = 210, alpha_points = 0.1, size_points = 7.5):
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
    ax.set_xlim(left = 1e1, right = right_data)
    ax.set_ylim(bottom = 10, top = top_data)
    


    if fof_no == 210:
        # csix = np.where((csfof == 0) |( csfof == 1) |( csfof == 2))[0]
        # cmix = np.where((cmfof == 0) | (cmfof == 1) | (cmfof == 2))[0]
        # psix = np.where((psfof == 0) | (psfof == 1) | (psfof == 2))[0]
        # pmix = np.where((pmfof == 0) | (pmfof == 1) | (pmfof == 2))[0]
        csix = np.where(csfof >-1)[0]
        cmix = np.where(cmfof >-1)[0]
        psix = np.where(psfof >-1)[0]
        pmix = np.where(pmfof >-1)[0]
    else:
        csix = np.where(csfof == fof_no)[0]
        cmix = np.where(cmfof == fof_no)[0]
        psix = np.where(psfof == fof_no)[0]
        pmix = np.where(pmfof == fof_no)[0]


    ax.scatter(csmstar_all[csix], np.array(csrh_all[csix]) * 1e3, marker = 's', fc = 'darkgreen', alpha = alpha_points, s = size_points, label = 'Survived', zorder = 200, edgecolor = 'darkgreen', linewidth = 0.7)
    ax.scatter(cmmstar_all[cmix], np.array(cmrh_all[cmix]) * 1e3, marker = 's', fc = 'purple', alpha = alpha_points, s = size_points, label = 'Merged', zorder = 200, edgecolor = 'purple', linewidth = 0.7)
    top_data = ax.get_ylim()[1]
    right_data = ax.get_xlim()[1]
    ax.plot(10**mspl_log, 1e3 * 10**get_lrh(mspl_log), color = 'k')
    ax.set_xlim(left = 1e1, right = right_data)
    ax.set_ylim(bottom = 10, top = top_data)

    ax2.scatter(psmstar_all[psix], np.array(psrh_all[psix]) * 1e3, marker = 's', fc = 'darkgreen', alpha = alpha_points, s = size_points, label = 'Survived', zorder = 200, edgecolor = 'darkgreen', linewidth = 0.7)
    ax2.scatter(pmmstar_all[pmix], np.array(pmrh_all[pmix]) * 1e3, marker = 's', fc = 'purple', alpha = alpha_points, s = size_points, label = 'Merged', zorder = 200, edgecolor = 'purple', linewidth = 0.7)
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

# plot_rh_vs_mstar(fof_no)



def plot_rh_vs_mstar_hist2d(fof_no, alpha_points = 0.1, size_points = 7.5):
    '''
    This is to plot the 2d histogram of size-mass relation
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
    # plot_lg_virgo(ax)
    # plot_lg_virgo(ax2)
    


    if fof_no == 210:
        csix = np.where((csfof == 0) | (csfof == 1 )| (csfof == 2))[0]
        cmix = np.where((cmfof == 0) |( cmfof == 1) | (cmfof == 2))[0]
        psix = np.where((psfof == 0) |( psfof == 1) | (psfof == 2))[0]
        pmix = np.where((pmfof == 0) | (pmfof == 1) | (pmfof == 2))[0]
    else:
        csix = np.where(csfof == fof_no)[0]
        cmix = np.where(cmfof == fof_no)[0]
        psix = np.where(psfof == fof_no)[0]
        pmix = np.where(pmfof == fof_no)[0]


    # ax.scatter(csmstar_all[csix], np.array(csrh_all[csix]) * 1e3, marker = 's', fc = 'darkgreen', alpha = alpha_points, s = size_points, label = 'Survived', zorder = 200, edgecolor = 'darkgreen', linewidth = 0.7)
    # ax.scatter(cmmstar_all[cmix], np.array(cmrh_all[cmix]) * 1e3, marker = 's', fc = 'purple', alpha = alpha_points, s = size_points, label = 'Merged', zorder = 200, edgecolor = 'purple', linewidth = 0.7)
    
    cmstar = np.append(csmstar_all[csix], cmmstar_all[cmix]) #msun, THis is all the subhalos in the cutoff model
    crh = np.append(csrh_all[csix], cmrh_all[cmix]) * 1e3 #kpc, This is rh of all the subhalos in the cutoff model
    cmstar = cmstar[crh > 0]
    crh = crh[crh > 0]


    pmstar = np.append(psmstar_all[psix], pmmstar_all[pmix])
    prh = np.append(psrh_all[psix], pmrh_all[pmix]) * 1e3
    pmstar = pmstar[prh > 0]
    prh = prh[prh > 0]


    y_space = np.logspace(np.log10(min(crh)), np.log10(max(crh)), 100)
    x_space = np.logspace(np.log10(min(cmstar)), np.log10(max(cmstar)), 100)


    y_space1 = np.logspace(np.log10(min(prh)), np.log10(max(prh)), 100)
    x_space1 = np.logspace(np.log10(min(pmstar)), np.log10(max(pmstar)), 100)

    hist1, _, _ = np.histogram2d(cmstar, crh, bins=(x_space, y_space))
    hist2, _, _ = np.histogram2d(pmstar, prh, bins=(x_space1, y_space1))

    vmin = min(hist1.min(), hist2.min())
    vmax = max(hist1.max(), hist2.max())

    print(vmin, vmax)

    ax.scatter(cmstar, crh, marker = 's', fc = 'darkblue', alpha = alpha_points, s = size_points, zorder = 0, edgecolor = 'darkblue', linewidth = 0.7)
    ax.hist2d(cmstar, crh, bins = (x_space, y_space), cmin = 9, norm = 'log', zorder = 100, vmin = 9, vmax = vmax)
    ax.plot(10**mspl_log, 1e3 * 10**get_lrh(mspl_log), color = 'k', zorder = 200)
    top_data = ax.get_ylim()[1]
    right_data = ax.get_xlim()[1]
    ax.set_xlim(left = 1e1, right = right_data)
    ax.set_ylim(bottom = 10, top = top_data)

    # ax2.scatter(psmstar_all[psix], np.array(psrh_all[psix]) * 1e3, marker = 's', fc = 'darkgreen', alpha = alpha_points, s = size_points, label = 'Survived', zorder = 200, edgecolor = 'darkgreen', linewidth = 0.7)
    # ax2.scatter(pmmstar_all[pmix], np.array(pmrh_all[pmix]) * 1e3, marker = 's', fc = 'purple', alpha = alpha_points, s = size_points, label = 'Merged', zorder = 200, edgecolor = 'purple', linewidth = 0.7)
    
    
    
    ax2.scatter(pmstar, prh, marker = 's', fc = 'darkblue', alpha = alpha_points, s = size_points, zorder = 0, edgecolor = 'darkblue', linewidth = 0.7)
    
    ax2.hist2d(pmstar, prh, bins = (x_space1, y_space1), cmin = 9, norm = 'linear', zorder = 100, vmin = 9, vmax = vmax)
    ax2.plot(10**mspl_log, 1e3 * 10**get_lrh(mspl_log), color = 'k', zorder = 200)
    top_data2 = ax2.get_ylim()[1]
    right_data2 = ax2.get_xlim()[1]
    ax2.set_xlim(left = 1e1, right = right_data2)
    ax2.set_ylim(bottom = 10, top = top_data2)


    if True: #This section is for lines of constant surface brightness
        angle = 62
        yval = 0.21*top_data
        ax.plot(10**mspl_log, get_line_of_constant_surfb(10**mspl_log, 24), ls = '--', color = 'gray')
        ax.annotate(r'24 mag arcsec$^{-2}$', xy = (inverse_get_line_of_constant_surfb(yval, 24), yval), xytext = (inverse_get_line_of_constant_surfb(yval, 24), 1.05* yval), 
                rotation=angle, color='gray', fontsize=10, rotation_mode='anchor')
        ax.plot(10**mspl_log, get_line_of_constant_surfb(10**mspl_log, 28), ls = '--', color = 'gray')
        ax.annotate(r'28 mag arcsec$^{-2}$', xy = (inverse_get_line_of_constant_surfb(yval, 28), yval), xytext = (inverse_get_line_of_constant_surfb(yval, 28), 1.05* yval), 
                rotation=angle, color='gray', fontsize=10, rotation_mode='anchor')
        ax.plot(10**mspl_log, get_line_of_constant_surfb(10**mspl_log, 35), ls = '--', color = 'gray')
        ax.annotate(r'35 mag arcsec$^{-2}$', xy = (inverse_get_line_of_constant_surfb(yval, 35), yval), xytext = (inverse_get_line_of_constant_surfb(yval, 35), 1.05* yval), 
                rotation=angle, color='gray', fontsize=10, rotation_mode='anchor')
        
        yval = 0.21*top_data2
        ax2.plot(10**mspl_log, get_line_of_constant_surfb(10**mspl_log, 24), ls = '--', color = 'gray')
        ax2.annotate(r'24 mag arcsec$^{-2}$', xy = (inverse_get_line_of_constant_surfb(yval, 24), yval), xytext = (inverse_get_line_of_constant_surfb(yval, 24), 1.05* yval), 
                rotation=angle, color='gray', fontsize=10, rotation_mode='anchor')
        ax2.plot(10**mspl_log, get_line_of_constant_surfb(10**mspl_log, 28), ls = '--', color = 'gray')
        ax2.annotate(r'28 mag arcsec$^{-2}$', xy = (inverse_get_line_of_constant_surfb(yval, 28), yval), xytext = (inverse_get_line_of_constant_surfb(yval, 28), 1.05* yval), 
                rotation=angle, color='gray', fontsize=10, rotation_mode='anchor')
        ax2.plot(10**mspl_log, get_line_of_constant_surfb(10**mspl_log, 35), ls = '--', color = 'gray')
        ax2.annotate(r'35 mag arcsec$^{-2}$', xy = (inverse_get_line_of_constant_surfb(yval, 35), yval), xytext = (inverse_get_line_of_constant_surfb(yval, 35), 1.05* yval), 
                rotation=angle, color='gray', fontsize=10, rotation_mode='anchor')
    

    # plot_lg_virgo_some(ax, alpha = 0.2)
    # plot_lg_virgo_some(ax2, alpha = 0.2)

    # # Example usage:
    # R = 100  # Half light radius in pc
    # S = 25  # Surface brightness
    # mstar_inverse = inverse_get_line_of_constant_surfb(R, S)
    # print(mstar_inverse)
    ax.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$', fontsize = label_font)
    ax.set_ylabel(r'$R_{\rm{h}}$ (pc)', fontsize = label_font)
    ax.text(0.01, 0.99, 'Cutoff', ha = 'left', va = 'top', transform=ax.transAxes)
    ax.legend(fontsize = 8, loc = 'lower right')
    ax.tick_params(axis='both', which = 'both', left=True, right=True, bottom = True, top = True, direction = 'in')

    ax2.set_xlabel(r'$M_{\bigstar}\,\rm{(M_\odot)}$', fontsize = label_font)
    ax2.set_ylabel(r'$R_{\rm{h}}$ (pc)', fontsize = label_font)
    ax2.text(0.01, 0.99, 'Power law', ha = 'left', va = 'top', transform=ax2.transAxes)
    ax2.legend(fontsize = 8, loc = 'lower right')
    ax2.tick_params(axis='both', which = 'both', left=True, right=True, bottom = True, top = True, direction = 'in')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    # fig.suptitle('FoF'+str(fof_no), fontsize = 14)
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'rh_vs_mstar_hist.png')
    plt.close()
    return None

plot_rh_vs_mstar_hist2d(210)





# rpl = np.logspace(1, 3.2, 100) #r = m





def plot_vr_vs_r():
    '''
    This is to plot the radial velocity versus the distance from the center, colored by fraction of the stellar mass remaining at z = 0
    '''
    ps_vr_ar = np.einsum('ij,ij->i', psvel_f_ar, pspos_f_ar)/psdist_f_ar
    pm_vr_ar = np.einsum('ij,ij->i', pmvel_f_ar, pmpos_ar)/pmdist_f_ar

    pdist_f_ar = np.append(psdist_f_ar, pmdist_f_ar)
    pvr_ar = np.append(ps_vr_ar, pm_vr_ar)
    p_log_fstar_ar = np.append(np.log10(psmstar_all/psmstar_max_all), np.log10(pmmstar_all/pmmstar_max_all))
    pfof_ar = np.append(psfof, pmfof)


    cut = (10 < pdist_f_ar) & (pdist_f_ar < 1000) & (pfof_ar == 0)
    pdist_f_ar = pdist_f_ar[cut]
    pvr_ar = pvr_ar[cut]
    p_log_fstar_ar = p_log_fstar_ar[cut]
    


    num_infinities = np.isinf(p_log_fstar_ar).sum()
    print("Number of infinities in p_log_fstar_ar:", num_infinities)
    p_log_fstar_ar[np.isinf(p_log_fstar_ar)] = np.nan

    fig, ax = plt.subplots(figsize = (6, 6))
    ax.hist(p_log_fstar_ar)
    ax.set_xlabel('log fstar')
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'trash/test.png')

    fig, ax = plt.subplots(figsize = (7.25, 6))    
    print(p_log_fstar_ar[:20])
    # print('min and max of fstar: ', p_log_fstar_ar.min(), p_log_fstar_ar.max())
    # print('min and max of vr: ', pvr_ar.min(), pvr_ar.max())
    # print('min and max of dist: ', pdist_f_ar.min(), pdist_f_ar.max())

    bin_avg_lfstar, xedges_sat, yedges_sat, binnumber = stats.binned_statistic_2d(pdist_f_ar, pvr_ar, p_log_fstar_ar, statistic='mean', bins=100) 
    max_mass = bin_avg_lfstar.max()/10
    min_mass = 50000 * 8/5 

    bin_avg_lfstar[np.isinf(bin_avg_lfstar)] = np.nan
    print('fstar min and max', bin_avg_lfstar)

    # print('min and max of mass bins: ', min_mass, max_mass)

    extent = xedges_sat[0], xedges_sat[-1], yedges_sat[0], yedges_sat[-1]
    im3 = ax.imshow(bin_avg_lfstar.T, cmap = 'viridis', extent = extent, origin='lower', interpolation = 'nearest', aspect = 'auto', vmin = -2)

    cb = fig.colorbar(im3, ax = ax)
    cb.set_label(r'$\log (M_{\rm{\bigstar, z= 0}} / M_{\rm{\bigstar, max}})$', fontsize = 14)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='3%', pad=0.05)
    # cb = fig.colorbar(im3, cax=cax, orientation="vertical")
    # cb.set_label(r'$\log (M_{\rm{\bigstar, z= 0}} / M_{\rm{\bigstar, max}})$', fontsize = 14)
    # cb.ax.xaxis.set_label_position('right')
    # cb.ax.xaxis.set_ticks_position('right')


    # sc = ax.scatter(pdist_f_ar, pvr_ar, c = p_log_fstar_ar, cmap = 'viridis', alpha = 0.5, s = 1, marker = 'o', label = 'Power law')


    # sc1 = ax.scatter(pmdist_f_ar, pm_vr_ar, c =, cmap = 'viridis', alpha = 0.5, s = 1, marker = 's', label = 'Power law killed')
    # ax.set_xscale('log')
    # cbar = plt.colorbar(sc, ax = ax)
    # cbar.set_label(r'$\log (M_{\rm{\bigstar, z= 0}} / M_{\rm{\bigstar, max}})$')
    ax.set_xlabel(r'$ r\,({\rm{kpc}})$')
    ax.set_ylabel(r'$v_r$ (km/s)')
    # ax.set_ylabel(r'$v_r$ (km/s)')
    # ax.legend(fontsize = 8)
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'vr_vs_r.png')
    plt.close()
    return None


plot_vr_vs_r()




def fstar_vs_dist():
    '''
    This is the plot for the remnant stellar fraction versus the distance from the center 
    Colored by the stellar mass at z = 0
    '''
    fig,ax = plt.subplots(figsize = (7.25, 6))
    sc = ax.scatter(psdist_f_ar, psmstar_all/psmstar_max_all, c = np.log10(psmstar_f_ar), cmap = 'viridis', alpha = 0.5, s = 10, label = 'Power law')
    sc1 = ax.scatter(psdist_f_ar[psmstar_max_all > 5e6], psmstar_f_ar_tng[psmstar_max_all > 5e6]/psmstar_max_all[psmstar_max_all > 5e6], color = 'gray', alpha = 0.5, s = 3, label = 'TNG')
    # ax.scatter(cmdist_f_ar, cmdf_ar, c = cmmstar_f_ar, cmap = 'viridis', alpha = 0.5, s = 10, label = 'Cutoff')
    cbar = plt.colorbar(sc, ax = ax)
    cbar.set_label(r'$\log M_{\rm{\bigstar, z= 0}}$')

    # cbar1 = 
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Distance from center (kpc)')
    ax.set_ylabel(r'$f_{\bigstar}$')
    ax.legend(fontsize = 8)
    ax.set_ylim(bottom = 1e-3)
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'fstar_vs_dist.png')

    return None

# fstar_vs_dist()


def plot_tng_proj(fig, ax):
    cat_sub = il.groupcat.loadSubhalos(basePath, 99, fields = ['SubhaloGrNr','SubhaloMassType', 'SubhaloMassInRadType', 'SubhaloMassInHalfRadType','SubhaloHalfmassRadType', 'SubhaloPos'])

    pos_sub = cat_sub['SubhaloPos']
    r_h_sub = cat_sub['SubhaloHalfmassRadType'][:,4] * 1/0.6774
    grnr = cat_sub['SubhaloGrNr']
    mstr_sub = cat_sub['SubhaloMassType'][:,4] * 1e10/0.6774

    sfid = np.arange(len(grnr))

    outpath  = '/rhome/psadh003/bigdata/tng50/output_files/'
    result_satellite = np.load(outpath + 'fof0_plot.npy')

    x_pos_satellite = result_satellite[:,0]
    y_pos_satellite = result_satellite[:,1]
    z_pos_satellite = result_satellite[:,2]
    mass_satellite = result_satellite[:,3]

    r_satellite = np.sqrt((x_pos_satellite**2)+(y_pos_satellite**2)+(z_pos_satellite**2))

    # aux = r_satellite < r_vir_cen
    # Testing different definitions of radii
    aux = (np.sqrt((x_pos_satellite**2)+(y_pos_satellite**2)) < 300)

    x_pos_satellite = x_pos_satellite[aux]
    y_pos_satellite = y_pos_satellite[aux]
    z_pos_satellite = z_pos_satellite[aux]
    mass_satellite = mass_satellite[aux]

    print(len(x_pos_satellite))
    print('max min mstar pp:', mass_satellite.max(), mass_satellite.min())
    mass_bin_sat, xedges_sat, yedges_sat, binnumber = stats.binned_statistic_2d(x_pos_satellite, y_pos_satellite, mass_satellite, statistic='sum', bins=500) #We are adding the mass of all the satellites in a given bin


    print('shape of mass_bin_sat in pp: ', mass_bin_sat.shape)
    # XX_sat, YY_sat = np.meshgrid(xedges_sat, yedges_sat) #This maybe just generates the cordinates for the plot

    max_mass = mass_bin_sat.max()/10
    min_mass = 50000 * 8/5

    extent = xedges_sat[0], xedges_sat[-1], yedges_sat[0], yedges_sat[-1]
    im3 = ax[0].imshow(mass_bin_sat.T, cmap = 'inferno', norm = 'log', vmin = 1e4, vmax = max_mass, interpolation='nearest', extent = extent, origin='lower')

    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('top', size='3%', pad=0.05)
    cb = fig.colorbar(im3, cax=cax, orientation="horizontal")
    cb.set_label(r'$\Sigma_{\rm*, \, sat}$ [M$_{\odot}$/kpc$^{2}$]', fontsize = 17)
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('top')



    angle = np.linspace( 0 , 2 * np.pi , 150 )
    x_vir = r_vir_cen * np.cos( angle )
    y_vir = r_vir_cen * np.sin( angle )

    x_r_h_str = 2 * r_h_cen * np.cos( angle )
    y_r_h_str = 2 * r_h_cen * np.sin( angle )

    for i in range (3):
            ax[i].plot(x_vir, y_vir, color = 'orange', lw = 2, label = '$r_{vir}$')
            ax[i].plot(x_r_h_str, y_r_h_str, color = 'magenta', lw = 2, label = r'$2 \times r_{half \,\, mass_{*}}$')

            ax[i].set_aspect(1)

            ax[i].set_xlim([-r_vir_cen, r_vir_cen])
            ax[i].set_ylim([-r_vir_cen, r_vir_cen])

            ax[i].xaxis.set_tick_params(labelsize=16)
            ax[i].yaxis.set_tick_params(labelsize=16)


    ax[0].set_xlabel('kpc', fontsize = 24)
    ax[0].set_ylabel('kpc', fontsize = 24)
    ax[1].set_xlabel('kpc', fontsize = 24)
    ax[2].set_xlabel('kpc', fontsize = 24)

    return None




def plot_fstar_range(ax, col = 'red', msmin = None, msmax = None):
    '''
    This is to plot the cumulative distribution of the remnant stellar mass fraction
    We are using power law here

    At some point, we need to break it based on the stellar mass that we have at z = 0
    '''
    # fig, ax = plt.subplots(figsize = (6, 6))
    
    pfstar = np.append(psmstar_all/psmstar_max_all, pmmstar_all/pmmstar_max_all) #This is the remnant stellar mass fraction
    pmstar_all = np.append(psmstar_all, pmmstar_all) #This is an array of all the stellar masses combined
    pfstar_tng = psmstar_f_ar_tng/psmstar_max_all #This is the remnant stellar mass fraction in TNG (we will only be looking at the surviving subhalos)
    if msmax is not None:
        cut = (pmstar_all > msmin) & (pmstar_all < msmax) 
        pfstar = pfstar[cut]
        cut_tng = (psmstar_f_ar_tng > msmin) & (psmstar_f_ar_tng < msmax)
        pfstar_tng = pfstar_tng[cut_tng] 

    fstar_ar = np.linspace(min(pfstar), max(pfstar), 100) #This is the range of fstar we are going to be looking at
    cum_fstar = [np.sum(pfstar < given_fstar) for given_fstar in fstar_ar] #This is the cumulative distribution of the remnant stellar mass fraction
    cum_fstar_tng = [np.sum(pfstar_tng < given_fstar) for given_fstar in fstar_ar] #This is the cumulative distribution of the remnant stellar mass fraction in TNG

    ax.plot(fstar_ar, cum_fstar/cum_fstar[-1], color = col, label = r'Power law in $\log M_{\rm{star}}$ [' + str(int(np.log10(msmin))) + ',' + str(int(np.log10(msmax))) + '] Msun') #This is the cumulative distribution of the remnant stellar mass fraction
    if msmin >= 1e6:
        ax.plot(fstar_ar, cum_fstar_tng/cum_fstar_tng[-1], color = col, ls = '--')
    
    
    # if msmax is not None:
    #     ax.set_title(r'$M_{\rm{\bigstar}}$ = ' + str(msmin) + ' - ' + str(msmax))
    #     plt.tight_layout()
    #     plt.savefig(this_fof_plotppath + 'cumulative_fstar' + str(int(np.log10(msmin))) + '_' + str(int(np.log10(msmax))) + '.png')
    # else:
    #     plt.tight_layout()
    #     plt.savefig(this_fof_plotppath + 'cumulative_fstar.png')
    return None


def cumulative_fstar():
    fig, ax = plt.subplots(figsize = (6, 6))
    # cumulative_fstar()
    plot_fstar_range(ax, col = 'pink', msmin = 1e4, msmax = 1e6)
    plot_fstar_range(ax, col = 'orange', msmin = 1e6, msmax = 1e8)
    plot_fstar_range( ax, col = 'red', msmin = 1e8, msmax = 1e14 )
    ax.set_xlabel(r'$f_{\rm{star}}$')
    ax.set_ylabel(r'$N(<f_{\rm{star}})$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(left = 9e-3)
    ax.legend(fontsize = 8)
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'cumulative_fstar.png')
    return None

cumulative_fstar()






def niusha_plot(rproj = 300, zmax = 50):
    '''
    This is to make the most good looking plot for the paper
    Planning to plot the distribution of the FoF0 as given by TNG based on Niusha's code

    We are making an xy projection plot

    To that, we add points of given position from the power law and cutoff models
    '''
    r_vir_cen = rvir_fof0
    plt.rcParams['axes.facecolor'] = 'black'
    fig, ax = plt.subplots(1,3, figsize = (18,6), sharex = True, sharey = True) #This line has been moved to final_plots.py
    plt.subplots_adjust(wspace = 0.02)
    for i,a in enumerate(ax.flat): #We are just formatting the ticks here
        a.tick_params(length = 8, width = 2, direction = 'inout')
        a.xaxis.tick_bottom()
        a.yaxis.tick_left()
    

    cat_sub = il.groupcat.loadSubhalos(basePath, 99, fields = ['SubhaloGrNr','SubhaloMassType', 'SubhaloMassInRadType', 'SubhaloMassInHalfRadType','SubhaloHalfmassRadType', 'SubhaloPos'])

    pos_sub = cat_sub['SubhaloPos']
    r_h_sub = cat_sub['SubhaloHalfmassRadType'][:,4] * 1/0.6774
    grnr = cat_sub['SubhaloGrNr']
    mstr_sub = cat_sub['SubhaloMassType'][:,4] * 1e10/0.6774
    mdm_sub = cat_sub['SubhaloMassType'][:,1] * 1e10/0.6774

    r_h_cen = r_h_sub[central_sfid_99_0]
    mdm_cen = mdm_sub[central_sfid_99_0]

    # mstr_sub = cat_sub['SubhaloMassType'][:,4] * 1e10/0.6774

    sfid = np.arange(len(grnr))

    outpath  = '/rhome/psadh003/bigdata/tng50/output_files/'
    result_satellite = np.load(outpath + 'fof0_plot.npy')

    x_pos_satellite = result_satellite[:,0]
    y_pos_satellite = result_satellite[:,1]
    z_pos_satellite = result_satellite[:,2]
    mass_satellite = result_satellite[:,3]

    print(f'x positions: {len(x_pos_satellite)}')
    print(f'x positions: {x_pos_satellite}')

    r_satellite = np.sqrt((x_pos_satellite**2)+(y_pos_satellite**2)+(z_pos_satellite**2))

    # aux = r_satellite < r_vir_cen
    # Testing different definitions of radii
    aux = (r_satellite > 2* r_h_cen) & (np.sqrt((x_pos_satellite**2)+(y_pos_satellite**2)) < rproj) & (np.absolute(z_pos_satellite) < zmax)

    x_pos_satellite = x_pos_satellite[aux]
    y_pos_satellite = y_pos_satellite[aux]
    z_pos_satellite = z_pos_satellite[aux]
    mass_satellite = mass_satellite[aux]

    print(len(x_pos_satellite))
    print('max min mstar pp:', mass_satellite.max(), mass_satellite.min())
    mass_bin_sat, xedges_sat, yedges_sat, binnumber = stats.binned_statistic_2d(x_pos_satellite, y_pos_satellite, mass_satellite, statistic='sum', bins=300) #We are adding the mass of all the satellites in a given bin


    print('shape of mass_bin_sat in pp: ', mass_bin_sat.shape)
    # XX_sat, YY_sat = np.meshgrid(xedges_sat, yedges_sat) #This maybe just generates the cordinates for the plot

    max_mass = mass_bin_sat.max()/10
    min_mass = 50000 * 8/5

    extent = xedges_sat[0], xedges_sat[-1], yedges_sat[0], yedges_sat[-1]

    for i in range(3):
        im3 = ax[i].imshow(mass_bin_sat.T, cmap = 'inferno', norm = 'log', vmin = 1e4, vmax = max_mass, interpolation='nearest', extent = extent, origin='lower', zorder = 100)

        # divider = make_axes_locatable(ax[i])
        # cax = divider.append_axes('top', size='3%', pad=0.05)
        # cb = fig.colorbar(im3, cax=cax, orientation="horizontal")
        # cb.set_label(r'$\Sigma_{\rm*, \, sat}$ [M$_{\odot}$/kpc$^{2}$]', fontsize = 17)
        # cb.ax.xaxis.set_label_position('top')
        # cb.ax.xaxis.set_ticks_position('top')

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im3, cax=cbar_ax)
    cbar.set_label(r'$\Sigma_{\rm*, \, sat}$ [M$_{\odot}$/kpc$^{2}$]', fontsize = 17)
    #Following lines are for drawing circles, brilliant!

    angle = np.linspace( 0 , 2 * np.pi , 150 )
    x_vir = rproj * np.cos( angle )
    y_vir = rproj * np.sin( angle )

    x_r_h_str = 2 * r_h_cen * np.cos( angle )
    y_r_h_str = 2 * r_h_cen * np.sin( angle )

    for i in range (3):
        if rproj == rvir_fof0:
            ax[i].plot(x_vir, y_vir, color = 'orange', lw = 2, label = '$r_{vir}$')
            # ax[i].plot(x_r_h_str, y_r_h_str, color = 'magenta', lw = 2, label = r'$2 \times r_{half \,\, mass_{*}}$')
        else:
            ax[i].plot(x_vir, y_vir, color = 'orange', lw = 2, label = '$r = $' + str(rproj) + ' kpc')
        ax[i].plot(x_r_h_str, y_r_h_str, color = 'magenta', lw = 2, label = r'$2 \times r_{half \,\, mass_{*}}$')

        ax[i].set_aspect(1)

        ax[i].set_xlim([-rproj, rproj])
        ax[i].set_ylim([-rproj, rproj])

        ax[i].xaxis.set_tick_params(labelsize=16)
        ax[i].yaxis.set_tick_params(labelsize=16)


    ax[0].set_xlabel('kpc', fontsize = 24)
    ax[0].set_ylabel('kpc', fontsize = 24)
    ax[1].set_xlabel('kpc', fontsize = 24)
    ax[2].set_xlabel('kpc', fontsize = 24)

    ax[1].set_xticks([-200, -100, 0, 100, 200])


    ps_cond = (np.sqrt(pspos_f_ar[:,0]**2 + pspos_f_ar[:,1]**2) < rproj) & (psfof == 0) & (np.absolute(pspos_f_ar[:,2]) < zmax)  &  (np.sqrt(pspos_f_ar[:,0]**2 + pspos_f_ar[:,1]**2 + pspos_f_ar[:,2]**2) > 2*0)
    # 
    ps_x = pspos_f_ar[:, 0][ps_cond]
    ps_y = pspos_f_ar[:, 1][ps_cond]
    ps_mstar = psmstar_all[ps_cond]
    ps_rh = psrh_all[ps_cond]

    pm_cond = (np.sqrt(pmpos_ar[:,0]**2 + pmpos_ar[:,1]**2) < rproj) & (pmfof == 0)  & (np.absolute(pmpos_ar[:,2]) < zmax)   &  (np.sqrt(pmpos_ar[:,0]**2 + pmpos_ar[:,1]**2 + pmpos_ar[:,2]**2) > 2*0)
    #
    pm_x = pmpos_ar[:, 0][pm_cond]
    pm_y = pmpos_ar[:, 1][pm_cond]
    pm_mstar = pmmstar_all[pm_cond]
    pm_rh = pmrh_all[pm_cond]

    pl_x = np.append(ps_x, pm_x)
    pl_y = np.append(ps_y, pm_y)
    pl_mstar = np.append(ps_mstar, pm_mstar)
    pl_rh = np.append(ps_rh, pm_rh)

    cs_cond = (np.sqrt(cspos_f_ar[:, 0]**2 + cspos_f_ar[:, 1]**2) < rproj) & (csfof == 0) & (np.absolute(cspos_f_ar[:,2]) < zmax) & (np.sqrt(cspos_f_ar[:, 0]**2 + cspos_f_ar[:, 1]**2 + cspos_f_ar[:, 2]**2) > 2*0)
    cs_x = cspos_f_ar[:, 0][cs_cond]
    cs_y = cspos_f_ar[:, 1][cs_cond]
    cs_mstar = csmstar_all[cs_cond]
    cs_rh = csrh_all[cs_cond]

    cm_cond = (np.sqrt(cmpos_ar[:, 0]**2 + cmpos_ar[:, 1]**2) < rproj) & (cmfof == 0) & (np.absolute(cmpos_ar[:,2]) < zmax) & (np.sqrt(cmpos_ar[:, 0]**2 + cmpos_ar[:, 1]**2 + cmpos_ar[:, 2]**2) > 2*0)
    cm_x = cmpos_ar[:, 0][cm_cond]
    cm_y = cmpos_ar[:, 1][cm_cond]
    cm_mstar = cmmstar_all[cm_cond]
    cm_rh = cmrh_all[cm_cond]

    cl_x = np.append(cs_x, cm_x)
    cl_y = np.append(cs_y, cm_y)
    cl_mstar = np.append(cs_mstar, cm_mstar)
    cl_rh = np.append(csrh_all, cmrh_all)
    

    print('len pl_x', len(pl_x))
    print('max min mstar fp', min(pl_mstar), max(pl_mstar))
    #This is the 2d histogram for power law model
    mass_bin_sat, xedges_sat, yedges_sat, binnumber = stats.binned_statistic_2d(pl_x, pl_y, pl_mstar, statistic='sum', bins=500) #We are adding the mass of all the satellites in a given bin

    print('mass_bin_sat size in fp: ', mass_bin_sat.shape)
    # XX_sat, YY_sat = np.meshgrid(xedges_sat, yedges_sat) #This maybe just generates the cordinates for the plot

    max_mass = mass_bin_sat.max()/10
    min_mass = 50000 * 8/5 

    print('min and max of mass bins: ', min_mass, max_mass)

    extent = xedges_sat[0], xedges_sat[-1], yedges_sat[0], yedges_sat[-1]
    # im3 = ax[1].imshow(mass_bin_sat.T, color = 'white', norm = 'log', vmin = 1e4, vmax = max_mass, interpolation='nearest', extent = extent, origin='lower')
    # im3 = ax[1].imshow(mass_bin_sat.T, cmap = 'inferno', norm = 'log', vmin = 1e4, vmax = max_mass, interpolation='nearest', extent = extent, origin='lower')

    # divider = make_axes_locatable(ax[1])
    # cax = divider.append_axes('top', size='3%', pad=0.05)
    # cb = fig.colorbar(im3, cax=cax, orientation="horizontal")
    # cb.set_label(r'$\Sigma_{\rm*, \,{power law}}$ [M$_{\odot}$/kpc$^{2}$]', fontsize = 17)
    # cb.ax.xaxis.set_label_position('top')
    # cb.ax.xaxis.set_ticks_position('top')


    #This is the 2d histogram for cutoff model
    mass_bin_cen, xedges_cen, yedges_cen, binnumber = stats.binned_statistic_2d(cl_x, cl_y, cl_mstar, statistic='sum', bins=500) #We are adding the mass of all the satellites in a given bin

    # XX_cen, YY_cen = np.meshgrid(xedges_cen, yedges_cen) #This maybe just generates the cordinates for the plot

    max_mass = mass_bin_cen.max()/10
    min_mass = 50000 * 8/5

    print('min and max of mass bins: ', min_mass, max_mass)

    extent = xedges_cen[0], xedges_cen[-1], yedges_cen[0], yedges_cen[-1]

    collection2 = mc.CircleCollection(20*cs_rh, offsets=np.column_stack((cs_x, cs_y)), transOffset=ax[2].transData, zorder = 200, 
                                      facecolors='none',  edgecolors='white', linewidths = 0.5)
    ax[2].add_collection(collection2)

    collection = mc.CircleCollection(20*pl_rh, offsets=np.column_stack((pl_x, pl_y)), transOffset=ax[1].transData, zorder = 200, 
                                      facecolors='none',  edgecolors='white', linewidths = 0.5)
    ax[1].add_collection(collection) 

    ax[0].text(0.05, 0.95, 'TNG50-1', color = 'white', fontsize = 10, transform=ax[0].transAxes)  
    ax[1].text(0.05, 0.95, 'Power law', color = 'white', fontsize = 10, transform=ax[1].transAxes)
    ax[2].text(0.05, 0.95, 'Cutoff', color = 'white', fontsize = 10, transform=ax[2].transAxes) 

    # ax[1].scatter(pl_x, pl_y, color = 'white', s = 1.2**(np.log10(pl_rh)), alpha = 0.5)
    # ax[2].scatter(cm_x, cm_y, color = 'white', s = 1.2**(np.log10(cl_rh)), alpha = 0.5)




    # im4 = ax[2].imshow(mass_bin_cen.T, color = 'white', norm = 'log', vmin = 1e4, vmax = max_mass, interpolation='nearest', extent = extent, origin='lower')
    # im4 = ax[2].imshow(mass_bin_cen.T, cmap = 'inferno', norm = 'log', vmin = 1e4, vmax = max_mass, interpolation='nearest', extent = extent, origin='lower')

    # divider = make_axes_locatable(ax[2])
    # cax = divider.append_axes('top', size='3%', pad=0.05)
    # cb = fig.colorbar(im4, cax=cax, orientation="horizontal")
    # cb.set_label(r'$\Sigma_{\rm*, \,{cutoff}}$ [M$_{\odot}$/kpc$^{2}$]', fontsize = 17)
    # cb.ax.xaxis.set_label_position('top')
    # cb.ax.xaxis.set_ticks_position('top')


    # print(type(1.2**(np.log10(pl_rh))))

    # patches = [plt.Circle(center, size) for center, size in zip(xy, sizes)]

    # fig, ax = plt.subplots()
    # ax[1].scatter(ps_x, pspos_f_ar[:, 1][(psdist_f_ar < rvir_fof0) & (psfof == 0)], color = 'white', s = 1, alpha = 0.5)
    # ax[1].scatter(pmpos_ar[:, 0][(pmdist_f_ar < rvir_fof0) & (pmfof == 0)], pmpos_ar[:, 1][(pmdist_f_ar < rvir_fof0) & (pmfof == 0)], color = 'white', s = 1, alpha = 0.5)
    # ax[2].scatter(cmpos_ar[:, 0][(cmdist_f_ar < rvir_fof0) & (cmfof == 0)], cmpos_ar[:, 1][(cmdist_f_ar < rvir_fof0) & (cmfof == 0)], color = 'white', s = 1, alpha = 0.5)
    


    fig.savefig(this_fof_plotppath +'niusha_plot.png', bbox_inches='tight')
    return None

niusha_plot()
plt.rcParams['axes.facecolor'] = 'white'



def checking_subhalos():
    '''
    This is to check the subhalos which are actually in TNG and not in model
    '''
    snpz0 = 99
    h_small = 0.6774
    simbase = '/rhome/psadh003/bigdata/L35n2160TNG_fixed/output'

    cat_halo = il.groupcat.loadHalos(simbase, snpz0, fields = ['Group_R_Crit200', 'Group_M_Crit200', 'GroupFirstSub'])

    idf = cat_halo['GroupFirstSub']
    mvir = cat_halo['Group_M_Crit200'] * 1.e10/h_small
    rvir = cat_halo['Group_R_Crit200'] * 1/h_small

    aux = mvir >= 1e8
    mvir_group = mvir[aux]
    idf_group = idf[aux]
    rvir_group = rvir[aux]

    cat_sub = il.groupcat.loadSubhalos(simbase, snpz0, fields = ['SubhaloGrNr','SubhaloMassType', 'SubhaloMassInRadType', 'SubhaloMassInHalfRadType','SubhaloHalfmassRadType', 'SubhaloPos'])

    pos_sub = cat_sub['SubhaloPos']/h_small
    r_h_sub = cat_sub['SubhaloHalfmassRadType'][:,4] * 1/h_small
    grnr = cat_sub['SubhaloGrNr']
    mstr_sub = cat_sub['SubhaloMassType'][:,4] * 1e10/h_small

    sfid = np.arange(len(grnr))

    # for i in range (1):
    i = 0 #This would be the FoF group number, this was one before, changing this to 0
    grnr_group = i
    print(i)
    sfid_cen = idf_group[i]
    print(sfid_cen)
    r_h_cen = r_h_sub[sfid_cen]
    print('r_h_cen: ', r_h_cen)
    r_vir_cen = rvir_group[i]
    print('r_vir_cen: ', r_vir_cen)


    x_cen_group = pos_sub[sfid_cen,0]
    y_cen_group = pos_sub[sfid_cen,1]
    z_cen_group = pos_sub[sfid_cen,2]

    #target subhalos(these are massive subhalos above 1e8 M_sun)

    aux = (grnr == grnr_group) & (mstr_sub > 5e6) #5e6 #1e10
    pos_sub_target = pos_sub[aux]
    x_sub_target = pos_sub_target[1:,0] - x_cen_group #We are excluding the central subhalo here
    y_sub_target = pos_sub_target[1:,1] - y_cen_group
    z_sub_target = pos_sub_target[1:,2] - z_cen_group
    mstr_sub_target = mstr_sub[aux]
    mstr_sub_target = np.delete(mstr_sub_target,0)
    sfid_target = sfid[aux]
    sfid_target = np.delete(sfid_target, 0)

    fig, ax = plt.subplots(1, 1, figsize = (6,6))
    aux2 = (np.sqrt((x_sub_target**2)+(y_sub_target**2)) < r_h_cen) & (np.absolute(z_sub_target) < 50)


    # POWER LAW MODEL
    rproj = r_h_cen
    zmax = 50
    ps_cond = (np.sqrt(pspos_f_ar[:,0]**2 + pspos_f_ar[:,1]**2) < rproj) & (psfof == 0) & (np.absolute(pspos_f_ar[:,2]) < zmax) & (np.sqrt(pspos_f_ar[:,0]**2 + pspos_f_ar[:,1]**2 + pspos_f_ar[:,2]**2) > 2 * r_h_cen)
    ps_x = pspos_f_ar[:, 0][ps_cond]
    ps_y = pspos_f_ar[:, 1][ps_cond]
    ps_mstar = psmstar_all[ps_cond]
    ps_rh = psrh_all[ps_cond]

    pm_cond = (np.sqrt(pmpos_ar[:,0]**2 + pmpos_ar[:,1]**2) < rproj) & (pmfof == 0) & (np.absolute(pmpos_ar[:,2]) < zmax) & (np.sqrt(pmpos_ar[:,0]**2 + pmpos_ar[:,1]**2 + pmpos_ar[:,2]**2) > 2 * r_h_cen)
    pm_x = pmpos_ar[:, 0][pm_cond]
    pm_y = pmpos_ar[:, 1][pm_cond]
    pm_mstar = pmmstar_all[pm_cond]
    pm_rh = pmrh_all[pm_cond]

    pl_x = np.append(ps_x, pm_x)
    pl_y = np.append(ps_y, pm_y)
    pl_mstar = np.append(ps_mstar, pm_mstar)
    pl_rh = np.append(ps_rh, pm_rh)







    ax.scatter(x_sub_target[aux2], y_sub_target[aux2], s = 5*np.log10(mstr_sub_target[aux2]), alpha = 0.5, color = 'darkblue')
    ax.scatter(ps_x, ps_y, s = 5*np.log10(ps_mstar), alpha = 0.5, color = 'orange')
    plt.savefig(this_fof_plotppath + 'checkinggg.png')




    

    print('These are from the simulation')
    for ix in range(len(mstr_sub_target[aux2])):
        print(x_sub_target[aux2][ix], y_sub_target[aux2][ix], np.log10(mstr_sub_target[aux2][ix]), sfid_target[aux2][ix])
    print('These are from the model')
    for ix in range(len(ps_x)):
        print(ps_x[ix], ps_y[ix], np.log10(ps_mstar[ix]))
    return None

checking_subhalos()



def check_udgs():
    '''
    This function is to check the amount of stripping that occurs for UDGs
    We define UDGs as the subhalos which end up with rh > 1kpc and sigma > 25 mag/arcsec^2
    '''
    rh_cut = 1 #kpc, because our rh is in kpc
    sigma_cut = 25
    psix = (psrh_all > rh_cut) & (pssigma_all > sigma_cut) & (psfof == 0)
    pmix = (pmrh_all > rh_cut) & (pmsigma_all > sigma_cut) & (pmfof == 0)
    csix = (csrh_all > rh_cut) & (cssigma_all > sigma_cut) & (csfof == 0)
    cmix = (cmrh_all > rh_cut) & (cmsigma_all > sigma_cut) & (cmfof == 0)

    print('Number of UDGs in power law model: ', len(psmstar_all[psix]) + len(pmmstar_all[pmix]))
    print('Number of UDGs in cutoff model: ', len(csmstar_all[csix]) + len(cmmstar_all[cmix]))
    plt.rcParams['axes.facecolor'] = 'white'
    plt.figure(figsize = (6.25,6))
    plt.hist(np.log10(psmstar_all[psix]/psmstar_max_all[psix]), bins = 10, histtype = 'step', color = 'blue', label = 'Power law surviving', alpha = 0.5)
    plt.hist(np.log10(pmmstar_all[pmix]/pmmstar_max_all[pmix]), bins = 10, histtype = 'step', color = 'purple', label = 'Power law merged', alpha = 0.5)
    plt.hist(np.log10(csmstar_all[csix]/csmstar_max_all[csix]), bins = 10, histtype = 'step', color = 'green', label = 'Cutoff surviving', alpha = 0.5)
    plt.hist(np.log10(cmmstar_all[cmix]/cmmstar_max_all[cmix]), bins = 10, histtype = 'step', color = 'red', label = 'Cutoff merged', alpha = 0.5)
    plt.legend()
    plt.xlabel(r'$\log (M_{\rm{\bigstar, z=0}}/M_{\rm{\bigstar, max}})$')
    plt.ylabel('Number of UDGs')
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'udg_hist.png')

    return None

check_udgs()


def check_extended_objects():
    '''
    This function checks the properties of the extended subhalos based on their size and stellar mass
    We define extended subhalos as the subhalos which end up with rh > 1.5 kpc and 1e7 < mstar < 1e9.

    We will be comparing Rh, check amount of stripping for stars and Mmx/Mmx0
    '''
    rh_cut = 1.5 #kpc
    mstar_min = 1e7
    mstar_max = 1e9
    psix = (psrh_all > rh_cut) & (psmstar_all > mstar_min) & (psmstar_all < mstar_max) & (psfof == 0)
    pmix = (pmrh_all > rh_cut) & (pmmstar_all > mstar_min) & (pmmstar_all < mstar_max) & (pmfof == 0)
    csix = (csrh_all > rh_cut) & (csmstar_all > mstar_min) & (csmstar_all < mstar_max) & (csfof == 0)
    cmix = (cmrh_all > rh_cut) & (cmmstar_all > mstar_min) & (cmmstar_all < mstar_max) & (cmfof == 0)

    print('Number of extended objects in power law model: ', len(psmstar_all[psix]) + len(pmmstar_all[pmix]))
    print('Number of extended objects in cutoff model: ', len(csmstar_all[csix]) + len(cmmstar_all[cmix]))

    fig, ax = plt.subplots(1, 3, figsize = (18,6))
    frh_all = np.append(psrh_all[psix]/psrh_max_all[psix], pmrh_all[pmix]/pmrh_max_all[pmix])
    fstar = np.append(psmstar_all[psix]/psmstar_max_all[psix], pmmstar_all[pmix]/pmmstar_max_all[pmix])
    fmmx = np.append(psmmx_f_ar[psix]/psmmx_if_ar[psix], pmmmx_f_ar[pmix]/pmmmx_if_ar[pmix])

    frh_tng = np.log10(psrh_f_ar_tng[psix]/psrh_max_ar[psix])
    fstar_tng = np.log10(psmstar_f_ar_tng[psix]/psmstar_max_ar[psix])
    fmmx_tng = np.log10(psmmx_f_ar_tng[psix]/psmmx_if_ar[psix])

    ax[0].hist(np.log10(frh_all), bins = 10, histtype = 'step', color = 'red', label = 'Power law', alpha = 0.5)
    ax[0].hist(frh_tng[np.isfinite(frh_tng)], bins = 10, histtype = 'step', color = 'black', label = 'TNG', alpha = 0.5)
    ax[0].set_xlabel(r'$\log (R_{\rm{h, z=0}}/R_{\rm{h, max}})$')
    ax[1].hist(np.log10(fstar), bins = 10, histtype = 'step', color = 'red', label = 'Power law', alpha = 0.5)
    ax[1].hist(fstar_tng[np.isfinite(fstar_tng)], bins = 10, histtype = 'step', color = 'black', label = 'TNG', alpha = 0.5)
    ax[1].set_xlabel(r'$\log (M_{\rm{star, z=0}}/M_{\rm{star, max}})$')
    ax[2].hist(np.log10(fmmx), bins = 10, histtype = 'step', color = 'red', label = 'Power law', alpha = 0.5)
    ax[2].hist(fmmx_tng[np.isfinite(fmmx_tng)], bins = 10, histtype = 'step', color = 'black', label = 'TNG', alpha = 0.5)
    ax[2].set_xlabel(r'$\log (M_{\rm{mx, z=0}}/M_{\rm{mx, inf}})$')
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'extended_hists.png')

    fig, ax = plt.subplots(1, 1, figsize = (6,6))
    
    fstar_comp = np.log10(psmstar_all[psix]/psmstar_max_all[psix])[np.isfinite(fstar_tng)]
    ax.scatter(fstar_tng[np.isfinite(fstar_tng)], fstar_comp, s = 1)
    ax.set_xlabel(r'$\log (M_{\rm{star, z=0}}/M_{\rm{star, max}})$ TNG')
    ax.set_ylabel(r'$\log (M_{\rm{star, z=0}}/M_{\rm{star, max}})$ Power law')
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'comparison_extended.png')

    return None


check_extended_objects()


def check_compact_objects():
    '''
    This is same as above but for compact objects defined as rh < 0.1 kpc and 1e6 < mstar < 1e10
    '''
    rh_cut = 0.1 #kpc
    mstar_min = 1e6
    mstar_max = 1e10
    psix = (psrh_all < rh_cut) & (psmstar_all > mstar_min) & (psmstar_all < mstar_max) & (psfof == 0)
    pmix = (pmrh_all < rh_cut) & (pmmstar_all > mstar_min) & (pmmstar_all < mstar_max) & (pmfof == 0)
    csix = (csrh_all < rh_cut) & (csmstar_all > mstar_min) & (csmstar_all < mstar_max) & (csfof == 0)
    cmix = (cmrh_all < rh_cut) & (cmmstar_all > mstar_min) & (cmmstar_all < mstar_max) & (cmfof == 0)

    print('Number of compact objects in power law model: ', len(psmstar_all[psix]) + len(pmmstar_all[pmix]))
    print('Number of compact objects in cutoff model: ', len(csmstar_all[csix]) + len(cmmstar_all[cmix]))
    
    fig, ax = plt.subplots(1, 3, figsize = (18,6))
    frh_all = np.append(psrh_all[psix]/psrh_max_all[psix], pmrh_all[pmix]/pmrh_max_all[pmix])
    fstar = np.append(psmstar_all[psix]/psmstar_max_all[psix], pmmstar_all[pmix]/pmmstar_max_all[pmix])
    fmmx = np.append(psmmx_f_ar[psix]/psmmx_if_ar[psix], pmmmx_f_ar[pmix]/pmmmx_if_ar[pmix])
    ax[0].hist(np.log10(frh_all), bins = 10, histtype = 'step', color = 'red', label = 'Power law', alpha = 0.5)
    ax[0].set_xlabel(r'$\log (R_{\rm{h, z=0}}/R_{\rm{h, max}})$')
    ax[1].hist(np.log10(fstar), bins = 10, histtype = 'step', color = 'red', label = 'Power law', alpha = 0.5)
    ax[1].set_xlabel(r'$\log (M_{\rm{star, z=0}}/M_{\rm{star, max}})$')
    ax[2].hist(np.log10(fmmx), bins = 10, histtype = 'step', color = 'red', label = 'Power law', alpha = 0.5)
    ax[2].set_xlabel(r'$\log (M_{\rm{mx, z=0}}/M_{\rm{mx, inf}})$')
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'compact_hists.png')

    return None

check_compact_objects()




def tidal_tracks():
    '''
    This is to plot the tidal tracks of the satellites based on Errani 22 or the plots given by Rapha
    '''
    
    fig, (ax, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize = (6,6), sharex = True)
    fpl = np.linspace(-6.4, 0, 100)

    # def get_first_set_l10rhbyrmx0(fpl_ar, Rh0byrmx0):
    #     '''
    #     fpl should be in log pleaseee
    #     '''
    #     l10rhbyrmx0_ar = np.zeros(0)
    #     for fpl in fpl_ar:
    #         if fpl > -2.5:
    #             if Rh0byrmx0 == 1/2:
    #                 l10rhbyrmx0_ar = np.append(l10rhbyrmx0_ar, l10rbyrmx0_1by2_spl(fpl))
    #             elif Rh0byrmx0 == 1/4:
    #                 l10rhbyrmx0_ar = np.append(l10rhbyrmx0_ar, l10rbyrmx0_1by4_spl(fpl))
    #             elif Rh0byrmx0 == 1/8:
    #                 l10rhbyrmx0_ar = np.append(l10rhbyrmx0_ar, l10rbyrmx0_1by8_spl(fpl))
    #             elif Rh0byrmx0 == 1/16:
    #                 l10rhbyrmx0_ar = np.append(l10rhbyrmx0_ar, l10rbyrmx0_1by16_spl(fpl))
    #         else:
    #             l10rhbyrmx0_ar = np.append(l10rhbyrmx0_ar, np.log10(get_rmxbyrmx0(10**fpl)))
    #     return l10rhbyrmx0_ar
    #     # if fpl > -2.5:
    #     #     if Rh0byrmx0 == 1/2:
    #     #         return l10rbyrmx0_1by2_spl(fpl)
    #     #     elif Rh0byrmx0 == 1/4:
    #     #         return l10rbyrmx0_1by4_spl(fpl)
    #     #     elif Rh0byrmx0 == 1/8:
    #     #         return l10rbyrmx0_1by8_spl(fpl)
    #     #     elif Rh0byrmx0 == 1/16:
    #     #         return l10rbyrmx0_1by16_spl(fpl)
    #     # else:
    #     #     return np.log10(get_rmxbyrmx0(10**fpl))
        
    def get_l10rhbyrmx0(fpl_ar, Rh0byrmx0):
        l10rhbyrmx0_ar = np.zeros(0)

        for fpl in fpl_ar:
            if (fpl > -5) and (Rh0byrmx0 == 1/66 or Rh0byrmx0 == 1/250):
                if Rh0byrmx0 == 1/66:
                    l10rhbyrmx0_ar = np.append(l10rhbyrmx0_ar, l10rbyrmx0_1by66_spl(fpl))
                elif Rh0byrmx0 == 1/250:
                    l10rhbyrmx0_ar = np.append(l10rhbyrmx0_ar, l10rbyrmx0_1by250_spl(fpl))
            elif (fpl > -2.5) and (Rh0byrmx0 in [1/2, 1/4, 1/8, 1/16]):
                if Rh0byrmx0 == 1/2:
                    l10rhbyrmx0_ar = np.append(l10rhbyrmx0_ar, l10rbyrmx0_1by2_spl(fpl))
                elif Rh0byrmx0 == 1/4:
                    l10rhbyrmx0_ar = np.append(l10rhbyrmx0_ar, l10rbyrmx0_1by4_spl(fpl))
                elif Rh0byrmx0 == 1/8:
                    l10rhbyrmx0_ar = np.append(l10rhbyrmx0_ar, l10rbyrmx0_1by8_spl(fpl))
                elif Rh0byrmx0 == 1/16:
                    l10rhbyrmx0_ar = np.append(l10rhbyrmx0_ar, l10rbyrmx0_1by16_spl(fpl))
            elif (fpl > -6.4) and (Rh0byrmx0 == 1/1000):
                    l10rhbyrmx0_ar = np.append(l10rhbyrmx0_ar, l10rbyrmx0_1by1000_spl(fpl))
            else:
                l10rhbyrmx0_ar = np.append(l10rhbyrmx0_ar, np.log10(get_rmxbyrmx0(10**fpl)))
        return l10rhbyrmx0_ar

        # return l10rhbyrmx0
    ax.plot(fpl, get_l10rhbyrmx0(fpl, 1/2), c = 'r', label = r'$R_{\rm{h0}}/r_{\rm{mx0}} = 1/2$')
    ax.plot(fpl, get_l10rhbyrmx0(fpl, 1/4), c = 'orange', label = r'$R_{\rm{h0}}/r_{\rm{mx0}} = 1/4$')
    ax.plot(fpl, get_l10rhbyrmx0(fpl, 1/8), c = 'skyblue', label = r'$R_{\rm{h0}}/r_{\rm{mx0}} = 1/8$')
    ax.plot(fpl, get_l10rhbyrmx0(fpl, 1/16), c = 'darkblue', label = r'$R_{\rm{h0}}/r_{\rm{mx0}} = 1/16$')
    ax.plot(fpl, get_l10rhbyrmx0(fpl, 1/66), c = 'purple', ls = ':')
    ax.plot(fpl, get_l10rhbyrmx0(fpl, 1/250), c = 'limegreen', ls = ':')
    ax.plot(fpl, get_l10rhbyrmx0(fpl, 1/1000), c = 'darkgreen', ls = ':')

    ax.plot(fpl, np.log10(get_rmxbyrmx0(10**fpl)), c = 'black', ls = '--')
    ax.annotate(r'$r_{\rm{mx}}$', xy = (fpl[-20], np.log10(get_rmxbyrmx0(10**fpl))[-20]), xytext = (fpl[-20], 0.8* np.log10(get_rmxbyrmx0(10**fpl))[-20]), 
                rotation=30, color='black', fontsize=10, rotation_mode='anchor')

    # ax.set_xlabel(r'$\log M_{\rm{mx}}/M_{\rm{mx0}}$', fontsize = 14)
    ax.set_ylabel(r'$\log R_{\rm{h}}/r_{\rm{mx0}}$', fontsize = 14)
    ax.legend(fontsize = 8, frameon=False)
    ax.tick_params(axis='y', which = 'both', left=True, right=True, direction = 'in')
    ax.tick_params(axis='x', which = 'both', direction = 'in')

    # print('Why are you like this!', get_LbyL0(fpl, 1/2))
    ax2.plot(fpl, np.log10(get_LbyL0(10 ** fpl, 1/2)), c = 'r')
    ax2.plot(fpl, np.log10(get_LbyL0(10 ** fpl, 1/4)), c = 'orange')
    ax2.plot(fpl, np.log10(get_LbyL0(10 ** fpl, 1/8)), c = 'skyblue')
    ax2.plot(fpl, np.log10(get_LbyL0(10 ** fpl, 1/16)), c = 'darkblue')
    ax2.plot(fpl, np.log10(get_LbyL0(10 ** fpl, 1/66)), c = 'purple', label = r'$R_{\rm{h0}}/r_{\rm{mx0}} = 1/66$', ls = ':')
    ax2.plot(fpl, np.log10(get_LbyL0(10 ** fpl, 1/250)), c = 'limegreen', label = r'$R_{\rm{h0}}/r_{\rm{mx0}} = 1/250$', ls = ':')
    ax2.plot(fpl, np.log10(get_LbyL0(10 ** fpl, 1/1000)), c = 'darkgreen', label = r'$R_{\rm{h0}}/r_{\rm{mx0}} = 1/1000$', ls = ':')

    ax2.set_xlabel(r'$\log M_{\rm{mx}}/M_{\rm{mx0}}$', fontsize = 14)
    ax2.set_ylabel(r'$\log M_{\rm{star}}/M_{\rm{star,0}}$', fontsize = 14)
    ax2.set_ylim(bottom = -3)
    ax2.legend(fontsize = 8, frameon=False)
    ax2.tick_params(axis='y', which = 'both', left=True, right=True, direction = 'in')
    ax2.tick_params(axis='x', which = 'both', direction = 'in')


    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'tidal_tracks.png')

    return None

tidal_tracks()


def surv_comp(cmap = 'turbo_r'):
    '''
    This is to compare mmx, rmx, mstar and rh between tng and the model
    '''
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols=2, figsize = (12*1/0.9, 12))
    alpha = 0.8
    msize = 8

    pix = (psmstar_max_ar > 10**7.5) & (psmstar_max_ar < 10**9.5)


    col_ar = pstinf_ar[pix]
    col_label = r'$t_{\rm{inf}}$'
    # col_ar = np.log10(psmstar_max_ar)[pix]
    # col_label = r'$\log M_{\rm{\bigstar, max}}$'
    sc = ax1.scatter(psmmx_f_ar[pix], psmmx_f_ar_tng[pix], c=col_ar, cmap=cmap, alpha = alpha, s = msize, marker='o', zorder = 20)

    # cbar = plt.colorbar(sc, ax = ax1)
    # cbar.set_label(col_label)

    dummy = np.linspace(ax1.get_xlim()[0], ax1.get_xlim()[1], 3) 
    ax1.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)

    ax1.set_xlabel(r'$M_{\rm{mx}}$ from model (M$_{\odot}$)')
    ax1.set_ylabel(r'$M_{\rm{mx}}$ from TNG (M$_{\odot}$)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(10**7.5, 1e11)
    ax1.set_ylim(10**7.5, 1e11)

    sc = ax2.scatter(psrmx_f_ar[pix]*1e3, psrmx_f_ar_tng[pix]*1e3, c=col_ar, cmap=cmap, alpha = alpha, s = msize, marker='o', zorder = 20)

    dummy = np.linspace(ax2.get_xlim()[0], ax2.get_xlim()[1], 3) 
    ax2.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)

    ax2.set_xlabel(r'$r_{\rm{mx}}$ from model (pc)')
    ax2.set_ylabel(r'$r_{\rm{mx}}$ from TNG (pc)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(10**2, 3*10**4)
    ax2.set_ylim(10**2, 3*10**4)

    sc = ax3.scatter(psmstar_f_ar[pix], psmstar_f_ar_tng[pix], c=col_ar, cmap=cmap, alpha = alpha, s = msize, marker='o', zorder = 20)

    dummy = np.linspace(ax3.get_xlim()[0], ax3.get_xlim()[1], 3)
    ax3.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)

    ax3.set_xlabel(r'$M_{\rm{star}}$ from model (M$_{\odot}$)')
    ax3.set_ylabel(r'$M_{\rm{star}}$ from TNG (M$_{\odot}$)')

    ax3.set_xscale('log')
    ax3.set_yscale('log')
    # ax3.set_xlim(1*10**8, 5*10**9)
    # ax3.set_ylim(1*10**8, 5*10**9)

    sc = ax4.scatter(psrh_f_ar[pix]*1e3, psrh_f_ar_tng[pix]*1e3, c=col_ar, cmap=cmap, alpha = alpha, s = msize, marker='o', zorder = 20)

    dummy = np.linspace(ax4.get_xlim()[0], ax4.get_xlim()[1], 3)
    ax4.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)

    ax4.set_xlabel(r'$R_{\rm{h}}$ from model (pc)')
    ax4.set_ylabel(r'$R_{\rm{h}}$ from TNG (pc)')

    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlim(10**2, 4*10**3)
    ax4.set_ylim(10**2, 4*10**3)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='y', which = 'both', left=True, right=True, direction = 'in')
        ax.tick_params(axis='x', which = 'both', direction = 'in')

    plt.tight_layout()

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label(col_label)


    plt.savefig(this_fof_plotppath + 'surv_comp.png')

    # The following part is to plot the fractional stellar mass remaining for TNG vs the model
    fig, ax = plt.subplots(1, 1, figsize = (6,6))
    ax.scatter(np.log10(psmstar_f_ar[pix]/psmstar_max_ar[pix]), np.log10(psmstar_f_ar_tng[pix]/psmstar_max_ar[pix]), s = 8, color = 'black', alpha = 0.4)
    dummy = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 3)
    ax.plot(dummy, dummy, 'k-', lw = 0.5, zorder = 0, alpha = 0.5)

    ax.set_xlabel(r'$\log (M_{\rm{star}}/M_{\rm{star, max}})$ from model')
    ax.set_ylabel(r'$\log (M_{\rm{star}}/M_{\rm{star, max}})$ from TNG')
    ax.set_xlim(-3, 0)
    ax.set_ylim(-3, 0)
    plt.tight_layout()
    plt.savefig(this_fof_plotppath + 'fstar_comp_validation.png')

    return None

surv_comp()