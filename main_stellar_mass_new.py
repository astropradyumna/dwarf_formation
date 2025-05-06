import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from errani_plus_tng_subhalo import ErraniSubhalo # This is the class that we want to use to save time on it 
from tqdm import tqdm
import illustris_python as il 
from joblib import Parallel, delayed
import sys
import os
sys.path.append(os.path.abspath('/bigdata/saleslab/psadh003/tng50-dark-analysis'))
from hydro_path_for_files import * 
from constants import * 
import re 
import ast

# Let us take the fofno as an input from the terminal screen 
fofno = int(sys.argv[1])
fofstr = 'fof' + str(int(fofno))

ages_df = pd.read_csv('ages_tng.csv', comment = '#')

all_snaps = np.array(ages_df['snapshot'])
all_redshifts = np.array(ages_df['redshift'])
all_ages = np.array(ages_df['age(Gyr)']) 

# ======================================

# Now we will try to get the V0 for the central at snapshot 99 
this_fof = il.groupcat.loadSingle(basePath, 99, haloID = fofno)
central_sfid_99 = this_fof['GroupFirstSub'] 
cen = il.groupcat.loadSingle(basePath, 99, subhaloID = central_sfid_99)
fields = ['SubhaloID']
central_mpb = il.sublink.loadTree(basePath, 99, central_sfid_99, fields = ['SubhaloID', 'SnapNum', 'SubfindID'], onlyMPB = True)
central_sfids = central_mpb['SubfindID']
central_snaps = central_mpb['SnapNum']
m200 = this_fof['Group_M_Crit200'] * 1e10 / h
r200 = this_fof['Group_R_Crit200'] / h
assert np.isclose(G, 4.3e-6) # This is in kpc, Msun and seconds
v200 = float(np.sqrt(G * m200 / r200)) # This is in km/s

# Let us not get the full tree of the central to see which of the Type-2s are directly getting merged into the central
cft = il.sublink.loadTree(basePath, 99, central_sfid_99, fields = ['SubhaloID', 'DescendantID', 'SnapNum', 'SubfindID', 'SubhaloIDMostbound'])
cft_shid = cft['SubhaloID']
cft_desc = cft['DescendantID']
cft_snap = cft['SnapNum']
cft_sfid = cft['SubfindID']
cft_mbpid = cft['SubhaloIDMostbound']

np.random.seed(42)


# ======================================
def get_mstar_co(lvmax, alpha = 3.5, mu = -3.1, M0 = 97765347):
    eta = 10**lvmax / 50
    mstar = eta**alpha *np.exp(-eta**mu) * M0
    return np.log10(mstar)



def get_mstar_pl(lvmaxar, m1 = 3.72, m2 = 8.3, b = 1.5):
    '''
    This is the power law model from Santos-Santos 2022
    Input is the log of Vmax
    '''
    vchange  = 57
    if isinstance(lvmaxar, (float, np.float64, np.float32)):
        if lvmaxar >= np.log10(vchange):
            lmstar = m1 * lvmaxar + b
        elif lvmaxar < np.log10(vchange):
            lmstar = m2 * lvmaxar + (m1 - m2)*np.log10(vchange) + b
        lmstarar = lmstar 
    else:
        lmstarar = np.zeros(0)
        for lvmax in lvmaxar:
            if lvmax >= np.log10(vchange):
                lmstar = m1 * lvmax + b
            elif lvmax < np.log10(vchange):
                lmstar = m2 * lvmax + (m1 - m2)*np.log10(vchange) + b
            lmstarar = np.append(lmstarar, lmstar)
    return lmstarar

def get_scatter(lvmaxar, sigma0 = 0.24, kappa = -1.26, V0 = 88.6):
    '''
    This function returns the scatter for both power law and the cutoff models
    '''
    vmaxar = 10**lvmaxar 
    if isinstance(vmaxar, float) or isinstance(vmaxar, np.float64):
        if vmaxar > 57:
            sigma = sigma0
        elif vmaxar <= 57:
            sigma = kappa * np.log10(vmaxar/V0)
        sigma_ar = sigma
    else:
        sigma_ar = np.zeros(0)
        for vmax in vmaxar:
            if vmax > 57:
                sigma = sigma0
            elif vmax <= 57:
                sigma = kappa * np.log10(vmax/V0)
            sigma_ar = np.append(sigma_ar, sigma)
    return sigma_ar

def get_mstar_pl_wsc(lvmaxar):
    '''
    This gives the stellar mass of a subhalos accounting for the scatter in the relation (as provided by Santos-Santos et al. 2022)
    '''
    mu_mstar = get_mstar_pl(lvmaxar) #this will be the mean for the gaussian distribution
    sig_mstar = get_scatter(lvmaxar) #this will be the scatter in the relation which is considered to be a gaussian
    # print(len(mu_mstar), len(sig_mstar), len(lvmaxar))
    if isinstance(lvmaxar, float) or isinstance(lvmaxar, np.float64) or isinstance(lvmaxar, np.float32):
        mstar = np.random.normal(mu_mstar, sig_mstar, size = 1) 
    else:
        mstar = np.random.normal(mu_mstar, sig_mstar, size = len(lvmaxar))
    return 10**mstar


def get_mstar_co_wsc(lvmaxar):
    '''
    This gives the stellar mass of a subhalos accounting for the scatter in the relation (as provided by Santos-Santos et al. 2022)
    '''
    # CURRENTLY USING TWICE THE SCATTER AS COMPARED TO ISABEL'S PAPER
    mu_mstar = get_mstar_co(lvmaxar) #this will be the mean for the gaussian distribution
    sig_mstar = get_scatter(lvmaxar) #this will be the scatter in the relation which is considered to be a gaussian
    if isinstance(lvmaxar, float) or isinstance(lvmaxar, np.float64) or isinstance(lvmaxar, np.float32):
        mstar = np.random.normal(mu_mstar, 2 * sig_mstar, size = 1) 
    else:
        mstar = np.random.normal(mu_mstar, 2 * sig_mstar, size = len(lvmaxar))
    return 10**mstar

def get_lrh(lmstar_ar, m1 = 0.178, m2 = 0.31, b = -1.49):
    '''
    This function returns the log rh for a given Mstar -- in what units?
    0.17832722702850887, 0.30555128418263083, -1.4929324569613338
    '''
    if isinstance(lmstar_ar, float):
        if lmstar_ar > 6.5:
            lrh = m1 * lmstar_ar + b 
        elif lmstar_ar <= 6.5:
            lrh = m2 * lmstar_ar + (m1 - m2) * 6.5 + b
        lrh_ar = lrh
    else:
        lrh_ar = np.zeros(0)
        # print(lmstar_ar)
        for lmstar in lmstar_ar:
            if lmstar > 6.5:
                lrh = m1 * lmstar + b 
            elif lmstar <= 6.5:
                lrh = m2 * lmstar + (m1 - m2) * 6.5 + b
            lrh_ar = np.append(lrh_ar, lrh)
    return lrh_ar


def get_rh_wsc(lmstar_ar):
    '''
    This function returns the half light radius for a given stellar mass
    '''
    mu_lrh = get_lrh(lmstar_ar)
    sig_lrh = 0.2
    if isinstance(lmstar_ar, float):
        lrh = np.random.normal(mu_lrh, sig_lrh, size=1)
    else:
        lrh = np.random.normal(mu_lrh, sig_lrh, size=len(lmstar_ar))
    return 10**lrh


def convert_to_float(value):
    try:
        if isinstance(value, float) or isinstance(value, int):
            return value

        if value == '[inf, inf, inf]' or value == '[inf inf inf]':
            return np.array([np.inf, np.inf, np.inf])

        if value == '[inf]':
            return np.inf
        
        if value == 'inf':
            return np.inf
        

        if value == '[-inf]':
            return -np.inf

        

        blah = ast.literal_eval(value)
        if isinstance(blah, list):
            if len(blah) == 1:
                blah2 = float(blah[0])   
            elif len(blah) == 3:
                blah2 = np.array([float(blah[0]), float(blah[1]), float(blah[2])])
        else:
            blah2 = float(blah)        
        return blah2
    except Exception as e:
        try:
            value_fixed = re.sub(r'(?<=\d)\s+(?=-?\d)', ', ', value)  # Handle spaces between numbers
            value_fixed = re.sub(r'(?<=\d)\s+(?=\d)', ', ', value_fixed)
            value_fixed = re.sub(r'(?<=\.)\s+(?=-\d)', ', ', value_fixed)
            value_fixed = re.sub(r'(?<=\.)\s+(?=\d)', ', ', value_fixed)
            # Now use ast.literal_eval
            # print(value_fixed)
            result = ast.literal_eval(value_fixed)
            return result
        except Exception as e:
            print(value)
            return value

# ======================================


suffix = '_Vpeak_newall' # This is the attempt where we have used the new power law
filepath = '/bigdata/saleslab/psadh003/tng50dark/tng_files/'

df = pd.read_csv(filepath + fofstr +  '_surviving_evolved_everything' + suffix + '.csv', delimiter= ',', low_memory=False)
df = df.applymap(convert_to_float)

df.columns = df.columns.str.replace('pl', 'pl1', regex=True) #This is to change the current power law to old power law 
df.columns = df.columns.str.replace('co', 'co1', regex=True) #This is to change the current cutoff to old cutoff


# Now the current power law and stellar mass of the Type-1s have been saved.

# Now, we are going to load som necessary columns for assigning the stellar mass and evolving the satellites

t1_mstar_max_ar = df['mstar_max_ar'].values # Remember that the max stellar mass decides whether a satellite is resolbed or unresolved 

t1_vpeak_ar = df['vpeak_ar'].values # This is the vpeak based on which we will be assigning the stellar mass 

t1_mmx_if_ar = df['mmx_if_ar'].values # This is the infall mass based on which we will be evolving the subhalos
t1_rmx_if_ar = df['rmx_if_ar'].values # This is the infall radius based on which we will be evolving the subhalos
t1_vmx_if_ar = df['vmx_if_ar'].values # This is the infall velocity based on which we will be evolving the subhalos
t1_rperi_ar = df['rperi_ar'].values # This is the pericentric radius based on which we will be evolving the subhalos
t1_rapo_ar = df['rapo_ar'].values # This is the apocentric radius based on which we will be evolving the subhalos
t1_torb_ar = df['torb_ar'].values # This is the orbital time based on which we will be evolving the subhalos
t1_tinf_ar = df['tinf_ar'].values # This is the infall time based on which we will be evolving the subhalos 

t1_vd_max_ar = df['vd_max_ar'].values # This is the maximum velocity dispersion based on which we will be evolving the subhalos


for ix in range(len(t1_mstar_max_ar[:50])):
    if t1_mstar_max_ar[ix] > 5e6:
        pass
    else:
        vpeak = t1_vpeak_ar[ix]
        mstar_max_pl = get_mstar_pl_wsc(np.log10(vpeak))
        rh_max_pl = get_rh_wsc(np.log10(mstar_max_pl))
        mstar_max_co = get_mstar_co_wsc(np.log10(vpeak))
        rh_max_co = get_rh_wsc(np.log10(mstar_max_co))
        if t1_torb_ar[ix] == np.inf:
            pass
        else:
            vpeak = t1_vpeak_ar[ix]
            mstar_max_pl = get_mstar_pl_wsc(np.log10(vpeak))
            rh_max_pl = get_rh_wsc(np.log10(mstar_max_pl))
            mstar_max_co = get_mstar_co_wsc(np.log10(vpeak))
            rh_max_co = get_rh_wsc(np.log10(mstar_max_co))
            if t1_torb_ar[ix] == np.inf:
                pass
            else:
                subh_pl = ErraniSubhalo(torb = float(t1_torb_ar[ix]), rperi = float(t1_rperi_ar[ix]), rapo = float(t1_rapo_ar[ix]), Rh = float(rh_max_pl), mstar0 = float(mstar_max_pl), rmx0 = float(t1_rmx_if_ar[ix]), mmx0 = float(t1_mmx_if_ar[ix]), vmx0 = float(t1_vmx_if_ar[ix]))
                _, _, _, _, rh_f_pl, mstar_f_pl = subh_pl.evolve_interp(tevol = float(all_ages[99]) - float(t1_tinf_ar[ix]), V0 = v200, min_mstarf = 0)
                subh_co = ErraniSubhalo(torb = float(t1_torb_ar[ix]), rperi = float(t1_rperi_ar[ix]), rapo = float(t1_rapo_ar[ix]), Rh = rh_max_co, mstar0 = mstar_max_co, rmx0 = float(t1_rmx_if_ar[ix]), mmx0 = float(t1_mmx_if_ar[ix]), vmx0 = float(t1_vmx_if_ar[ix]))
                _, _, _, _, rh_f_co, mstar_f_co = subh_co.evolve_interp(tevol = float(all_ages[99]) - float(t1_tinf_ar[ix]), V0 = v200, min_mstarf = 0)


# sys.exit(0)
        


def get_new_mstar_t1(ix):
    '''
    This function will return the new stellar mass of the Type-1 subhalos. 
    
    Returns:
    mstar_max_pl_ar, rh_max_pl_ar, mstar_max_co_ar, rh_max_co_ar, mstar_f_pl_ar, rh_f_pl_ar, mstar_f_co_ar, rh_f_co_ar
    '''
    if t1_mstar_max_ar[ix] > 5e6: # This is the resolved case
        return -1, -1, -1, -1, -1, -1, -1, -1
    else: # These are the unresolved cases
        vpeak = t1_vpeak_ar[ix]
        # print(vpeak, t1_mstar_max_ar[ix], t1_rperi_ar[ix], t1_rapo_ar[ix], t1_torb_ar[ix], t1_tinf_ar[ix], t1_rmx_if_ar[ix], t1_mmx_if_ar[ix], t1_vmx_if_ar[ix])
        
        # return -1, -1, -1, -1, -1, -1, -1, -1
        mstar_max_pl = get_mstar_pl_wsc(np.log10(vpeak))
        rh_max_pl = get_rh_wsc(np.log10(mstar_max_pl))
        mstar_max_co = get_mstar_co_wsc(np.log10(vpeak))
        rh_max_co = get_rh_wsc(np.log10(mstar_max_co))
        if t1_torb_ar[ix] == np.inf:
            return mstar_max_pl, rh_max_pl, mstar_max_co, rh_max_co, -1, -1, -1, -1 # This is the case where the orbital time is too high
        subh_pl = ErraniSubhalo(torb = float(t1_torb_ar[ix]), rperi = float(t1_rperi_ar[ix]), rapo = float(t1_rapo_ar[ix]), Rh = float(rh_max_pl), mstar0 = float(mstar_max_pl), rmx0 = float(t1_rmx_if_ar[ix]), mmx0 = float(t1_mmx_if_ar[ix]), vmx0 = float(t1_vmx_if_ar[ix])) 
        # _, _, _, _, rh_f_pl, mstar_f_pl = subh_pl.evolve(tevol = float(all_ages[99]) - float(t1_tinf_ar[ix]), V0 = v200, min_mstarf = 0)
        _, _, _, _, rh_f_pl, mstar_f_pl = subh_pl.evolve_interp(tevol = float(all_ages[99]) - float(t1_tinf_ar[ix]), V0 = v200, min_mstarf = 0)
        subh_co = ErraniSubhalo(torb = float(t1_torb_ar[ix]), rperi = float(t1_rperi_ar[ix]), rapo = float(t1_rapo_ar[ix]), Rh = rh_max_co, mstar0 = mstar_max_co, rmx0 = float(t1_rmx_if_ar[ix]), mmx0 = float(t1_mmx_if_ar[ix]), vmx0 = float(t1_vmx_if_ar[ix]))
        # _, _, _, _, rh_f_co, mstar_f_co = subh_co.evolve(tevol = float(all_ages[99]) - float(t1_tinf_ar[ix]), V0 = v200, min_mstarf = 0) 
        _, _, _, _, rh_f_co, mstar_f_co = subh_co.evolve_interp(tevol = float(all_ages[99]) - float(t1_tinf_ar[ix]), V0 = v200, min_mstarf = 0) 
    return mstar_max_pl, rh_max_pl, mstar_max_co, rh_max_co, mstar_f_pl, rh_f_pl, mstar_f_co, rh_f_co

        
    



results = Parallel(n_jobs = 32, pre_dispatch = '1.5*n_jobs')(delayed(get_new_mstar_t1)(ix) for ix in tqdm(range(len(t1_mstar_max_ar)))) 

df['mstar_max_pl_ar'] = [value[0] for value in results]
df['rh_max_pl_ar'] = [value[1] for value in results]
df['mstar_max_co_ar'] = [value[2] for value in results]
df['rh_max_co_ar'] = [value[3] for value in results]
df['mstar_f_pl_ar'] = [value[4] for value in results]
df['rh_f_pl_ar'] = [value[5] for value in results]
df['mstar_f_co_ar'] = [value[6] for value in results]
df['rh_f_co_ar'] = [value[7] for value in results]


def type1_additional_details(ix):
    '''
    This function will return the additional details for the Type-1 subhalos. 

    Returns:
    hof_flag
    '''
    def hof(lvpeak, x0 = 1.29, x1 = 0.05):
        return 0.5 * (1 + np.tanh((lvpeak - x0) / x1)) 
    lvpeak = np.log10(t1_vpeak_ar[ix])
    this_hof = hof(lvpeak)
    hof_flag = np.random.rand() < this_hof
    return hof_flag

results = Parallel(n_jobs = 32, pre_dispatch = '1.5*n_jobs')(delayed(type1_additional_details)(ix) for ix in tqdm(range(len(t1_mstar_max_ar))) )

df['hof_flag'] = [value for value in results]


# suffix = '_vpeak_bestfits'
suffix = '_interpolation'
# suffix = '_interpolation2'
df.to_csv(filepath + fofstr +  '_surviving_evolved_everything' + suffix + '.csv', index = False)


# ======================================
# We will now move on to Type-2s 
# ======================================

suffix = '_Vpeak_newall'

df = pd.read_csv(filepath + fofstr +  '_merged_evolved_everything' + suffix + '.csv', delimiter= ',', low_memory=False)
df = df.applymap(convert_to_float)

df.columns = df.columns.str.replace('pl', 'pl1', regex=True) #This is to change the current power law to old power law
df.columns = df.columns.str.replace('co', 'co1', regex=True) #This is to change the current cutoff to old cutoff

t2_mstar_max_ar = df['mstar_max_ar'].values # Remember that the max stellar mass decides whether a satellite is resolbed or unresolved

t2_vpeak_ar = df['vpeak_ar'].values # This is the vpeak based on which we will be assigning the stellar mass

t2_mmx_if_ar = df['mmx_if_ar'].values # This is the infall mass based on which we will be evolving the subhalos
t2_rmx_if_ar = df['rmx_if_ar'].values # This is the infall radius based on which we will be evolving the subhalos
t2_vmx_if_ar = df['vmx_if_ar'].values # This is the infall velocity based on which we will be evolving the subhalos
t2_rperi_ar = df['rperi_ar'].values # This is the pericentric radius based on which we will be evolving the subhalos
t2_rapo_ar = df['rapo_ar'].values # This is the apocentric radius based on which we will be evolving the subhalos
t2_torb_ar = df['torb_ar'].values # This is the orbital time based on which we will be evolving the subhalos
t2_tinf_ar = df['tinf_ar'].values # This is the infall time based on which we will be evolving the subhalos

t2_vd_max_ar = df['vd_max_ar'].values # This is the maximum velocity dispersion based on which we will be evolving the subhalos

t2_snap_if_ar = df['snap_if_ar'].values
t2_sfid_if_ar = df['sfid_if_ar'].values

def get_new_mstar_t2(ix):
    '''
    This function will return the new stellar mass of the Type-2 subhalos.
    '''
    if t2_mstar_max_ar[ix] > 5e6: # This is the resolved case
        return -1, -1, -1, -1, -1, -1, -1, -1
    else: # These are the unresolved cases
        vpeak = t2_vpeak_ar[ix]
        # print(vpeak, t2_mstar_max_ar[ix], t2_rperi_ar[ix], t2_rapo_ar[ix], t2_torb_ar[ix], t2_tinf_ar[ix], t2_rmx_if_ar[ix], t2_mmx_if_ar[ix], t2_vmx_if_ar[ix])
        mstar_max_pl = get_mstar_pl_wsc(np.log10(vpeak))
        rh_max_pl = get_rh_wsc(np.log10(mstar_max_pl))
        mstar_max_co = get_mstar_co_wsc(np.log10(vpeak))
        rh_max_co = get_rh_wsc(np.log10(mstar_max_co))
        if t2_torb_ar[ix] == np.inf:
            return mstar_max_pl, rh_max_pl, mstar_max_co, rh_max_co, -1, -1, -1, -1 # This is the case where the orbital time is too high
        subh_pl = ErraniSubhalo(torb = float(t2_torb_ar[ix]), rperi = float(t2_rperi_ar[ix]), rapo = float(t2_rapo_ar[ix]), Rh = float(rh_max_pl), mstar0 = float(mstar_max_pl), rmx0 = float(t2_rmx_if_ar[ix]), mmx0 = float(t2_mmx_if_ar[ix]), vmx0 = float(t2_vmx_if_ar[ix]))
        # _, _, _, _, rh_f_pl, mstar_f_pl = subh_pl.evolve(tevol = float(all_ages[99]) - float(t2_tinf_ar[ix]), V0 = v200, min_mstarf = 0)
        _, _, _, _, rh_f_pl, mstar_f_pl = subh_pl.evolve_interp(tevol = float(all_ages[99]) - float(t2_tinf_ar[ix]), V0 = v200, min_mstarf = 0)
        subh_co = ErraniSubhalo(torb = float(t2_torb_ar[ix]), rperi = float(t2_rperi_ar[ix]), rapo = float(t2_rapo_ar[ix]), Rh = rh_max_co, mstar0 = mstar_max_co, rmx0 = float(t2_rmx_if_ar[ix]), mmx0 = float(t2_mmx_if_ar[ix]), vmx0 = float(t2_vmx_if_ar[ix]))
        # _, _, _, _, rh_f_co, mstar_f_co = subh_co.evolve(tevol = float(all_ages[99]) - float(t2_tinf_ar[ix]), V0 = v200, min_mstarf = 0)
        _, _, _, _, rh_f_co, mstar_f_co = subh_co.evolve_interp(tevol = float(all_ages[99]) - float(t2_tinf_ar[ix]), V0 = v200, min_mstarf = 0)
    return mstar_max_pl, rh_max_pl, mstar_max_co, rh_max_co, mstar_f_pl, rh_f_pl, mstar_f_co, rh_f_co

results = Parallel(n_jobs = 32, pre_dispatch = '1.5*n_jobs')(delayed(get_new_mstar_t2)(ix) for ix in tqdm(range(len(t2_mstar_max_ar))) )

df['mstar_max_pl_ar'] = [value[0] for value in results]
df['rh_max_pl_ar'] = [value[1] for value in results]
df['mstar_max_co_ar'] = [value[2] for value in results]
df['rh_max_co_ar'] = [value[3] for value in results]
df['mstar_f_pl_ar'] = [value[4] for value in results]
df['rh_f_pl_ar'] = [value[5] for value in results]
df['mstar_f_co_ar'] = [value[6] for value in results]
df['rh_f_co_ar'] = [value[7] for value in results]



# ======================================

# We are now going to check if the descendant of each of the subhalos is indeed the central. 

df0 = pd.read_csv(filepath + 'hydrofofno_' + str(fofno) + '_t2.csv', comment = '#')
par_snap_if_ar = np.array(df0['snap_if_ar'].values, dtype = int)
par_sfid_if_ar = np.array(df0['sfid_if_ar'].values, dtype = int)
par_snap_merger_ar = np.array(df0['snap_merger_ar'].values, dtype = int)
par_sfid_merger_ar = np.array(df0['sfid_merger_ar'].values, dtype = int)
par_posx_ar = np.array(df0['posx_ar'].values)
par_posy_ar = np.array(df0['posy_ar'].values)
par_posz_ar = np.array(df0['posz_ar'].values)

def type2_additional_details(ix):
    '''
    This function will return the additional details for the Type-2 subhalos. 

    Returns:
    merger_snap, merger_sfid, posx, posy, posz, desc_flag, desc_flag2, mbpID, hof_flag
    '''
    # For each subhalo in the evolved catalog, lets get the subhalo from the parent catalog
    snap_if = t2_snap_if_ar[ix]
    sfid_if = t2_sfid_if_ar[ix]

    # look for these in the par_snap_if_ar and par_sfid_if_ar
    ix2 = np.where((par_snap_if_ar == snap_if) & (par_sfid_if_ar == sfid_if))[0][0]
    merger_snap = par_snap_merger_ar[ix2]
    merger_sfid = par_sfid_merger_ar[ix2]

    subh_tree = il.sublink.loadTree(basePath, merger_snap, merger_sfid, fields = ['SubhaloID', 'DescendantID', 'SnapNum', 'SubfindID', 'SubhaloIDMostbound'], onlyMDB = True)
    subh_sfid = subh_tree['SubfindID']
    subh_snap = subh_tree['SnapNum']
    subh_desc = subh_tree['DescendantID']
    subh_shid = subh_tree['SubhaloID']
    if len(subh_snap[subh_snap == merger_snap+1]) > 1 :
        desc_flag2 = 0
    else:
        desc_flag2 = 1

    ix_merged = np.where((cft_snap == merger_snap) & (cft_sfid == merger_sfid))[0][0]
    ix_desc = np.where(cft_shid == cft_desc[ix_merged])[0][0] # This is the index of the descendant in the cft
    desc_sfid = cft_sfid[ix_desc]
    desc_snap = cft_snap[ix_desc]
    if desc_sfid == central_sfids[central_snaps == desc_snap]:
        desc_flag = 1
    else: 
        desc_flag = 0
    # this_subh_desc = subh_desc[-1] # This will be the descendant of the current subhalo
    # ix3 = np.where(subh_shid == this_subh_desc)[0][0]
    # # subh_immediate_desc = subh_sfid[subh_snap == merger_snap+1]
    # # lets check if this is the central sfid at this snapshot
    # if int(subh_immediate_desc) == central_sfids[central_snaps == merger_snap+1]:
    #     desc_flag = 1
    # else:
    #     desc_flag = 0


    # desc_flag
    def hof(lvpeak, x0 = 1.29, x1 = 0.05):
        return 0.5 * (1 + np.tanh((lvpeak - x0) / x1))
    lvpeak = np.log10(t2_vpeak_ar[ix])
    this_hof = hof(lvpeak)
    hof_flag = np.random.rand() < this_hof

    return merger_snap, merger_sfid, par_posx_ar[ix2], par_posy_ar[ix2], par_posz_ar[ix2], desc_flag, desc_flag2, cft_mbpid[ix_merged] , hof_flag

results = Parallel(n_jobs = 32, pre_dispatch = '1.5*n_jobs')(delayed(type2_additional_details)(ix) for ix in tqdm(range(len(t2_mstar_max_ar))) )

df['merger_snap'] = [value[0] for value in results]
df['merger_sfid'] = [value[1] for value in results]
df['posx'] = [value[2] for value in results]
df['posy'] = [value[3] for value in results]
df['posz'] = [value[4] for value in results]
df['desc_flag'] = [value[5] for value in results]
df['desc_flag2'] = [value[6] for value in results]
df['mbpID'] = [value[7] for value in results]
df['hof_flag'] = [value[8] for value in results]

# suffix = '_vpeak_bestfits'
suffix = '_interpolation'
# suffix = '_interpolation2'
df.to_csv(filepath + fofstr +  '_merged_evolved_everything' + suffix + '.csv', index = False)
# ======================================
