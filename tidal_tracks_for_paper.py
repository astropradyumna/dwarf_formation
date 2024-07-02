'''
This part is to obtain the tidal tracks for different subhalos
This goes into figure one of the paper, showing the tidal tracks that Errani22 gives

We will be saving the files based on the evolved parameters -- Mstar and Rh is what we are looking at
'''



import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import illustris_python as il
from orbit_calculator_preamble import *
from galpy.potential import NFWPotential, TimeDependentAmplitudeWrapperPotential
from galpy.orbit import Orbit
from astropy import units as u
from testing_errani import get_rot_curve, get_rmxbyrmx0, get_vmxbyvmx0, get_mxbymx0, get_LbyL0, l10rbyrmx0_1by4_spl,l10rbyrmx0_1by2_spl, l10rbyrmx0_1by8_spl, l10rbyrmx0_1by16_spl, l10vbyvmx0_1by2_spl, l10vbyvmx0_1by4_spl, l10vbyvmx0_1by8_spl, l10vbyvmx0_1by16_spl, l10rbyrmx0_1by66_spl, l10rbyrmx0_1by250_spl, l10rbyrmx0_1by1000_spl, l10vbyvmx0_1by66_spl, l10vbyvmx0_1by250_spl, l10vbyvmx0_1by1000_spl
from tng_subhalo_and_halo import TNG_Subhalo
from matplotlib import gridspec
from colossus.cosmology import cosmology
from colossus.halo import concentration
from scipy.signal import argrelmin
import warnings
from populating_stars import *


cosmology.setCosmology('planck18')

h = 0.6774
mass_dm = 3.07367708626464e-05 * 1e10/h #This is for TNG50-1
G = 4.5390823753559603e-39 #This is in kpc, Msun and seconds

mvir = 1.5e8
conc=concentration.concentration(0.6744 * mvir, 'vir', 0, 'ludlow16')
print(conc)



def get_converted_nfw_params(m_vir, c_vir):
    '''
    Converts virial mass and concentration of an NFW profile to rho_s and r_s parameters.
    
    Args:
        m_vir (float): Virial mass in solar masses.
        c_vir (float): Concentration parameter.
        
    Returns:
        rho_s (float): Scale density in solar masses per cubic kiloparsec.
        r_s (float): Scale radius in kiloparsecs.
    '''
    # Define constants
    G = 4.302e-6 # Gravitational constant in units of kpc / Msun (km/s)^2
    H0 = h * 0.1 # Hubble constant in units of km/s / kpc
    
    # Calculate critical density of the Universe
    rho_crit = 3 * H0**2 / (8 * np.pi * G)
    
    # Calculate virial radius
    r_vir = (3 * m_vir / (4 * np.pi * c_vir**3 * rho_crit))**(1/3)
    
    # Calculate scale radius
    r_s = r_vir / c_vir
    
    # Calculate scale density
    rho_s = m_vir / (4 * np.pi * r_s**3 * (np.log(1 + c_vir) - c_vir / (1 + c_vir)))

    rmx0 = 2.16*r_s
    vmx0 = (1.65*r_s)*np.sqrt(G*rho_s)
    
    return rmx0, vmx0


rmx0, vmx0 = get_converted_nfw_params(mvir, conc)

mstar = 10**get_mstar_pl(np.log10(vmx0)) #This would be the stellar mass corresponding to the virial mass in power law model
rh = 10**get_lrh(np.log10(mstar))/2

print(np.log10(mstar), rh) #This is the initial stellar mass and rh for the subhalo

def get_rh0byrmx0(Rh, rmx0):
    '''
    This is a function to calculate the initial rh0burmx0 for the subhalo
    '''
    values = [1/2, 1/4, 1/8, 1/16]
    Rh0 = Rh
    
    closest_value = min(values, key=lambda x: abs(np.log10(x) - np.log10(Rh0/rmx0)))
    # print(closest_value)
    return closest_value

rh0byrmx0 = get_rh0byrmx0(rh, rmx0)

print(rh0byrmx0, 'versus', rh/rmx0)

frem_ar = np.logspace(-4, 0, 100)
mstar_ar = np.zeros(0)
rh_ar = np.zeros(0)

for frem in frem_ar:
    rmx = get_rmxbyrmx0(frem) * rmx0
    mstar_evolved = get_LbyL0(frem, rh0byrmx0) * mstar
    mstar_ar = np.append(mstar_ar, mstar_evolved)
    if np.log10(frem) >= -2.5:
        if rh0byrmx0 == 0.25:
            rh_now = 10 ** (l10rbyrmx0_1by4_spl(np.log10(frem))) * rh/rh0byrmx0
        elif rh0byrmx0 == 0.125:
            rh_now = 10 ** (l10rbyrmx0_1by8_spl(np.log10(frem))) * rh/rh0byrmx0
        elif rh0byrmx0 == 0.5:
            rh_now = 10 ** (l10rbyrmx0_1by2_spl(np.log10(frem))) * rh/rh0byrmx0
        elif rh0byrmx0 == 0.0625:
            rh_now = 10 ** (l10rbyrmx0_1by16_spl(np.log10(frem))) * rh/rh0byrmx0


    elif np.log10(frem) < -2.5:
        rh_now = rmx

    rh_ar = np.append(rh_ar, rh_now)

rh_ar = rh_ar * 1e3 #coverting the rh to parsecs

df = pd.DataFrame({'mstar': mstar_ar, 'rh': rh_ar})


filepath = '/bigdata/saleslab/psadh003/misc_files/'

df.to_csv(filepath + '1e6_tidal_tracks.csv', index = False)

# plt.plot(np.log10(mstar_ar), np.log10(rh_ar))
# plt.show()





# fig, ax = plt.subplots(figsize = ())