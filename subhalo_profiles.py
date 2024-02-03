'''
This program has been taken from the HPCC, this can be used to create the exponential profile and compare it with the subhalos
'''

import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import IPython
import warnings


filepath = '/rhome/psadh003/bigdata/ucd_formation_errani_files/'


def get_surf_dens_in_mag_per_arcsec2_wiki(sigma):
    '''
    This function is to return the surface density of a given profile in mag per arcsec^2

    Args:
    sigma: This is the array of surface density in Msun/(pc)^2

    Returns:
    sigma_mag: The array of surface density in mag/arcsec^2
    
    This relation is from Wikipedia
    '''
    S = 4.83 + 21.527 - 2.5 * np.log10(sigma)
    return S

def get_king_k(M, r_c, c = 1.5):
    '''
    This function returns the central surface density (k) of the King profile 

    Args:
    M [Msun]: mass of the system in Msun
    r_c [pc]: the core radius of the system in pc
    c : The concentration parameter to assume (This is typically between 1 and 2 for UCDs)
    FIXME: Concentration cannot be fixed from the paper to test mixing of the profiles

    Returns:
    k
    '''
    r_t = r_c * 10 ** c
    def get_king_mass2(k, r_c, r_t):
        '''
        This function is zero at r_t for the right value of k
        '''
        r = r_t #at tidal raduis, we expect the maximum mass
        m = 2*np.pi*k*r_c**2*np.log(abs(r_c*np.sqrt(r**2+r_c**2)+np.abs(r_c)*r))+(8*k*r_c**2*np.arctan(r/r_c)-8*k*r_c*r)/np.sqrt(r_t**2/r_c**2+1)-(2*np.pi*k*r_c*r)/np.sqrt(r**2/r_c**2+1)
        m = m - 4 * np.pi * k * r_c ** 2 * np.log(r_c)
        return m - M
    k = fsolve(get_king_mass2, 1000, args = (r_c, r_t))[0]

    return k



G = 4.5390823753559603e-39 #This is in kpc, Msun and seconds

def get_H(z = 0, h = 0.6774):
    '''
    Calculates the Hubble constant as a function of redshift z in km/s/Mpc
    '''
    # Cosmological model parameters
    hubble_constant = h*100  # Hubble constant in km/s/Mpc
    matter_density = 0.31  # Density of matter in the universe
    dark_energy_density = 0.69  # Density of dark energy in the universe

    # Calculate the Hubble parameter
    hubble_parameter = hubble_constant * np.sqrt(matter_density * (1 + z)**3 + dark_energy_density)

    return hubble_parameter


def get_critical_dens(z):
    '''
    Returns the critical density of the universe at a given redshift in Msun/kpc^3
    '''
    H = get_H(z)*3.24078e-20 #in s^-1
    
    return 3*H**2/(8*np.pi*G) #Msun, kpc



class StellarProfile:
    def surface_brightness_mag(self, r):
        '''
        This function is to return the surface brightness in mag/arcsec^2

        Args:
        r (float)[pc]: The radius where surface brightness needs to be caluclated
        '''
        return get_surf_dens_in_mag_per_arcsec2_wiki(self.surface_brightness(r))
    
    def rh(self):
        '''
        This function returns the half light radius for the given King profile
        '''
        rad = fsolve(lambda r:self.mass(r) - 0.5 * self.M, 100)[0] # the radius where mass / light is half of the max mass
        return rad


class KingProfile(StellarProfile):
    def __init__(self, M, r_c, c):
        self.M = M
        self.r_c = r_c
        self.r_t = 10**c * r_c
        self.k = get_king_k(M, r_c, c)

    def mass(self, rad):
        """
        Calculate the mass enclosed within a given radius in pc.

        Parameters:
        - radius (float): The radius at which to calculate the enclosed mass.

        Returns:
        - float: The enclosed mass.
        """
        k = self.k
        r_c = self.r_c 
        r_t = self.r_t
        if isinstance(rad, float):
            r = rad
            M = 2*np.pi*k*r_c**2*np.log(abs(r_c*np.sqrt(r**2+r_c**2)+np.abs(r_c)*r))+(8*k*r_c**2*np.arctan(r/r_c)-8*k*r_c*r)/np.sqrt(r_t**2/r_c**2+1)-(2*np.pi*k*r_c*r)/np.sqrt(r**2/r_c**2+1)
            M = M - 4 * np.pi * k * r_c ** 2 * np.log(r_c)
            return M
        M = np.zeros(len(rad))
        r = rad[rad<= r_t]
        M[rad<=r_t] = 2*np.pi*k*r_c**2*np.log(abs(r_c*np.sqrt(r**2+r_c**2)+np.abs(r_c)*r))+(8*k*r_c**2*np.arctan(r/r_c)-8*k*r_c*r)/np.sqrt(r_t**2/r_c**2+1)-(2*np.pi*k*r_c*r)/np.sqrt(r**2/r_c**2+1)
        M = M - 4 * np.pi * k * r_c ** 2 * np.log(r_c)
        M[rad>r_t] = self.M * np.ones(len(rad[rad>r_t]))
        return M
        

    def density(self, r):
        """
        Calculate the density at a given radius.

        Parameters:
        - r (float): The radius at which to calculate the density in pc.

        Returns:
        - float: The density at the given radius.
        """
        k = self.k
        r_c = self.r_c 
        r_t = self.r_t
        dens = (1/np.pi)*(k*r_c*(np.pi*r_c*np.sqrt(r_c**2+r**2)*np.sqrt(r_t**2/r_c**2+1)-4*r_c**2-4*r**2))/(2*(r_c**4+2*r**2*r_c**2+r**4)*np.sqrt(r_t**2/r_c**2+1))
        return dens
        
        

    def surface_brightness(self, rad_ar):
        """
        Calculate the surface brightness at a given radius.

        Parameters:
        - r (float): The radius at which to calculate the surface brightness in pc.

        Returns:
        - sb (float): The surface brightness at the given radius in Msun/pc^2.
        """
        if isinstance(rad_ar, float):
            if rad_ar > self.r_t:
                return 0
            else:
                r = rad_ar
                return self.k * ((1/np.sqrt(1 + (r/self.r_c) ** 2)) - (1/np.sqrt(1 + ( self.r_t / self.r_c )**2)))**2
        sb = np.zeros(len(rad_ar))
        for (i, r) in enumerate(rad_ar):
            if r < self.r_t:
                sb[i] = self.k * ((1/np.sqrt(1 + (r/self.r_c) ** 2)) - (1/np.sqrt(1 + ( self.r_t / self.r_c )**2)))**2
            else:
                sb[i] = 0  
        return sb
    
    def mean_density(self, r):
        '''
        This function returns the mean density inside a radius r        
        '''
        return self.mass(r)/ ((4/3.) * np.pi* r**3 )
    

class ExponentialProfile(StellarProfile):
    '''
    This class is to create an object following exponential profile
    '''
    def __init__(self, M, rh):
        self.M = M
        self.rh = rh

    def density(self, r):
        '''
        Returns the exopential density profile
        '''
        rstar_s = self.rh / 2
        return np.exp(-r / rstar_s) * self.M/(2.39 * self.rh**3)
    
    def surface_brightness(self, R_ar):
        '''
        Returns the surface brightness profile
        '''
        somebigradius = 1e5
        sb = np.zeros(0)
        for R in R_ar:
            sb = np.append(sb, 2*quad(lambda r: self.density(r) * r / np.sqrt(r**2 - R**2),R,somebigradius)[0])
        return sb

    def mass(self, r):
        '''
        This returns the mass
        '''
        r_s = self.rh / 2
        return 4*np.pi* self.M/(2.39 * self.rh**3) *(2*r_s**3-(r_s*r**2+2*r_s**2*r+2*r_s**3)*np.exp(-r/r_s))
    


class NFWProfile():
    def __init__(self, z, Mvir = None, cvir = None, rmx = None, vmx = None):
        self.z = z
        if (rmx is None and vmx is None) and (Mvir is not None and cvir is not None):
            self.Mvir = Mvir
            self.cvir = cvir
        else:
            self.rmx = rmx
            self.vmx = vmx
            self.Mvir, self.cvir = self.get_mvircvir_from_mx()

    def get_mvircvir_from_mx(self):
        '''
        This function converts the rmx and vmx values into Mvir and cvir
        First proceeds through calculation of rho0 and rs
        '''
        rs = self.rmx/2.16 #hopeflly in kpc
        rho0 = 1e9 * (self.vmx  / (1.64 * rs * 1e3)) ** 2 * (1/4.3e-3) #this would be in Msun/kpc^3
        def g(c):
            return 1 / ( np.log(1 + c) - (c / (1 + c)) )
        
        with warnings.catch_warnings(record=True) as w:
            cvir = fsolve(lambda c: np.log10(rho0 * 3 / (200 * get_critical_dens(self.z))) -  np.log10(c ** 3 * g(c)),  10, xtol=1.49012e-12)[0]
            if len(w) > 0:
                with warnings.catch_warnings(record=True) as w:
                    cvir = fsolve(lambda c: np.log10(rho0 * 3 / (200 * get_critical_dens(self.z))) -  np.log10(c ** 3 * g(c)),  15, xtol=1.49012e-12)[0]
                    if len(w) > 0:
                        cvir = fsolve(lambda c: np.log10(rho0 * 3 / (200 * get_critical_dens(self.z))) -  np.log10(c ** 3 * g(c)),  20, xtol=1.49012e-12)[0]
        
        # print(rho0 * 3 / (200 * get_critical_dens(self.z)) -  cvir ** 3 * g(cvir))
        Mvir = 4 * np.pi * rho0 * rs ** 3  / g(cvir)

        # print(Mvir, cvir)
        # G2 = 4.302e-6 # Gravitational constant in units of kpc / Msun (km/s)^2
    
        # Calculate critical density of the Universe
        # rho_crit = 3 * 0.06744**2 / (8 * np.pi * G2)
        
        # Calculate virial radius
        r_vir = (3 * Mvir / (4 * np.pi * 200 * get_critical_dens(self.z)))**(1/3)
        
        # Calculate scale radius
        r_s = r_vir / cvir
        
        # Calculate scale density
        rho_s = Mvir / (4 * np.pi * r_s**3 * (np.log(1 + cvir) - cvir / (1 + cvir)))

        # rs = self.rmx/2.16 #hopeflly in kpc
        vmx_calc= 1.64 * r_s * 1e3 * np.sqrt(4.3e-3 * rho_s/1e9)  #this would be in Msun/kpc^3
        print(f'Calcuated rmx / input rmx = {2.16 * r_s / self.rmx:.2f} and vmx = {vmx_calc / self.vmx:.2f}')

        return Mvir, cvir


    def density(self, r):
        '''
        This function returns the density 
        based on https://ui.adsabs.harvard.edu/abs/2001MNRAS.321..155L/abstract
        '''
        Mvir = self.Mvir 
        cvir = self.cvir 
        z = self.z


        rvir = (Mvir / (4 / 3. * np.pi * 200 * get_critical_dens(z))) ** (1/3.) #kpc
        s = r/rvir
        
        def g(c):
            return 1 / ( np.log(1 + c) - c / (1 + c) )
        
        dens = get_critical_dens(self.z) * 200 * cvir ** 2 * g(cvir) / (3 * s * (1 + cvir * s)**2)
        return dens

    def mass(self, r):
        ''' 
        This functio returns the mass based on the previous paper again
        '''
        Mvir = self.Mvir 
        cvir = self.cvir 
        z = self.z

        rvir = (Mvir / (4 / 3. * np.pi * 200 * get_critical_dens(z))) ** (1/3.) #kpc
        s = r/rvir
        def g(c):
            return 1 / ( np.log(1 + c) - c / (1 + c) )
        mass = Mvir * g(cvir) * (np.log( 1+ cvir * s) - cvir * s / (1 + cvir * s))
        return mass
    
    # def velocity2(self, r):
    #     v =  np.sqrt(G * self.mass(r)/r) / 3.24078e-17
    #     return v

    def velocity(self, r):
        '''
        This is to get the circular velocity curve for the NFW from the same paper as above
        '''
        rvir = (self.Mvir / (4 / 3. * np.pi * 200 * get_critical_dens(self.z))) ** (1/3.) #kpc
        vv = 3.086e+16 *np.sqrt (G * self.Mvir / rvir) #km/s
        s = r / rvir
        c = self.cvir
        def g(c):
            return 1 / ( np.log(1 + c) - c / (1 + c) )
        # print(rvir, vv, s, c, g(c))
        v = vv * np.sqrt((g(c)/s) * (np.log(1 + c * s) - (c * s/(1 + c*s))))

        return v #km/s

    def mean_density(self, r):
        '''
        This function returns the mean density in a given radius 
        '''

        rho_mean = self.mass(r)/(4./3 * np.pi * r**3)
        return rho_mean




def get_rh_total(M, rh, M_ucd = 1e6, r_c_ucd = 10, c_ucd = 1.5):
    env = ExponentialProfile(M, rh) # defining an instance of the ExponentialProfile class for the envelope
    ucd = KingProfile(M_ucd, r_c_ucd, c_ucd)
    rh_combined = fsolve(lambda r:(ucd.mass(r) + env.mass(r)) - 0.5 * (env.M + ucd.M), 100)[0]
    return rh_combined


def plot_subhalo(M, rh, M_ucd = 1e6, r_c_ucd = 10, c_ucd = 1.5):

    ucd_col = 'magenta'
    env_col = 'green'
    combined_col = 'red'
    rpl = np.logspace(0, np.log10(r_t), 500)
    rpl2 = np.logspace(0, 3, 500)
    env = ExponentialProfile(M, rh) # defining an instance of the ExponentialProfile class for the envelope
    ucd = KingProfile(M_ucd, r_c_ucd, c_ucd)
    r_t = 10**c_ucd * r_c_ucd #this would be the tidal radius

    fig, [ax1, ax2, ax3] = plt.subplots(figsize = (15, 5), nrows = 1, ncols = 3)
    ax1.plot(rpl, ucd.surface_brightness_mag(rpl), c = ucd_col, ls = '--', label = 'Nucleus')
    ax1.plot(rpl2, env.surface_brightness_mag(rpl2), c = env_col, ls = '--', label = 'Envelope')
    ax1.plot(rpl2, get_surf_dens_in_mag_per_arcsec2_wiki(ucd.surface_brightness(rpl2) + env.surface_brightness(rpl2)), c= combined_col, ls = '--', label = 'Combined')
    ax1.set_xscale('log')
    ax1.invert_yaxis()
    ax1.set_ylim(bottom = 30, top = 17.5)
    ax1.set_ylabel(r'$\Sigma\,\rm{(mag/arcsec^2)}$')
    ax1.set_xlabel(r'Radius $\, \rm{(pc)}$')
    ax1.legend()
    ax1.set_title('Surface brightness')

    ax2.plot(rpl, ucd.density(rpl), c = ucd_col, ls = '--')
    ax2.plot(rpl2, env.density(rpl2), c = env_col, ls = '--')
    ax2.set_ylim(bottom = 1e-3, top = 1e2)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylabel(r'Density $\rm{(M_\odot/pc^3)}$')
    ax2.set_xlabel(r'Radius $\rm{(pc)}$')
    ax2.set_title('Mass density')

    ax3.plot(rpl, ucd.mass(rpl), c = ucd_col, ls = '--')
    ax3.plot(rpl2, env.mass(rpl2), c = env_col, ls = '--')
    ax3.plot(rpl2, ucd.mass(rpl2) + env.mass(rpl2), c = combined_col, ls = '--')
    ax3.axvline(ucd.rh(), ls = ':', c = ucd_col, label = r'$r_{\rm{h, nuc}}$')
    ax3.axvline(env.rh, ls = ':', c = env_col, label = r'$r_{\rm{h, env}}$')
    rh_combined = fsolve(lambda r:(ucd.mass(r) + env.mass(r)) - 0.5 * (env.M + ucd.M), 100)[0]
    ax3.axvline(rh_combined, ls = ':', c = combined_col, label = r'$r_{\rm{h, tot}}$')
    ax3.set_title('Mass profile')
    ax3.set_ylabel(r'Mass $\rm{(M_\odot)}$')
    ax3.set_xlabel(r'Radius $\rm{(pc)}$')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.legend(fontsize = 8)

    fig.suptitle(f'King profile of log(M/Msun) = {np.log10(ucd.M):.1f} and core radius = {r_c:.1f} pc\nEnvelope of log(M/Msun) = {np.log10(env.M):.1f} and r_h = {env.rh:.1f}', fontsize=16, bbox=dict(facecolor='orange', alpha=0.2, edgecolor='black'))
    plt.tight_layout()
    plt.savefig(filepath + 'test.png')

    return None





"""

def king(r, k, rc, rt):
    '''
    This returns the surface brightness of King profile

    k, rc and rt are the parameters for the King profile (see King 1962)
    '''
    sb = k * ((1/np.sqrt(1 + (r/rc) ** 2)) - (1/np.sqrt(1 + ( rt / rc )**2)))**2
    return sb



def get_deriv_king(r, k, rc, rt):
    dk = (2 * k *r / ( rc ** 2 * ( 1 + (r/rc) ** 2 ) ** 1.5 )) * ((1/np.sqrt(1 + (r/rc) ** 2)) - (1/np.sqrt(1 + ( rt / rc )**2)))
    return dk

def get_king_density2(r, k, r_c, r_t):
    '''
    This is the analytical form of the density profile corresponding to the King profile
    '''
    dens = (1/np.pi)*(k*r_c*(np.pi*r_c*np.sqrt(r_c**2+r**2)*np.sqrt(r_t**2/r_c**2+1)-4*r_c**2-4*r**2))/(2*(r_c**4+2*r**2*r_c**2+r**4)*np.sqrt(r_t**2/r_c**2+1))
    return dens

def get_king_mass2(r, k, r_c, r_t):
    M = 2*np.pi*k*r_c**2*np.log(abs(r_c*np.sqrt(r**2+r_c**2)+np.abs(r_c)*r))+(8*k*r_c**2*np.arctan(r/r_c)-8*k*r_c*r)/np.sqrt(r_t**2/r_c**2+1)-(2*np.pi*k*r_c*r)/np.sqrt(r**2/r_c**2+1)
    M = M - 4 * np.pi * k * r_c ** 2 * np.log(r_c)
    return M

def get_king_density(r_ar, k, rc, rt):
    '''
    This function returns the density profile corresopnding to the King surface density based on Eq. 1.79 from B&T
    '''
    somebignumber = 1000
    if isinstance(r_ar, float):
        return (1/np.pi)*quad(lambda R: ( R ** 2 - r_ar ** 2)**(-0.5) * get_deriv_king(R, k, rc, rt), r_ar , somebignumber)[0]
    
    j_ar = np.zeros(0)
    for r in r_ar:
        j = (1/np.pi)*quad(lambda R: ( R ** 2 - r ** 2)**(-0.5) * get_deriv_king(R, k, rc, rt), r , somebignumber)[0]
        j_ar = np.append(j_ar, j)
    return j_ar

def get_king_mass(r_ar, k, rc, rt):
    # somebignumber = 1000
    m_ar = np.zeros(0)
    for r in r_ar:
        m = 4 * np.pi * quad(lambda a: get_king_density2(a, k, rc, rt) * a ** 2, 0, r)[0]
        m_ar = np.append(m_ar, m)
    return m_ar

rt = 30
rpl = np.linspace(0.1, rt, 100)
S = get_surf_dens_in_mag_per_arcsec2_wiki(king(rpl, 24, 1, 30))
fig, [ax1, ax2, ax3] = plt.subplots(figsize = (15, 5), nrows = 1, ncols = 3)
ax1.plot(rpl, S, c = 'k')
ax1.set_xscale('log')
ax1.invert_yaxis()
ax1.set_title('Surface brightness')

# ax2.plot(rpl, get_king_density(rpl, 24, 1, 30), c = 'k', alpha = 0.5)
ax2.plot(rpl, get_king_density2(rpl, 24, 1, 30), ls = '-', c = 'k')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title('Mass density')

ax3.plot(rpl, get_king_mass(rpl, 24, 1, 30), c = 'k', alpha = 0.3)
ax3.plot(rpl, get_king_mass2(rpl, 24, 1, 30), c = 'k', ls = '--')
ax3.set_title('Mass profile')
ax3.set_xscale('log')
ax3.set_yscale('log')

plt.tight_layout()
# plt.title('KING profile')
# plt.show()
plt.savefig(filepath + 'test.pdf')

"""