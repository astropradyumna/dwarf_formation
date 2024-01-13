import numpy as np
import matplotlib.pyplot as plt 
import os 
os.environ["USE_LZMA"] = "0"
import pandas as pd
from errani_plus_tng_subhalo import Subhalo
from tqdm import tqdm
from scipy.optimize import fsolve
import IPython
from testing_errani import get_rot_curve, get_rmxbyrmx0, get_vmxbyvmx0, get_mxbymx0, get_LbyL0, l10rbyrmx0_1by4_spl,l10rbyrmx0_1by2_spl, l10rbyrmx0_1by8_spl, l10rbyrmx0_1by16_spl


class ErraniSubhalo():
    def __init__(self, rmx0, vmx0, mmx0,  rperi, rapo, torb):
        self.rmx0 = rmx0 
        self.vmx0 = vmx0 
        self.mmx0 = mmx0
        self.rperi = rperi 
        self.rapo = rapo 
        self.torb = torb

    def evolve(self, tevol, V0):
        '''
        This is a function that evolves the subhalo using Errani models

        Args:
        t (float): The time for which evolution must take place
        V0: This is a parameter of the host whn paramterized using the isothermal profile
        '''
        rmx0 = self.rmx0
        vmx0 = self.vmx0
        rperi = self.rperi
        rapo = self.rapo

        if any([rmx0, vmx0, rperi, rapo]) == None:
            raise ValueError('Some of the required values are None, recheck if they have been updated in the Object')

        tmx0 = 2 * np.pi *  ( rmx0 / vmx0 ) * 3.086e16 * 3.17098e-8 * 1e-9 #This would be in Gyrs assuming r to be in kpc and v to be in km/s
        tperi = 2 * np.pi * ( rperi / V0) * 3.086e16 * 3.17098e-8 * 1e-9 #This is the tperi that is calculated in Errani+21
        
        x = rapo / rperi
        fecc = (2 * x / (x + 1)) ** 3.2

        torb = self.torb* fecc #Gyr, this is after accounting for the ellipticity of the orbit
        
        def get_tmx_t(t):
            '''
            This is the Tmx value at a given time t. 
            We can calculate the real time t given a Tmx (for a given value of Mmx)
            '''
            
            if tmx0/tperi >= 2/3: #Heavy mass loss regime
                tasy = 0.22 * tperi
                y0 = (tmx0 - tasy) / tperi
                tau_asy = 0.65 * torb
                tau = tau_asy / y0 
                eta = 1 - np.exp( - 2.5 * y0 )
                inner_term = 1 + (t/tau)**eta 
                tmx = tasy + tperi * y0 * (inner_term)**(-1/eta)
            else: #modest mass loss regime
                tasyp =  ( tmx0 / (1 + (tmx0/tperi))**2.2)
                etap = 0.67
                # yp = (tmx - tasyp)/tperi 
                y0p = (tmx0 - tasyp)/tperi 
                taup = torb * 1.2 * (tmx0 / tperi)**(-0.5)
                inner_term = 1 + (t/taup)**etap
                tmx = tasyp + tperi * y0p * (inner_term)**(-1/etap)
            return tmx
        
        def get_tmx(rmx):
            ''' 
            This is to calculate the Tmx value from rmx and vmx
            '''
            vmx = vmx0 * get_vmxbyvmx0(rmx/rmx0)
            return 2 * np.pi * rmx / vmx * 3.086e16 * 3.17098e-8 * 1e-9 
        
        
        def get_time(rmx):
            '''
            I want to obtain the time for a given value of rmx
            '''
            t = fsolve(lambda t: get_tmx_t(t) - get_tmx(rmx), 2)[0]
            return t
        
        tmx = get_tmx_t(tevol)
        rmx = fsolve(lambda rmx: get_tmx(rmx) - tmx, rmx0/100)[0]
        print(rmx/rmx0)
        frem = get_mxbymx0(rmx/rmx0) 
        # print(f'frem = {100 * frem:.2f} % ')
        
        return frem
    
IPython.embed()

subh = ErraniSubhalo(mmx0 = 1e6, rmx0 = 0.47, rapo = 200, rperi = 40, vmx0 = 3, torb = 2.5)
frem = subh.evolve(tevol = 5 * subh.torb, V0 = 220)
print(f'This is the resultant Rh = {10 ** (l10rbyrmx0_1by2_spl(np.log10(frem)))}')
print(get_LbyL0())