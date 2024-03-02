import numpy as np 



def get_mstar_co(lvmax, alpha = 2.93, mu = -1.39, M0 = 2.43e8):
    eta = 10**lvmax / 50
    mstar = eta**alpha *np.exp(-eta**mu) * M0
    return np.log10(mstar)



def get_mstar_pl(lvmaxar, m1 = 3.01, m2 =  4.6, b = 3.14):
    '''
    This is the power law model from Santos-Santos 2022
    '''
    if isinstance(lvmaxar, float):
        if lvmaxar >= np.log10(87):
            lmstar = m1 * lvmaxar + b
        elif lvmaxar < np.log10(87):
            lmstar = m2 * lvmaxar + (m1 - m2)*np.log10(87) + b
        lmstarar = lmstar 
    else:
        lmstarar = np.zeros(0)
        for lvmax in lvmaxar:
            if lvmax >= np.log10(87):
                lmstar = m1 * lvmax + b
            elif lvmax < np.log10(87):
                lmstar = m2 * lvmax + (m1 - m2)*np.log10(87) + b
            lmstarar = np.append(lmstarar, lmstar)
    return lmstarar


def get_scatter(lvmaxar, sigma0 = 0.24, kappa = -1.26, V0 = 88.6):
    '''
    This function returns the scatter for both power law and the cutoff models
    '''
    vmaxar = 10**lvmaxar 
    if isinstance(lvmaxar, float):
        if vmaxar > 57:
            sigma = sigma0
        elif vmaxar <= 57:
            sigma = kappa * np.log10(vmax/V0)
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
    mu_mstar = get_mstar_pl(lvmaxar, m1 = 2.9, m2 =  5.622, b = 3.41) #this will be the mean for the gaussian distribution
    sig_mstar = get_scatter(lvmaxar) #this will be the scatter in the relation which is considered to be a gaussian
    print(len(mu_mstar), len(sig_mstar), len(lvmaxar))
    mstar = np.random.normal(mu_mstar, sig_mstar, size = len(lvmaxar))
    return 10**mstar


def get_mstar_co_wsc(lvmaxar):
    '''
    This gives the stellar mass of a subhalos accounting for the scatter in the relation (as provided by Santos-Santos et al. 2022)
    '''
    mu_mstar = get_mstar_co(lvmaxar) #this will be the mean for the gaussian distribution
    sig_mstar = get_scatter(lvmaxar) #this will be the scatter in the relation which is considered to be a gaussian
    mstar = np.random.normal(mu_mstar, sig_mstar, size = len(lvmaxar))
    return 10**mstar




def get_lrh(lmstar_ar, m1 = 0.178, m2 = 0.31, b = -1.49):
    '''
    This function returns the log rh for a given Mstar
    0.17832722702850887, 0.30555128418263083, -1.4929324569613338
    '''
    lrh_ar = np.zeros(0)
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
    lrh =  np.random.normal(mu_lrh, sig_lrh, size = len(lmstar_ar))
    return 10**lrh
