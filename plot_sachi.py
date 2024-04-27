import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sachi_path = '/bigdata/saleslab/psadh003/sachi/'



def get_L_from_Mv(Mv):
    '''
    this function is to obtain the Sachi luminosity from the magnitude, conversion from their paper
    '''
    x = Mv
    y = -0.4 * x + 0.016 #this is the line equation
    L = 10**y
    return L

df_nb = np.loadtxt(sachi_path + 'n_body.csv', delimiter = ',')
Mv_nb = df_nb[:, 0]
mstar_nb = get_L_from_Mv(Mv_nb)
N_nb = df_nb[:, 1]

df_obs = np.loadtxt(sachi_path + 'cenA_obs.csv', delimiter = ',')
Mv_obs = df_obs[:, 0]
mstar_obs = get_L_from_Mv(Mv_obs)
N_obs = df_obs[:, 1]

df_eps_5e12 = np.loadtxt(sachi_path + 'eps_5e12.csv', delimiter = ',')
Mv_eps_5e12 = df_eps_5e12[:, 0]
mstar_eps_5e12 = get_L_from_Mv(Mv_eps_5e12)
N_eps_5e12 = df_eps_5e12[:, 1]

df_eps_1e13 = np.loadtxt(sachi_path + 'eps_1e13.csv', delimiter = ',')
Mv_eps_1e13 = df_eps_1e13[:, 0]
mstar_eps_1e13 = get_L_from_Mv(Mv_eps_1e13)
N_eps_1e13 = df_eps_1e13[:, 1]

def plot_sachi(ax, alpha = 0.5, ls= '--'):
    ax.plot(mstar_nb, N_nb, color = 'green', label = r'N-body ($M_{\rm{host}} = 1.67 \times 10^{13}\,M_{\odot}$)', alpha = alpha, ls= '--')
    ax.plot(mstar_obs, N_obs, color = 'orange', label = r'Observations Cen-A', alpha = alpha, ls= '--')
    ax.plot(mstar_eps_5e12, N_eps_5e12, color = 'aqua', label = r'EPS ($M_{\rm{host}} = 5 \times 10^{12}\,M_{\odot}$)', alpha = alpha, ls= '--')
    ax.plot(mstar_eps_1e13, N_eps_1e13, color = 'dodgerblue', label = r'EPS ($M_{\rm{host}} = 1 \times 10^{13}\,M_{\odot}$)', alpha = alpha, ls= '--')


# fig, ax = plt.subplots(figsize = (5, 5))
# plot_sachi(ax)
# ax.legend(fontsize = 8)
# plt.tight_layout()
# plt.loglog()
# plt.show()