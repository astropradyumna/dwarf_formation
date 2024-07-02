import numpy as np
import illustris_python as il
import math
import pdb
import matplotlib.pyplot as plt
# import get_intersect_1d
import matplotlib.cm as cm
import h5py
import matplotlib.colors as colors
from scipy import stats
from matplotlib.image import NonUniformImage
from mpl_toolkits.axes_grid1 import make_axes_locatable

# for dealing with the periodicity:
################
### Function ###
################
def fold_pos(x,Lbox):
    aux = x > Lbox / 2.
    x[aux] = x[aux] - Lbox
    return x
##################
def fold_pos2(x,Lbox):
    aux = x < - Lbox / 2.
    x[aux] = x[aux] + Lbox
    return x
##################

simbase = '/rhome/psadh003/bigdata/L35n2160TNG_fixed/output'

### load additional files

filein1 = '/bigdata/saleslab/psadh003/tng50/tng_files/time_table_TNG50.dat'
tt = np.loadtxt(filein1)
a_scale = tt[:,0] # scale factor at every snapshot
z_red = tt[:,1] # redshift at every snapshot
t_cosmic = tt[:,2] # time in Gyr at every snapshot


lbox = 35000
snpz0 = 99
h_small = 0.677

print('starting with catalog ...')

cat_halo = il.groupcat.loadHalos(simbase, snpz0, fields = ['Group_R_Crit200', 'Group_M_Crit200', 'GroupFirstSub'])

idf = cat_halo['GroupFirstSub']
mvir = cat_halo['Group_M_Crit200'] * 1.e10/h_small
rvir = cat_halo['Group_R_Crit200'] * 1/h_small

aux = mvir >= 5e13
mvir_group = mvir[aux]
idf_group = idf[aux] #This is the ID of the central subhalo (most massive subhalo)
rvir_group = rvir[aux]

cat_sub = il.groupcat.loadSubhalos(simbase, snpz0, fields = ['SubhaloGrNr','SubhaloMassType', 'SubhaloMassInRadType', 'SubhaloMassInHalfRadType','SubhaloHalfmassRadType', 'SubhaloPos'])

pos_sub = cat_sub['SubhaloPos']
r_h_sub = cat_sub['SubhaloHalfmassRadType'][:,4] * 1/h_small
grnr = cat_sub['SubhaloGrNr']
mstr_sub = cat_sub['SubhaloMassType'][:,4] * 1e10/h_small

sfid = np.arange(len(grnr))


i = 0 #What is this?
grnr_group = i
print(i)
sfid_cen = idf_group[i] #Why are we using the ID of FoF1 here? The data is for FoF0
print(sfid_cen)
r_h_cen = r_h_sub[sfid_cen]
print('r_h_cen: ', r_h_cen)
r_vir_cen = rvir_group[i]
print('r_vir_cen: ', r_vir_cen)


###############
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
aux = (np.sqrt((x_pos_satellite**2)+(y_pos_satellite**2)) < 300)

x_pos_satellite = x_pos_satellite[aux]
y_pos_satellite = y_pos_satellite[aux]
z_pos_satellite = z_pos_satellite[aux]
mass_satellite = mass_satellite[aux]


#####################
##Binning
#####################

print(len(x_pos_satellite))
print('max min mstar pp:', mass_satellite.max(), mass_satellite.min())
mass_bin_sat, xedges_sat, yedges_sat, binnumber = stats.binned_statistic_2d(x_pos_satellite, y_pos_satellite, mass_satellite, statistic='sum', bins=500) #We are adding the mass of all the satellites in a given bin


print('shape of mass_bin_sat in pp: ', mass_bin_sat.shape)
# XX_sat, YY_sat = np.meshgrid(xedges_sat, yedges_sat) #This maybe just generates the cordinates for the plot

max_mass = mass_bin_sat.max()/10
min_mass = 50000 * 8/5

print('min and max of mass bins: ', min_mass, max_mass)

########################################
##Plotting
########################################
# plt.rcParams['axes.facecolor'] = 'black'
# fig, ax = plt.subplots(1,3, figsize = (18,6), sharex = True, sharey = True) #This line has been moved to final_plots.py
# plt.subplots_adjust(wspace = 0.02)


def plot_tng_subhalos(fig, ax):
        for i,a in enumerate(ax.flat): #We are just formatting the ticks here
                a.tick_params(length = 8, width = 2, direction = 'inout')
                a.xaxis.tick_bottom()
                a.yaxis.tick_left()


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

        


