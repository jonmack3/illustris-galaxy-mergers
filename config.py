# -*- coding: utf-8 -*-
"""
Multiple mergers.py configuration variables.
Created on Mon Jul 3 01:20:03 2017
@author: Jonathan Mack

Notes
-----
All functions beyond setmergers are designed to be executed one per
each combination of ilnum and snapnum, with all other parameters combined.
"""

functions = {0: 'setmergers',
             1: 'setmergersmult',
             2: 'viewmergerdata',
             3: 'setmassmax',
             4: 'setfs',
             5: 'viewfvmdata',
             6: 'setdtmax',
             7: 'setdts',
             8: 'viewdtdata',
             9: 'setfmin',
             10: 'createfvmplots',
             11: 'createfvm_mlt_plot',
             12: 'createfvmratioplots',
             13: 'create_single_dtplots',
             14: 'create_all_dt2dplot',
             15: 'createpubplots',
             16: 'test'}

testcfg = 0
fnum = 16

if not testcfg:
    debug = 0
    subhalostart = 0
    subhalo_end = -1
    
    mbinsnumraw = 8
    RGms_num = 100
    fxs_num = 500
    setmergers_arylen = 1000000

    ilnums = [3]
    # ilnums_mlt = [3, 100]
    ilnums_mlt = [1, 3, 100, 300]
    snapnumsOG = [127]
    # snapnumsOGmlt = [68]
    snapnumsOGmlt = [49, 54, 60, 64, 68, 75, 85, 103, 120, 127]
    snapnumsTNG = []
    # snapnumsTNGmlt = [33]
    snapnumsTNGmlt = [17, 21, 25, 29, 33, 40, 50, 67, 84, 91]
    # mu_maxes = [4]
    mu_maxes = [2, 4, 10]
    # virtualprogs = [1]
    virtualprogs = [0, 1]
    # SubLink_gals = [1]
    SubLink_gals = [0, 1]
    # Trefs = ['analysis', 'merger']
    Trefs = ['analysis', 'merger', 'snapwidth']
    # Tfacs = [1]
    Tfacs = [0.5, 1, 2]
    
else:
    debug = 1
    subhalostart = 0
    subhalo_end = 10
    
    mbinsnumraw = 3
    RGms_num = 10
    fxs_num = 10
    setmergers_arylen = 200
    
    ilnums = [3]
    ilnums_mlt = [1, 3, 100, 300]
    snapnumsOG = [76]
    snapnumsOGmlt = [49, 54, 60, 64, 68, 75, 85, 103, 120, 127]
    snapnumsTNG = []
    snapnumsTNGmlt = [17, 21, 25, 29, 33, 40, 50, 67, 84, 91]
    mu_maxes = [4]
    # mu_maxes = [2, 4, 10]
    virtualprogs = [1]
    # virtualprogs = [0, 1]
    SubLink_gals = [1]
    # SubLink_gals = [0, 1]
    Trefs = ['merger']
    # Trefs = ['analysis', 'merger']
    # Trefs = ['analysis', 'merger', 'snapwidth']
    Tfacs = [1]
    # Tfacs = [0.5, 1, 2]

# setmergers only parameters
ilnum = 100
snapnum = 31
mu_max = 4
virtualprog = 1
SubLink_gal = 1

mminvirt = 0.01

mmin = 0.1
mmrglst3 = 0
mlogspace = 1
dtbinwdthopt = 0.2
bin_pdng = 0.01  # since value = max bin edge causes error when digitize
fminmin = 1e-5
KDEmult = 10

# f vs m multiple z/sim plot-related
fvm_mlt_ilnums = [1, 100]
fvm_mlt_snapsOG = [60, 127]
fvm_mlt_snapsTNG = [25, 91]

# plot-related
m_axis_maxmanual = 0
plot_toconsole = 0
plot_tofile = 1
ploterrorbars = 1

plotcml = 0
plot_fKDE = 0
mu_maxes_to_plot = [4]
# mu_maxes_to_plot = [2, 4, 10]
# mu_maxes_to_plot_mlt = [2]
mu_maxes_to_plot_mlt = [2, 4, 10]
virtualprogs_to_plot = [1]
# virtualprogs_to_plot = [0, 1]
# virtualprogs_to_plot_mlt = [1]
virtualprogs_to_plot_mlt = [0, 1]
SubLink_gals_to_plot = [1]
# SubLink_gals_to_plot = [0, 1]
# SubLink_gals_to_plot_mlt = [1]
SubLink_gals_to_plot_mlt = [0, 1]
# Trefs_to_plot = ['merger']
Trefs_to_plot = ['analysis', 'merger']
# Trefs_to_plot_mlt = ['merger']
Trefs_to_plot_mlt = ['analysis', 'merger']
Tfacs_to_plot = [1]
# Tfacs_to_plot_mlt = [2]
Tfacs_to_plot_mlt = [0.5, 1, 2]
Tfacs_to_plot_dt = [0.5, 1, 2]
ratio_numbins = 50
ratio_plt_avgs = 1
ratio_axes_log = 1
ratio_log_min = 3e-3
ratio_log_avg_min = 0.01
ratio_log_avg_len_min = 25
dt_mbins_to_plot = [-1] # include -1 in list to plot all mass bins
dt_mbins_to_plot_mlt = [-1] # include -1 in list to plot all mass bins