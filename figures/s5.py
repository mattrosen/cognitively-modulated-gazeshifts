################################################################################
#
# f5.py
#
# Author: Matt Rosen
# Modified: 6/24
#
# Code to generate Supplementary Figure 5: Alignment of category-coding 
#   directions to a shift-coding direction estimated from small gaze shifts in 
#   the initial fixation period of DMC
#
################################################################################

import numpy as np
import os, glob, pprint
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import scipy.io as sio
import pickle, types
import pymatreader
from pycircstat import tests
from scipy import signal, stats, ndimage
from .. import *
from scipy import ndimage
import more_itertools as mit

import matplotlib as mpl


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.use('Agg')
params = {'mathtext.default'  : 'regular',
          'figure.figsize'    : [2.0, 2.0],
          'axes.labelsize'    : 16,
          'legend.fontsize'   : 12,
          'xtick.labelsize'   : 12,
          'ytick.labelsize'   : 12,
          'font.family'       : 'Arial',
          'pdf.fonttype'      : 42}          
plt.rcParams.update(params)

np.set_printoptions(precision=3, suppress=True)
DEGREE_CHAR = u'\N{DEGREE SIGN}'

class s5(object):

    def __init__(self, troop, kwargs={}):
        self.troop = troop
        self.figpath = "./figures/EyeMovementAnalysis/s5/"
        if not os.path.exists(self.figpath):
            os.makedirs(self.figpath)

        self.group_colors = {'eye'    : 'slategray',
                             'neural' : 'royalblue'}
        self.area_colors = {'lip'  : 'darkgoldenrod',
                            'sc'   : 'royalblue'}



        # A. Timecourse of angle between SD and CD, Stanton
        # B. Timecourse of angle between SD and CD, Neville
        b, c = self.perform_gaze_mode_analysis()
        b.fig.savefig(self.figpath + "b.pdf", bbox_inches='tight')
        c.fig.savefig(self.figpath + "c.pdf", bbox_inches='tight')

        plt.close('all')

        return

    def perform_gaze_mode_analysis(self, area='sc',
        animals=['stanton', 'neville'],
        sd_period='DMC',
        alpha=0.05, 
        nboot=1000):

        results  = []

        for a in self.troop.animals:
            if a.name not in animals:
                continue

            # Results for main paper, using MGS-defined shift axes
            res = a.msd.gaze_mode_analysis_DMC(area, n_reps=nboot, alpha=alpha)
            results.append(res[0])

        return results

