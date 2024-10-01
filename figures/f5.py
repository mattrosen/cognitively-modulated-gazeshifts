################################################################################
#
# f5.py
#
# Author: Matt Rosen
# Modified: 6/24
#
# Code to generate Figure 5: SC’s population encoding of categories and gaze 
#   shifts become aligned before category-correlated eye movements.  
#
# TODO:
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

class f5(object):

    def __init__(self, troop, kwargs={}):
        self.troop = troop
        self.figpath = "./figures/EyeMovementAnalysis/f5/"
        if not os.path.exists(self.figpath):
            os.makedirs(self.figpath)

        self.group_colors = {'eye'    : 'slategray',
                             'neural' : 'royalblue'}
        self.area_colors = {'lip'  : 'darkgoldenrod',
                            'sc'   : 'royalblue'}


        # A. Analysis schematic (one part for shift-code direction,
        # another part for category-coding direction)

        # B. Timecourse of angle between SD and CD, Stanton
        # C. Timecourse of angle between SD and CD, Neville
        # D. Timecourse of theta with decoding of category from eye movements, Stanton
        # E. Timecourse of theta with decoding of category from eye movements, Neville
        b, c, d, e = self.perform_gaze_mode_analysis()
        b.fig.savefig(self.figpath + "b.pdf", bbox_inches='tight')
        c.fig.savefig(self.figpath + "c.pdf", bbox_inches='tight')
        d.fig.savefig(self.figpath + "d.pdf", bbox_inches='tight')
        e.fig.savefig(self.figpath + "e.pdf", bbox_inches='tight')

        if hasattr(b, 'test'):
            with open(self.figpath + "b.txt", 'w') as f_:
                f_.write(str(b.test))
        if hasattr(c, 'test'):
            with open(self.figpath + "c.txt", 'w') as f_:
                f_.write(str(c.test))
        if hasattr(d, 'test'):
            with open(self.figpath + "d.txt", 'w') as f_:
                f_.write(str(d.test))
        if hasattr(e, 'test'):
            with open(self.figpath + "e.txt", 'w') as f_:
                f_.write(str(e.test))


        plt.close('all')

        return

    def perform_gaze_mode_analysis(self, area='sc',
        animals=['stanton', 'neville'],
        alpha=0.05, 
        nboot=1000):

        timecourse_results  = []
        eyedec_comp_results = []


        for a in self.troop.animals:
            if a.name not in animals:
                continue

            # Results for main paper, using MGS-defined shift axes
            res = a.msd.gaze_mode_analysis_MGS(area, n_reps=nboot, alpha=alpha)
            timecourse_results.append(res[0])
            eyedec_comp_results.append(res[1])
            

        results = [*timecourse_results, *eyedec_comp_results]

        return results
