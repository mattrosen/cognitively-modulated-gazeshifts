################################################################################
#
# s4.py
#
# Author: Matt Rosen
# Modified: 6/24
#
# Code to generate Supplementary Figure 4: A double category boundary 
#   discourages similarity-based strategies during motion categorization tasks 
#   by increasing (decreasing) the average angular distance between pairs of 
#   directions in the same (opposite) category.  
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

class s4(object):

    def __init__(self, troop, kwargs={}):
        self.troop = troop
        self.figpath = "./figures/EyeMovementAnalysis/s4/"
        if not os.path.exists(self.figpath):
            os.makedirs(self.figpath)

        # A. Boundary schematic
        # B. Distance from boundary
        b = self.plot_average_between_within_dist()
        b.fig.savefig(self.figpath + "b.pdf", bbox_inches='tight')

        plt.close('all')

        return

    def plot_average_between_within_dist(self,
        animals=['bootsy','bootsy1', 'denson', 'jb1', 'maceo1', 'stanton', 'stimpy', 'hobbes2']):

        # Make figure quantifying ratio of distance among between vs. within-category pairs,
        # highlighting the dual-boundary ratio
        markers = ['+','x','o','*','v','^','<','>']
        colors = ['black','black', 'black', 'black', 'black', 'red', 'black', 'black']

        fig, ax = plt.subplots(1)
        for ki, k in enumerate(animals):

            cat = self.troop.cat_labels[k]
            dirs = self.troop.dir_labels[k]

            # In general - average distance from boundary
            print(k, dbounds[k].mean())

            within, between = [], []

            for i in range(len(dirs)):
                for j in range(i, len(dirs)):

                    if cat[i] == -1 or cat[j] == -1:
                        continue
                    dif_ij = np.abs(dirs[i]+360 - dirs[j])% 360
                    dif_ij = np.minimum(dif_ij, 360 - dif_ij)

                    if min([cat[i], cat[j]]) >= 0:
                        if cat[i] == cat[j]:
                            within.append(dif_ij)
                        else:
                            between.append(dif_ij)


            within = np.array(within)
            between = np.array(between)

            ax.scatter(within.mean(), between.mean(), marker=markers[ki], color=colors[ki], clip_on=False)

        ax.set(xlabel='Mean dif.,\nsame category', ylabel='Mean dif.,\nopp. category')
        t = ax.text(0, 0, 'Peysakhovich et al. 2024\ndouble boundary', fontsize=12, color='red', ha='left',
                va='bottom', transform=ax.transAxes)
        t.set_bbox(dict(facecolor='white', alpha=0.75, linewidth=0))
        extreme_lo = min([ax.get_xlim()[0], ax.get_ylim()[0]])
        extreme_hi = max([ax.get_ylim()[1], ax.get_ylim()[1]])
        plt.axis('square')
        ax.set(xlim=[extreme_lo, extreme_hi], ylim=[extreme_lo, extreme_hi])
        ax.axline((extreme_lo, extreme_lo), (extreme_hi, extreme_hi), color='grey', linestyle='--', 
                  linewidth=1, zorder=-10)
        sns.despine(fig, offset=5)
        plt.show()

        return fig