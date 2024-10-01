################################################################################
#
# f1.py
#
# Author: Matt Rosen
# Modified: 6/24
#
# Code to generate Figure 1: Abstract categories are most strongly encoded in 
#   oculomotor/spatial orienting areas
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
from scipy import signal, stats, ndimage
import Animal, Microsaccade, Troop, util
from scipy import ndimage
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


class f1(object):

    def __init__(self, troop, kwargs={}):
        self.troop = troop
        self.figpath = "./figures/EyeMovementAnalysis/f1/"
        if not os.path.exists(self.figpath):
            os.makedirs(self.figpath)

        # Color key for areas:
        self.area_colors = {'lip': 'darkgoldenrod',
                            'sc' : 'royalblue',
                            'area5': 'tomato',
                            'mt'   : 'firebrick',
                            'mst'  : 'seagreen',
                            'pfc'  : 'peru'}

        # A. Task schematic
        # B. Brain diagram w/ areas labeled/colored

        # C. Summary CTI, MT
        c = self.compute_cti(area='mt', animals=['stimpy','hobbes'])
        c.fig.savefig(self.figpath + "c.pdf", bbox_inches='tight')
        with open(self.figpath + "c.txt", 'w') as f_:
            f_.write("MT CTI:\n")
            f_.write(str(c.test))

        # D. Summary CTI, MST
        d = self.compute_cti(area='mst', animals=['quincy', 'maceo2'])
        d.fig.savefig(self.figpath + "d.pdf", bbox_inches='tight')
        with open(self.figpath + "d.txt", 'w') as f_:
            f_.write("MST CTI:\n")
            f_.write(str(d.test))
        
        # E. Summary CTI, LIP
        e = self.compute_cti(area='lip')
        e.fig.savefig(self.figpath + "e.pdf", bbox_inches='tight')
        with open(self.figpath + "e.txt", 'w') as f_:
            f_.write("LIP CTI:\n")
            f_.write(str(e.test))

        # F. Summary CTI, PFC
        f = self.compute_cti(area='pfc', animals=['bootsy', 'jb'])
        f.fig.savefig(self.figpath + "f.pdf", bbox_inches='tight')
        with open(self.figpath + "f.txt", 'w') as f_:
            f_.write("PFC CTI:\n")
            f_.write(str(f.test))

        # G. Summary CTI, SC
        g = self.compute_cti(area='sc', animals=['stanton', 'neville'])
        g.fig.savefig(self.figpath + "g.pdf", bbox_inches='tight')
        with open(self.figpath + "g.txt", 'w') as f_:
            f_.write("SC CTI:\n")
            f_.write(str(g.test))

        plt.close('all')

        return

    

    def compute_cti(self, area, dirmod_only=False,
        alpha=0.01, animals=None):

        ctis, dirmods = {}, {}
        for a in self.troop.animals:
            
            if animals is not None and a.name not in animals:
                continue

            print(a.name)

            results = a.compute_cti(area=area)
            if results is None:
                continue
            ctis[a.name]    = results[0]
            dirmods[a.name] = results[1]

        # Concatenate together + select visually modulated units
        ctis    = np.concatenate(list(ctis.values()))
        dirmods = np.concatenate(list(dirmods.values()))

        if dirmod_only:
            dirmod_sig = np.where(dirmods <= alpha)[0]
            ctis = ctis[dirmod_sig]
        else:
            enough_trials_spk = np.where(dirmods < 1)[0]
            print(len(enough_trials_spk), dirmods.shape, ctis.shape)
            ctis = ctis[enough_trials_spk]

        # Plot CTI distributions -- for each unit,
        # max of (CTI sample, CTI delay)
        labels = ['sample', 'delay', 'all']
        for i in [0,1]:
            print(labels[i])
            f_0, ax = plt.subplots(1, figsize=(2,2))
            perunit_delay = ctis[:,i]
            bins = np.linspace(-1, 1, 21)
            sns.histplot(perunit_delay, bins=bins, element='step', 
                color=self.area_colors[area], linewidth=0, rasterized=True)
            ax.set(xlabel='CTI', ylabel='# neurons')
            ax.axvline(0, linewidth=1, color='black', 
                zorder=90, linestyle='--')
            ax.text(0.05,1,area.upper(),ha='left',va='top',fontsize=12,
                color=self.area_colors[area], transform=ax.transAxes)
            ax.text(1,1,f"N = {len(perunit_delay)} neur.",
                ha='right', va='bottom', color='black', fontsize=12,
                transform=ax.transAxes)
            y = ax.get_ylim()[1]
            ax.scatter(np.mean(perunit_delay), y, 
                marker='v', 
                color=self.area_colors[area])
            ax.text(np.mean(perunit_delay)+0.1, y, 
                f"{np.mean(perunit_delay):.2f}", ha='left',
                va='center', color=self.area_colors[area],
                fontsize=12)
            sns.despine(f_0)
            plt.show()

            # Compute statistical test of difference from 0 mean (t-test)
            sig = stats.ttest_1samp(perunit_delay, 0, 
                alternative='greater')

        # Bind results together and return
        results = types.SimpleNamespace()
        results.fig = f_0
        results.test = sig

        return results