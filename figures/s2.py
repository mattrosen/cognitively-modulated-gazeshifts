################################################################################
#
# f4.py
#
# Author: Matt Rosen
# Modified: 6/24
#
# Code to generate Figure 4: Reversible inactivation of SC but not LIP 
#   diminishes category-correlated eye movements.
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
from . import *

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn import metrics
from sklearn.metrics import silhouette_score, confusion_matrix, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeCV, LassoCV, Lasso, Ridge
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline



from scipy import ndimage
import more_itertools as mit

from statannotations.Annotator import Annotator
import matplotlib as mpl
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FormatStrFormatter, FixedFormatter
from matplotlib.colors import LinearSegmentedColormap


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

class s2(object):

    def __init__(self, troop, kwargs={}):
        self.troop = troop
        self.figpath = "./figures/EyeMovementAnalysis/s2/"
        if not os.path.exists(self.figpath):
            os.makedirs(self.figpath)

        self.area_colors = {'lip': 'darkgoldenrod',
                            'sc' : 'royalblue'}

        self.group_colors = {'inactivation': 'black',
                             'control'     : 'grey'}

        animals = ['maceo2', 'quincy', 'stanton', 'neville']

        # Supplementary: 
        # SA-SD. Distribution of eye movement vector magnitudes for each animal
        # SE. Summary of differences in eye movement distributions during
        # inactivation 
        supps = self.compute_inact_eyemvmt_effect(
            animals=animals,
            alpha=0.001)
        suppnames = ['a', 'b', 'c', 'd']
        for i, (k, sn) in enumerate(zip(animals, suppnames)):

            supps[i].fig.savefig(self.figpath + sn + ".pdf", bbox_inches='tight')
            with open(self.figpath + f"{sn}.txt", 'w') as f_:
                f_.write(f"{k}, difference in eye movement magnitude distributions,\n")
                f_.write("Epps-Singleton 2-sample test, p-values by session:")
                f_.write(str(supps[i].test) + "\n")
                f_.write(f"Summary: significant at P < 0.001 in " + \
                    f"{sum(supps[i].test < 0.001)} of {len(supps[i].test)} sessions")


        supps[-1].fig.savefig(self.figpath + "e.pdf", bbox_inches='tight')

        plt.close('all')

        return


    def compute_inact_eyemvmt_effect(self, animals=None, alpha=0.001):

        # Set up record arrays for returning results
        results = []

        f_, a_ = plt.subplots(ncols=2, figsize=(4,2), sharey=True, layout='constrained')

        for a in self.troop.animals:

            print(a.name)

            if animals is not None and a.name not in animals:
                continue

            result = types.SimpleNamespace()

            area = 'sc' if a.name in ['stanton', 'neville'] else 'lip'

            ctrl_mags, inact_mags = a.compute_inact_eyemvmt_effect(do_concat= area == 'lip')

            # Plot ctrl/inact mags relative to one another
            pvals_by_session = np.array([stats.epps_singleton_2samp(ctrl_mags[s], inact_mags[s]).pvalue for s in range(len(ctrl_mags))])
            pvals_by_session = stats.false_discovery_control(pvals_by_session)

            print(pvals_by_session)
            
            print(a.name, f"{sum(pvals_by_session < alpha)} of {len(pvals_by_session)} signif. (Epps-Singleton test, 2 sample, P < {alpha})")
            ctrl_mags = np.concatenate(ctrl_mags)
            inact_mags = np.concatenate(inact_mags)

            allmags = np.concatenate([ctrl_mags, inact_mags])
            magbins = np.geomspace(np.percentile(allmags, 2.5), np.percentile(allmags, 97.5), 11)
            centers = (magbins[1:] + magbins[:-1])/2
            ctrl_ = np.histogram(ctrl_mags, bins=magbins)[0]
            inact_ = np.histogram(inact_mags, bins=magbins)[0]
            fig, ax = plt.subplots(1)
            ax.plot(centers, ctrl_/ctrl_.sum(), color=self.group_colors['control'])
            ax.plot(centers, inact_/inact_.sum(), color=self.group_colors['inactivation'])
            sns.despine(fig, offset=5)
            ax.set(xlabel='Mag. (dva)', ylabel='Prob.', xscale='log')

            result.fig = fig
            result.test = pvals_by_session
            results.append(result)

            # Plot difference trace
            if area == 'lip':
                i = 0
            else:
                i = 1
            a_[i].plot(centers, (inact_/inact_.sum() - ctrl_/ctrl_.sum()), 
                color=self.area_colors[area], linewidth=1)

        sns.despine(f_, offset=5)
        y_ex = max([np.abs(a_[i].get_ylim()).max() for i in range(len(a_))])
        a_[0].set(ylabel=r'$P_{inact} - P_{ctrl}$', 
            ylim=[-y_ex, y_ex], xscale='log')
        a_[1].set(ylim=[-y_ex, y_ex], xscale='log')
        f_.supxlabel("Mag. (dva)", fontsize=16)
        a_[0].text(0, 0.0, 'LIP', color=self.area_colors['lip'],
            va='bottom', ha='left', fontsize=12, transform=a_[0].transAxes)
        a_[1].text(0, 0., 'SC', color=self.area_colors['sc'],
            va='bottom', ha='left', fontsize=12,
            transform=a_[1].transAxes)
        a_[0].axhline(0, color='lightgrey', zorder=-90, linestyle='--')
        a_[1].axhline(0, color='lightgrey', zorder=-90, linestyle='--')
        plt.show()

        result = types.SimpleNamespace()
        result.fig = f_
        results.append(result)

        return results