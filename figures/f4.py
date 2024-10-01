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

class f4(object):

    def __init__(self, troop, kwargs={}):
        self.troop = troop
        self.figpath = "./figures/EyeMovementAnalysis/f4/"
        if not os.path.exists(self.figpath):
            os.makedirs(self.figpath)

        self.area_colors = {'lip': 'darkgoldenrod',
                            'sc' : 'royalblue'}

        self.group_colors = {'inactivation': 'black',
                             'control'     : 'grey'}

        animals = ['maceo2', 'quincy', 'stanton', 'neville']


        # A. Example animal, LIP inactivation sessions vs. control sessions, 
        # timecourse of separation in eye movements (inactivated VF trials)

        # B. Example animal, SC inactivation sessions vs. control sessions, 
        # timecourse of separation in eye movements according to category 
        # (treatment trials)
        a, b, c  = self.compute_inact_eyesep(
            tasks=['DMC'],
            animals=animals,
            resolution=20)
        a.fig.savefig(self.figpath + "a.pdf", bbox_inches='tight')
        b.fig.savefig(self.figpath + "b.pdf", bbox_inches='tight')
        c.fig.savefig(self.figpath + "c.pdf", bbox_inches='tight')

        d, _ = self.compute_inact_eyecat(
            tasks=['DMC'],
            animals=animals)
        d.fig.savefig(self.figpath + "d.pdf", bbox_inches='tight')
        with open(self.figpath + f"d.txt", 'w') as f_:
            f_.write(d.test)

        plt.close('all')

        return

    def compute_inact_eyecat(self,
        tasks=['DMC'],
        animals=None,
        example_animals=['quincy','stanton'],
        resolution=50,
        alpha=0.05):

        # Loop through animals, computing categorization from 
        # eye movements, treatment vs control; also w.r.t. choice,
        # by including error trials
        scores, scores_comp = defaultdict(list), defaultdict(list)
        for a in self.troop.animals:
            print(a.name)

            if animals is not None and a.name not in animals:
                continue

            results = a.compute_eyecat_inact(tasks=tasks,
                resolution=resolution)

            # Obtain changes 
            if a.name in ['stanton', 'neville']:
                scores['sc'].append([a.name, results[0]])
                scores_comp['sc'].append([a.name, results[1]])
            else:
                scores['lip'].append([a.name, results[0]])
                scores_comp['lip'].append([a.name, results[1]])

        results = [types.SimpleNamespace(), types.SimpleNamespace()]

        # Plot together on the same plot - 
        f_0, ax = plt.subplots(1, figsize=(1.5,1.5))
        sigs = []
        names = []
        quantification_string = []
        for area in scores.keys():
            for name, s in scores[area]:
                sig = stats.mannwhitneyu(s['control'].mean(-1), 
                    s['inactivation'].mean(-1), alternative='greater')
                sigs.append(sig.pvalue)
                ax.errorbar(s['control'].mean(), s['inactivation'].mean(),
                    xerr=s['control'].mean(-1).std(), 
                    yerr=s['inactivation'].mean(-1).std(),
                    linewidth=0, elinewidth=1, color=self.area_colors[area])
                ax.scatter(s['control'].mean(), s['inactivation'].mean(),
                    marker='.', color=self.area_colors[area])

                # Quantification: mean +/- sd (sessions)
                mean_c = s['control'].mean()
                mean_i = s['inactivation'].mean()
                sd_c = s['control'].mean(-1).std()
                sd_i = s['inactivation'].mean(-1).std()
                quantification_string.append(", ".join([
                    name, area, 
                    f"Control (mean +/- sd): {mean_c:.3f} (+/- {sd_c:.3f})",
                    f"Inact.  (mean +/- sd): {mean_i:.3f} (+/- {sd_i:.3f})"
                ]))

        quantification_string = "\n".join(quantification_string)

        sigs = stats.false_discovery_control(np.array(sigs))

        # Bind to names
        sigs_areas = np.column_stack((sigs, np.repeat(list(scores.keys()), 2)))


        sns.despine(f_0, offset=5)
        ax.set(xlabel='Decoding acc.\n(control)',
               ylabel='Decoding acc.\n(inactivation)')
        ax.text(0.15,1,'SC',color=self.area_colors['sc'],
            ha='left', va='top', transform=ax.transAxes,
            fontsize=10)
        ax.text(0.15,0.85,'LIP',color=self.area_colors['lip'],
            ha='left', va='top', transform=ax.transAxes,
            fontsize=10)
        
        ax.axline((0.5,0.5), (1,1), color='grey', linestyle='--', linewidth=1)
        ax.axvline(0.5, color='lightgrey', linestyle='--', linewidth=1)
        ax.axhline(0.5, color='lightgrey', linestyle='--', linewidth=1)
        ax.set_aspect('equal','box')
        plt.show()

        results[0].test = "\n".join(["Mann-Whitney U test p-values, control vs. inactivation, 1-sided, FDR-controlled:",
            str(sigs_areas), "\n", quantification_string])
        results[0].fig = f_0

        # Same but for all completed trials, using inferred category 
        # (consistent w/ animals' choice)
        f_1, ax = plt.subplots(1, figsize=(1.5,1.5))
        sigs = []
        names = []
        for area in scores_comp.keys():
            for name, s in scores_comp[area]:
                sig = stats.mannwhitneyu(s['control'].mean(-1), 
                    s['inactivation'].mean(-1), alternative='greater')
                sigs.append(sig.pvalue)
                ax.errorbar(s['control'].mean(), s['inactivation'].mean(),
                    xerr=s['control'].mean(-1).std(), 
                    yerr=s['inactivation'].mean(-1).std(),
                    linewidth=0, elinewidth=1, color=self.area_colors[area])
                ax.scatter(s['control'].mean(), s['inactivation'].mean(),
                    marker='.', color=self.area_colors[area])
        sigs = stats.false_discovery_control(np.array(sigs))

        # Bind to names
        sigs_areas = np.column_stack((sigs, np.repeat(list(scores_comp.keys()), 2)))

        sns.despine(f_1, offset=5)
        ax.set(xlabel='Decoding acc.\n(control)',
               ylabel='Decoding acc.\n(inactivation)')
        ax.text(0.15,1,'SC',color=self.area_colors['sc'],
            ha='left', va='top', transform=ax.transAxes,
            fontsize=10)
        ax.text(0.15,0.85,'LIP',color=self.area_colors['lip'],
            ha='left', va='top', transform=ax.transAxes,
            fontsize=10)
        
        ax.axline((0.5,0.5), (1,1), color='grey', linestyle='--', linewidth=1)
        ax.axvline(0.5, color='lightgrey', linestyle='--', linewidth=1)
        ax.axhline(0.5, color='lightgrey', linestyle='--', linewidth=1)
        ax.set_aspect('equal','box')
        plt.show()

        results[1].test = "\n".join(["Mann-Whitney U test p-values, control vs. inactivation, 1-sided, FDR-controlled:",
            str(sigs_areas)])
        results[1].fig = f_1


        return results

    def compute_inact_eyesep(self, 
        tasks=['DMC'],
        animals=None,
        example_animals=['quincy', 'neville'],
        resolution=50,
        alpha=0.05):


        # Loop through animals, computing separation b/w
        # eye movements in each condition (inact/control)
        ratio_by_c = {a: {} for a in animals}
        figs = []
        for a in self.troop.animals:

            if animals is not None and a.name not in animals:
                continue

            results = a.compute_eyesep_inact(tasks=tasks,
                resolution=resolution)

            median_dur, median_sample, median_delay = results[1:]

            # Plot average trace across sessions
            x = np.arange(-(median_dur - median_sample - median_delay), 
                (median_sample + median_delay + 100))[::resolution]

            f_0, ax = plt.subplots(1, figsize=(2,1.5))
            
            sigs = {}

            for i, (k,v) in enumerate(results[0].items()):
                v = np.array(v)#.squeeze()
                print(v.shape)
                v = v.transpose((1,0,2,3,4)) # becomes [group, session, B/W, boot, time]

                # Record significance of difference b/w BCD, WCD
                sig = stats.ttest_rel(v.mean(3)[1,:,0,:], v.mean(3)[1,:,1,:], 
                    axis=0, alternative='greater').pvalue
                sig = stats.false_discovery_control(sig)
                sigs[k] = sig

                # BCD - WCD
                v = v[:,:,0] - v[:,:,1]

                # Plot average separation in eye movements during 
                # inactivation, control
                mean_ = np.nanmean(v[1].mean(-2),0)
                sem_ = stats.sem(v[1].mean(-2),axis=0)

                if k == 'inactivation':
                    if a.name in ['quincy', 'maceo1']:
                        color = self.area_colors['lip']
                    else:
                        color = self.area_colors['sc']
                else:
                    color = self.group_colors[k]

                ax.plot(x, mean_, color=color)
                ax.fill_between(x, mean_ - sem_, mean_ + sem_,
                    alpha=0.1, color=color)
                
                # Also record the maximum separation during the delay
                delay_sep = v[...,-int((100+median_delay)//resolution):int(-100//resolution)]


                ratio_by_c[a.name][k] = delay_sep[1].max(-1).mean(-1)

            for k, grp in enumerate(results[0].keys()):
                if grp == 'inactivation':
                    if a.name in ['quincy', 'maceo1']:
                        color = self.area_colors['lip']
                    else:
                        color = self.area_colors['sc']
                else:
                    color = self.group_colors[grp]
                ax.text(0, 1 - 0.1*k, grp, color=color,
                    va='bottom', ha='left', transform=ax.transAxes,
                    fontsize=12)
                
                ax = util.annotate_signif(x, sig, ax, 
                    color=color)

            ax.set(xlabel='ms from sample on')
            ax.set(ylabel='BCD - WCD (dva)')
            ax.axvline(0, color='lightgrey', linestyle='--', 
                zorder=-90, linewidth=1)
            ax.axvline(median_sample, color='lightgrey',
                linestyle='--', zorder=-90, linewidth=1)
            ax.axvline(median_sample+median_delay, 
                color='lightgrey', linestyle='--', 
                zorder=-90, linewidth=1)
            
            sns.despine(f_0, offset=5)
            plt.show()

            if a.name in example_animals:
                figs.append(f_0)

            plt.close(f_0)


        # Compute change for inactivation relative to control trials
        rng = np.random.default_rng(10)
        f_1, ax = plt.subplots(1, figsize=(1.5,1.5))

        sigdict = {}

        for j, a in enumerate(animals):

            # Extract treatment, control
            treatment = ratio_by_c[a]['inactivation']
            control_keys = np.setdiff1d(list(ratio_by_c[a].keys()), ['inactivation'])
            control = np.concatenate([ratio_by_c[a][k] for k in control_keys])

            print(treatment.shape, control.shape)
            sig = stats.ranksums(control, treatment).pvalue
            print(a, sig)

            if a in ['stanton', 'neville']:
                color = self.area_colors['sc']
            else:
                color = self.area_colors['lip']

            # Generate error bars by bootstrapping - 
            boot_t = rng.choice(treatment.shape[0], [1000, treatment.shape[0]], replace=True)
            boot_c = rng.choice(control.shape[0], [1000, control.shape[0]], replace=True)

            tb = treatment[boot_t].mean(1)
            cb = control[boot_c].mean(1)
            boot_dif = (tb - cb)/cb
            serr = np.std(boot_dif)

            ax.bar(j+0.5*(j >= 2), 100*(treatment.mean() - control.mean())/(control.mean()), 
                yerr=serr*100,
                color=color, width=0.7,
                linewidth=0)

            annot = 'n.s.'
            va = 'center'
            if sig < 0.05:
                annot = '*'
                va = 'top'
            if sig < 0.01:
                annot = '**'
            if sig < 0.001:
                annot = '***'

            ax.text(j + 0.5 * (j >= 2), 30, annot, color='black', 
                fontsize=12, ha='center', va=va)

            # Retain significance for saving
            sigdict[a] = stats.ranksums(control, treatment)

        # Label areas
        ax.set(xlabel='Monkey', ylabel=r'$\Delta$ peak sep. (dva)',
            yticks=[0,-50,-100],
            xticks=[0,1,2.5,3.5], xticklabels=['M', 'Q', 'S', 'N'],
            ylim=[-100, ax.get_ylim()[1]])
        sns.despine(f_1, offset=5)
        ax.text(0, 0.1, 'LIP', color=self.area_colors['lip'],
            va='bottom', ha='left', fontsize=12, transform=ax.transAxes)
        ax.text(0, 0., 'SC', color=self.area_colors['sc'],
            va='bottom', ha='left', fontsize=12,
            transform=ax.transAxes)

        plt.show()

        # Return results
        results = [types.SimpleNamespace(), 
            types.SimpleNamespace(), types.SimpleNamespace()]
        results[0].fig = figs[0]
        results[1].fig = figs[1]
        results[2].fig = f_1
        results[2].test = sigdict

        return results


