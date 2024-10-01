################################################################################
#
# s3.py
#
# Author: Matt Rosen
# Modified: 6/24
#
# Code to generate Supplementary Figure S3: Neural activity in SC predicts 
#   upcoming miniature eye movements.
#
################################################################################

import numpy as np
import os, glob, pprint
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import scipy.io as sio
import pickle
import shutil
import pymatreader, types
from pycircstat import tests
from scipy import signal, stats, ndimage
from . import *
from .. import util


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

class f3(object):

    def __init__(self, troop, kwargs={}):
        self.troop = troop
        self.figpath = "./figures/EyeMovementAnalysis/f3/"
        if not os.path.exists(self.figpath):
            os.makedirs(self.figpath)

        self.area_colors = {'lip'  : 'darkgoldenrod',
                            'sc'   : 'royalblue'}

        
        ####
        # TEXT: mean displacement, across all delay-period gaze shifts,
        # +/- SEM across animals (with N)
        text = self.get_misc_text()
        with open(self.figpath + "misc_text.txt", 'w') as f_:
            f_.write(text)

        # A. Example gaze-shift extraction from eye traces, 1 trial
        a = self.copy_figure(["example_ms0_hobbes2_fixation.pdf"],
            [self.figpath + "a.pdf"])

        # B. Example raster aligned to gaze shifts, 1 SC neuron
        # C. Example raster aligned to gaze shifts, 1 LIP neuron
        b, c = self.copy_figure(
            ['denson/lip/raster/neuron_0.pdf', 'neville/sc/raster/neuron_201.pdf'],
            [self.figpath + "b.pdf", self.figpath + "c.pdf"])
        
        # D. Heatmap of responses for modulated units, SC
        # E. Heatmap of responses for modulated units, LIP
        # G. % shift-modulated units that show significant modulation 
        #   before shift onset (LIP vs. SC, bootstrap CI on proportion)
        # H. Quantification of timing of peak response by area for units
        #   modulated significantly before gaze shift
        d, e, g, h = self.compute_shift_modulations()
        d.fig.savefig(self.figpath + "d.pdf", bbox_inches='tight')
        e.fig.savefig(self.figpath + "e.pdf", bbox_inches='tight')
        g.fig.savefig(self.figpath + "g.pdf", bbox_inches='tight')
        h.fig.savefig(self.figpath + "h.pdf", bbox_inches='tight')
        if hasattr(d, 'test'):
            with open(self.figpath + "d.txt", 'w') as f_:
                f_.write(str(d.test))
        if hasattr(e, 'test'):
            with open(self.figpath + "e.txt", 'w') as f_:
                f_.write(str(e.test))
        if hasattr(g, 'test'):
            with open(self.figpath + "g.txt", 'w') as f_:
                f_.write(str(g.test))
        if hasattr(h, 'test'):
            with open(self.figpath + "h.txt", 'w') as f_:
                f_.write(str(h.test))

        # F. Decoding of gaze shift direction from LIP/SC
        f = self.compute_shift_axes(animals=['stanton','neville'])
        f.fig.savefig(self.figpath + "f.pdf", bbox_inches='tight')
        if hasattr(f, 'test'):
            with open(self.figpath + "f.txt", 'w') as f_:
                f_.write(str(h.test))


        plt.close('all')

        return

    def get_misc_text(self):

        mean_amps = np.array([a.msd.get_mean_amp() for a in self.troop.animals])
        text = f"Mean gaze shift amplitude: {np.nanmean(mean_amps):.3f} +/- {stats.sem(mean_amps, nan_policy='omit'):.3f} dva (mean +/- SEM, N = {np.isfinite(mean_amps).sum()})"
        return text

    def copy_figure(self, fns, new_fns, base="/home/mattrosen/RCDST-analysis"):
        """
        Copy figure generated during gaze shift extraction.
        """
        for ofn, nfn in zip(fns, new_fns):
            shutil.copy(os.path.join(base, "ms/figures/", ofn),
                nfn)

        return new_fns


    def compute_shift_modulations(self, 
        areas=['lip', 'sc'],
        alpha=0.05, 
        n_boot=10000):

        # Set up record arrays
        pvs    = defaultdict(list)
        fr     = defaultdict(list)

        results = [types.SimpleNamespace(), types.SimpleNamespace(),
                   types.SimpleNamespace(), types.SimpleNamespace()]

        # Loop through animals, load or generate response modulations
        # locked to gaze shifts
        for a in self.troop.animals:
            print(a.name)

            res = a.compute_shift_modulated(areas, epoch='fixation')
            pvs_by_area, fr_by_area, _, _ = res
            for k,v in pvs_by_area.items():
                pvs[k].append(v)
                fr[k].append(fr_by_area[k])

        # Concatenate together units across datasets
        pvs    = {k: np.concatenate(v, axis=0) for k,v in pvs.items()}
        fr     = {k: np.concatenate(v, axis=0) for k,v in fr.items()}

        # Set up indicators for statistical test on difference in prop. modulated
        indicators = {}

        # Store bootstrap estimates of proportion modulated and 
        # latency to peak modulation for plot including both areas
        premod_props, mod_peaklatencies = [], []

        rng = np.random.default_rng(10)

        # For each area, select modulated units; among these, compute
        # proportion w/ significant modulation before shift vs. after
        for i, (k, v) in enumerate(pvs.items()):

            mins = v.min(1)

            modulated = np.where(stats.false_discovery_control(mins) < alpha)[0]
            pre_mod = np.where(stats.false_discovery_control(v[:,0]) < alpha)[0]
            post_mod = np.where(stats.false_discovery_control(v[:,1]) < alpha)[0]

            pre_frac = len(np.intersect1d(modulated, pre_mod))/len(modulated)
            post_frac = len(np.intersect1d(modulated, post_mod))/len(modulated)

            # Store indicator of modulation
            indicators[k] = np.int32(v[modulated,0] < alpha)
            
            # Heatmap of these normed firing rates
            f_0, ax = plt.subplots(1, figsize=(3,3), layout='constrained')
            fr_mod =  fr[k][modulated]-fr[k][modulated].mean(1,keepdims=True)
            fr_premod = fr[k][pre_mod]-fr[k][pre_mod].mean(1,keepdims=True)
            order = np.argsort(np.abs(fr_mod).argmax(1))
            fr_mod = fr_mod[order]
            sns.heatmap(fr_mod,
                center=0, linewidth=0, 
                cbar_kws={'label': r"$\Delta$ FR (Hz)"},
                vmin=-30, vmax=30, rasterized=True)
            ax.set_yticks([0, fr_mod.shape[0]], labels=[0, fr_mod.shape[0]], rotation=0)
            ax.axvline(fr_mod.shape[1]//2, color='black', linestyle='--', linewidth=1)
            ax.set(xlabel='ms peri-shift',
                   ylabel='Unit #',
                   xticks=np.linspace(0, fr_mod.shape[1], 3),
                   xticklabels=[-fr_mod.shape[1]//2, 0, fr_mod.shape[1]//2])
            plt.show()

            results[i].fig = f_0
            results[i].test = "\n".join([f'{k}\n{len(pre_mod)}/{len(modulated)} ({100*pre_frac:.2f}%) pre,',
                                f'{len(post_mod)}/{len(modulated)} ({100*post_frac:.2f}%) post',
                                f'{len(modulated)}/{v.shape[0]} ({100*len(modulated)/v.shape[0]}%) modulated overall'])

            # Bootstrap estimate of variation in mean proportion modulated
            boot = rng.choice(v.shape[0], [n_boot, v.shape[0]], replace=True)
            boot_ps = v[boot] # n_boot x n_neur x 2
            boot_mod = np.array([len(np.where(stats.false_discovery_control(bpi.min(1)) < alpha)[0]) for bpi in boot_ps])
            boot_premod = np.array([len(np.where(stats.false_discovery_control(bpi[:,0]) < alpha)[0]) for bpi in boot_ps])
            mod = boot_premod / boot_mod
            mod[~np.isfinite(mod)] = 0
            premod_props.append(mod)
 
            # Compute time to peak modulation
            peakmod_t = np.argmax(np.abs(fr_mod), axis=1)

            mod_peaklatencies.append(peakmod_t-fr_mod.shape[1]//2)


        # Statistical test on difference in proportion modulated pre-shift
        test = self._twoproportion_z_test(indicators['sc'], indicators['lip'])

        # Plot proportion of modulated units that show modulations *before* gaze shift
        bar_x = [-0.25, 0.25]
        fig, ax = plt.subplots(1, figsize=(0.65, 2))
        ax.set(ylim=[0,80])
        for i in range(len(premod_props)):
            ax.bar(bar_x[i], premod_props[i].mean()*100, 
                yerr=premod_props[i].std()*100,
                color=self.area_colors[areas[i]],
                width=0.5)
            ax.text(0, 0.67-0.1*i, areas[i].upper(),
                fontsize=12, ha='left',
                va='top', color=self.area_colors[areas[i]],
                transform=ax.transAxes)

        
        sns.despine(fig, offset=5, bottom=True)
        ax.get_xaxis().set_visible(False)
        ax.set(ylabel='% mod. pre-shift')

        # Annotate w/ significance
        annot = ''
        if test.pvalue < 0.05:
            annot = '*'
        if test.pvalue < 0.01:
            annot = '**'
        if test.pvalue < 0.001:
            annot = '***'
        ax.text(0, 75, annot, fontsize=14, color='black',
            ha='center', va='top')
        plt.show()
        results[2].fig = fig
        results[2].test = test


        # Plot summary of latency distribution for each group
        fig, ax = plt.subplots(1, figsize=(0.65, 2))
        for i in range(len(premod_props)):
            ax.bar(bar_x[i], mod_peaklatencies[i].mean(), 
                yerr=stats.sem(mod_peaklatencies[i]),
                color=self.area_colors[areas[i]],
                width=0.5)
            ax.text(1, 0.9-0.1*i, areas[i].upper(),
                fontsize=12, ha='right',
                va='top', color=self.area_colors[areas[i]],
                transform=ax.transAxes)
        sig = stats.mannwhitneyu(*mod_peaklatencies, alternative='greater').pvalue
        annot = ''
        if sig < 0.05:
            annot = '*'
        if sig < 0.01:
            annot = '**'
        if sig < 0.001:
            annot = '***'
        ax.text(0, ax.get_ylim()[1]*1.05, annot, fontsize=14, color='black',
            ha='center', va='top')
        sns.despine(fig, offset=5, bottom=True)
        ax.get_xaxis().set_visible(False)
        ax.set(ylim=[0, ax.get_ylim()[1]],
               ylabel='ms to peak mod.')
        plt.show()
        results[3].fig = fig
        results[3].test = [(stats.mannwhitneyu(*mod_peaklatencies, alternative='greater'), 'LIP vs. SC'), 
                           (stats.ttest_1samp(mod_peaklatencies[0], 0, alternative='greater'), 'LIP mean vs. 0'),
                           (stats.ttest_1samp(mod_peaklatencies[1], 0), 'SC mean vs. 0'),
                           (mod_peaklatencies[0].mean(), 'LIP mean time to peak'),
                           (mod_peaklatencies[1].mean(), 'SC mean time to peak')]

        
        return results

    def compute_shift_axes(self, animals=['stanton', 'neville'], 
        areas=['sc'], nboot=1000):

        results = []
        labels = ['Sta', 'N']

        j = 0

        f_0, ax = plt.subplots(nrows=1, ncols=len(animals), 
            figsize=(3,2),
            sharey=True)

        for a in self.troop.animals:
            print(a.name)
            if animals is not None and a.name not in animals:
                continue
            res = a.msd.compute_shift_axes(areas, nboot=nboot)

            scores, times_scores = res

            # 1. LIP vs. SC classification of eye movement dir.
            for k, v in scores.items():
                v *= 100

                # Index into true results (vs. label shuffle)
                data = v[:,0]
                shuf = v[:,1]
                ax[j].plot(times_scores, data.mean(0), color=self.area_colors[k])
                ax[j].fill_between(times_scores, 
                    data.mean(0)-data.std(0), 
                    data.mean(0)+data.std(0),
                    color=self.area_colors[k], alpha=0.2)

                ax[j].plot(times_scores, shuf.mean(0), color='grey')
                ax[j].fill_between(times_scores, 
                    shuf.mean(0)-shuf.std(0), 
                    shuf.mean(0)+shuf.std(0),
                    color='grey', alpha=0.2)
            ax[j].set(xlabel='ms to shift',
                ylabel='Accuracy (%)' if j == 0 else None,
                ylim=[35, 100])

            ax[j].axhline(50, color='dimgrey', linestyle='--', linewidth=1)


            # Annotate w/ moments of significance, determined via bootstrap

            # Significance of difference from chance
            for i,area in enumerate(areas):
                sig = np.sum(scores[area][:,1] >= scores[area][:,0].mean(0), axis=0)/nboot
                sig = stats.false_discovery_control(sig)
                ax[j] = util.annotate_signif(times_scores, sig, ax[j], 
                    color=self.area_colors[area],
                    height=100 - 2*i)

            # Label with animal name
            ax[j].text(1, 0, f'Monkey {labels[j]}', ha='right',
                va='bottom', transform=ax[j].transAxes, fontsize=12,
                color='black')

            if j == 0:
                ax[j].text(0, 0.7, 'data', ha='left',
                    va='bottom', color=self.area_colors[area],
                    fontsize=12, transform=ax[j].transAxes)
                ax[j].text(0, 0.8, 'shuffle', ha='left',
                    va='bottom', color='grey',
                    fontsize=12, transform=ax[j].transAxes)
            j += 1

        sns.despine(f_0, offset=5)
        sns.despine(ax=ax[1], left=True)
        ax[1].get_yaxis().set_visible(False)
        plt.show()

        results = types.SimpleNamespace()
        results.fig = f_0


        return results


    def _twoproportion_z_test(self, i0, i1):
        """
        Compute two-proportion z-test of difference b/w proportions,
        one-sided [alternative hypothesis: prop(i0) > prop(i1)]

        """

        # Compute z-statistic
        n0 = len(i0)
        n1 = len(i1)
        p0_hat = i0.sum() / n0
        p1_hat = i1.sum() / n1
        p_hat = (i0.sum() + i1.sum()) / (n0 + n1)
        z = (p0_hat - p1_hat) / np.sqrt(p_hat * (1 - p_hat) * (1/n0 + 1/n1))

        # Compute p-value, return object
        p = 1 - stats.norm.cdf(z, 0, 1)
        result = types.SimpleNamespace()
        result.z_statistic = z
        result.pvalue = p 

        return result

