################################################################################
#
# f6.py
#
# Author: Matt Rosen
# Modified: 7/24
#
# Code to generate Figure 6: Oculomotor areas are specifically recruited by 
#   tasks requiring abstract cognition, not just matching or short-term memory.  
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
from deco import *


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

class f6(object):

    def __init__(self, troop, kwargs={}):
        self.troop = troop
        self.figpath = "./figures/EyeMovementAnalysis/f6/"
        if not os.path.exists(self.figpath):
            os.makedirs(self.figpath)

        self.task_colors = {'DMS'     : 'slategray',
                            'DMCearly': 'royalblue',
                            'DMC'     : 'darkblue',}


        # A. DMS task schematic

        # B. Summary of separation analysis across animals for DMS
        # (Quincy/Wahwah/Herbie/Denson)
        b = self.compute_eyesep_timecourse(
            tasks=['DMS'],
            animals=['quincy1','wahwah','herbie','denson'])
        b.fig.savefig(self.figpath + "b.pdf", bbox_inches='tight')
        if hasattr(b, 'test'):
            with open(self.figpath + "b.txt", 'w') as f_:
                f_.write(str(b.test))

        # C. Classification of direction dichotomies from eye positions, DMS
        c = self.compute_eyecat()
        c.fig.savefig(self.figpath + "c.pdf", bbox_inches='tight')
        if hasattr(c, 'test'):
            with open(self.figpath + "c.txt", 'w') as f_:
                f_.write(str(c.test))

        # D. Separation through time across variants,
        # DMS/DMCearly/DMC (Denson)
        d = self.compare_eyesep_by_task(
            tasks=['DMS', 'DMCearly','DMC'],
            example_animal='denson')
        d.fig.savefig(self.figpath + "d.pdf", bbox_inches='tight')
        
        # E. Fraction direction selective for LIP neurons 
        # during DMS, DMCearly, DMC
        e = self.compute_fraction_direction_selective(area='lip',
            tasks=['DMS','DMCearly','DMC'], 
            animals=['herbie', 'denson'])
        e.fig.savefig(self.figpath + "e.pdf", bbox_inches='tight')
        if hasattr(e, 'test'):
            with open(self.figpath + "e.txt", 'w') as f_:
                for i in range(len(e.test)):
                    f_.write(str(e.test[i]) + "\n")
        

        plt.close('all')

        return

    def compute_eyesep_timecourse(self, tasks=['DMC'], animals=None):

        """
        Compute separation through time b/w eye movements for
        pairs of conditions, relative to control (shuffle).
        """
        true_seps, shuf_seps = {}, {}
        delay_inds = {}, {}

        for a in self.troop.animals:
            print(a.name)

            if animals is not None and a.name not in animals:
                continue
            results = a.compute_condwise_eyesep_timecourse(tasks=tasks, 
                split_by_dist=True,
                split_by_cat=False)
            if results is None:
                continue

            ts, ss, di, si = results[:-1]

            true_seps[a.name] = ts
            shuf_seps[a.name] = ss
            delay_inds[a.name] = di

        # Summary quantification: delay-period difference
        # between true and category-shuffled traces for each animal
        peak_delay_sep = []
        for k in true_seps.keys():
            all_inds = np.union1d(delay_inds[k], sample_inds[k])
            ts = true_seps[k][:,delay_inds[k]].mean(0).max()
            ss = shuf_seps[k][:,delay_inds[k]].mean(0).max()

            peak_delay_sep.append([ts, ss])


        peak_delay_sep = np.array(peak_delay_sep)

        # Barplot of average true vs shuffle separation peaks
        f_1, ax = plt.subplots(1, figsize=(1,2))
        ax.bar(0, peak_delay_sep[:,1].mean(), yerr=stats.sem(peak_delay_sep[:,1]),
            color='grey', linewidth=0,
            width=1)
        ax.bar(1, peak_delay_sep[:,0].mean(), yerr=stats.sem(peak_delay_sep[:,0]),
            color='dodgerblue', linewidth=0,
            width=1)

        # Individual scatters
        for j in peak_delay_sep:
            ax.plot([0,1], j[::-1], linewidth=0.5,
                color='dimgrey') 

        ax.set(ylabel='Sep. (dva)', 
               ylim=[0, 0.15],
               xticks=[],
               xlim=[-1, 2])
        ax.text(0.1,0.6,'same', color='grey', ha='left', 
            va='bottom', fontsize=12, transform=ax.transAxes)
        ax.text(0.1,0.7,'opp.', color='dodgerblue', ha='left',
            va='bottom', fontsize=12, transform=ax.transAxes)

        # Add significance annotation
        sig = stats.wilcoxon(peak_delay_sep[:,0], peak_delay_sep[:,1], 
            alternative='greater').pvalue

        annot = 'n.s.'
        if sig < 0.05:
            annot = '*'
        if sig < 0.01:
            annot = '**'
        if sig < 0.001:
            annot = '***'

        y = ax.get_ylim()[1]
        ax.text(0.5, y, annot, fontsize=12,
            ha='center', va='top', color='black')
        ax.plot([0, 1], [y - 0.015, y - 0.015], color='black', linewidth=1)

        sns.despine(f_1)
        plt.show()

        # Bind results together and return
        result = types.SimpleNamespace()
        result.fig = f_1
        result.test = sig

        return result

    
    def compute_fraction_direction_selective(self, area,
        animals=None,
        tasks=['DMS', 'DMCearly', 'DMC'], 
        alpha=0.001):

        dm_by_task = defaultdict(list)

        for a in self.troop.animals:
            if animals is None or a.name in animals:
                results = a.compute_fraction_direction_selective(area,
                    tasks=tasks)
                for k,v in results.items():
                    dm_by_task[k].append(v)

        # Plot bootstrap estimates of proportion selective
        f_0, ax = plt.subplots(ncols=2, figsize=(3.5, 1.5), layout='constrained')
        rng = np.random.default_rng()
        props = []
        indicators = {}
        for i, k in enumerate(tasks):
            dm = np.concatenate(dm_by_task[k], axis=-1).T
            print(k, np.sum(dm < alpha, axis=0))

            # Bootstrap resample to generate distribution
            # of proportions
            boot = rng.choice(dm.shape[0], [10000, dm.shape[0]], replace=True)
            dm_boot = dm[boot]

            sig_boot = np.sum(dm_boot < alpha, axis=1)
            prop_sample = sig_boot[...,0] / dm.shape[0]
            prop_delay = sig_boot[...,1] / dm.shape[0]
            ax[0].bar(i, prop_sample.mean(), yerr=prop_sample.std(), 
                color=self.task_colors[k], width=0.7, alpha=0.4,
                linewidth=0)
            ax[1].bar(i, prop_delay.mean(), yerr=prop_delay.std(), 
                color=self.task_colors[k], width=0.7,
                linewidth=0)

            props.append(prop_delay)
            
            ax[1].text(0., 1 - 0.1 * i, k, ha='left',va='top',
                fontsize=12, color=self.task_colors[k],
                transform=ax[1].transAxes)

            # Store indicators for statistical test on difference of proportions
            indicators[k] = (np.int32(dm[...,0] < alpha), np.int32(dm[...,1] < alpha))

        proptests = []
        test = self._twoproportion_z_test(indicators['DMC'][0], indicators['DMS'][0])
        test.name = "Sample, DMC vs. DMS"
        proptests.append(test)

        test = self._twoproportion_z_test(indicators['DMC'][0], indicators['DMCearly'][0])
        test.name = "Sample, DMC vs. DMCearly"
        proptests.append(test)

        test = self._twoproportion_z_test(indicators['DMC'][1], indicators['DMS'][1])
        test.name = "Delay, DMC vs. DMS"
        proptests.append(test)

        test = self._twoproportion_z_test(indicators['DMC'][1], indicators['DMCearly'][1])
        test.name = "Delay, DMC vs. DMCearly"
        proptests.append(test)

        ax[0].set(ylabel='Frac. DS')
        sns.despine(f_0, offset=5)
        ax[0].get_xaxis().set_visible(False)
        ax[1].get_xaxis().set_visible(False)
        ax[0].spines['bottom'].set_visible(False)
        ax[1].spines['bottom'].set_visible(False)
        plt.show()


        # Same thing, but just delay period
        f_1, ax = plt.subplots(1, figsize=(1.25, 2))
        for i, k in enumerate(tasks):
            dm = np.concatenate(dm_by_task[k], axis=-1).T

            # Bootstrap resample to generate distribution
            # of proportions
            boot = rng.choice(dm.shape[0], [10000, dm.shape[0]], replace=True)
            dm_boot = dm[boot]

            sig_boot = np.sum(dm_boot < alpha, axis=1)
            prop_delay = sig_boot[...,1] / dm.shape[0]
            ax.bar(i, prop_delay.mean(), yerr=prop_delay.std(), 
                color=self.task_colors[k], width=0.5,
                linewidth=0)
            
            ax.text(0., 1 - 0.1 * i, k, ha='left',va='top',
                fontsize=12, color=self.task_colors[k],
                transform=ax.transAxes)

        ax.set(ylabel='Frac. DS')
        sns.despine(f_1, offset=5)
        ax.get_xaxis().set_visible(False)
        ax.spines['bottom'].set_visible(False)
        plt.show()


        # Bind results together and return
        results = types.SimpleNamespace()
        results.fig = f_1
        results.test = proptests

        return results
        

    def compare_eyesep_by_task(self, 
        tasks=['DMS', 'DMCearly', 'DMC'],
        animals=['herbie', 'denson'],
        example_animal='denson',
        resolution=50,
        alpha=0.05):


        # Loop through animals, computing separation b/w
        # eye movements in each task condition
        ratio_by_t = defaultdict(list)
        for a in self.troop.animals:

            if animals is not None and a.name not in animals:
                continue

            # Compute eye separation on pairs of directions separated
            # by 180 degrees
            results = a.compute_eyesep_oppositepairs(tasks=tasks,
                resolution=resolution)

            median_dur, median_sample, median_delay = results[1:]

            # Quantify by ratio of ODS to SDS -- for each animal,
            # for each task, compute the distribution of this ratio,
            # and render 95% confidence intervals
            for k,v in results[0].items():
                ratio_by_t[k].append(v[0]/v[1].mean(0, keepdims=True))

            if a.name == example_animal:
                f_0, ax = plt.subplots(1, figsize=(3,2))
                x = np.arange(-500//resolution, (median_dur - 500 + 100)//resolution + 1)*resolution
                sigs = {}
                for k,v in results[0].items():
                    mean_ = np.nanmean(v[0] - v[1],0)
                    std_ = np.nanstd(v[0] - v[1],0)
                    ax.plot(x, mean_, color=self.task_colors[k])
                    ax.fill_between(x, mean_ - std_, mean_ + std_,
                        alpha=0.1, color=self.task_colors[k])

                    # Compute significance via bootstrap
                    sig = np.nansum(v[1] > v[0], axis=0)/v[0].shape[0]
                    sig = stats.false_discovery_control(sig)
                    sig_periods = consecutive(np.where(sig < alpha)[0])
                    sigs[k] = [sp for sp in sig_periods if len(sp) > 1]
                
                for k, v in sigs.items():
                    y = ax.get_ylim()[1]
                    for vi in v:
                        ax.plot([x[vi[0]], x[vi[-1]]], np.repeat(y, 2), linewidth=2, 
                            color=self.task_colors[k])

                
                ax.set(ylabel='opp. - same (dva)', 
                    xlabel='ms from sample onset')
                ax.axvline(median_sample + median_delay, color='lightgrey', 
                    linestyle='--', zorder=-90, linewidth=1)
                ax.axvline(median_sample, color='lightgrey', 
                    linestyle='--', zorder=-90, linewidth=1)
                ax.axvline(0, color='lightgrey', linestyle='--', 
                    zorder=-90, linewidth=1)

                for i in range(len(tasks)):
                    ax.text(0., 1 - 0.1 * i, tasks[i],
                        ha='left', va='top', fontsize=12,
                        color=self.task_colors[tasks[i]],
                        transform=ax.transAxes)
                sns.despine(f_0, offset=5)

                plt.show()

        # Plot peak ratio during the delay by task
        f_1, ax1 = plt.subplots(1, figsize=(2,2))
        for j, (k,v) in enumerate(ratio_by_t.items()):

            # Take average ratio
            sigs = []
            means = []
            sds = []

            for i in range(len(v)):
                means.append(np.nanmean(v[i], 0))
                sds.append(np.nanstd(v[i], 0))

                # Extract peak ratio in delay
                delay_inds = np.where((x >= median_sample) & (x <= median_sample + median_delay))[0]#np.where((x >= -median_delay) & (x <= 0))[0]
                peak_delay = np.argmax(means[-1][delay_inds])

                inds = np.where((x >= median_sample // 2) & (x <= median_sample + median_delay//2))[0]
                peak_ind = np.argmax(means[-1][inds])
                #inds = np.where((x >= -(median_sample//2+median_delay)) & 
                #    (x <= -median_delay//2))[0]
                #peak_ind = np.argmax(means[-1][inds])

                bar_x = 3*i + j
                if i == 1:
                    bar_x += 1

                ax1.bar(bar_x, v[i][:,inds[peak_ind]].mean(), 
                    yerr=v[i][:,inds[peak_ind]].std(),
                    color=self.task_colors[k], 
                    width=0.75, linewidth=0)

                print()
                

        ax1.set(ylim=[0.85, ax1.get_ylim()[1]*1.2])

        for j,k in enumerate(tasks):
            ax1.text(1, 1-0.1*j, k, 
                color=self.task_colors[k],
                transform=ax1.transAxes,
                ha='right', va='top',
                fontsize=12)

        sns.despine(f_1)
        ax1.set(ylabel='Peak dif. ratio', 
            ylim=[0.85, ax1.get_ylim()[1]],
            xlabel='Monkey',
            xticks=[1, 5],
            xticklabels=['D', 'He'])

        plt.show()

        results = [types.SimpleNamespace(), types.SimpleNamespace()]
        results[0].fig = f_0
        results[1].fig = f_1

        return results

    def compute_eyecat(self, animals=['denson','herbie','quincy1','wahwah'],
        tasks=['DMS'],
        nboot=1000):

        true_decodes   = {}
        shuf_decodes   = {}

        test_info = []

        ps = {}
        for a in self.troop.animals:
            print(a.name)
            if a.name not in animals:
                continue

            results = a.compute_eyecat_static_DMS(nboot=nboot)
            if results is None:
                continue
            true_decodes[a.name]   = results[0]
            shuf_decodes[a.name]   = results[1]

            p_shuf = np.sum(results[1] >= results[0].mean(1,keepdims=True), axis=1)/nboot
            print(stats.false_discovery_control(p_shuf), sum(stats.false_discovery_control(p_shuf) < 0.05))
            p_shuf = stats.wilcoxon(true_decodes[a.name].mean(1), 
                                    shuf_decodes[a.name].mean(1), 
                                    alternative='greater').pvalue

            print(a.name, true_decodes[a.name].mean(1), shuf_decodes[a.name].mean(1), p_shuf)
            ps[a.name] = [p_shuf]

        # Control for FDR
        all_shuf_p = np.array([v[0] for v in ps.values()])

        corr_shuf_p = stats.false_discovery_control(all_shuf_p)
        test_info.append("Wilcoxon signed-rank test on true label decode vs. shuffle, FDR-corrected, for each dataset:")
        test_info.append(str(corr_shuf_p))

        # Plot results
        means, serrs = [], []
        shuf_means, shuf_serrs = [], []
        f_1, ax = plt.subplots(1, figsize=(1,2))
        for k, v in true_decodes.items():
            means.append(v.mean((0,1)))
            serrs.append(stats.sem(v.mean(1), axis=0))
        for k, v in shuf_decodes.items():
            shuf_means.append(v.mean((0,1)))
            shuf_serrs.append(stats.sem(v.mean(1), axis=0))

        means = np.array(means)
        shuf_means = np.array(shuf_means)

        ax.bar(0, shuf_means.mean(), 
            yerr=stats.sem(shuf_means), 
            color='lightgrey', width=0.7)
        ax.bar(1, means.mean(), 
            yerr=stats.sem(means), 
            color='dodgerblue', width=0.7)

        for j in np.column_stack((shuf_means, means)):
            ax.plot([0,1], j, linewidth=0.5,
                color='grey') 

        ########################################################################
        # 1. Test of difference in means, true vs. shuffle
        test = stats.wilcoxon(means, shuf_means, alternative='greater')
        annot1 = 'n.s.'
        if test.pvalue < 0.001:
            annot1 = '***'
        elif test.pvalue < 0.01:
            annot1 = '**'
        elif test.pvalue < 0.05:
            annot1 = '*'
        test_info.append("Wilcoxon rank-sum test, difference in location b/w true vs. shuffle labels, across animals:")
        test_info.append(str(test))

        ax.axhline(0.5, linestyle='--', linewidth=1, color='black')
        ax.set(xticks=[0,1], ylim=[0.4, 1.0],
            xticklabels=['shuf.', 'true'], ylabel='Accuracy')

        # Annotate with difference significance
        h2 = 1.05

        # Second: annot1, centered at 1, slightly higher
        ax.text(0.5, h2, annot1, ha='center', va='bottom', fontsize=14)
        ax.plot([0,1], [h2 + 0.01, h2 + 0.01], color='black', linewidth=1,
            clip_on=False)


        # Bind results together and return
        result = types.SimpleNamespace()
        result.fig = f_1
        result.test = "\n".join(test_info)

        print("\n".join(test_info))
        sns.despine(f_1)
        plt.show()
        

        return result



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


