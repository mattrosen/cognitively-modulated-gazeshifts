################################################################################
#
# f2.py
#
# Author: Matt Rosen
# Modified: 6/24
#
# Code to generate Figure 2: Small eye movements reflect abstract categories.
#
################################################################################

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pickle, types
from scipy import stats
from . import *


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

class f2(object):

    def __init__(self, troop, kwargs={}):
        self.troop = troop
        self.figpath = "./figures/EyeMovementAnalysis/f2/"
        if not os.path.exists(self.figpath):
            os.makedirs(self.figpath)

        # Palette information:
        self.cat_pal = sns.color_palette(['#3182bd', '#bd3182'])

        # A. Example animal, eye positions during delay
        a = self.visualize_delay_eyepositions(
            tasks=['DMC'],
            animal='jb')
        a.fig.savefig(self.figpath + "a.pdf", bbox_inches='tight')
        if hasattr(a, 'test'):
            with open(self.figpath + "a.txt", 'w') as f_:
                f_.write(str(a.test))

        # B. Example decoding timecourse, 1 animal 
        #    (+/- s.e.m. across sessions)
        # C. Summary (static decoding across sessions,
        #    true vs. perpendicular counterfactual boundary)
        b, c = self.compute_eyecat(tasks=['DMC'],
            example_animal='jb')
        b.fig.savefig(self.figpath + "b.pdf", bbox_inches='tight')
        c.fig.savefig(self.figpath + "c.pdf", bbox_inches='tight')
        if hasattr(b, 'test'):
            with open(self.figpath + "b.txt", 'w') as f_:
                f_.write(str(b.test))
        if hasattr(c, 'test'):
            with open(self.figpath + "c.txt", 'w') as f_:
                f_.write(str(c.test))

        # D. Error trial eye movements on average, example animal
        # E. Congruence index timecourse, example animal
        # F. Congruence index summary (last 100ms of delay, 
        #    held-out correct vs. error trials)
        d, e, f = self.compute_congruence_index(
            tasks=['DMC','RCDSTVGS'],
            example_animal='stanton')

        d.fig.savefig(self.figpath + "d.pdf", bbox_inches='tight')
        e.fig.savefig(self.figpath + "e.pdf", bbox_inches='tight')
        f.fig.savefig(self.figpath + "f.pdf", bbox_inches='tight')
        
        if hasattr(d, 'test'):
            with open(self.figpath + "d.txt", 'w') as f_:
                f_.write(str(d.test))
        if hasattr(e, 'test'):
            with open(self.figpath + "e.txt", 'w') as f_:
                f_.write(str(e.test))
        if hasattr(f, 'test'):
            with open(self.figpath + "f.txt", 'w') as f_:
                f_.write(str(f.test))
        

        # TEXT: print metadata summary
        # - number correct completed trials per animal/area
        # - number correct completed trials, total
        # - number sessions of DMC behavior per animal/area
        # - number sessions of DMC behavior, total
        text = self.get_misc_text()
        with open(self.figpath + "misc_text.txt", 'w') as f_:
            for l in text:
                f_.write(l)
                f_.write("\n")
        
        plt.close('all')

        return

    def visualize_delay_eyepositions(self, tasks=['DMC'],
        animal='bootsy',
        session=50):

        # Set up results record
        results = types.SimpleNamespace()

        # For the specified animal, obtain eye movements on single trials 
        # during the delay period
        for a in self.troop.animals:
            if a.name != animal:
                continue

            res = a.obtain_delay_eyemovements(
                tasks=tasks, session=session) 
            eye = res[0] # trials x time x 2
            cat = res[1] # trials 

            break

        # Plot these eye movements through time
        fig, ax = plt.subplots(1)
        for i in range(eye.shape[0]):
            ax.plot(*eye[i, 3*eye.shape[1]//4:].T, color=self.cat_pal[cat[i]], linewidth=1,
                alpha=0.5)

        sns.despine(fig, offset=5)

        extreme = max([np.abs(ax.get_ylim()).max(), 
            np.abs(ax.get_xlim()).max()])
        ax.set(xlim=[-extreme, extreme],
               ylim=[-extreme, extreme])
        ax.set(xlabel='Azim. (dva)',
               ylabel='Elev. (dva)')

        # Label category colors
        for j in range(len(self.cat_pal)):
            ax.text(0, 0.1 - j*0.1, f'c{j}', fontsize=12, ha='left',
                va='bottom', color=self.cat_pal[j],
                transform=ax.transAxes)



        plt.show()

        results.fig = fig

        return results

    def compute_congruence_index(self, tasks=['DMC'],
        example_animal='stanton'):
        """
        Compute timecourse of congruence index for each dataset,
        as well as summary (congruence index @ end of delay).
        """
        ci = {}
        ci_eval = {}

        f_0, f_1 = None, None

        for a in self.troop.animals:
            print(a.name)
            results = a.compute_congruence_index(tasks=tasks)

            if results is None:
                continue

            c0ci, c1ci, c0cie, c1cie = results[:-2]
            if a.name == example_animal:
                f_0, f_1 = results[-2:]

            ci[a.name] = np.column_stack((c0ci, c1ci)).mean(1)
            ci_eval[a.name] = np.column_stack((c0cie, c1cie)).mean(1)

        # Average congruence index at the end of the delay
        # for incorrect trials
        average_ci = np.array([v[-5:].mean() for v in ci.values()])
        average_cie = np.array([v[-5:].mean() for v in ci_eval.values()])
        f_2, ax = plt.subplots(1, figsize=(1,2))
        ax.bar(0.3, average_cie.mean(), yerr=stats.sem(average_cie),
            color='lightgrey', linewidth=0,
            width=0.4)
        ax.bar(0.7, average_ci.mean(), yerr=stats.sem(average_ci),
            color='black', linewidth=0,
            width=0.4)

        # Individual scatters
        averages = np.column_stack((average_cie, average_ci))
        for j in averages:
            ax.plot([0.3,0.7], j, linewidth=0.5,
                color='dimgrey') 

        ax.text(0, 0, 'error', color='lightgrey',
            fontsize=12, ha='left', va='bottom',
            transform=ax.transAxes)
        ax.text(0, 0.1, 'correct', color='black',
            fontsize=12, ha='left', va='bottom',
            transform=ax.transAxes)

        ax.set(ylabel='Congr. ind. (a.u.)', 
               xlim=[0,1],
               ylim=[-1,1])

        # Add significance annotation
        sig = stats.wilcoxon(*averages.T, alternative='greater').pvalue
        annot = 'n.s.'
        if sig < 0.05:
            annot = '*'
        if sig < 0.01:
            annot = '**'
        if sig < 0.001:
            annot = '***'

        if len(annot) > 0:
            ax.text(0.5, ax.get_ylim()[1], annot, fontsize=14,
                ha='center', va='top', color='black')
        sns.despine(f_2)
        plt.show()

        # Bind results together and return
        results = [types.SimpleNamespace(), types.SimpleNamespace(), types.SimpleNamespace()]
        results[0].fig = f_0
        results[1].fig = f_1
        results[2].fig = f_2
        results[2].test = sig

        return results

    def compute_eyecat(self, tasks=['DMC'],
        example_animal='hobbes1', nboot=500):

        true_decodes   = {}
        shuf_decodes   = {}
        ctfctl_decodes = {}

        test_info = []

        f_0 = None
        ps = {}
        for a in self.troop.animals:
            print(a.name)
            if a.name == example_animal:
                f_0 = a.compute_eyecat_timecourse(tasks=tasks, area=None)

            results = a.compute_eyecat_static(tasks=tasks,
                do_counterfactual=True, do_shuffle=True,
                nboot=nboot)
            if results is None:
                continue
            true_decodes[a.name]   = results[0]
            ctfctl_decodes[a.name] = results[1]
            shuf_decodes[a.name]   = results[2]

            p_shuf = stats.wilcoxon(results[0].mean(1), 
                results[2].mean(1), alternative='greater').pvalue
            p_ctfctl = stats.wilcoxon(results[0].mean(1), 
                results[1].mean(1), alternative='greater').pvalue
            ps[a.name] = [p_shuf, p_ctfctl]

        # Correct p values using BH
        all_shuf_p = np.array([v[0] for v in ps.values()])
        all_ctfctl_p = np.array([v[1] for v in ps.values()])

        corr_shuf_p = stats.false_discovery_control(all_shuf_p)
        corr_ctfctl_p = stats.false_discovery_control(all_ctfctl_p)
        test_info.append("Wilcoxon signed-rank test on true label decode vs. shuffle, FDR-corrected, for each dataset:")
        test_info.append(str(corr_shuf_p))
        test_info.append("Wilcoxon signed-rank test on true label decode vs. counterfactual labels, FDR-corrected, for each dataset:")
        test_info.append(str(corr_ctfctl_p))


        # Plot results
        means, serrs = [], []
        ctfctl_means, ctfctl_serrs = [], []
        shuf_means, shuf_serrs = [], []
        f_1, ax = plt.subplots(1, figsize=(1.5,2))
        for k, v in true_decodes.items():
            means.append(v.mean((0,1)))
            serrs.append(stats.sem(v.mean(1), axis=0))
        for k, v in ctfctl_decodes.items():
            ctfctl_means.append(v.mean((0,1)))
            ctfctl_serrs.append(stats.sem(v.mean(1), axis=0))
        for k, v in shuf_decodes.items():
            shuf_means.append(v.mean((0,1)))
            shuf_serrs.append(stats.sem(v.mean(1), axis=0))

        means = np.array(means)
        shuf_means = np.array(shuf_means)
        ctfctl_means = np.array(ctfctl_means)

        ax.bar(0, shuf_means.mean(), 
            yerr=stats.sem(shuf_means), 
            color='lightgrey', width=0.7)
        ax.bar(1, ctfctl_means.mean(),
            yerr=stats.sem(ctfctl_means),
            color='darkgrey', width=0.7)
        ax.bar(2, means.mean(), 
            yerr=stats.sem(means), 
            color='dodgerblue', width=0.7)

        for j in np.column_stack((shuf_means, ctfctl_means, means)):
            ax.plot([0,1,2], j, linewidth=0.5,
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

        # 2. Test of difference in means, true vs. ctfctl
        test = stats.wilcoxon(means, ctfctl_means, alternative='greater')
        annot2 = 'n.s.'

        if test.pvalue < 0.001:
            annot2 = '***'
        elif test.pvalue < 0.01:
            annot2 = '**'
        elif test.pvalue < 0.05:
            annot2 = '*'
        test_info.append("Wilcoxon rank-sum test, difference in location b/w true vs. ctfctl labels, across animals:")
        test_info.append(str(test))

        # Test of difference in means, shuffle vs. ctfctl
        test = stats.wilcoxon(shuf_means, ctfctl_means, alternative='greater')
        annot3 = 'n.s.'
        if test.pvalue < 0.001:
            annot3 = '***'
        elif test.pvalue < 0.01:
            annot3 = '**'
        elif test.pvalue < 0.05:
            annot3 = '*'
        test_info.append("Wilcoxon rank-sum test, difference in location b/w shuffle vs. ctfctl labels, across animals:")
        test_info.append(str(test))

        ax.axhline(0.5, linestyle='--', linewidth=1, color='black')
        ax.set(xticks=[0,1,2], ylim=[0.4, 1],
            xticklabels=['shuf.', r'$\perp$', 'true'], ylabel='Accuracy')

        # Annotate with difference significance
        h1 = 0.95
        h2 = 1.05
        h3 = 0.85

        # First: annot2, centered at 1.5, slightly lower
        ax.text(1.5, h1, annot2, ha='center', va='bottom', fontsize=14)
        ax.plot([1,2], [h1 + 0.01, h1 + 0.01], color='black', linewidth=1)

        # Second: annot1, centered at 1, slightly higher
        ax.text(1, h2, annot1, ha='center', va='bottom', fontsize=14)
        ax.plot([0,2], [h2 + 0.01, h2 + 0.01], color='black', linewidth=1,
            clip_on=False)

        # Third: annot3, centered at 0.5, slightly lower
        ax.text(0.5, h3, annot3, ha='center', va='bottom', fontsize=14)
        ax.plot([0,1], [h3 + 0.01, h3 + 0.01], color='black', linewidth=1)
        sns.despine(f_1)
        plt.show()

        # Bind results together and return
        results = [types.SimpleNamespace(), types.SimpleNamespace()]
        results[0].fig = f_0
        results[1].fig = f_1
        results[1].test = "\n".join(test_info)

        return results

    def get_misc_text(self, 
        areas=['lip', 'pfc', 'mt', 'mst', 'fef', 'sc', 'area5']):

        md = self.troop.summarize_recording_metadata(areas=areas,
                                   tasks=['DMC'])
        return md

