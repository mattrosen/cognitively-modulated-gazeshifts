################################################################################
#
# f3.py
#
# Author: Matt Rosen
# Modified: 6/24
#
# Code to generate Figure 3: Category-correlated eye movements contribute 
#   weakly to the evolution of category encoding in LIP and SC.  
#
################################################################################

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import types
from scipy import stats
from .. import util

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

class f3(object):

    def __init__(self, troop, kwargs={}):
        self.troop = troop
        self.figpath = "./figures/EyeMovementAnalysis/f3/"
        if not os.path.exists(self.figpath):
            os.makedirs(self.figpath)

        self.area_colors = {'lip'  : 'darkgoldenrod',
                            'sc'   : 'royalblue'}

        # A. Analysis schematic

        # B. Average eye positions, 1 session, example animal,
        #   colored by category; separately for MP and LP
        # C. Peak distance b/w average eye positions for trials of
        #   different categories in MP and LP, SC sessions
        # D. Peak distance b/w average eye positions for trials of
        #   different categories in MP and LP, LIP sessions
        # E. Decoding of category from MP and LP, SC sessions
        # F. Decoding of category from MP and LP, LIP sessions
        e, c = self.perform_eyestrat_analysis(areas=['sc'], 
            animals=['stanton', 'neville'])
        e.fig.savefig(self.figpath + "e.pdf", bbox_inches='tight')
        c.fig.savefig(self.figpath + "c.pdf", bbox_inches='tight')
        if hasattr(e, 'test'):
            with open(self.figpath + "e.txt", 'w') as f_:
                f_.write(str(e.test))

        b, d, f = self.perform_eyestrat_analysis()
        b.fig.savefig(self.figpath + "b.pdf", bbox_inches='tight')
        d.fig.savefig(self.figpath + "c.pdf", bbox_inches='tight')
        f.fig.savefig(self.figpath + "d.pdf", bbox_inches='tight')
        if hasattr(d, 'test'):
            with open(self.figpath + "d.txt", 'w') as f_:
                f_.write(str(d.test))
        

        plt.close('all')

        return

    def perform_eyestrat_analysis(self, 
        areas=['lip'],
        animals=None,
        alpha=0.05, 
        example_animal='bootsy',
        nboot=1000):

        # Collect results
        by_ep = defaultdict(list)
        al    = defaultdict(list)

        f_0, f_1, f_2 = None, None, None
        for a in self.troop.animals:

            print(a.name)
            if animals is not None and a.name not in animals:
                continue
            
            res = a.test_cs_analysis(areas, nboot=nboot)

            if res is not None:
                for area in res[0].keys():
                    by_ep[area].append(res[0][area])
                    al[area].append(res[1][area])

                # Save example figures
                if a.name == example_animal:
                    f_0 = res[2][0]

        # Form aggregate plots:
        # 1. Scatterplot (with error bars) of category encoding
        #    from MP vs. LP pseudopopulations, dataset-by-dataset
        # 2. Scatterplot of category axis length for eye movements 
        #    (at largest separation moment), dataset by dataset
        results = [types.SimpleNamespace(), types.SimpleNamespace()]
        epochs = ['fix', 'sample', 'delay (early)', 'delay (late)', 'test']
        labels = ['Fix', 'Sample', f'Delay\n(early)', f'Delay\n(late)', 'Test']
        for area in by_ep.keys():

            # Differences b/w MP, LP across task epochs
            f_1, ax_1 = plt.subplots(nrows=1, ncols=4, figsize=(4,1), sharex=True, sharey=True)
            any_sig = []
            sig_by_e = np.ones((len(by_ep[area]),len(epochs)-1))
            for i, ep in enumerate(epochs[1:]):
                close = np.array([bea['close'][ep] for bea in by_ep[area]])
                far   = np.array([bea['far'][ep] for bea in by_ep[area]])

                # 2-sided test for difference using bootstrap -- % of samples in close
                # that are 
                sig = np.sum(
                    np.abs(close - close.mean(1, keepdims=True)) >= 
                        np.abs(far.mean(1) - close.mean(1))[:,None], axis=1) / close.shape[1]

                sig = stats.false_discovery_control(sig)
                any_sig.extend(np.where(sig < alpha)[0])
                sig_by_e[:,i] = sig

                # Compute significance of difference b/w close and far,
                # and use this to determine whether marker filled or not
                for j in range(close.shape[0]):
                    color = 'black' if sig[j] >= alpha else 'red'

                    ax_1[i].errorbar(close[j].mean(), far[j].mean(),
                        xerr=close[j].std(), yerr=far[j].std(), color=color,
                        linewidth=0, elinewidth=1)
                ax_1[i].set(xlim=[0.4, 1], ylim=[0.4, 1])
                
                ax_1[i].axline((0.4,0.4), (1,1), color='lightgrey', linewidth=1, 
                    linestyle='--', zorder=-90)
                ax_1[i].set_aspect('equal', 'box')
                if i > 0:
                    ax_1[i].axis('off')
                sns.despine(ax=ax_1[i], left=i > 0, bottom=i > 0)


            ax_1[0].set(xlabel='LP dec.',
                ylabel='MP dec.')
            for i in range(len(ax_1)):
                ax_1[i].text(1, 0.3, labels[i+1], 
                    ha='right', va='top', color='black',
                    transform=ax_1[i].transAxes)
            plt.show()

            results[0].fig = f_1

            # Axis length summary
            f_2, ax_2 = plt.subplots(1, figsize=(1.5, 1.5))
            ala_c = [ala['close'] for ala in al[area]]
            ala_f = [ala['far'] for ala in al[area]]

            # Loop through animals, plot eye movement sep.
            # by animal for MP, LP
            for i in range(len(ala_c)):
                color = 'black' if i not in any_sig else 'red'
                ax_2.errorbar(ala_c[i].mean(), ala_f[i].mean(),
                    xerr=ala_c[i].std(), yerr=ala_f[i].std(),
                    color=color, marker='.', linewidth=0,
                    elinewidth=1)
            min_ = min([ax_2.get_xlim()[0], ax_2.get_ylim()[0]])
            max_ = max([ax_2.get_xlim()[1], ax_2.get_ylim()[1]])

            ax_2.axline((0,0), (1,1), color='lightgrey', linewidth=1,
                linestyle='--', zorder=-90)
            ax_2.set_xlabel(r"$|e_{c0}-e_{c1}|$ (LP)", fontsize=14)
            ax_2.set_ylabel(r"$|e_{c0}-e_{c1}|$ (MP)", fontsize=14)
            ax_2.set(xlim=[min_, max_], ylim=[min_, max_])
            ax_2.set_aspect('equal','box')
            sns.despine(f_2, offset=5)
            plt.show()

            results[1].fig = f_2

            # Statistical tests/numerical reporting
            pv_al = np.array([stats.wilcoxon(ala['close'], ala['far'], alternative='less').pvalue for ala in al[area]])
            pv_al = stats.false_discovery_control(pv_al)
            text = "\n".join([f"Datasets w/ significant difference: {any_sig}, numbering {len(any_sig)} of {len(ala_c)} total.",
                            f"MP vs. LP, peak separation b/w mean eye positions by category, BH-corrected P values: {pv_al}, max P of {pv_al.max()};",
                            f"N < {alpha}, {sum(pv_al < alpha)} of {len(pv_al)}"])


            results[0].test = text

            # If appropriate, add example session MP/LP eye data, 
            # colored by category
            if f_0 is not None:
                results = [types.SimpleNamespace(), results[0], results[1]]
                results[0].fig = f_0


        return results
