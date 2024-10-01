import numpy as np
import os, glob, pprint, scipy, copy, time, math
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
import scipy.io as sio
import dill as pickle
import pandas as pd
import pymatreader
from scipy import ndimage, stats, signal, linalg
import binascii, itertools
import warnings
import multiprocessing
from deco import *

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
import sklearn.model_selection as ms
from sklearn import metrics
from functools import reduce


from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import warnings
import more_itertools as mit
from inspect import signature
from scipy.signal import butter, lfilter, sosfilt

from dataset_info import DatasetArup, DatasetBarbara, DatasetChris, DatasetDave, \
    DatasetKrithika, DatasetSruthi, DatasetYang, InactivationBehavior, \
    DatasetJamie, DatasetNick

import util, Microsaccade
from statannotations.Annotator import Annotator

# Plotting imports
import seaborn as sns
import matplotlib as mpl
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FormatStrFormatter
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.use('Agg')
params = {'mathtext.default'  : 'regular',
          'figure.figsize'    : [3.0, 3.0],
          'axes.labelsize'    : 14,
          'legend.fontsize'   : 12,
          'xtick.labelsize'   : 12,
          'ytick.labelsize'   : 12,
          'font.family'       : 'Arial',
          'pdf.fonttype'      : 42}          
plt.rcParams.update(params)

warnings.filterwarnings('ignore')
np.set_printoptions(precision=3, suppress=True)


class Animal(object):

    def __init__(self, animal_name, investigator_name, 
        areas, cat_labels, dir_labels, dbounds,
        dicd_splits):

        super().__init__()

        # Bind critical identifying information
        self.name = animal_name
        self.investigator_name = investigator_name
        self.areas = areas
        self.cat_pal = sns.color_palette(['#3182bd', '#bd3182'])

        # Bind labels for plotting
        self.cat_labels = cat_labels
        self.dir_labels = dir_labels
        self.dbounds = dict(zip(dir_labels, dbounds))
        self.dbounds[-1] = -1
        self.dicd_splits = dicd_splits

        # Plot for diagnostic - 
        full_color_spectrum = LinearSegmentedColormap.from_list('', [self.cat_pal[0], self.cat_pal[1]])
        full_circle = [full_color_spectrum(np.linspace(0, 0.5, 180)), 
                       full_color_spectrum(np.linspace(0.5, 1, 180))]
        max_d = np.amax(list(self.dbounds.values()))

        dir_pal = {}
        for c1_dir in self.dir_labels[np.where(self.cat_labels == 0)[0]]:
            dir_pal[c1_dir] = full_color_spectrum(0.5 + int(self.dbounds[c1_dir]) / (2 * max_d))

        for c2_dir in self.dir_labels[np.where(self.cat_labels == 1)[0]]:
            dir_pal[c2_dir] = full_color_spectrum(0.5 - int(self.dbounds[c2_dir]) / (2 * max_d))

        for boundary_dir in self.dir_labels[np.where((self.cat_labels != 0) & (self.cat_labels != 1))[0]]:
            dir_pal[boundary_dir] = full_color_spectrum(0.5)

        
        for k, v in dir_pal.items():
            alpha = min([0.9, self.dbounds[k]/max_d + 0.7])
            dir_pal[k] = [*v[:-1], alpha]
        

        dir_pal_dict = copy.deepcopy(dir_pal)

        dpo = np.argsort(np.array(list(dir_pal.keys())))
        dir_pal = np.array(list(dir_pal.values()))
        self.dir_pal = np.array(dir_pal)[dpo]

        # Use investigator name to determine what kind of dataset object to invoke
        dpath = f"/media/graphnets/pseudopopulation_datasets/{investigator_name.lower()}/data/"
        dobj = eval(f"Dataset{investigator_name}.Dataset{investigator_name}")
        self.dataset = dobj(self.name, areas, dpath)
        self.dataset.extract()

        self.msd = Microsaccade.MicrosaccadeDataset(self)

        ########################################################################
        # If Stan/Neville: also load inactivation behavior
        if self.name in ['stanton', 'neville']:
            trt = ['inactivation', 'sham', 'saline']
            inact_dpath = f"/media/graphnets/pseudopopulation_datasets/{investigator_name.lower()}/sc_inactivation_barbara/"
            self.inact_bhv = InactivationBehavior.InactivationBehavior(self.name, trt, 
                inact_dpath)
            self.inact_bhv.extract()

        # If Quincy/Maceo2: also load inactivation behavior
        elif self.name in ['quincy', 'maceo2']:
            trt = ['inactivation', 'control']
            inact_dpath = f"/media/graphnets/pseudopopulation_datasets/{investigator_name.lower()}/data/"
            bobj = eval(f"Dataset{investigator_name}.InactivationBehavior")
            self.inact_bhv = bobj(self.name, trt, 
                inact_dpath)
            self.inact_bhv.extract()

        return

    def get_recording_metadata(self, areas, tasks=['DMC', 'RCDSTVGS']):
        """
        Get number of sessions, number of neurons recorded for 
        this animal in this area.
        """
        self.dataset.load()

        total_s = defaultdict(int)
        total_n = defaultdict(int)
        total_t = 0

        for s in range(self.dataset.get_n_sessions()):
            tasks_s = np.unique(self.dataset.trial_type[s])
            area_s = self.dataset.areas[s][0]

            if area_s in areas and len(np.intersect1d(tasks_s, tasks)) > 0:
                total_s[area_s] += 1

                # Don't consider duplicate units -- 
                # compute spike count correlation across stimulus
                # presentations, eliminate units where value is too high
                rel_t = np.where((self.dataset.trial_error[s] == 0) & 
                                 (np.isin(self.dataset.trial_type[s],tasks)))[0]
                sample = self.dataset.direction[s][rel_t]
                spk_s = self.dataset.spikes[s][rel_t].astype(np.float64)
                spk_s = spk_s.sum(1) # Trial x Neuron

                # Compute last trial with spike for each unit
                z = np.where(np.maximum(0, spk_s).sum(0) == 0)[0]

                available = np.ones(spk_s.shape[-1])

                spkg0 = copy.deepcopy(spk_s)
                spkg0[np.where(spkg0 < 0)] = np.nan
                spkg0m = np.nanmean(spkg0,0)
                
                available[z] = 0

                # For each unique sample direction -- if any unit doesnt have 5 nonzero
                # spikecounts, make it not available
                for k in range(spk_s.shape[-1]):
                    #if k in z:
                    #    continue

                    for d in np.unique(sample[sample >= 0]):
                        
                        where_d = np.where(sample == d)[0]
                        sc_on_d = spk_s[where_d, k]
                        n_g0_on_d = np.sum(sc_on_d >= 0)

                        if n_g0_on_d < 5:
                            available[k] = 0

                isavailable = np.where(available > 0)[0]
                available[spkg0m == 0] = 0
                
                total_n[area_s] += sum(available)
                total_t += len(rel_t)

        self.dataset.stow()

        return total_s, total_n, total_t

    def get_n_trials(self, tes=[0],
        tasks=['DMC','DMC1','RCDSTVGS'], 
        areas=['lip','pfc','area5','sc','mt','mst','it','fef','lpfc','bhv'],
        dirs=np.arange(0,360,1,dtype=np.int32)):

        self.dataset.load()
        n_trials = []
        for s in range(self.dataset.get_n_sessions()):
            rel_t = np.where((np.isin(self.dataset.trial_type[s],tasks) > 0) & 
                             (np.isin(self.dataset.trial_error[s],tes) > 0) & 
                             (np.isin(self.dataset.direction[s],dirs) > 0))[0]
            if len(rel_t) < 10 or self.dataset.areas[s][0] not in areas:
                continue
            else:
                n_trials.append(len(rel_t))
        self.dataset.stow()

        return np.sum(n_trials)

    def compute_eyecat_static(self,
        sess_to_do=None,
        tasks=['DMC'],
        tes=[0],
        resolution=50, 
        nboot=100,
        do_shuffle=True,
        do_counterfactual=False,
        samp_only=False):

        self.dataset.load()

        eyedata_by_sess = defaultdict(dict)

        conds = None

        median_dur = None
        median_delay = None 

        if sess_to_do is None:
            sess_to_do = np.arange(len(self.dataset.spikes))

        for s in sess_to_do:

            # Obtain trials
            catcond = self.dataset.category[s] >= 0 if 'DMC' in tasks else True 
            corr_t = np.where((np.isin(self.dataset.trial_type[s],tasks) > 0) & 
                             (np.isin(self.dataset.trial_error[s], [0])) & 
                             catcond)[0]

            if len(corr_t) <= 1:
                continue

            task = self.dataset.trial_type[s][corr_t][0]

            # For OIC analysis - only use first 500ms (common sample period)
            if samp_only and 'OIC' in np.unique(self.dataset.trial_type[s]):
                # Select relevant periods
                corr_samp = self.dataset.times[f'{task}_sample_onset'][s][corr_t]
                if task == 'OIC':
                    corr_test = self.dataset.times[f'OIC_target_onset'][s][corr_t]
                else:
                    corr_test = corr_samp + 500
                median_dur  = np.median(corr_test - corr_samp)

            else:
                if task == 'DMCearly':
                    t_ = 'DMC'
                else:
                    t_ = task
                # Select relevant periods
                corr_samp = self.dataset.times[f'{t_}_sample_onset'][s][corr_t]
                corr_delay = self.dataset.times[f'{t_}_sample_offset'][s][corr_t]
                if t_ == 'RCDSTVGS':
                    corr_test = self.dataset.times[f'{t_}_test_onset'][s][corr_t]
                else:
                    corr_test = self.dataset.times[f'{t_}_test_onset'][s][corr_t]


                # Compute median duration -- median of fixation+sample+delay
                if median_dur is None:
                    median_delay = np.median(corr_test - corr_delay)
                    median_sample = np.median(corr_delay - corr_samp)
                    median_dur = median_delay# + median_sample

            # Make relevant time-axis indices for each trial
            delay_rng = np.array([np.arange(to - median_dur, to, dtype=np.int32) for to in corr_test])

            # Low-pass filter eye position data
            ex = self.dataset.eye_x[s][corr_t]
            ey = self.dataset.eye_y[s][corr_t]
            

            corr_eye = np.concatenate([ex[..., None],
                                       ey[..., None]], 
                                       axis=-1)

            conds = np.unique(self.dataset.direction[s][corr_t])
            min_ct_corr = min([len(np.where(self.dataset.direction[s][corr_t] == d)[0]) for d in conds])
            if min_ct_corr < 5:
                continue

            corr_eye_d = corr_eye[np.indices(delay_rng.shape)[0], delay_rng]

            # Assemble eye data
            for i, d in enumerate(conds):
                in_d = np.where(self.dataset.direction[s][corr_t] == d)[0]
            
                # Save eye data for this session
                rel_ = corr_eye_d[in_d]
                if resolution == 0:
                    eyedata_by_sess[s][d] = rel_.mean(1, keepdims=True)
                else:
                    eyedata_by_sess[s][d] = rel_[:,::resolution]

        self.dataset.stow()

        if len(eyedata_by_sess.keys()) == 0:
            return None

        # Set up for classification:
        clg0 = np.where(self.cat_labels >= 0)[0]
        labels = self.cat_labels[clg0]
        results = np.zeros((len(eyedata_by_sess.keys()), nboot))

        if do_counterfactual:
            overlaps = []
            for boundary in range(1, len(clg0)//2):
                labels_ = np.concatenate((labels[boundary:], labels[:boundary]))#np.concatenate((labels[boundary:], labels[-boundary:]))
                overlap = np.maximum(np.mean(labels ^ labels_), 1 - np.mean(labels ^ labels_))
                overlaps.append(overlap)
            ctfctl = np.argmin(overlaps) + 1
            results = np.zeros((len(eyedata_by_sess.keys()), nboot, 2))

        if do_shuffle:
            shuffle = np.zeros(results.shape[:2])

        # Loop through sessions, classify from eye positions in each session
        allres = generate_results_eyecat(eyedata_by_sess, nboot, 1, labels, ctfctl, 
            do_shuf=do_shuffle, do_static=True)
        for s in range(len(eyedata_by_sess.keys())):
            if do_counterfactual:
                #print([allres[(s,n)][0].shape for n in range(nboot)])
                results[s,:,0] = np.array([allres[(s,n)][0] for n in range(nboot)]).squeeze()
                results[s,:,1] = np.array([allres[(s,n)][1] for n in range(nboot)]).squeeze()
            else:
                results[s] = np.array([allres[(s,n)] for n in range(nboot)]).squeeze()

            if do_shuffle:
                shuffle[s] = np.array([allres[(s,n)][-1] for n in range(nboot)]).squeeze()


        if do_shuffle:
            if do_counterfactual:
                return results[...,0], results[...,1], shuffle
            else:
                return results, shuffle

        return {k: results[i] for i,k in enumerate(eyedata_by_sess.keys())}

    def compute_eyecat_static_DMS(self,
        sess_to_do=None,
        tes=[0],
        resolution=50, 
        nboot=100):

        self.dataset.load()

        eyedata_by_sess = defaultdict(dict)

        conds = None

        median_dur = None
        median_delay = None 

        if sess_to_do is None:
            sess_to_do = np.arange(len(self.dataset.spikes))

        for s in sess_to_do:

            # Obtain trials
            corr_t = np.where((np.isin(self.dataset.trial_type[s],['DMS']) > 0) & 
                             (np.isin(self.dataset.trial_error[s], [0])))[0]

            if len(corr_t) <= 1:
                continue

            task = self.dataset.trial_type[s][corr_t][0]

            # Select relevant periods
            corr_samp  = self.dataset.times[f'{task}_sample_onset'][s][corr_t]
            corr_delay = self.dataset.times[f'{task}_sample_offset'][s][corr_t]
            corr_test  = self.dataset.times[f'{task}_test_onset'][s][corr_t]

            # Compute median duration -- median of fixation+sample+delay
            if median_dur is None:
                median_delay = np.median(corr_test - corr_delay)
                median_sample = np.median(corr_delay - corr_samp)
                median_dur = median_delay

            # Make relevant time-axis indices for each trial
            delay_rng = np.array([np.arange(to - median_dur, to, dtype=np.int32) for to in corr_test])

            ex = self.dataset.eye_x[s][corr_t]
            ey = self.dataset.eye_y[s][corr_t]

            corr_eye = np.concatenate([ex[..., None],
                                       ey[..., None]], 
                                       axis=-1)

            conds = np.unique(self.dataset.direction[s][corr_t])
            min_ct_corr = min([len(np.where(self.dataset.direction[s][corr_t] == d)[0]) for d in conds])
            if min_ct_corr < 5:
                continue

            corr_eye_d = corr_eye[np.indices(delay_rng.shape)[0], delay_rng]

            # Assemble eye data
            for i, d in enumerate(conds):
                in_d = np.where(self.dataset.direction[s][corr_t] == d)[0]
            
                # Save eye data for this session
                rel_ = corr_eye_d[in_d]
                if resolution == 0:
                    eyedata_by_sess[s][d] = rel_.mean(1, keepdims=True)
                else:
                    eyedata_by_sess[s][d] = rel_[:,::resolution]

        self.dataset.stow()

        if len(eyedata_by_sess.keys()) == 0:
            return None

        # Set up for classification:
        labels  = conds
        allres = generate_results_eyecat_DMS(eyedata_by_sess, nboot, 1, labels, 
            do_static=True)

        '''
        results = np.zeros((len(eyedata_by_sess.keys()), len(conds)//2, nboot))
        shuffle = np.zeros((len(eyedata_by_sess.keys()), len(conds)//2, nboot))

        # Loop through sessions, classify from eye positions in each session
        for s in range(len(eyedata_by_sess.keys())):
            results[s] = np.array([allres[(s,n)][0] for n in range(nboot)]).squeeze().T
            shuffle[s] = np.array([allres[(s,n)][1] for n in range(nboot)]).squeeze().T
        '''
        results = np.zeros((len(eyedata_by_sess.keys()), nboot))
        shuffle = np.zeros((len(eyedata_by_sess.keys()), nboot))
        for s in range(len(eyedata_by_sess.keys())):
            results[s] = np.array([allres[(s,n)][0] for n in range(nboot)]).squeeze()
            shuffle[s] = np.array([allres[(s,n)][1] for n in range(nboot)]).squeeze()

        return results, shuffle



    def compute_eyecat_timecourse(self,
        tasks=['DMC'],
        tes=[0],
        area='lip',
        resolution=50, 
        nboot=100, # 100
        sess_to_do=None,
        return_raw=False,
        do_load=True,
        do_stow=True):

        if do_load:
            self.dataset.load()

        eyedata_by_sess = defaultdict(dict)

        median_dur    = None
        median_delay  = None 
        median_sample = None

        if sess_to_do is None:
            sess_to_do = np.arange(self.dataset.get_n_sessions())

        accs = {}

        for s in sess_to_do:

            # Obtain trials
            catcond = self.dataset.category[s] >= 0
            corr_t = np.where((np.isin(self.dataset.trial_type[s],tasks) > 0) & 
                             (self.dataset.trial_error[s] == 0) & 
                             catcond)[0]

            if len(corr_t) <= 1:
                continue

            task = self.dataset.trial_type[s][corr_t][0]
            if task == 'DMCearly':
                task = "DMC"

            if area is not None and self.dataset.areas[s][0] != area:
                continue

            # Select relevant periods
            if 'OIC' in np.unique(self.dataset.trial_type[s]):
                # Select relevant periods
                corr_samp = self.dataset.times[f'{task}_sample_onset'][s][corr_t]
                if task == 'OIC':
                    corr_test = self.dataset.times[f'OIC_target_onset'][s][corr_t]
                else:
                    corr_test = corr_samp + 510
                median_dur  = np.median(corr_test - corr_samp) + 500 

            else:
                # Select relevant periods
                corr_samp = self.dataset.times[f'{task}_sample_onset'][s][corr_t]
                corr_delay = self.dataset.times[f'{task}_sample_offset'][s][corr_t]
                corr_test = self.dataset.times[f'DMC_test_onset'][s][corr_t]
                if task == 'SPA':
                    self.dataset.times[f'SPA_test_onset'][s][corr_t]

                # Compute median duration -- median of fixation+sample+delay
                if median_dur is None:
                    median_delay = np.median(corr_test - corr_delay)
                    median_sample = np.median(corr_delay - corr_samp)
                    median_dur = median_delay + median_sample + 500

            # Make relevant time-axis indices for each trial
            delay_rng = np.array([np.arange(to - median_dur, to+100, dtype=np.int32) for to in corr_test])

            # Extract eye position data - correct and error trials
            ex = self.dataset.eye_x[s]
            ey = self.dataset.eye_y[s]

            corr_eye = np.concatenate([ex[corr_t][...,None],
                                       ey[corr_t][...,None]], 
                                       axis=-1)

            dmc_conds = np.unique(self.dataset.direction[s][corr_t])
            min_ct_corr = min([len(np.where(self.dataset.direction[s][corr_t] == d)[0]) for d in dmc_conds])
            if min_ct_corr < 5:
                continue

            corr_eye_d = corr_eye[np.indices(delay_rng.shape)[0], delay_rng]
            corr_eye_d -= corr_eye_d[:,:5].mean(1,keepdims=True)

            # Assemble eye data by condition for current session
            for i, d in enumerate(dmc_conds):
                in_d = np.where(self.dataset.direction[s][corr_t] == d)[0]
            
                # Save eye data for this session
                rel_ = corr_eye_d[in_d]
                eyedata_by_sess[s][d] = rel_[:,::resolution]



        if do_stow:
            self.dataset.stow()

        # Set up for classification:
        n_t = int((median_dur + 100)// resolution) 
        clg0 = np.where(self.cat_labels >= 0)[0]
        labels = self.cat_labels[clg0]
        results = np.zeros((len(eyedata_by_sess.keys()), nboot, 2, n_t))

        # Find counterfactual boundary closest to orthogonal
        overlaps = []
        for boundary in range(1, len(clg0)//2):
            labels_ = np.concatenate((labels[boundary:],labels[:boundary]))
            overlap = np.maximum(np.mean(labels ^ labels_), 1 - np.mean(labels ^ labels_))
            overlaps.append(overlap)
        ctfctl = np.argmin(overlaps) + 1

        # Loop through sessions, classify from eye positions in each session
        allres = generate_results_eyecat(eyedata_by_sess, 
            nboot, n_t, labels, ctfctl, do_shuf=True)
        for s in range(len(eyedata_by_sess.keys())):
            results[s,:,0] = [allres[(s,n)][0] for n in range(nboot)]
            results[s,:,1] = [allres[(s,n)][2] for n in range(nboot)]

        fig, ax = plt.subplots(1, figsize=(3,2))
        x = np.arange(-500//resolution, (median_dur - 500 + 100)//resolution)*resolution

        # Plot mean classification by true category split
        ax.plot(x, results[:,:,0].mean((0,1)), color='dodgerblue', linewidth=2)
        ax.fill_between(x, results[:,:,0].mean((0,1)) - stats.sem(results[:,:,0].mean(1),axis=(0)),
            results[:,:,0].mean((0,1)) + stats.sem(results[:,:,0].mean(1),axis=(0)), 
            color='dodgerblue', alpha=0.1)

        # Plot mean classification on counterfactual split
        ax.plot(x, results[:,:,1].mean((0,1)), color='grey', linewidth=2)
        ax.fill_between(x, results[:,:,1].mean((0,1)) - stats.sem(results[:,:,1].mean(1),axis=(0)),
            results[:,:,1].mean((0,1)) + stats.sem(results[:,:,1].mean(1),axis=(0)), 
            color='grey', alpha=0.1)

        ax.set(xlabel='ms from sample onset', ylabel='Accuracy',
               ylim=[0.1, 1])
        ax.text(0.5, 0., 'true boundary', ha='right', va='bottom', color='dodgerblue', fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 0.1, r'$\perp$ boundary', ha='right', va='bottom', color='grey', fontsize=12, transform=ax.transAxes)

        ax.axhline(0.5, linestyle='--', color='black')
        ax.axvline(0, linestyle='--', color='lightgrey', linewidth=1)

        # Annotate significance wrt sessions -- average across the bootstrap
        # dimension, and at each timepoint, one-sided t-test w/ alternative 
        # hypothesis that true is greater than counterfactual boundary
        true_by_sess = results[:,:,0].mean(1)
        ctfctl_by_sess = results[:,:,1].mean(1)
        ps = stats.ttest_rel(true_by_sess, ctfctl_by_sess, axis=0, alternative='greater').pvalue
        ps = stats.false_discovery_control(ps)
        ax = util.annotate_signif(x, ps, ax, color='dodgerblue')

        try:
            ax.axvline(median_sample+median_delay, linestyle='--', color='lightgrey', linewidth=1)
            ax.axvline(median_sample, linestyle='--', color='lightgrey', linewidth=1)
        except:
            pass
        
        sns.despine(fig, offset=5)
        plt.show()

        print(f"N sessions: {results.shape[0]}")

        if return_raw:
            return x, ps, results

        return fig



    def determine_dirpairs(self):

        # Determine pairs of sample directions to include
        # in BCD/WCD computation
        # Generate all pairings of sample directions, and compute the maximal
        # difference between directions in the same category; winnow down the 
        # set of pairings of directions in different categories to the set that
        # are separated by at most this amount

        clg0 = np.where(self.cat_labels >= 0)[0]
        dirs = self.dir_labels[clg0]
        cats = self.cat_labels[clg0]
        dirs[np.where(dirs == 0)] = 360
        c0_dir = np.unique(dirs[np.where(cats == 0)[0]])
        c1_dir = np.unique(dirs[np.where(cats == 1)[0]])

        c0_range = (np.amax(c0_dir) - np.amin(c0_dir)) % 180
        c1_range = (np.amax(c1_dir) - np.amin(c1_dir)) % 180

        func = lambda x: np.abs((x[1] - x[0] + 180) % 360 - 180) <= c0_range

        same_cat_pairs = list(itertools.combinations(c0_dir, 2))
        same_cat_pairs.extend(list(itertools.combinations(c1_dir, 2)))
        same_cat_pairs = np.array(list(filter(func, same_cat_pairs)))

        dif_cat_pairs  = list(itertools.product(c0_dir,c1_dir))
        dif_cat_pairs = np.array(list(filter(func, dif_cat_pairs)))

        # Identify unique differences in both BCD and WCD comparisons;
        # return dict for each difference level w/ corresponding pairs
        same_difs = np.minimum(
            np.abs(same_cat_pairs[:,0] - same_cat_pairs[:,1]),
            360 - np.abs(same_cat_pairs[:,0] - same_cat_pairs[:,1]))
        dif_difs = np.minimum(
            np.abs(dif_cat_pairs[:,0] - dif_cat_pairs[:,1]),
            360 - np.abs(dif_cat_pairs[:,0] - dif_cat_pairs[:,1]))
        unq_difs_same = np.unique(same_difs)
        unq_difs_dif = np.unique(dif_difs)

        unq_difs = np.intersect1d(unq_difs_same, unq_difs_dif)
        same, dif = {}, {}
        for ud in unq_difs:
            same[ud] = same_cat_pairs[np.where(same_difs == ud)[0]]
            dif[ud] = dif_cat_pairs[np.where(dif_difs == ud)[0]]

        return same, dif

    def compute_cti(self, area, tasks=['DMC'],
        window=500, shift=50, n_shuf=10):

        # Load data, save spikes and directions for all trials,
        # aligned to sample onset; store for each session
        spk_by_s = {}
        self.dataset.load()
        median_dur = None
        for s in np.arange(len(self.dataset.spikes)):

            catcond = self.dataset.category[s] >= 0
            rel_t = np.where((np.isin(self.dataset.trial_type[s], tasks)) & 
                             (self.dataset.trial_error[s] == 0) & 
                             catcond)[0]
            dirs = self.dataset.direction[s][rel_t]

            # Check if correct area, else continue
            if self.dataset.areas[s][0] != area or len(rel_t) < 50:
                continue

            task = self.dataset.trial_type[s][rel_t[0]]
            if task == 'DMCearly':
                task = 'DMC'

            # For each epoch, compute CTI in windows, shifted by 
            # shift ms, and report the max CTI across windows in each epoch
            # (a) sample epoch (50ms after sample to 550ms after sample)
            # (b) delay epoch (500ms pre-test to test on)
            sample_on = self.dataset.times[f'{task}_sample_onset'][s][rel_t]
            sample_off = self.dataset.times[f'{task}_sample_offset'][s][rel_t]
            test_on = self.dataset.times[f'{task}_test_onset'][s][rel_t]
            s_rng = np.array([np.arange(so + 50, so + 550, dtype=np.int32) 
                for so in sample_on])
            d_rng = np.array([np.arange(to - 500, to, dtype=np.int32) 
                for to in test_on]) 
            if median_dur is None:
                median_dur = np.median(test_on - sample_on)
            a_rng = np.array([np.arange(to - median_dur, to, dtype=np.int32) for to in test_on])
            spk = self.dataset.spikes[s][rel_t].astype(np.float32)
            spk[np.where(spk < 0)] = np.nan

            # Set trials w/ no spikes at all, from start to end, to nan as well
            for n in range(spk.shape[-1]):
                spk[np.where(spk[...,n].sum(1) == 0)[0],:,n] = np.nan

            # Compute spike counts for each window
            s_spk = spk[np.indices(s_rng.shape)[0], s_rng]
            d_spk = spk[np.indices(d_rng.shape)[0], d_rng]
            a_spk = spk[np.indices(a_rng.shape)[0], a_rng]

            s_spk = np.concatenate(
                [np.nansum(s_spk[:,t:t+window], axis=1, keepdims=True) for t in np.arange(0, s_spk.shape[1] - window + 1, shift)],
                axis=1)
            d_spk = np.concatenate(
                [np.nansum(d_spk[:,t:t+window], axis=1, keepdims=True) for t in np.arange(0, d_spk.shape[1] - window + 1, shift)],
                axis=1)
            a_spk = np.nansum(spk[np.indices(a_rng.shape)[0], a_rng],
                1)
            '''
            s_spk = np.nansum(spk[np.indices(s_rng.shape)[0], s_rng], 
                1, keepdims=True)
            d_spk = np.nansum(spk[np.indices(d_rng.shape)[0], d_rng],
                1, keepdims=True)
            a_spk = np.nansum(spk[np.indices(a_rng.shape)[0], a_rng],
                1, keepdims=True)
            '''

            spk_by_s[s] = [[s_spk, d_spk], a_spk, dirs]


        self.dataset.stow()
        if len(spk_by_s.keys()) == 0:
            return

        # Compute CTI, direction modulation for each unit
        # (direction modulation defined here by significance of 1-way ANOVA
        # across directions)
        same, dif = self.determine_dirpairs()
        ctis, dirmod, pvalues = [], [], []

        # Compute weighting factors for each difference to account
        # for differing numbers of pairs by direction difference
        # in BCD/WCD
        n_by_dif_same = [len(v) for v in same.values()]
        n_by_dif_dif = [len(v) for v in dif.values()]
        
        lcm_same = math.lcm(*n_by_dif_same)
        lcm_dif = math.lcm(*n_by_dif_dif)

        factors_same = lcm_same // np.array(n_by_dif_same)
        factors_dif  = lcm_dif // np.array(n_by_dif_dif)


        rng = np.random.default_rng(10)
        
        for s, ds in spk_by_s.items():

            spikes, allspikes, dirs = ds
            unq_d = np.unique(dirs)
            shuf_dirs = np.tile(copy.deepcopy(dirs), (n_shuf, 1))
            [rng.shuffle(k) for k in shuf_dirs]

            cti = []
            pv = []

            # Loop through epochs
            for ep in range(2):

                # Compute BCD/WCD for each unit
                BCD = np.zeros((len(same.keys()), 
                    spikes[ep].shape[1], spikes[ep].shape[-1]))
                WCD = np.zeros((len(same.keys()), 
                    spikes[ep].shape[1], spikes[ep].shape[-1]))

                BCD_shuf = np.zeros((n_shuf, len(same.keys()), 
                    spikes[ep].shape[1], spikes[ep].shape[-1]))
                WCD_shuf = np.zeros_like(BCD_shuf)

                for n in range(2):
                    for j, k in enumerate(same.keys()):
                        same_k = same[k]
                        dif_k = dif[k]
                        same_i, dif_i = [], []
                        for i, pair in enumerate(same_k):
                            if n == 0:
                                p0 = np.where(dirs == pair[0] % 360)[0]
                                p1 = np.where(dirs == pair[1] % 360)[0]
                                same_i.append(np.abs(np.nanmean(spikes[ep][p0],0) - 
                                    np.nanmean(spikes[ep][p1],0)) * factors_same[j])
                            else: 
                                p0 = np.array([np.where(sd == pair[0] % 360)[0] for sd in shuf_dirs])
                                p1 = np.array([np.where(sd == pair[1] % 360)[0] for sd in shuf_dirs])
                                same_i.append(np.abs(np.nanmean(spikes[ep][p0],1) - 
                                    np.nanmean(spikes[ep][p1],1)) * factors_same[j])
                            

                        for i, pair in enumerate(dif_k):
                            if n == 0:
                                p0 = np.where(dirs == pair[0] % 360)[0]
                                p1 = np.where(dirs == pair[1] % 360)[0]
                                dif_i.append(np.abs(np.nanmean(spikes[ep][p0],0) - 
                                    np.nanmean(spikes[ep][p1],0)) * factors_dif[j])
                            else:
                                p0 = np.array([np.where(sd == pair[0] % 360)[0] for sd in shuf_dirs])
                                p1 = np.array([np.where(sd == pair[1] % 360)[0] for sd in shuf_dirs])
                                dif_i.append(np.abs(np.nanmean(spikes[ep][p0],1) - 
                                    np.nanmean(spikes[ep][p1],1)) * factors_dif[j])
                            
                        if n == 0:
                            WCD[j] = np.sum(np.array(same_i), 0)
                            BCD[j] = np.sum(np.array(dif_i), 0)
                        else:
                            WCD_shuf[:,j] = np.sum(np.array(same_i), 0)
                            BCD_shuf[:,j] = np.sum(np.array(dif_i), 0)

                BCD /= (lcm_dif * len(n_by_dif_dif))
                WCD /= (lcm_same * len(n_by_dif_same))

                BCD_shuf /= (lcm_dif * len(n_by_dif_dif))
                WCD_shuf /= (lcm_same * len(n_by_dif_same))

                # Compute the CTI - ratio of dif. b/w BCD and WCD and their sum
                CTI = (BCD.mean(0) - WCD.mean(0)) / (BCD.mean(0) + WCD.mean(0))
                shuf_CTI = (BCD_shuf.mean(1) - WCD_shuf.mean(1)) / (BCD_shuf.mean(1) + WCD_shuf.mean(1))

                # Take peak CTI, p-value, and store
                peak_window = np.argmax(CTI,axis=0)
                peak_CTI = CTI[peak_window, np.arange(CTI.shape[1])]
                shuf_at_peak = shuf_CTI[:,peak_window, np.arange(CTI.shape[1])] # n_shuf x n_neur
                pvals = np.sum(shuf_at_peak >= peak_CTI[None,:], axis=0)/n_shuf

                cti.append(peak_CTI)
                pv.append(pvals)

            ctis.append(np.array(cti))
            pvalues.append(np.array(pv))

            # Compute direction modulation
            dirmod_s = []
            for n in range(allspikes.shape[-1]):
                by_dir = [allspikes[np.where(dirs == d)[0],n] for d in unq_d]
                #by_dir = [bd[np.where(bd > 0)[0]] for bd in by_dir]

                # Filter out units w/ very low spiking
                if np.mean([bd.mean()*1000/median_dur for bd in by_dir]) < 3: # 5
                    p = 1
                elif min([sum(bd > 0) for bd in by_dir]) < 5:
                    p = 1
                else:
                    try:
                        p = stats.kruskal(*by_dir)[1]
                    except:
                        p = 1
                dirmod_s.append(p)
            dirmod.append(dirmod_s)

        cti = np.concatenate(ctis, axis=1).T
        dirmod = np.concatenate(dirmod)
        pvalues = np.array(pvalues).squeeze()

        # Select non-nan elements before returning
        to_return = np.where(np.isfinite(cti.max(1)))[0]

        return cti[to_return], dirmod[to_return], pvalues[to_return]

    def compute_eyesep_inact(self, tasks=['DMC'],
        resolution=50, nboot=1000):

        eyesep_by_c = defaultdict(list)

        median_dur = None
        median_sample = None
        median_delay = None
        conds = None

        self.inact_bhv.load()

        for s in np.arange(self.inact_bhv.get_n_sessions()):

            #print(s, np.unique(self.inact_bhv.group[s]))

            # Obtain trials, ctrl/inactivation
            treatment = self.inact_bhv.treatment[s][0]
            if treatment in ['sham','saline']:
                treatment = 'control'

            data = {k: {} for k in ['ctrl', 'treatment']}
            eyesep_by_group = []

            for group in ['ctrl', 'treatment']:

                group_grp = [group]
                if group == 'ctrl':
                    group_grp = ['ctrl', 'ipsi-ctrl']
                if group == 'treatment':
                    group_grp = ['treatment']#, 'contra-ctrl']

                corr_t = np.where((np.isin(self.inact_bhv.trial_type[s],tasks) > 0) & 
                                 (self.inact_bhv.trial_error[s] == 0) & 
                                 (self.inact_bhv.category[s] >= 0) & 
                                 (np.isin(self.inact_bhv.group[s],group_grp)>0))[0]

                if len(corr_t) <= 1:
                    continue

                # Select relevant periods
                corr_samp = self.inact_bhv.times[f'DMC_sample_onset'][s][corr_t]
                corr_delay = self.inact_bhv.times[f'DMC_sample_offset'][s][corr_t]
                corr_test = self.inact_bhv.times[f'DMC_test_onset'][s][corr_t]

                # Compute median duration -- median of fixation+sample+delay
                if median_dur is None:
                    median_sample = np.median(corr_delay - corr_samp)
                    median_delay = np.median(corr_test - corr_delay)
                    median_dur = median_sample + median_delay + 500

                # Make relevant time-axis indices for each trial
                corr_rng = np.array([np.arange(ct - median_dur, ct+100, dtype=np.int32) for ct in corr_test])

                ####################################################################
                # Extract eye data
                ex = self.inact_bhv.eye_x[s][corr_t]
                ey = self.inact_bhv.eye_y[s][corr_t]

                corr_eye = np.concatenate([ex[..., None],
                                           ey[..., None]], 
                                          axis=-1)

                ####################################################################
                # Set up task conditions for joining eye data
                conds = [0,1]
                ce = corr_eye[np.indices(corr_rng.shape)[0], corr_rng]

                # Assemble eye data
                for i, d in enumerate(conds):
                    in_d = np.where(self.inact_bhv.category[s][corr_t] == d)[0]

                    if len(in_d) == 0:
                        continue

                    data[group][d] = ce[in_d,::resolution]


                # Record arrays
                n_c = len(data[group].keys())
                n_t = list(data[group].values())[0].shape[1]
                true_sep = np.full((nboot, n_t, n_c, n_c), np.nan)
                within_sep = np.full(true_sep.shape, np.nan)

                rng = np.random.default_rng(10)
                clg0 = np.where(self.cat_labels >= 0)[0]
                cat_labs = self.cat_labels[clg0]

                for n in range(nboot):

                    # Sample minsamp trials w/ replacement from each category
                    te_trials = {ki: rng.choice(vi.shape[0], vi.shape[0]//2, replace=False) for ki,vi in data[group].items()}
                    tr_trials = {ki: np.setdiff1d(np.arange(vi.shape[0]), te_trials[ki]) for ki,vi in data[group].items()}#{ki: np.arange(vi.shape[0]) for ki,vi in data[group].items()}#

                    min_t = min([v.shape for v in te_trials.values()])

                    cbd = [data[group][ki][vi[rng.choice(np.arange(vi.shape[0]), vi.shape[0], replace=True)]] for ki,vi in tr_trials.items()]
                    cbde = [data[group][ki][vi[rng.choice(np.arange(vi.shape[0]), vi.shape[0], replace=True)]] for ki,vi in te_trials.items()]

                    # True means
                    cm = np.concatenate([vi.mean(0, keepdims=True) for vi in cbd], axis=0)
                    cme = np.concatenate([vi.mean(0, keepdims=True) for vi in cbde], axis=0)

                    for i in range(n_c):
                        for j in range(n_c):

                            dif_ij = cm[i] - cme[j]
                            if i == j:
                                within_sep[n,:,i,j] = np.sqrt(np.sum(dif_ij**2,axis=1))
                            else:
                                true_sep[n,:,i,j] = np.sqrt(np.sum(dif_ij**2,axis=1))

                # Join together and return
                within_sep = np.nanmean(within_sep, (-2,-1))
                true_sep = np.nanmean(true_sep,(-2,-1))

                eyesep_by_group.append([true_sep, within_sep])

            eyesep_by_c[treatment].append(eyesep_by_group)
        self.inact_bhv.stow()

        return eyesep_by_c, median_dur, median_sample, median_delay

    def compute_eyecat_inact(self, tasks=['DMC'],
        resolution=50, nboot=1000):

        eyecat = defaultdict(list)
        eyecat_comp = defaultdict(list)

        median_dur    = None
        median_sample = None
        median_delay  = None
        conds         = None

        self.inact_bhv.load()

        for s in np.arange(self.inact_bhv.get_n_sessions()):

            # Obtain trials, ctrl/inactivation
            treatment = self.inact_bhv.treatment[s][0]

            # Group sham/saline controls together for Neville/Stan
            if treatment in ['sham','saline']:
                treatment = 'control'

            eyecat_by_group = []
            eyecat_by_group_comp = []
            for group in ['ctrl', 'treatment']:

                group_grp = [group]
                if group == 'ctrl':
                    group_grp = ['ctrl', 'ipsi-ctrl']
                if group == 'treatment':
                    group_grp = ['treatment']

                # Do this for correct and completed trials, to see if it makes a difference
                # whether we decode this vs. true assigned category (e.g. does it 
                # reflect the animal's choice, even if the choice is wrong)
                corr_t = np.where((np.isin(self.inact_bhv.trial_type[s],tasks) > 0) & 
                                 (self.inact_bhv.trial_error[s] == 0) & 
                                 (self.inact_bhv.category[s] >= 0) & 
                                 (np.isin(self.inact_bhv.group[s],group_grp)>0))[0]

                comp_t = np.where((np.isin(self.inact_bhv.trial_type[s],tasks) > 0) & 
                                 (self.inact_bhv.trial_error[s] % 6 == 0) & #(self.inact_bhv.times['DMC_test_onset'][s] > 0) & 
                                 (self.inact_bhv.category[s] >= 0) & 
                                 (np.isin(self.inact_bhv.group[s],group_grp)>0))[0]

                if len(corr_t) <= 1:
                    continue

                # Select relevant periods
                corr_samp  = self.inact_bhv.times[f'DMC_sample_onset'][s][corr_t]
                corr_delay = self.inact_bhv.times[f'DMC_sample_offset'][s][corr_t]
                corr_test  = self.inact_bhv.times[f'DMC_test_onset'][s][corr_t]

                comp_samp = self.inact_bhv.times[f'DMC_sample_onset'][s][comp_t]
                comp_delay = self.inact_bhv.times[f'DMC_sample_offset'][s][comp_t]
                comp_test  = self.inact_bhv.times[f'DMC_test_onset'][s][comp_t]

                # Compute median duration -- median of fixation+sample+delay
                if median_dur is None:
                    median_sample = np.median(corr_delay - corr_samp)
                    median_delay = np.median(corr_test - corr_delay)
                    median_dur = median_sample + median_delay + 500

                # Make relevant time-axis indices for each trial
                corr_rng = np.array([np.arange(ct - median_dur, ct+100, dtype=np.int32) for ct in corr_test])
                comp_rng = np.array([np.arange(ct - median_dur, ct+100, dtype=np.int32) for ct in comp_test])

                ####################################################################
                # Extract eye data
                ex = self.inact_bhv.eye_x[s][corr_t]
                ey = self.inact_bhv.eye_y[s][corr_t]

                corr_eye = np.concatenate([ex[..., None],
                                           ey[..., None]], 
                                          axis=-1)[np.indices(corr_rng.shape)[0], corr_rng]

                ex_comp = self.inact_bhv.eye_x[s][comp_t]
                ey_comp = self.inact_bhv.eye_y[s][comp_t]

                comp_eye = np.concatenate([ex_comp[..., None],
                                           ey_comp[..., None]], 
                                          axis=-1)[np.indices(comp_rng.shape)[0], comp_rng]

                corr_eye -= corr_eye[:,:5].mean(1, keepdims=True)
                comp_eye -= comp_eye[:,:5].mean(1, keepdims=True)

                # Decode category from true labels vs. chosen labels
                cats = self.inact_bhv.category[s]
                tes = self.inact_bhv.trial_error[s]
                tes_comp = tes[comp_t]

                true_cat_corr = cats[corr_t]
                true_cat_comp = cats[comp_t]
                chosen_cat = copy.deepcopy(true_cat_comp)
                chosen_cat[np.where(tes_comp != 0)[0]] += 1
                chosen_cat = chosen_cat % 2

                # Classify results
                corr_true = generate_results_eyecat_inact(
                    corr_eye[:,::resolution], true_cat_corr,
                    nboot)
                comp_true = generate_results_eyecat_inact(
                    comp_eye[:,::resolution], chosen_cat,
                    nboot)

                corr_true = np.array(list(corr_true.values()))
                comp_true = np.array(list(comp_true.values()))

                # Same but for shuffled data
                '''
                corr_shuf = generate_results_eyecat_inact(
                    corr_eye[:,::resolution], true_cat_corr,
                    nboot, do_shuffle=True)
                comp_shuf = generate_results_eyecat_inact(
                    comp_eye[:,::resolution], chosen_cat,
                    nboot, do_shuffle=True)


                corr_shuf = np.array(list(corr_shuf.values()))
                comp_shuf = np.array(list(comp_shuf.values()))
                '''
                eyecat_by_group.append([corr_true, corr_true])
                eyecat_by_group_comp.append([comp_true, comp_true])

            eyecat[treatment].append(eyecat_by_group)
            eyecat_comp[treatment].append(eyecat_by_group_comp)

        self.inact_bhv.stow()

        # MAKE INTO WELL-SHAPED ARRAY BEFORE RETURNING;
        # Print shapes to debug
        scores = {}
        scores_comp = {}
        for k in eyecat.keys():
            ek = np.array(eyecat[k])

            print(k, ek.shape) # sessions x groups x true vs shuf x nboot
            scores[k] = ek[:,1,0]
            scores_comp[k] = np.array(eyecat_comp[k])[:,1,0]

        

        #print(0/0)

        return scores, scores_comp

    def compute_eyesep_oppositepairs(self, tasks=['DMC'],
        resolution=50, nboot=10000):

        corr_eyedata = defaultdict(lambda: defaultdict(list))

        median_dur = None
        median_sample = None
        median_delay = None
        conds = None

        self.dataset.load()

        for s in np.arange(self.dataset.get_n_sessions()):

            # Obtain trials, correct and incorrect
            corr_t = np.where((np.isin(self.dataset.trial_type[s],tasks) > 0) & 
                             (self.dataset.trial_error[s] == 0))[0]

            if len(corr_t) <= 1:
                continue

            ###################
            task = self.dataset.trial_type[s][corr_t][0]
            if task == 'DMCearly':
                t_ = "DMC"
            else:
                t_ = task

            # Select relevant periods
            corr_samp = self.dataset.times[f'{t_}_sample_onset'][s][corr_t]
            corr_delay = self.dataset.times[f'{t_}_sample_offset'][s][corr_t]
            corr_test = self.dataset.times[f'{t_}_test_onset'][s][corr_t]

            # Compute median duration -- median of fixation+sample+delay
            if median_dur is None:
                median_sample = np.median(corr_delay - corr_samp)
                median_delay = np.median(corr_test - corr_delay)
                median_dur = median_sample + median_delay + 500

            # Make relevant time-axis indices for each trial
            corr_rng = np.array([np.arange(ct - median_dur, ct+100, dtype=np.int32) for ct in corr_test])

            ####################################################################
            # Extract eye data
            ex = self.dataset.eye_x[s][corr_t]
            ey = self.dataset.eye_y[s][corr_t]

            corr_eye = np.concatenate([ex[..., None],
                                      ey[..., None]], 
                                      axis=-1)

            ####################################################################
            # Set up task conditions for joining eye data
            conds = np.unique(self.dataset.direction[s][corr_t])
            ce = corr_eye[np.indices(corr_rng.shape)[0], corr_rng]

            # Mean-subtract for all trials in the session
            #ce -= ce.mean((0,1),keepdims=True)
            ce -= ce[:,:5].mean(1,keepdims=True)

            # Assemble eye data
            for i, d in enumerate(conds):
                in_d = np.where(self.dataset.direction[s][corr_t] == d)[0]

                if len(in_d) == 0:
                    continue

                corr_eyedata[task][d].append(ce[in_d,::resolution])

        self.dataset.stow()

        ####################################################################
        
        eyesep_by_t = {}
        for task in tasks:
            corr_eyedata[task] = {ki: np.concatenate(v, axis=0).astype(np.float32) 
                                for ki,v in corr_eyedata[task].items()}

            # Record arrays
            n_c = len(corr_eyedata[task].keys())
            n_t = list(corr_eyedata[task].values())[0].shape[1]
            true_sep = np.full((nboot, n_t, n_c, n_c), np.nan)
            within_sep = np.full(true_sep.shape, np.nan)

            rng = np.random.default_rng(10)
            clg0 = np.where(self.cat_labels >= 0)[0]
            cat_labs = self.cat_labels[clg0]
            
            minsamp = 200
            for n in range(nboot):

                # Sample minsamp trials w/ replacement from each dir.
                te_trials = {ki: rng.choice(vi.shape[0], vi.shape[0]//2, replace=False) for ki,vi in corr_eyedata[task].items()}
                tr_trials = {ki: np.setdiff1d(np.arange(vi.shape[0]), te_trials[ki]) for ki,vi in corr_eyedata[task].items()}

                cbd = [corr_eyedata[task][ki][vi[rng.choice(vi.shape[0], minsamp, replace=True)]] for ki,vi in tr_trials.items()]
                cbde = [corr_eyedata[task][ki][vi[rng.choice(vi.shape[0], minsamp, replace=True)]] for ki,vi in te_trials.items()]

                # True means
                cm = np.concatenate([vi.mean(0, keepdims=True) for vi in cbd], axis=0)
                cme = np.concatenate([vi.mean(0, keepdims=True) for vi in cbde], axis=0)
                for i in range(n_c):
                    for j in range(i,n_c):

                        dd = np.abs(conds[i]+360 - conds[j])% 360
                        dd = np.minimum(dd, 360 - dd)

                        if i == j:
                            dif_ii = cm[i] - cme[j]
                            within_sep[n,:,i,j] = np.sqrt(np.sum(dif_ii**2,axis=1))

                        elif dd == 180:
                            dif_ij = cm[i] - cm[j]
                            true_sep[n,:,i,j] = np.sqrt(np.sum(dif_ij**2,axis=1))


            # Join together and return
            within_sep = np.nanmean(within_sep, (-2,-1))
            true_sep = np.nanmean(true_sep,(-2,-1))

            eyesep_by_t[task] = [true_sep, within_sep]

        return eyesep_by_t, median_dur, median_sample, median_delay


    
    def compute_fraction_direction_selective(self, area,
        tasks=['DMC']):

        self.dataset.load()

        # Set up record array
        dirmod_by_task = defaultdict(list)

        median_dur = None

        # Collect activity from all relevant sessions/neurons
        for s in np.arange(len(self.dataset.spikes)):

            # Select relevant trials
            rel_t = np.where((np.isin(self.dataset.trial_type[s], tasks)) & 
                             (self.dataset.trial_error[s] == 0))[0]
            if len(rel_t) < 50:
                continue
            if self.dataset.areas[s][0] != area:
                continue

            # Select relevant timeperiod
            task = self.dataset.trial_type[s][rel_t[0]]
            if task == 'DMCearly':
                t_ = 'DMC'
            else:
                t_ = task
            samp_off = self.dataset.times[f'{t_}_sample_offset'][s][rel_t]
            test_on = self.dataset.times[f'{t_}_test_onset'][s][rel_t]
            samp_on = self.dataset.times[f'{t_}_sample_onset'][s][rel_t]

            if median_dur is None:
                median_dur = np.median(test_on - samp_on)
            full_rng = np.array([np.arange(to - median_dur - 500, to, dtype=np.int32) for to in test_on])

            fullspk = self.dataset.spikes[s][rel_t][np.indices(full_rng.shape)[0], full_rng].astype(np.float32)

            # Extract direction for each trial to set up labels
            dirs = self.dataset.direction[s][rel_t]
            unq_dirs = np.unique(dirs)

            # Loop through neurons, computing Kruskal H test on spikecounts by direction
            for n in range(fullspk.shape[-1]):
                
                active_trials = np.where(fullspk[...,n].min(1) >= 0)[0]
                if len(active_trials) < 50:
                    continue

                dir_inds = [np.where(dirs[active_trials] == x)[0] for x in unq_dirs]

                # Direction modulation p value (sample, delay)
                sample_spkc = [np.nansum(fullspk[active_trials[x],600:1100,n],1) for x in dir_inds]
                delay_spkc = [np.nansum(fullspk[active_trials[x],-500:,n],1) for x in dir_inds]
                p_s = stats.kruskal(*sample_spkc)[1]
                p_d = stats.kruskal(*delay_spkc)[1]

                dirmod_by_task[task].append([p_s, p_d])

        self.dataset.stow()

        dirmod_by_task = {k: stats.false_discovery_control(np.array(v).T,axis=1) for k,v in dirmod_by_task.items()}

        #print([f"{k}, {v.shape}" for k,v in dirmod_by_task.items()])

        return dirmod_by_task

    def compute_shift_modulated(self, areas, epoch='delay'):
        """
        (Wrapper on analogous function bound to the MicrosaccadeDataset 
        class)

        Apply ZETA test (eLife 2021) to determine significance
        of modulation around gaze shifts.
        """
        return self.msd.compute_shift_modulated(areas, epoch=epoch)


    def obtain_delay_eyemovements(self,
        tasks=['DMC'],
        session=0,
        resolution=10):

        self.dataset.load()


        rel_t = np.where((np.isin(self.dataset.trial_type[session], tasks)) & 
                         (np.isin(self.dataset.trial_error[session], [0])) & 
                         (self.dataset.category[session] >= 0))[0]
        if len(rel_t) < 50:
            return None, None

        # Select eyemovements during the delay
        delay_end = self.dataset.times['DMC_test_onset'][session][rel_t]
        delay_start = self.dataset.times['DMC_sample_offset'][session][rel_t]
        median_dur = np.median(delay_end - delay_start)
        delay_rng = np.array([np.arange(de - median_dur, de, dtype=np.int32) for de in delay_end])
        ex = self.dataset.eye_x[session][rel_t]
        ey = self.dataset.eye_y[session][rel_t]

        
        sos = signal.butter(7, 70, 'lp', fs=1000, output='sos')
        ex = signal.sosfilt(sos, ex, axis=1)
        ey = signal.sosfilt(sos, ey, axis=1)
        eye = np.concatenate((ex[np.indices(delay_rng.shape)[0], delay_rng,None],
            ey[np.indices(delay_rng.shape)[0], delay_rng,None]), axis=-1)
        eye = eye[:,::resolution]

        # Store category for each trial
        cat = self.dataset.category[session][rel_t].astype(np.int32)
        self.dataset.stow()

        return eye, cat

    
    def compute_inact_eyemvmt_effect(self, do_concat=False):

        self.inact_bhv.load()

        # Record arrays
        ctrl_mags = []
        inact_mags = []


        for s in np.arange(self.inact_bhv.get_n_sessions()):

            ctrl_group = ['ctrl', 'ipsi-ctrl', 'contra-ctrl']

            # Load eye movements in fixation period
            completed_t = np.where((self.inact_bhv.category[s] >= 0) & 
                                   np.isin(self.inact_bhv.trial_error[s], [0,5,6]) & 
                                   (self.inact_bhv.trial_type[s] == 'DMC'))[0]
            inact_t = np.where(self.inact_bhv.group[s] == 'treatment')[0]
            ctrl_t = np.where(np.isin(self.inact_bhv.group[s], ctrl_group))[0]
            corr_t = np.where(self.inact_bhv.trial_error[s] == 0)[0]
            treatment = self.inact_bhv.treatment[s][0]

            # For SC inactivations, don't use control sessions
            if treatment != 'inactivation' and not do_concat:
                continue

            inact_comp = np.intersect1d(completed_t, inact_t)
            ctrl_comp = np.intersect1d(completed_t, ctrl_t)

            inact_corr = np.intersect1d(inact_comp, corr_t)
            ctrl_corr = np.intersect1d(ctrl_comp, corr_t)

            sample_onset = self.inact_bhv.times['DMC_sample_onset'][s]

            fix_rng_inact = np.array([np.arange(so - 500, so, dtype=np.int32) for so in sample_onset[inact_comp]])
            fix_rng_ctrl = np.array([np.arange(so - 500, so, dtype=np.int32) for so in sample_onset[ctrl_comp]])

            ex = self.inact_bhv.eye_x[s]
            ey = self.inact_bhv.eye_y[s]
            
            eye = np.concatenate((ex[...,None],
                ey[...,None]), axis=-1)

            ctrl_eye = eye[ctrl_comp][np.indices(fix_rng_ctrl.shape)[0], fix_rng_ctrl]
            inact_eye = eye[inact_comp][np.indices(fix_rng_inact.shape)[0], fix_rng_inact]

            # Compute difference in average eye movement amount
            ctrl_vecs = np.gradient(ctrl_eye[:,::50],axis=1)
            inact_vecs = np.gradient(inact_eye[:,::50],axis=1)

            # Store magnitudes
            if do_concat:

                #all_vecs = ctrl_vecs if treatment != 'inactivation' else inact_vecs
                all_vecs = np.concatenate((ctrl_vecs, inact_vecs), axis=0)
                all_mags = np.linalg.norm(all_vecs, axis=-1).flatten()
                if treatment != 'inactivation':
                    ctrl_mags.append(all_mags)
                else:
                    inact_mags.append(all_mags)

            else:

                inact_mags.append(np.linalg.norm(inact_vecs, axis=-1).flatten())
                ctrl_mags.append(np.linalg.norm(ctrl_vecs, axis=-1).flatten())


        self.inact_bhv.stow()

        # If LIP inactivation, concatenate distributions across sessions
        if do_concat:
            ctrl_mags = [np.concatenate(ctrl_mags, axis=-1)]
            inact_mags = [np.concatenate(inact_mags, axis=-1)]

        # Return data
        return ctrl_mags, inact_mags
 

    def compute_condwise_eyesep_timecourse(self,
        tasks=['DMS', 'SPA'],
        area=None,
        samp_only=False,
        resolution=20,
        nboot=1000,
        nshuf=1000,
        split_by_cat=True,
        split_by_dist=False,
        time_rel_to='sample',
        return_raw=False,
        SEED=10,
        do_load=True,
        do_stow=True,
        sess_to_do=None,
        minsamp=200):

        if do_load:
            self.dataset.load()
    
        corr_eyedata = defaultdict(list)

        median_dur = None
        median_sample = None
        median_delay = None
        conds = None

        # Allow option to specify which sessions to do this for
        if sess_to_do is None:
            sess_to_do = np.arange(self.dataset.get_n_sessions())

        for s in sess_to_do:

            if area is not None and self.dataset.areas[s][0] != area:
                continue

            # Obtain trials, correct and incorrect
            if split_by_cat:
                catcond = self.dataset.category[s] >= 0 
            else:
                catcond = True
            corr_t = np.where((np.isin(self.dataset.trial_type[s],tasks) > 0) & 
                             (self.dataset.trial_error[s] == 0) & 
                             catcond)[0]

            if len(corr_t) <= 1:
                continue

            ###################
            task = self.dataset.trial_type[s][corr_t][0]
            if task == 'DMCearly':
                task = "DMC"

            # Select relevant periods
            if 'OIC' in np.unique(self.dataset.trial_type[s]) and samp_only:
                corr_samp = self.dataset.times[f'{task}_sample_onset'][s][corr_t]
                median_dur = 500
                corr_rng = np.array([np.arange(cs, cs+500, dtype=np.int32) for cs in corr_samp])
            else:
                corr_samp = self.dataset.times[f'{task}_sample_onset'][s][corr_t]
                corr_delay = self.dataset.times[f'{task}_sample_offset'][s][corr_t]
                corr_test = self.dataset.times[f'{task}_test_onset'][s][corr_t]

                # Compute median duration -- median of fixation+sample+delay
                if median_dur is None:
                    median_sample = np.median(corr_delay - corr_samp)
                    median_delay = np.median(corr_test - corr_delay)
                    median_dur = np.median(corr_test - corr_samp) + 500 # 500 for initial fixation

                # Make relevant time-axis indices for each trial
                corr_rng = np.array([np.arange(ct - median_dur, ct + 100, dtype=np.int32) for ct in corr_test])

            ####################################################################
            # Extract eye data
            ex = self.dataset.eye_x[s][corr_t]
            ey = self.dataset.eye_y[s][corr_t]

            corr_eye = np.concatenate([ex[..., None],
                                      ey[..., None]], 
                                      axis=-1)

            ####################################################################
            # Set up task conditions for joining eye data
            conds = np.unique(self.dataset.direction[s][corr_t])
            ce = corr_eye[np.indices(corr_rng.shape)[0], corr_rng]

            # Mean-subtract for all trials in the session
            #ce -= ce.mean((0,1),keepdims=True)
            ce -= ce[:,:5].mean(1, keepdims=True)

            # Assemble eye data
            for i, d in enumerate(conds):
                in_d = np.where(self.dataset.direction[s][corr_t] == d)[0]

                if len(in_d) == 0:
                    continue

                corr_eyedata[d].append(ce[in_d,::resolution])

        if do_stow:
            self.dataset.stow()

        ####################################################################
        # Compute bootstrapped estimate of mean -- across all sessions
        if len(corr_eyedata.keys()) == 0:
            return None

        corr_eyedata = {ki: np.concatenate(v, axis=0).astype(np.float32) 
                            for ki,v in corr_eyedata.items()}

        # Record arrays
        n_c = len(corr_eyedata.keys())
        n_t = list(corr_eyedata.values())[0].shape[1]
        true_sep = np.full((nboot, n_t, n_c, n_c), np.nan)
        shuf_sep = np.full(true_sep.shape, np.nan)
        within_sep = np.full(true_sep.shape, np.nan)

        rng = np.random.default_rng(SEED)
        clg0 = np.where(self.cat_labels >= 0)[0]
        cat_labs = self.cat_labels[clg0]
        if 'SPA' in tasks:
            cat_labs = np.array([0,0,1,1,2,2])
        
        for n in range(nboot):

            # Sample minsamp trials w/ replacement from each dir.
            te_trials = {ki: rng.choice(vi.shape[0], vi.shape[0]//2, replace=False) for ki,vi in corr_eyedata.items()}
            tr_trials = {ki: np.setdiff1d(np.arange(vi.shape[0]), te_trials[ki]) for ki,vi in corr_eyedata.items()}

            #cbd = [vi[rng.choice(vi.shape[0], minsamp, replace=True)] for ki,vi in corr_eyedata.items()]
            cbd = [corr_eyedata[ki][vi[rng.choice(vi.shape[0], minsamp, replace=True)]] for ki,vi in tr_trials.items()]
            cbde = [corr_eyedata[ki][vi[rng.choice(vi.shape[0], minsamp, replace=True)]] for ki,vi in te_trials.items()]

            # True means
            cm = np.concatenate([vi.mean(0, keepdims=True) for vi in cbd], axis=0)
            cme = np.concatenate([vi.mean(0, keepdims=True) for vi in cbde], axis=0)
            for i in range(n_c):
                for j in range(i,n_c):

                    dif = np.sqrt(np.sum((cm[i] - cme[j])**2,axis=1))

                    if i == j:
                        #dif_ii = cm[i] - cme[j]
                        within_sep[n,:,i,j] = dif#np.sqrt(np.sum(dif_ii**2,axis=1))
                        continue


                    if split_by_cat:
                        #dif_ij = cm[i] - cm[j]
                        if cat_labs[i] == cat_labs[j]:
                            within_sep[n,:,i,j] = dif#np.sqrt(np.sum(dif_ij**2,axis=1))
                        else:
                            true_sep[n,:,i,j] = dif#np.sqrt(np.sum(dif_ij**2,axis=1))

                        continue

                    elif split_by_dist:
                        dd = np.abs(conds[i]+360 - conds[j])% 360
                        dd = np.minimum(dd, 360 - dd)
                        if dd < 180:
                            continue
                    #dif_ij = cm[i] - cm[j]
                    true_sep[n,:,i,j] = dif#np.sqrt(np.sum(dif_ij**2,axis=1))


            # Shuffle means: concatenate all data, shuffle, split
            '''
            all_cbd = np.concatenate(cbd, axis=0)
            rng.shuffle(all_cbd)
            cm = np.concatenate([all_cbd[i*minsamp:(i+1)*minsamp].mean(0, keepdims=True) for i in range(n_c)],axis=0)
            for i in range(n_c):
                for j in range(i, n_c):
                    if i == j:
                        continue
                    if split_by_cat:
                        if cat_labs[i] == cat_labs[j]:
                            continue
                    elif split_by_dist:
                        dd = np.abs(conds[i]+360 - conds[j])% 360
                        dd = np.minimum(dd, 360 - dd)
                        if dd < 180:
                            continue
                    dif_ij = cm[i] - cm[j]
                    shuf_sep[n,:,i,j] = np.sqrt(np.sum(dif_ij**2,axis=1))
            '''

        # Compute p values for difference via bootstrap
        mean_sep = np.nanmean(true_sep,(0,-2,-1))
        pv_t = np.nansum(np.nanmean(within_sep, (-2,-1)) >= mean_sep, axis=0)/nboot

        # Correct for false discovery rate
        pv_t = stats.false_discovery_control(pv_t)

        true_sep = np.nanmean(true_sep,(-2,-1))
        within_sep = np.nanmean(within_sep, (-2,-1))

        # Plot timecourse
        if time_rel_to == 'test':
            x = np.arange(-median_dur//resolution, 100//resolution)*resolution
        elif time_rel_to == 'sample':
            x = np.arange(-median_dur//resolution, 100//resolution)*resolution + median_dur - 500
        if samp_only:
            x = np.arange(-median_dur//resolution, 0)*resolution

        if return_raw:
            return true_sep, within_sep, x, pv_t

        fig, ax = plt.subplots(1, figsize=(3,2))

        #dif_tc = true_sep - within_sep
        #ax.plot(x, dif_tc.mean(0), color='dodgerblue', linewidth=2)

        '''
        ax.fill_between(x, 
                        dif_tc.mean(0) - dif_tc.std(0),
                        dif_tc.mean(0) + dif_tc.std(0),
                        color='dodgerblue',
                        alpha=0.1)
        '''
        ax.plot(x, true_sep.mean(0), color='dodgerblue', linewidth=2)
        ax.fill_between(x, 
                        true_sep.mean(0) - true_sep.std(0),
                        true_sep.mean(0) + true_sep.std(0),
                        color='dodgerblue',
                        alpha=0.1)
        ax.plot(x, within_sep.mean(0), color='grey', linewidth=2)
        ax.fill_between(x, 
                        within_sep.mean(0) - within_sep.std(0),
                        within_sep.mean(0) + within_sep.std(0),
                        color='grey',
                        alpha=0.1)

        ax = util.annotate_signif(x, pv_t, ax, color='dodgerblue')

        xlabel = f'ms from {time_rel_to} onset'
        ax.set(xlabel=xlabel, ylabel='Separation (dva)')
        ax.text(0, 1, 'BCD', ha='left', va='top',
            color='dodgerblue', fontsize=12, transform=ax.transAxes)
        ax.text(0, 0.9, 'WCD', ha='left', va='top',
            color='grey', fontsize=12, transform=ax.transAxes)
        s_on = {'sample': 0, 'test': -(median_sample + median_delay)}[time_rel_to]
        s_off = {'sample': median_sample, 'test': -median_delay}[time_rel_to]
        t_on = {'sample': median_sample + median_delay, 'test': 0}[time_rel_to]
        if not samp_only:
            ax.axvline(s_on, color='lightgrey', linestyle='--', zorder=-90, linewidth=1)
            ax.axvline(s_off, color='lightgrey', linestyle='--', zorder=-90, linewidth=1)
            ax.axvline(t_on, color='lightgrey', linestyle='--', zorder=-90, linewidth=1)

        sns.despine(fig, offset=5)
        plt.show()

        # Return values + indices for delay
        if not samp_only:
            delay_inds = range(-int((median_delay+100)//resolution), -int(100//resolution))
            sample_inds = range(-int(median_delay+median_sample+100)//resolution, 
                -1-int((median_delay+100)//resolution))
        else:
            delay_inds = None
            sample_inds = None

        return true_sep, within_sep, delay_inds, sample_inds, fig


    def compute_congruence_index(self,
        tasks=['DMC','RCDSTVGS'],
        tes=[0],
        resolution=20,
        nboot=1000,
        nshuf=1000):

        self.dataset.load()
    
        corr_eyedata = defaultdict(list)
        err_eyedata  = defaultdict(list)

        median_dur = None
        median_sample = None
        median_delay = None

        ct = 0

        for s in np.arange(self.dataset.get_n_sessions()):

            errcond = np.isin(self.dataset.trial_error[s], [6])
            if 'stimpy' in self.name or 'hobbes' in self.name or 'ivan' in self.name:
                errcond1 = (self.dataset.trial_error[s] == 3) & (self.dataset.match[s] == 1)
                errcond2 = (self.dataset.trial_error[s] == 7) & (self.dataset.match[s] == 0)
                errcond = errcond1 ^ errcond2

            # Obtain trials, correct and incorrect
            catcond = self.dataset.category[s] >= 0 if 'DMC' in tasks else True

            # Also: stratify by distance from boundary (if <=30 degrees, don't use)
            dirs = self.dataset.direction[s]
            bdd = np.array([self.dbounds[d] for d in dirs])
            dcond = bdd <= 30

            corr_t = np.where((np.isin(self.dataset.trial_type[s],tasks) > 0) & 
                             (self.dataset.trial_error[s] == 0) & 
                             dcond &
                             catcond)[0]

            err_t = np.where((np.isin(self.dataset.trial_type[s],tasks) > 0) & 
                             errcond & 
                             dcond &
                             catcond)[0]

            if len(corr_t) <= 1 or len(err_t) <= 1:
                continue

            ct += 1

            task = self.dataset.trial_type[s][corr_t][0]
            if task == 'DMCearly':
                continue

            # Select relevant periods
            corr_samp = self.dataset.times[f'{task}_sample_onset'][s][corr_t]
            corr_delay = self.dataset.times[f'{task}_sample_offset'][s][corr_t]
            err_samp = self.dataset.times[f'{task}_sample_onset'][s][err_t]
            err_delay = self.dataset.times[f'{task}_sample_offset'][s][err_t]

            
            corr_test = self.dataset.times[f'{task}_test_onset'][s][corr_t]
            err_test = self.dataset.times[f'{task}_test_onset'][s][err_t]

            # Compute median duration -- median of fixation+sample+delay
            if median_dur is None:
                median_sample = np.median(corr_delay - corr_samp)
                median_delay = np.median(corr_test - corr_delay)
                median_dur = median_sample + median_delay + 500 # 500 for initial fixation

            # Make relevant time-axis indices for each trial
            corr_rng = np.array([np.arange(ct - median_dur, ct + 100, dtype=np.int32) for ct in corr_test])
            err_rng = np.array([np.arange(et - median_dur, et + 100, dtype=np.int32) for et in err_test])

            ####################################################################
            # Extract eye data
            ex = self.dataset.eye_x[s][corr_t]
            ey = self.dataset.eye_y[s][corr_t]

            corr_eye = np.concatenate([ex[..., None],
                                      ey[..., None]], 
                                      axis=-1)

            ex = self.dataset.eye_x[s][err_t]
            ey = self.dataset.eye_y[s][err_t]

            err_eye = np.concatenate([ex[..., None],
                                      ey[..., None]], 
                                      axis=-1)

            ####################################################################
            # Set up task conditions for joining eye data
            dmc_conds = np.unique(self.dataset.direction[s][corr_t])

            ce = corr_eye[np.indices(corr_rng.shape)[0], corr_rng]
            ee = err_eye[np.indices(err_rng.shape)[0], err_rng]

            # Mean-subtract for all trials in the session
            #mean_ = np.concatenate((ce, ee), axis=0).mean((0,1), keepdims=True)
            ce -= ce.mean((0,1),keepdims=True)
            ee -= ee.mean((0,1),keepdims=True)

            # Assemble eye data
            for i, d in enumerate(dmc_conds):
                in_d = np.where(self.dataset.direction[s][corr_t] == d)[0]
                in_d_err = np.where(self.dataset.direction[s][err_t] == d)[0]

                if len(in_d) == 0:
                    continue

                corr_eyedata[d].append(ce[in_d,::resolution])
                err_eyedata[d].append(ee[in_d_err, ::resolution])

        self.dataset.stow()

        if ct == 0:
            return None

        ####################################################################
        # Compute bootstrapped estimate of mean -- across all sessions
        corr_eyedata = {ki: np.concatenate(v, axis=0).astype(np.float32) for ki,v in corr_eyedata.items()}
        err_eyedata = {ki: np.concatenate(v, axis=0).astype(np.float32) for ki,v in err_eyedata.items()}

        n_t = list(corr_eyedata.values())[0].shape[1]

        # Record arrays
        c0m_corr = np.zeros((nboot, n_t, 2))
        c1m_corr = np.zeros_like(c0m_corr)
        c0m_err = np.zeros_like(c0m_corr)
        c1m_err = np.zeros_like(c0m_corr)

        # Evaluation set for correct trials -- average correct trace 
        c0m_corr_eval = np.zeros_like(c0m_corr)
        c1m_corr_eval = np.zeros_like(c0m_corr)

        rng = np.random.default_rng(10)

        # When selecting category labels, use only 
        # directions sufficiently far from the boundary
        clg0 = np.where(self.cat_labels >= 0)[0]
        ffb = np.where(np.array(list(self.dbounds.values())) > 0)[0]
        rel_d = np.intersect1d(clg0,ffb)
        cat_labs = self.cat_labels[rel_d]
        
        minsamp = 200
        for n in range(nboot):

            # Sample minsamp trials w/ replacement from each dir.
            fit_inds = {ki: rng.choice(vi.shape[0], vi.shape[0]//2, replace=False) for ki,vi in corr_eyedata.items()}
            eval_inds = {ki: np.setdiff1d(np.arange(vi.shape[0]), fit_inds[ki]) for ki,vi in corr_eyedata.items()}

            cbd = [vi[rng.choice(fit_inds[ki], minsamp, replace=True)] for ki,vi in corr_eyedata.items()]
            cbde = [vi[rng.choice(eval_inds[ki], minsamp, replace=True)] for ki, vi in corr_eyedata.items()]

            # For error trials: sample with replacement from pool by category rather than by direction,
            # where the numbers get iffy
            ebc0 = np.concatenate([vi for i,vi in enumerate(err_eyedata.values()) if i in np.where(cat_labs == 0)[0]], axis=0)
            ebc1 = np.concatenate([vi for i,vi in enumerate(err_eyedata.values()) if i in np.where(cat_labs == 1)[0]], axis=0)
            
            # True means
            if n == 0:
                print(len(cbd), cat_labs, np.where(cat_labs == 0)[0])
            c0 = np.concatenate([cbd[vi] for vi in np.where(cat_labs == 0)[0]],axis=0)
            c1 = np.concatenate([cbd[vi] for vi in np.where(cat_labs == 1)[0]],axis=0)
            c0m_corr[n] = c0.mean(0)
            c1m_corr[n] = c1.mean(0)

            # Error trial means
            c0e = ebc0[rng.choice(ebc0.shape[0], minsamp, replace=True)]
            c1e = ebc1[rng.choice(ebc1.shape[0], minsamp, replace=True)]
            c0m_err[n] = c0e.mean(0)
            c1m_err[n] = c1e.mean(0)

            # Save evaluation traces
            c0m_corr_eval[n] = np.concatenate([cbde[vi] for vi in np.where(cat_labs == 0)[0]],axis=0).mean(0)
            c1m_corr_eval[n] = np.concatenate([cbde[vi] for vi in np.where(cat_labs == 1)[0]],axis=0).mean(0)

        # Return values + indices for delay
        delay_inds = range(-int((median_delay+100)//resolution), -int(100//resolution))
        sample_delay_inds = range(-int(median_delay+median_sample+100)//resolution, -int(100//resolution))
        sample_inds = range(-c0m_err.shape[1], 
            -int(median_delay+100)//resolution)

        # Plot mean traces in 2D for correct/error trials of each category
        fig, ax = plt.subplots(1)
        ax.plot(*c0m_corr.mean(0)[delay_inds].T, color=self.cat_pal[0], alpha=1, linewidth=2)
        ax.plot(*c1m_corr.mean(0)[delay_inds].T, color=self.cat_pal[1], alpha=1, linewidth=2)
        ax.plot(*c0m_err.mean(0)[delay_inds].T, color=self.cat_pal[0], alpha=0.5, linewidth=2, linestyle='--')
        ax.plot(*c1m_err.mean(0)[delay_inds].T, color=self.cat_pal[1], alpha=0.5, linewidth=2, linestyle='--')
        sns.despine(fig, offset=5)
        ax.text(0.09, 1, 'c0', ha='right', va='top', color=self.cat_pal[0], transform=ax.transAxes)
        ax.text(0.09, 0.9, 'c1', ha='right', va='top', color=self.cat_pal[1], transform=ax.transAxes)
        ax.text(0.1, 1, 'corr.', ha='left', va='top', alpha=1, color='black', transform=ax.transAxes)
        ax.text(0.1, 0.9, 'err.', ha='left', va='top', alpha=0.5, color='black', transform=ax.transAxes)
        ax.set(xlabel='Azim. (dva)', ylabel='Elev. (dva)')
        plt.show()

        # Plot timecourse of proximity index for error trials in each category
        # (distance to right category - distance to wrong category over their sum --
        # when closer to the wrong category, index closer to 1; when closer to the 
        # right category, index closer to -1; when equivalent, index = 0)
        c0e_c0c = np.linalg.norm(c0m_err - c0m_corr, axis=-1)
        c0e_c1c = np.linalg.norm(c0m_err - c1m_corr, axis=-1)
        c1e_c0c = np.linalg.norm(c1m_err - c0m_corr, axis=-1)
        c1e_c1c = np.linalg.norm(c1m_err - c1m_corr, axis=-1)
        c0_pi = (c0e_c1c - c0e_c0c) / (c0e_c1c + c0e_c0c)
        c1_pi = (c1e_c0c - c1e_c1c) / (c1e_c0c + c1e_c1c)

        # Same, but for evaluation trials
        c0e_c0c_eval = np.linalg.norm(c0m_corr_eval - c0m_corr, axis=-1)
        c0e_c1c_eval = np.linalg.norm(c0m_corr_eval - c1m_corr, axis=-1)
        c1e_c0c_eval = np.linalg.norm(c1m_corr_eval - c0m_corr, axis=-1)
        c1e_c1c_eval = np.linalg.norm(c1m_corr_eval - c1m_corr, axis=-1)
        c0_pi_eval = (c0e_c1c_eval - c0e_c0c_eval) / (c0e_c1c_eval + c0e_c0c_eval)
        c1_pi_eval = (c1e_c0c_eval - c1e_c1c_eval) / (c1e_c0c_eval + c1e_c1c_eval)

        fig1, ax1 = plt.subplots(1)
        x = np.arange(-500//resolution, (median_dur - 500 + 100)//resolution + 1)*resolution
        mean_pi = np.concatenate((c0_pi[...,None], c1_pi[...,None]),axis=-1).mean(-1)
        mean_pi_eval = np.concatenate((c0_pi_eval[...,None], c1_pi_eval[...,None]),axis=-1).mean(-1)
        ax1.plot(x, mean_pi.mean(0), color='black')
        ax1.fill_between(x, mean_pi.mean(0) - mean_pi.std(0), 
            mean_pi.mean(0) + mean_pi.std(0),
            color='black', alpha=0.1)
        ax1.plot(x, mean_pi_eval.mean(0), color='grey')
        ax1.fill_between(x, mean_pi_eval.mean(0) - mean_pi_eval.std(0), 
            mean_pi_eval.mean(0) + mean_pi_eval.std(0),
            color='grey', alpha=0.1)
        ax1.set(ylim=[-1,1], xlabel='ms from sample onset', ylabel='Congr. ind. (a.u.)')
        ax1.axvline(median_sample + median_delay, color='lightgrey', linestyle='--', zorder=-90, linewidth=1)
        ax1.axvline(median_sample, color='lightgrey', linestyle='--', zorder=-90, linewidth=1)
        ax1.axvline(0, color='lightgrey', linestyle='--', zorder=-90, linewidth=1)
        ax1.axhline(0, color='grey', linewidth=1, zorder=-90)
        ax1.text(0,1,'error',color='black',ha='left',va='top',transform=ax1.transAxes)
        ax1.text(0,0.9,'correct',color='grey',ha='left',va='top',transform=ax1.transAxes)
        sns.despine(fig1, offset=5)
        plt.show()

        return c0_pi.mean(0), c1_pi.mean(0), c0_pi_eval.mean(0), c1_pi_eval.mean(0), fig, fig1




    def test_cs_analysis(self, areas, nboot=1000, resolution=50, 
        test_win=250, alpha=0.05):

        X = defaultdict(list)
        y = defaultdict(list)
        eye = defaultdict(list)
        RT = defaultdict(list)

        n_by_session = defaultdict(int)

        self.dataset.load()

        median_dur  = defaultdict(lambda: None)
        sample_durs = defaultdict(lambda: None)
        delay_durs  = defaultdict(lambda: None)
        test_durs   = defaultdict(lambda: None)

        sess_to_do = defaultdict(list)

        # Load activity, eye movements
        for s in range(self.dataset.get_n_sessions()):

            if 'DMC' not in np.unique(self.dataset.trial_type[s]):
                continue

            # Load DMC trials
            rel_t = np.where((self.dataset.trial_type[s] == "DMC") & 
                             (self.dataset.trial_error[s] == 0) & 
                             (self.dataset.category[s] >= 0) & 
                             (self.dataset.testcategory[s] >= 0) & 
                             (np.isin(self.dataset.areas[s], areas) > 0) & 
                             (self.dataset.times['DMC_test_onset'][s] + test_win < self.dataset.spikes[s].shape[1]))[0]

            area = self.dataset.areas[s][0]
            if len(rel_t) < 50:
                continue

            spk = self.dataset.spikes[s][rel_t]
            cat = self.dataset.category[s][rel_t]
            tcat = self.dataset.testcategory[s][rel_t]
            dirs = self.dataset.direction[s][rel_t]

            s_onset = self.dataset.times['DMC_sample_onset'][s][rel_t]
            s_offset = self.dataset.times['DMC_sample_offset'][s][rel_t]
            t_onset = self.dataset.times['DMC_test_onset'][s][rel_t]
            if median_dur[area] is None:
                sample_durs[area] = np.median(s_offset - s_onset)
                delay_durs[area] = np.median(t_onset - s_offset)
                median_dur[area] = np.median(t_onset - s_onset) + 500


            t_rng = np.array([np.arange(to-median_dur[area], to+test_win, dtype=np.int32) for to in t_onset])
            spk = spk[np.indices(t_rng.shape)[0], t_rng].astype(np.float32)
            spk_smooth = ndimage.gaussian_filter1d(spk, 20, axis=1) 
            ex = self.dataset.eye_x[s][rel_t]
            ey = self.dataset.eye_y[s][rel_t]
            eye_ = np.concatenate((ex[...,None], ey[...,None]), axis=-1)
            eye_rng = np.array([np.arange(to-median_dur[area]+500, to, dtype=np.int32) for to in t_onset])
            eye_ = eye_[np.indices(eye_rng.shape)[0], eye_rng]

            rel_dirs = np.unique(np.concatenate([np.concatenate(x) for x in self.dicd_splits]).flatten())

            for n in range(spk.shape[-1]):
                rel_n = np.where(spk[...,n].min(1) >= 0)[0]
                if len(rel_n) < 10:
                    continue

                #if np.mean(spk[rel_n,:,n].sum(1)) < 3 * (spk.shape[1]/1000):
                #    continue

                if min([len(np.where(dirs[rel_n] == i)[0]) for i in np.unique(rel_dirs)]) < 10:
                    continue

                # If including any unit - mark this as a session to use in computing 
                # latency of category-correlated eye movements
                if s not in sess_to_do[area]:
                    sess_to_do[area].append(s)

                # Store spiking, eye movement data
                n_by_session[len(X[area])] = s
                X[area].append(spk_smooth[rel_n,::resolution,n])
                y[area].append(np.column_stack((dirs[rel_n],cat[rel_n])))
                eye[area].append(eye_[rel_n,::resolution])

        self.dataset.stow()

        # If no data for this animal, return None
        if len(X.keys()) == 0:
            return None

        ########################################################################
        # Set up records for analysis, for each area
        al    = defaultdict(dict) # axis length b/w cat. centers in eye movements
        by_ep = defaultdict(dict) # DICD by epoch

        # Assess whether unit encodes sample category during test period
        cba = {'mt': 'red', 'mst': 'orange', 'lip': 'goldenrod', 'pfc': 'green', 'sc': 'blue'}
        for k,v in X.items():

            print(k)

            X_close, X_far = [], []
            y_close, y_far = [], []

            axlen_close  = {}
            axlen_far    = {}

            RT_sep_corrs = []

            for s in range(len(v)):

                eye_s = eye[k][s]

                c0_s = np.where(y[k][s][:,1] == 0)[0]
                c1_s = np.where(y[k][s][:,1] == 1)[0]

                eye_ax = eye_s[c0_s].mean(0) - eye_s[c1_s].mean(0)
                max_norm = np.argmax(np.linalg.norm(eye_ax, axis=-1))

                #by_trial = eye_s[:,max_norm] @ eye_ax[max_norm]

                # Obtain projections of each trial onto this axis
                by_trial = np.einsum('ijk,jk->ij', eye_s, eye_ax).mean(1)

                zbt = stats.zscore(by_trial)


                # Take top half vs. bottom half by eye separation 
                # separately for each sample direction
                close = []
                far = []
                for sd in np.unique(y[k][s][:,0]):
                    in_d = np.where(y[k][s][:,0] == sd)[0]
                    cat = y[k][s][in_d[0],1]

                    sorted_proj = in_d[np.argsort(by_trial[in_d])]

                    if cat == 0:
                        close.append(sorted_proj[:int(len(in_d)//2)])
                        far.append(sorted_proj[-int(len(in_d)//2):])
                    elif cat == 1:
                        far.append(sorted_proj[:int(len(in_d)//2)])
                        close.append(sorted_proj[-int(len(in_d)//2):])

                close = np.concatenate(close)
                far   = np.concatenate(far)

                # For first unit included - plot eye movements in MP, LP,
                # colored by category
                s_thresh = 0
                if len(v) > 100:
                    s_thresh = 100
                if s == s_thresh:
                    f_0, ax = plt.subplots(nrows=1,ncols=2,figsize=(2,1),sharex=True,sharey=True)
                    c0_close = np.where(y[k][s][close,1] == 0)[0]
                    c1_close = np.where(y[k][s][close,1] == 1)[0]
                    c0_far   = np.where(y[k][s][far,1] == 0)[0]
                    c1_far   = np.where(y[k][s][far,1] == 1)[0]

                    ax[0].scatter(*eye_s[close][c0_close].mean(1).T, color=self.cat_pal[0],
                        marker='.', alpha=0.50)
                    ax[0].scatter(*eye_s[close][c1_close].mean(1).T, color=self.cat_pal[1],
                        marker='.', alpha=0.5)
                    ax[1].scatter(*eye_s[far][c0_far].mean(1).T, color=self.cat_pal[0],
                        marker='.', alpha=0.5)
                    ax[1].scatter(*eye_s[far][c1_far].mean(1).T, color=self.cat_pal[1],
                        marker='.', alpha=0.5)
                    plt.axis('square')
                    sns.despine(f_0)
                    ax[1].axis('off')
                    ax[1].text(0, 1.05, 'c0', fontsize=10, color=self.cat_pal[0],
                        ha='right', va='top', transform=ax[1].transAxes)
                    ax[1].text(0, 0.9, 'c1', fontsize=10, color=self.cat_pal[1],
                        ha='right', va='top', transform=ax[1].transAxes)
                    ax[0].set(xlabel='Azim. (dva)', ylabel='Elev. (dva)')
                    plt.show()

                # Compute length of cat axis on close vs far trials
                close_ax = eye_s[close][np.where(y[k][s][close,1] == 0)[0]].mean(0) - eye_s[close][np.where(y[k][s][close,1] == 1)[0]].mean(0)
                far_ax = eye_s[far][np.where(y[k][s][far,1] == 0)[0]].mean(0) - eye_s[far][np.where(y[k][s][far,1] == 1)[0]].mean(0)
                
                if n_by_session[s] not in axlen_close.keys():
                    axlen_close[n_by_session[s]] = np.linalg.norm(close_ax[max_norm])
                    axlen_far[n_by_session[s]] = np.linalg.norm(far_ax[max_norm])
                

                y_close.append(y[k][s][close])
                y_far.append(y[k][s][far])

                # Store activity on close/far trials
                X_close.append(v[s][close])
                X_far.append(v[s][far])

            # Generate epochs to structure results to return
            f  = 500 
            s  = f + sample_durs[k]
            de = s + delay_durs[k]//2
            dl = de + delay_durs[k]//2
            t  = dl + test_win
            epochs = {'fix'   : [0, int(f//resolution)],
                      'sample': [int(f//resolution), int(s//resolution)],
                      'delay (early)' : [int(s//resolution), int(de//resolution)],
                      'delay (late)' : [int(de//resolution), int(dl//resolution)],
                      'test'  : [int(dl//resolution), int(t//resolution)]}

            results_far = generate_results_pp(X_far, y_far, 
                nboot, len(X_far), self.dicd_splits)
            results_close = generate_results_pp(X_close, y_close, 
                nboot, len(X_close), self.dicd_splits)
            scores_close, scores_far = [], []
            for sp in range(len(self.dicd_splits)):
                scores_close.append(np.array([results_close[(sp,n)] for n in range(nboot)]))
                scores_far.append(np.array([results_far[(sp,n)] for n in range(nboot)]))
  
            scores_close = np.array(scores_close).mean(0)
            scores_far   = np.array(scores_far).mean(0)

            # Bind together eye metrics
            axlen_close  = np.array(list(axlen_close.values()))
            axlen_far    = np.array(list(axlen_far.values()))
            al[k]['close'] = axlen_close
            al[k]['far']   = axlen_far

            # Compute significance of difference b/w MP, LP
            # (two-sided) 
            sig = np.sum(
                np.abs(scores_close - scores_close.mean(0, keepdims=True)) >= 
                    np.abs(scores_far.mean(0) - scores_close.mean(0)), axis=0) / nboot

            sig = stats.false_discovery_control(sig)

            # Plot results
            f_1, ax = plt.subplots(1)
            x = np.arange(scores_close.shape[1])*resolution - 500
            ax.axhline(0.5, color='grey', linestyle='--', linewidth=1, zorder=-90)
            ax.plot(x, scores_close.mean(0), color='black')
            ax.fill_between(x, scores_close.mean(0) - scores_close.std(0),
                scores_close.mean(0) + scores_close.std(0), color='black',
                alpha=0.1)
            ax.plot(x, scores_far.mean(0), color=cba[k])
            ax.fill_between(x, scores_far.mean(0) - scores_far.std(0),
                scores_far.mean(0) + scores_far.std(0), color=cba[k],
                alpha=0.1)
            sns.despine(f_1, offset=5)
            ax.set(ylim=[0.2, 1], xlabel='ms from sample onset',
                ylabel='Decoding acc.')
            ax.text(0, 1, 'MP', ha='left', va='top',
                transform=ax.transAxes, color=cba[k], fontsize=10)
            ax.text(0, 0.9, 'LP', ha='left', va='top',
                transform=ax.transAxes, color='black', fontsize=10)
            ax.axvline(0, color='lightgrey', linestyle='--', linewidth=1, zorder=-90)
            ax.axvline(s - 500, color='lightgrey', linestyle='--', linewidth=1, zorder=-90)
            ax.axvline(dl - 500, color='lightgrey', linestyle='--',linewidth=1, zorder=-90)

            # Annotate w/ significance
            ax = util.annotate_signif(x, sig, ax, color=cba[k])
            plt.show()

            # Average results by epoch and plot
            close_by_ep, far_by_ep = {}, {}
            for ep, v in epochs.items():
                close_by_ep[ep] = scores_close[:,v[0]:v[-1]].mean(1)
                far_by_ep[ep]   = scores_far[:,v[0]:v[-1]].mean(1)

            f_2, ax = plt.subplots(1, figsize=(3.5,2))
            for j, ep in enumerate(epochs.keys()):
                ax.errorbar(j - 0.1, close_by_ep[ep].mean(0), yerr=close_by_ep[ep].std(0),
                    color='black', linewidth=0, marker='.', elinewidth=1)
                ax.errorbar(j + 0.1, far_by_ep[ep].mean(0), yerr=far_by_ep[ep].std(0),
                    color=cba[k], linewidth=0, marker='.', elinewidth=1)

            sns.despine(f_2, offset=5)
            ax.set(ylim=[0.2, 1], xticks=np.arange(len(epochs.keys())),
                xticklabels=['Fix', 'Sample', f'Delay\n(early)', f'Delay\n(late)', 'Test'],
                ylabel='Decoding acc.', xlabel='Epoch')
            ax.axhline(0.5, color='lightgrey', linewidth=1, linestyle='--', zorder=-90)
            plt.show()

            by_ep[k]['close'] = close_by_ep
            by_ep[k]['far']   = far_by_ep


        return by_ep, al, (f_0, f_1, f_2)

################################################################################
# Parallelized classification methods
# 
# 1. Classifying category from activity in MP/LP 
@concurrent(processes=10)
def classify_fn_pp(X, y, n, splits, split, n_feat, n_pseudotri):

    pl = make_pipeline(SVC(kernel='linear',class_weight='balanced'))
    rng = np.random.default_rng(n)

    n_t = X[0].shape[1]

    # Sample n_pseudotri trials w/ replacement from
    # each unique condition for each unit
    if len(y[0].shape) < 2:
        conds = np.unique(y[0][:,None], axis=0)
    else:
        conds = np.unique(y[0], axis=0)

    n_time = n_t

    ########################################################################
    # Preallocate X/y
    trd, ted = splits[split]
    n_ptri_tr = n_pseudotri*len(trd)
    n_ptri_te = n_pseudotri*len(ted)


    X_tr = np.zeros((n_ptri_tr, n_time, n_feat))
    y_tr = np.zeros((n_ptri_tr,))
    X_te = np.zeros((n_ptri_te, n_time, n_feat))
    y_te = np.zeros((n_ptri_te,))

    # Bootstrap over units to include
    X_to_inc = rng.choice(np.arange(n_feat), n_feat, replace=True)
    rng.shuffle(X_to_inc)

    ########################################################################
    # Loop through units, conditions
    for k, i in enumerate(X_to_inc):

        # Loop through training directions
        for j, trj in enumerate(trd):

            rel = np.where(y[i][:,0] == trj)[0]
            trt = rng.choice(rel, n_pseudotri, replace=True)

            X_tr[j*n_pseudotri:(j+1)*n_pseudotri, :, k] = X[i][trt]
            y_tr[j*n_pseudotri:(j+1)*n_pseudotri] = y[i][trt,1]

        # Loop through testing directions
        for j, tej in enumerate(ted):

            rel = np.where(y[i][:,0] == tej)[0]
            tet = rng.choice(rel, n_pseudotri, replace=True)

            X_te[j*n_pseudotri:(j+1)*n_pseudotri, :, k] = X[i][tet]
            y_te[j*n_pseudotri:(j+1)*n_pseudotri] = y[i][tet,1]


    # Classify category from inputs
    return [pl.fit(X_tr[:,t], y_tr).score(X_te[:,t], y_te) for t in range(n_t)]

@synchronized
def generate_results_pp(X, y, nboot, n_feat, splits, n_pseudotri=10):

    allres = {}

    for s in range(len(splits)):
        for n in range(nboot):
            allres[(s,n)] = classify_fn_p(X, y, n, splits, s, 
                n_feat, n_pseudotri)
    
    return allres

################################################################################
# 2. Classifying category from eye position data (static/timecourse)
@concurrent(processes=10)
def classify_fn_eyecat(corr_eyedata, 
    n, n_t, labels, ctfctl, do_shuf, do_static):

    pl = make_pipeline(SVC(kernel='linear'))
    rng = np.random.default_rng(n)

    # Split train/test halves
    te_trials = {ki: rng.choice(vi.shape[0], vi.shape[0]//2, replace=False) for ki,vi in corr_eyedata.items()}
    tr_trials = {ki: np.setdiff1d(np.arange(vi.shape[0]), te_trials[ki]) for ki,vi in corr_eyedata.items()}

    X_tr = np.concatenate([vi[tr_trials[ki]] for ki,vi in corr_eyedata.items()],axis=0)
    X_te = np.concatenate([vi[te_trials[ki]] for ki,vi in corr_eyedata.items()],axis=0)

    if do_static:
        X_tr = X_tr.reshape((X_tr.shape[0], -1))
        X_te = X_te.reshape((X_te.shape[0], -1))

    results = []

    
    # Do true/counterfactual boundaries
    for boundary in range(2):

        # True boundary
        if boundary == 0:
            labels_tr = np.concatenate([np.repeat(labels[i], len(v)) for i,v in enumerate(tr_trials.values())])
            labels_te = np.concatenate([np.repeat(labels[i], len(v)) for i,v in enumerate(te_trials.values())])
        else:
            labels_ = np.concatenate((labels[ctfctl:], labels[:ctfctl]))
            labels_tr = np.concatenate([np.repeat(labels_[i], len(v)) for i,v in enumerate(tr_trials.values())])
            labels_te = np.concatenate([np.repeat(labels_[i], len(v)) for i,v in enumerate(te_trials.values())])

        # Subsample test set to be of equal category proportions
        order = np.arange(X_te.shape[0])
        rng.shuffle(order)
        X_te = X_te[order]
        labels_te = labels_te[order]
        min_by_c = min([sum(labels_te == 0), sum(labels_te == 1)])
        rel_0 = np.where(labels_te == 0)[0][:min_by_c]
        rel_1 = np.where(labels_te == 1)[0][:min_by_c]
        rel = np.concatenate((rel_0, rel_1))

        # Same for training
        order = np.arange(X_tr.shape[0])
        rng.shuffle(order)
        X_tr = X_tr[order]
        labels_tr = labels_tr[order]
        min_by_c = min([sum(labels_tr == 0), sum(labels_tr == 1)])
        rel_0 = np.where(labels_tr == 0)[0][:min_by_c]
        rel_1 = np.where(labels_tr == 1)[0][:min_by_c]
        rel_tr = np.concatenate((rel_0, rel_1))

        if n_t > 1:
            results_ = []
            for t in range(n_t):
                pl.fit(X_tr[rel_tr, t], labels_tr[rel_tr])
                results_.append(pl.score(X_te[rel,t],labels_te[rel]))
            results.append(results_)
        else:
            pl.fit(X_tr[rel_tr], labels_tr[rel_tr])
            results.append([pl.score(X_te[rel], labels_te[rel])])


    # Do label-shuffle control
    if do_shuf:

        labels_tr = np.concatenate([np.repeat(labels[i], len(v)) for i,v in enumerate(tr_trials.values())])
        labels_te = np.concatenate([np.repeat(labels[i], len(v)) for i,v in enumerate(te_trials.values())])

        rng.shuffle(labels_tr)
        rng.shuffle(labels_te)

        # Subsample test set to be of equal category proportions
        order = np.arange(X_te.shape[0])
        rng.shuffle(order)
        X_te = X_te[order]
        labels_te = labels_te[order]
        min_by_c = min([sum(labels_te == 0), sum(labels_te == 1)])
        rel_0 = np.where(labels_te == 0)[0][:min_by_c]
        rel_1 = np.where(labels_te == 1)[0][:min_by_c]
        rel = np.concatenate((rel_0, rel_1))

        # Same for training
        order = np.arange(X_tr.shape[0])
        rng.shuffle(order)
        X_tr = X_tr[order]
        labels_tr = labels_tr[order]
        min_by_c = min([sum(labels_tr == 0), sum(labels_tr == 1)])
        rel_0 = np.where(labels_tr == 0)[0][:min_by_c]
        rel_1 = np.where(labels_tr == 1)[0][:min_by_c]
        rel_tr = np.concatenate((rel_0, rel_1))

        if n_t > 1:
            results_ = []
            for t in range(n_t):
                pl.fit(X_tr[rel_tr, t], labels_tr[rel_tr])
                results_.append(pl.score(X_te[rel,t],labels_te[rel]))
            results.append(results_)
        else:
            pl.fit(X_tr[rel_tr], labels_tr[rel_tr])
            results.append([pl.score(X_te[rel], labels_te[rel])])

    results = np.array(results)

    return results

@synchronized
def generate_results_eyecat(eyedata_by_sess, 
    nboot, n_t, labels, ctfctl, do_shuf=False, do_static=False):

    allres = {}

    # Loop through sessions, classify from eye positions, activity in each session
    for s, k in enumerate(eyedata_by_sess.keys()):
        corr_eyedata = eyedata_by_sess[k]

        # Bootstrapped classification procedure: 
        for n in range(nboot):
            allres[(s,n)] = classify_fn_eyecat(corr_eyedata,
                n, n_t, labels, ctfctl, do_shuf, do_static)

    return allres

################################################################################
# 3. Classifying category from eye position data during inactivation
@concurrent(processes=10)
def classify_fn_eyecat_inact(X, y, n, do_shuffle):

    pl = make_pipeline(SVC(kernel='linear'))
    rng = np.random.default_rng(n)

    # Split train/test halves
    min_t = min([sum(y == i) for i in [0,1]])
    y0 = np.where(y == 0)[0]
    y1 = np.where(y == 1)[0]
    tr_0 = rng.choice(y0, min_t//2, replace=False)
    tr_1 = rng.choice(y1, min_t//2, replace=False)

    te_0 = rng.choice(np.setdiff1d(y0, tr_0), min_t//2, replace=False)
    te_1 = rng.choice(np.setdiff1d(y1, tr_1), min_t//2, replace=False)

    tr = np.concatenate((tr_0, tr_1))
    te = np.concatenate((te_0, te_1))

    # Reshape data
    X_tr = X[tr].reshape((len(tr), -1))
    X_te = X[te].reshape((len(te), -1))

    y_tr = y[tr]
    y_te = y[te]

    if do_shuffle:
        rng.shuffle(y_tr)
        rng.shuffle(y_te)

    # Classify
    return pl.fit(X_tr, y_tr).score(X_te, y_tr)

@synchronized
def generate_results_eyecat_inact(X, y, nboot, do_shuffle=False):

    allres = {}
    # Bootstrapped classification procedure: 
    for n in range(nboot):
        allres[n] = classify_fn_eyecat_inact(X, y, n, do_shuffle)

    return allres

################################################################################
# 4. Classifying sample direction dichotomies from DMS data
@concurrent(processes=10)
def classify_fn_eyecat_DMS(corr_eyedata, n, n_t, labels, do_static):

    pl = make_pipeline(SVC(kernel='linear'))
    rng = np.random.default_rng(n)

    # Split train/test halves
    te_trials = {ki: rng.choice(vi.shape[0], vi.shape[0]//2, replace=False) for ki,vi in corr_eyedata.items()}
    tr_trials = {ki: np.setdiff1d(np.arange(vi.shape[0]), te_trials[ki]) for ki,vi in corr_eyedata.items()}

    X_tr = np.concatenate([vi[tr_trials[ki]] for ki,vi in corr_eyedata.items()],axis=0)
    X_te = np.concatenate([vi[te_trials[ki]] for ki,vi in corr_eyedata.items()],axis=0)

    if do_static:
        X_tr = X_tr.reshape((X_tr.shape[0], -1))
        X_te = X_te.reshape((X_te.shape[0], -1))



    results = np.zeros((2, len(labels)//2, n_t))

    for i in range(2):

        for j in range(results.shape[1]): # True, shuffle

            lab_inds = np.arange(len(labels))
            labels_ = np.concatenate((lab_inds[j+1:], lab_inds[:j+1]))
            labels_tr = np.concatenate([np.repeat(labels_[i], len(v)) for i,v in enumerate(tr_trials.values())])
            labels_te = np.concatenate([np.repeat(labels_[i], len(v)) for i,v in enumerate(te_trials.values())])

            # Make binary
            labels_tr = labels_tr//int(len(labels)//2)
            labels_te = labels_te//int(len(labels)//2)
            
            if i == 1:
                rng.shuffle(labels_tr)
                rng.shuffle(labels_te)

            # Subsample test set to be of equal category proportions
            order = np.arange(X_te.shape[0])
            rng.shuffle(order)
            X_te = X_te[order]
            labels_te = labels_te[order]
            min_by_c = min([sum(labels_te == x) for x in np.unique(labels_te)])
            rel_by_c = [np.where(labels_te == x)[0][:min_by_c] for x in np.unique(labels_te)]
            rel = np.concatenate(rel_by_c)

            # Same for training
            order = np.arange(X_tr.shape[0])
            rng.shuffle(order)
            X_tr = X_tr[order]
            labels_tr = labels_tr[order]
            min_by_c = min([sum(labels_tr == x) for x in np.unique(labels_tr)])
            rel_by_c = [np.where(labels_tr == x)[0][:min_by_c] for x in np.unique(labels_tr)]
            rel_tr = np.concatenate(rel_by_c)

            if n_t > 1:
                results[i,j] = [pl.fit(X_tr[rel_tr,t], labels_tr[rel_tr]).score(X_te[rel,t], labels_te[rel]) for t in range(n_t)]
            else:
                results[i,j] = [pl.fit(X_tr[rel_tr], labels_tr[rel_tr]).score(X_te[rel], labels_te[rel])]

    return results.mean(1)

@synchronized
def generate_results_eyecat_DMS(eyedata_by_sess, nboot, n_t, labels, 
    do_static=False):

    allres = {}

    # Loop through sessions, classify from eye positions, activity in each session
    for s, k in enumerate(eyedata_by_sess.keys()):
        corr_eyedata = eyedata_by_sess[k]

        # Bootstrapped classification procedure: 
        for n in range(nboot):
            allres[(s,n)] = classify_fn_eyecat_DMS(corr_eyedata, n, n_t, labels, do_static)

    return allres

