################################################################################
# util.py
# 
# Author     : Matt Rosen
# Modified   : 8/2024
# Description: Utility functions for plotting etc.
################################################################################
import matplotlib.pyplot as plt 
import numpy as np
from scipy import stats

# Annotate w/ moments of significance, determined via bootstrap
def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def annotate_signif(x, y, ax, color='black', height=None):

    xdif = (x[1] - x[0])/2
    if height is None:
        height = ax.get_ylim()[1]

    # Annotate provided axis object w/ moments of significance
    for i in range(len(x)):
        lw = 0
        if y[i] <= 0.05:
            lw = 0.5
        if y[i] <= 0.01:
            lw = 1
        if y[i] <= 0.001:
            lw = 2
        ax.plot((x[i]-xdif, x[i]+xdif), 
                (height, height),
                color=color,
                lw=lw,
                clip_on=False)

    return ax