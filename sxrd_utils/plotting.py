import numpy as np

from sxrd_utils.experiment import SXRDExperiment
from sxrd_utils.ctr import CTR
from sxrd_utils.scan import SXRDScan

import matplotlib.pyplot as plt

def plot_sxrd(experiments, semilog=True, plot_kwargs=None, fig_size_factor=5):
    # if experiments is a single experiment, make it into a tuple
    if isinstance(experiments, SXRDExperiment):
        experiments = (experiments, )
    h_max = max(exp.max_hk[0] for exp in experiments)
    k_max = max(exp.max_hk[1] for exp in experiments)

    figure = plt.figure(figsize=(fig_size_factor*h_max, fig_size_factor*k_max))
    grid_space = figure.add_gridspec(h_max+1, k_max+1, hspace=0, wspace=0)
    axes = grid_space.subplots(sharex="col", sharey="row")

    # iterate over all (h, k)
    for h in range(0, h_max+1):
        for k in range(0, k_max+1):
            hk = (h, k)
            axis = axes[h, k]

            # set title
            axis.set_title(str(hk))

            # plot the rod
            plot_rod_onto_axis(hk, axis, experiments,
                               semilog=semilog,
                               plot_kwargs=plot_kwargs)

    return figure

def plot_rod_onto_axis(hk, axis, experiments, semilog=True, plot_kwargs=None):
    if plot_kwargs is None:
        plot_kwargs =[{},]*len(experiments)
    for exp, kwargs in zip(experiments, plot_kwargs):
        if hk not in exp.ctrs:
            # skip if we don't have a fit for this hk
            continue
        for fit in exp.ctrs[hk].fits:
            print(kwargs)
            axis.plot(fit.l_values, fit.values, **kwargs)
    if any("label" in kwargs.keys() for kwargs in plot_kwargs):
        axis.legend()
    if semilog:
        axis.set_yscale("log")
