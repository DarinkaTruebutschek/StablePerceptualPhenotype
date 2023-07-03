#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 15:55:37 2022

@author: darinka

#Purpose: All functions necessary to perform important control analyses.
#Author: Darinka Truebutschek
#Date created: 5/12/2022
#Date last modified: 5/12/2022
#Python version: 3.7.1

"""

def plot_quickScatter(x, y, my_title, col, markers, sizes, xticks, xticklabels, xlabel,
            yticks, yticklabels, ylabel, ylim_bottom, ylim_top,
            axisBreak, figsize, factor_x, factor_y, ax, leg_labels):
    """
    :param x: number of bars to plot
    :param y: height of individual bars
    :param my_title: title for the entire plot
    :param col: color to be used for plotting
    :param markers: which markers to use
    :param sizes: which marker sizes to use
    :param xticks: xticks to be used
    :param xticklabels: xticklabels to be used (if None, xticks == xticklabels)
    :param xlabel: label to be used for x-axis
    :param yticks: yticks to be used
    :param yticklabels: yticklabels to be used (if None, yticks == yticklabels)
    :param ylabel: label to be used for y-axis  
    :param ylim_bottom: where to end lower axis in case of break
    :param ylim_top: where to start upper axis in case of break
    :param axisBreak: should there be a break in the axis or not    
    :param figsize: size of figure
    :param factor_x: by how many percent to extend x-axis
    :param factor_y: by how many percent to extend y-axis
    :param ax: which axis to plot into
    :param leg_labels: labels for legend
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats as scipy
    import seaborn as sns
    
    from JoV_Analysis_basicPlottingFuncs import basicFigProps, add_stats, breakAxis, pretty_plot
    
    #Set style
    rc = basicFigProps()
    plt.rcParams.update(rc)
    
    #Determine necessary variables
    if xticklabels == None:
        xticklabels = xticks
    
    if yticklabels == None:
        yticklabels = yticks
    
    #Determine whether there will be a single axis or 1 with a break
    if figsize is not None:
        if axisBreak == 0:
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=rc['savefig.dpi'])
            ax = np.array(ax, ndmin=1) #necessary step to make axis object iterable 
        else:
            fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [20, 1]},
                                figsize=figsize, dpi=rc['savefig.dpi'])
            fig.subplots_adjust(hspace = 0.05)  #adjust space between axes
        fontsize=2
    else:
        if axisBreak == 0:
            ax = np.array(ax, ndmin=1)
            
        fontsize=4
        
    #Plot group data
    for ax_i, axis in enumerate(ax):
        for ploti, _ in enumerate(np.arange(len(x))):
            #Plot study means
            ax[ax_i].scatter(x[ploti], y[ploti], c=col[ploti], marker=markers[ploti], s=sizes[ploti],
                             label=leg_labels[ploti])
        
        #Adjust ylims specifically if we plot p-values (to account for the added space)
        factor = np.max(yticks)*factor_y 
        ax[ax_i].set_ylim((np.min(yticks)-factor, np.max(yticks)+factor))
        
        if factor_x is not None:
            factor = np.max(x)*factor_x 
            ax[ax_i].set_xlim((np.min(ax[ax_i].get_xticks())-factor, np.max(ax[ax_i].get_xticks())+factor))
        
        #Add axes ticks, labels, etc
        ax[ax_i].set_xticks(xticks)
        ax[ax_i].set_xticklabels(xticklabels, font=rc['font.family'], fontsize=rc['font.size']-fontsize)
        if xlabel:
            ax[ax_i].set_xlabel(xlabel, font=rc['font.family'], fontsize=rc['font.size']-fontsize)
        
        ax[ax_i].set_yticks(yticks)
        ax[ax_i].set_yticklabels(yticklabels, font=rc['font.family'], fontsize=rc['font.size']-fontsize)
        if ylabel:
            ax[0].set_ylabel(ylabel, font=rc['font.family'], fontsize=rc['font.size']-fontsize)
        
        #Prettify axes
        ax[ax_i] = pretty_plot(ax[ax_i])
        
        #Set title
        ax[0].set_title(my_title, fontdict={'fontfamily': rc['font.family'], 'fontsize': rc['font.size']-fontsize, 
                                            'fontweight': 'bold'})
        
        #Add legend
        ax[0].legend(prop={'family': rc['font.family'], 'size': rc['font.size']-4})
            
        #Insert axis break if wanted
        if axisBreak:
            breakAxis(ax=ax, ylim_b=(ax[0].get_ylim()[0], ylim_bottom), ylim_t=(ylim_top, ax[0].get_ylim()[1]))

    return ax

def run_control_dependenceR2(data, model, path_results):
    
    """
    :param data: raw data from which to extract trial numbers
    :param model: which model did we run?
    :param path_results: where to plot the data
    """
    
    #Imports
    import os.path
    
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    from pathlib import Path
    
    from JoV_Analysis_controls import plot_quickScatter
    from JoV_Analysis_basicFitting import dvm
    from JoV_Analysis_basicPlottingFuncs import saveFig
    
    '''
    ====================================================
    Extract all of the necessary data: aka, number of trials
    available/session
    ====================================================
    '''
    ntrials_evBound = len(data[0])
    ntrials_menRot = len(data[1])
    
    ntrials_evBound_Sess1 = len(data[0][data[0].Session==0])
    ntrials_evBound_Sess2 = len(data[0][data[0].Session==1])
    ntrials_menRot_Sess1 = len(data[1][data[1].Session==0])
    ntrials_menRot_Sess2 = len(data[1][data[1].Session==1])
    
    ntrials = np.squeeze(np.array(([ntrials_evBound, ntrials_menRot, ntrials_evBound_Sess1,
                              ntrials_evBound_Sess2, ntrials_menRot_Sess1, ntrials_menRot_Sess2])))
    
    '''
    ====================================================
    Load actual R2 values
    ====================================================
    '''
    filename_gof_evBound = 'Group/Fits/Study_evBound_' + model + '_GoF_pooled_perms_1000_evBound.csv'
    gof_evBound = pd.read_csv(path_results / filename_gof_evBound, index_col=0)
    
    filename_gof_menRot = 'Group/Fits/Study_menRot_' + model +'_GoF_pooled_perms_1000_menRot.csv'
    gof_menRot = pd.read_csv(path_results / filename_gof_menRot, index_col=0)
    
    filename_gof_evBound_Sess1 = 'Group/Fits/Study_evBound_' + model +'_GoF_pooled_perms_1000_evBound_Session1.csv'
    gof_evBound_Sess1 = pd.read_csv(path_results / filename_gof_evBound_Sess1, index_col=0)
    
    filename_gof_evBound_Sess2 = 'Group/Fits/Study_evBound_' + model +'_GoF_pooled_perms_1000_evBound_Session2.csv'
    gof_evBound_Sess2 = pd.read_csv(path_results / filename_gof_evBound_Sess2, index_col=0)
    
    filename_gof_menRot_Sess1 = 'Group/Fits/Study_menRot_' + model +'_GoF_pooled_perms_1000_menRot_Session1.csv'
    gof_menRot_Sess1 = pd.read_csv(path_results / filename_gof_menRot_Sess1, index_col=0)
    
    filename_gof_menRot_Sess2 = 'Group/Fits/Study_menRot_' + model +'_GoF_pooled_perms_1000_menRot_Session2.csv'
    gof_menRot_Sess2 = pd.read_csv(path_results / filename_gof_menRot_Sess2, index_col=0)

    gofs = np.squeeze(np.array(([gof_evBound.RSquared.values, gof_menRot.RSquared.values, gof_evBound_Sess1.RSquared.values,
                              gof_evBound_Sess2.RSquared.values, gof_menRot_Sess1.RSquared.values, gof_menRot_Sess2.RSquared.values])))
    
    '''
    ====================================================
    Load actual amplitudes
    ====================================================
    '''
    filename_evBound = 'Group/Fits/Study_evBound_' + model +'_bestParams_pooled_perms_1000_evBound.csv'
    evBound = pd.read_csv(path_results / filename_evBound, index_col=0)
    
    filename_menRot = 'Group/Fits/Study_menRot_' + model +'_bestParams_pooled_perms_1000_menRot.csv'
    menRot = pd.read_csv(path_results / filename_menRot, index_col=0)
    
    filename_evBound_Sess1 = 'Group/Fits/Study_evBound_' + model +'_bestParams_pooled_perms_1000_evBound_Session1.csv'
    evBound_Sess1 = pd.read_csv(path_results / filename_evBound_Sess1, index_col=0)
    
    filename_evBound_Sess2 = 'Group/Fits/Study_evBound_' + model +'_bestParams_pooled_perms_1000_evBound_Session2.csv'
    evBound_Sess2 = pd.read_csv(path_results / filename_evBound_Sess2, index_col=0)
    
    filename_menRot_Sess1 = 'Group/Fits/Study_menRot_' + model +'_bestParams_pooled_perms_1000_menRot_Session1.csv'
    menRot_Sess1 = pd.read_csv(path_results / filename_menRot_Sess1, index_col=0)
    
    filename_menRot_Sess2 = 'Group/Fits/Study_menRot_' + model +'_bestParams_pooled_perms_1000_menRot_Session2.csv'
    menRot_Sess2 = pd.read_csv(path_results / filename_menRot_Sess2, index_col=0)
    
    if model == 'DoG':
        fits = np.squeeze(np.array(([evBound.Amplitude.values, menRot.Amplitude.values, evBound_Sess1.Amplitude.values,
                              evBound_Sess2.Amplitude.values, menRot_Sess1.Amplitude.values, menRot_Sess2.Amplitude.values])))
    else:
        fit_evBound = dvm(np.deg2rad(np.linspace(-90, 90, 181)), evBound.Amplitude.values, evBound.Kappa.values, 0)
        fit_evBound = np.rad2deg(fit_evBound) 
        peak2peak_evBound = np.sign(evBound.Amplitude.values) * (fit_evBound.max() - fit_evBound.min())
        peak2peak_evBound = peak2peak_evBound / 2
        
        fit_menRot = dvm(np.deg2rad(np.linspace(-90, 90, 181)), menRot.Amplitude.values, menRot.Kappa.values, 0)
        fit_menRot = np.rad2deg(fit_menRot) 
        peak2peak_menRot = np.sign(menRot.Amplitude.values) * (fit_menRot.max() - fit_menRot.min())
        peak2peak_menRot = peak2peak_menRot / 2
        
        fit_evBound_one = dvm(np.deg2rad(np.linspace(-90, 90, 181)), evBound_Sess1.Amplitude.values, evBound_Sess1.Kappa.values, 0)
        fit_evBound_one = np.rad2deg(fit_evBound_one) 
        peak2peak_evBound_one = np.sign(evBound_Sess1.Amplitude.values) * (fit_evBound_one.max() - fit_evBound_one.min())
        peak2peak_evBound_one = peak2peak_evBound_one / 2
        
        fit_evBound_two = dvm(np.deg2rad(np.linspace(-90, 90, 181)), evBound_Sess2.Amplitude.values, evBound_Sess2.Kappa.values, 0)
        fit_evBound_two = np.rad2deg(fit_evBound_two) 
        peak2peak_evBound_two = np.sign(evBound_Sess2.Amplitude.values) * (fit_evBound_two.max() - fit_evBound_two.min())
        peak2peak_evBound_two = peak2peak_evBound_two / 2
        
        fit_menRot_one = dvm(np.deg2rad(np.linspace(-90, 90, 181)), menRot_Sess1.Amplitude.values, menRot_Sess1.Kappa.values, 0)
        fit_menRot_one = np.rad2deg(fit_menRot_one) 
        peak2peak_menRot_one = np.sign(menRot_Sess1.Amplitude.values) * (fit_menRot_one.max() - fit_menRot_one.min())
        peak2peak_menRot_one = peak2peak_menRot_one / 2
        
        fit_menRot_two = dvm(np.deg2rad(np.linspace(-90, 90, 181)), menRot_Sess2.Amplitude.values, menRot_Sess2.Kappa.values, 0)
        fit_menRot_two = np.rad2deg(fit_menRot_two) 
        peak2peak_menRot_two = np.sign(menRot_Sess2.Amplitude.values) * (fit_menRot_two.max() - fit_menRot_two.min())
        peak2peak_menRot_two = peak2peak_menRot_two / 2
        
        fits = np.squeeze(np.array(([peak2peak_evBound, peak2peak_menRot, peak2peak_evBound_one,
                             peak2peak_evBound_two, peak2peak_menRot_one, peak2peak_menRot_two])))

    '''
    ====================================================
    Plot GOF
    ====================================================
    '''
    ax = plot_quickScatter(x=ntrials, y=gofs, my_title='R-squared as a function of trial counts', 
                      col=['#b11226', '#0061B5', '#b11226', '#b11226', '#0061B5', '#0061B5'], 
                      markers=['*', '*', 'o', 's', 'o', 's'], sizes=[80, 80, 36, 36, 36, 36],
                      xticks=[0, 2000, 4000, 6000, 8000, 10000], xticklabels=['0', '2k', '4k', '6k', '8k', '10k'], 
                      xlabel='Pooled trial counts', yticks=[0, 0.005, 0.01, 0.015, 0.02, 0.025], yticklabels=None, 
                      ylabel='R-squared', ylim_bottom=False, ylim_top=False,
                      axisBreak=0, figsize=(4, 3), factor_x=0, factor_y=0, ax=None, 
                      leg_labels=['Study1', 'Study2', 'Study1 - Session1', 'Study1 - Session2', 
                                  'Study2 - Session1', 'Study2 - Session2'])
    
    if model == 'DoG':
        filename_tmp = 'Figures/ControlAnalysis_RSquaredxTrialCounts.svg'
    else:
        filename_tmp = 'Figures/ControlAnalysis_DvM_RSquaredxTrialCounts.svg'
    format_tmp = 'svg'
    
    filename = Path(path_results / filename_tmp)
    saveFig(plt.gcf(), filename, format=format_tmp)    
    
    '''
    ====================================================
    Plot Fits
    ====================================================
    '''
    ax = plot_quickScatter(x=ntrials, y=fits, my_title='Amplitude as a function of trial counts', 
                      col=['#b11226', '#0061B5', '#b11226', '#b11226', '#0061B5', '#0061B5'], 
                      markers=['*', '*', 'o', 's', 'o', 's'], sizes=[80, 80, 36, 36, 36, 36],
                      xticks=[0, 2000, 4000, 6000, 8000, 10000], xticklabels=['0', '2k', '4k', '6k', '8k', '10k'], 
                      xlabel='Pooled trial counts', yticks=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], yticklabels=None, 
                      ylabel='Amplitude (in deg)', ylim_bottom=False, ylim_top=False,
                      axisBreak=0, figsize=(4, 3), factor_x=0, factor_y=0, ax=None, 
                      leg_labels=['Study1', 'Study2', 'Study1 - Session1', 'Study1 - Session2', 
                                  'Study2 - Session1', 'Study2 - Session2'])
    
    if model == 'DoG':
        filename_tmp = 'Figures/ControlAnalysis_AmplitudesxTrialCounts.svg'
    else:
        filename_tmp = 'Figures/ControlAnalysis_DvM_AmplitudesxTrialCounts.svg'
    format_tmp = 'svg'
    
    filename = Path(path_results / filename_tmp)
    saveFig(plt.gcf(), filename, format=format_tmp)   

    return