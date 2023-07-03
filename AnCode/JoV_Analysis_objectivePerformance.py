#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 14:28:47 2022

@author: darinka
"""

#Purpose: All functions necessary to analyze objective performance on the reproduction task.
#Author: Darinka Truebutschek
#Date created: 22/11/2022
#Date last modified: 22/11/2022
#Python version: 3.7.1

import numpy as np

def plotBox(x, y, my_title, col, xticks, xticklabels, xlabel,
            yticks, yticklabels, ylabel, ylim_bottom, ylim_top,
            axisBreak, figsize, p_val, plotStats, factor_x, factor_y, ax):
    """
    :param x: number of bars to plot
    :param y: height of individual bars
    :param my_title: title for the entire plot
    :param col: color to be used for plotting
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
    :param p_val: p values to plot
    :param plotStats: all info necessary to plot the stats
    :param factor_x: by how many percent to extend x-axis
    :param factor_y: by how many percent to extend y-axis
    :param ax: which axis to plot into
    """
    
    import matplotlib.pyplot as plt
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
        ax[ax_i] = sns.boxplot(data=np.transpose(y), palette=col, fliersize=0, showcaps=0, 
                         medianprops=dict(color="k", alpha=1, linewidth=1),
                         flierprops=dict(color='gray', alpha=1, linewidth=2), ax=ax[ax_i])
                  
        #Overlay individual data in form of scatter, connected by lines
        jittered_xPos = np.tile(x, (len(y[ax_i]), 1))+np.random.uniform(-0.15, 0.15, (len(y[ax_i]), 1))
        for x_i, x_pos in enumerate(x):
            ax[ax_i].scatter(x=jittered_xPos[:, x_i], y=y[x_i, :], 
                        c=col[x_i], edgecolor='dimgray', linewidth=1, zorder=3)
        
        #Adjust ylims specifically if we plot p-values (to account for the added space)
        factor = np.max(yticks)*factor_y 
        ax[ax_i].set_ylim((np.min(yticks)-factor, np.max(yticks)+factor))
        
        if factor_x is not None:
            factor = np.max(x)*factor_x 
            ax[ax_i].set_xlim((np.min(ax[ax_i].get_xticks())-factor, np.max(ax[ax_i].get_xticks())+factor))
        
        #Add axes ticks, labels, etc
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
            
        #Significance
        if p_val is not None:
            n_comps = np.sum(plotStats['alpha']<.05)
            
            for compi, comps in enumerate(np.arange(n_comps)):
                if n_comps == 1:
                    alpha = plotStats['alpha']
                else:
                    alpha = plotStats['alpha'][compi]
                    
                add_stats(ax=ax[0], alpha=alpha, x1=plotStats['x1'][compi], 
                          x2=plotStats['x2'][compi], y=plotStats['y'][compi], h=plotStats['h'][compi], 
                          vertLines=plotStats['vertLines'][compi], rotation=plotStats['rotation'][compi])
            
        #Insert axis break if wanted
        if axisBreak:
            breakAxis(ax=ax, ylim_b=(ax[0].get_ylim()[0], ylim_bottom), ylim_t=(ylim_top, ax[0].get_ylim()[1]))

    return ax

def plotLine(x, y, my_title, col, xticks, xticklabels, xlabel, xlim_split,
            yticks, yticklabels, ylabel, ylim_bottom, ylim_top,
            leg_labels, axisBreak, axisXBreak, figsize, p_val, plotStats, factor_x, factor_y):
    """
    :param x: number of scatters to plot (different rows belong to different groups)
    :param y: height of individual scatters
    :param my_title: title for the entire plot
    :param col: color to be used for plotting
    :param xticks: xticks to be used
    :param xticklabels: xticklabels to be used (if None, xticks == xticklabels)
    :param xlabel: label to be used for x-axis
    :param xlim_split: how to split x data 
    :param yticks: yticks to be used
    :param yticklabels: yticklabels to be used (if None, yticks == yticklabels)
    :param ylabel: label to be used for y-axis   
    :param ylim_bottom: where to end lower axis in case of break
    :param ylim_top: where to start upper axis in case of break
    :param leg_labels: which label to use for the legend
    :param axisBreak: should there be a break in the axis or not
    :param axisXBreak: should there be a break in the x axis or not
    :param figsize: size of figure
    :param p_val: p values to plot
    :param plotStats: all info necessary to plot the stats
    :param factor_x: by how many percent to extend x-axis
    :param factor_y: by how many percent to extend y-axis
    """
    
    import matplotlib.pyplot as plt
    import scipy.stats as scipy
    import seaborn as sns
    
    from JoV_Analysis_basicPlottingFuncs import basicFigProps, add_stats, breakAxis, breakXAxis, pretty_plot
    
    #Set style
    rc = basicFigProps()
    plt.rcParams.update(rc)
    
    #Determine necessary variables
    if xticklabels == None:
        xticklabels = xticks
    
    if yticklabels == None:
        yticklabels = yticks
    
    #How many categories do we want tom plot?
    if leg_labels is None:
        categories = 1
    else:
        categories = len(leg_labels)
    
    #Determine whether there will be a single axis or 1 with a break
    if (axisBreak == 0) & (axisXBreak == 0):
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=rc['savefig.dpi'])
        ax = np.array(ax, ndmin=1) #necessary step to make axis object iterable 
    elif (axisBreak == 1) & (axisXBreak == 0):
        fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [20, 1]},
                                figsize=figsize, dpi=rc['savefig.dpi'])
        fig.subplots_adjust(hspace = 0.05)  #adjust space between axes
    elif (axisBreak == 0) & (axisXBreak == 1):
        fig, ax = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [1, 1]},
                                figsize=figsize, dpi=rc['savefig.dpi'])
        fig.subplots_adjust(wspace = 0.05)  #adjust space between axes
    elif (axisBreak == 1) & (axisXBreak == 1):
            fig, ax = plt.subplots(2, 2, gridspec_kw={'height_ratios': [20, 1], 'width_ratios': [1, 1]},
                                    figsize=figsize, dpi=rc['savefig.dpi'])
            fig.subplots_adjust(wspace = 0.05)  #adjust space between axes
            ax = ax.flatten()
    
    #Plot group data
    for ax_i, axis in enumerate(ax):
        for categi, categ in enumerate(np.arange(categories)):
            
            #Compute mean & sem
            y_mean = np.nanmean(y[categi], axis=0)
            y_err = scipy.sem(y[categi], nan_policy='omit')
            
            #Plot mean with errorbar
            ax[ax_i] = sns.lineplot(x=x, y=y_mean, color=col[categi], linewidth=1.5, ax=ax[ax_i])
            plt.fill_between(x=x, y1=y_mean-y_err, y2=y_mean+y_err, color=col[categi], alpha=.2)
                                    
        #Adjust ylims specifically if we plot p-values (to account for the added space)
        factor = np.max(yticks)*factor_y 
        ax[ax_i].set_ylim((np.min(yticks)-factor, np.max(yticks)+factor))
        
        factor = np.max(x)*factor_x 
        ax[ax_i].set_xlim((np.min(xticks)-factor, np.max(xticks)+factor))

        #Add axes ticks, labels, etc
        ax[ax_i].set_xticks(xticks)
        
        ax[ax_i].set_xticklabels(xticklabels, font=rc['font.family'], fontsize=rc['font.size']-2)
        if xlabel:
            if axisXBreak == 0:
                ax[ax_i].set_xlabel(xlabel, font=rc['font.family'], fontsize=rc['font.size']-2)
            else:
                fig.supxlabel(xlabel, y= .05, verticalalignment='top', fontfamily=rc['font.family'], 
                              fontsize=rc['font.size']-2, color='dimgray')
        
        ax[ax_i].set_yticks(yticks)
        ax[ax_i].set_yticklabels(yticklabels, font=rc['font.family'], fontsize=rc['font.size']-2)
        if ylabel:
            if axisBreak == 0:
                ax[ax_i].set_ylabel(ylabel, font=rc['font.family'], fontsize=rc['font.size']-2)
            else:
                ax[0].set_ylabel(ylabel, font=rc['font.family'], fontsize=rc['font.size']-2)
        
        #Prettify axes
        ax[ax_i] = pretty_plot(ax[ax_i])
        
        #Plot legend
        if leg_labels != None:
            ax[0].legend(leg_labels, prop={'family': rc['font.family'], 'size': rc['font.size']-4})
    
        #Set title
        if axisXBreak == 0:
            ax[0].set_title(my_title, fontdict={'fontfamily': rc['font.family'], 'fontsize': rc['font.size']-1, 
                                            'fontweight': 'bold'})
        else:
            plt.suptitle(my_title, y=.9, verticalalignment='bottom',
                          fontfamily=rc['font.family'], fontsize=rc['font.size']-1, fontweight='bold')

        #Significance
        if p_val is not None:
            n_comps = np.sum(plotStats['alpha']<.05)
            
            for compi, comps in enumerate(np.arange(n_comps)):
                add_stats(ax=ax[0], alpha=plotStats['alpha'][compi], x1=plotStats['x1'][compi], 
                          x2=plotStats['x2'][compi], y=plotStats['y'][compi], h=plotStats['h'][compi], 
                          vertLines=plotStats['vertLines'][compi], rotation=plotStats['rotation'][compi])
            
        #Insert axis break if wanted
        if axisBreak:
            if (axisXBreak==0) | (axisXBreak==1):
                breakAxis(ax=ax, ylim_b=(ax[0].get_ylim()[0], ylim_bottom), ylim_t=(ylim_top, ax[0].get_ylim()[1]))
                        
        if axisXBreak:
            if (axisBreak==0) | (axisBreak==1):
                breakXAxis(ax=ax, xlim_b=(ax[0].get_xlim()[0], xlim_split[0][1]), xlim_t=(xlim_split[1][0], ax[0].get_xlim()[1]))
          
    return ax

### Run analysis ###
def run_Analysis_objectivePerformance(data, studies, respError_var, angle_bins, path_results):
    """
    :param data: pandas dataFrame of entire group
    :param studies: which studies to include
    :param respError_var: which specific errors to plot (i.e., raw errors, outliers removed, mean-centered)
    :param angle_bins: how to bin angular space
    :param path_results: where to save the data
    """
    
    ### Imports ###
    import copy 
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import pycircstat
    import scipy.stats as scipy
    
    from JoV_Analysis_basicFuncs import computeRespError, dim
    
    #from colorama import Fore, Style
    from pathlib import Path
    
    from JoV_Analysis_objectivePerformance import plotBox, plotLine
    from JoV_Analysis_basicPlottingFuncs import saveFig
    
    '''
    ====================================================
    Compute response error
    ====================================================
    '''
    
    if respError_var == 'RespError':
        data = computeRespError(data)
    
    '''
    ====================================================
    Select data
    ====================================================
    '''
    
    if respError_var == 'RespError':
        selData = data[(data.incl_trials==1) & (data.incl_JoV==1)]
    else:
        selData = data
    
    '''
    ====================================================
    Prepare data for plotting
    ====================================================
    '''
    if len(studies)==2:
        data_sel = [selData[(selData.Study=='evBound')],
                    selData[(selData.Study=='menRot')]]
        
        curr_studies = ['evBound', 'menRot']
    elif len(studies)==3:
        data_sel = [selData[(selData.Study=='evBound')],
                    selData[(selData.Study=='funcStates')],
                    selData[(selData.Study=='menRot')]]
        curr_studies = ['evBound', 'funcStates', 'menRot']
    
    '''
    ====================================================
    Prepare the histograms and the data for the insets
    ====================================================
    '''
    dat2plot_all = []
    inset1_all = []
    inset2_all = []
    
    for condi, cond in enumerate(np.arange(np.shape(data_sel)[0])): #loop over studies
        dat2plot = np.zeros((len(np.unique(data_sel[condi].Subject)), len(angle_bins)-1))
        
        inset1 = np.zeros((len(np.unique(data_sel[condi].Subject)))) #mean absolute error
        inset2 = np.zeros((len(np.unique(data_sel[condi].Subject)))) #precision (i.e., 1/std)
        
        for subi, sub in enumerate(np.unique(data_sel[condi].Subject)): #loop over subject
            tmp = data_sel[condi][data_sel[condi].Subject==sub][respError_var].values #extract data from each subject for each condition
            
            inset1[subi] = np.mod(np.rad2deg(pycircstat.mean(np.deg2rad(np.abs(tmp[~np.isnan(tmp)])))), 180)
            inset2[subi] = np.rad2deg(pycircstat.std(np.deg2rad(tmp[~np.isnan(tmp)])))
            
            tmp, _ = np.histogram(tmp, bins=angle_bins, density=False)
            dat2plot[subi, :] = tmp / np.sum(tmp)
        
        dat2plot_all.append(dat2plot)
        inset1_all.append(inset1)
        inset2_all.append(inset2)
    
    '''
    ====================================================
    Prepare insets for stats (exported to JASP)
    ====================================================
    '''
    if len(studies) == 2:
        subjects = [np.unique(selData.Subject[selData.Study=='evBound']),
                    np.unique(selData.Subject[selData.Study=='menRot'])]
    elif len(studies) == 3:
        subjects = [np.unique(selData.Subject[selData.Study=='evBound']),
                    np.unique(selData.Subject[selData.Study=='funcStates']),
                    np.unique(selData.Subject[selData.Study=='menRot'])]
        
    #Inset 1
    inset1_df = pd.DataFrame({'Study': np.repeat(curr_studies, dim(subjects)),
                              'Subject': np.concatenate(subjects),
                              'MeanAbsError': np.concatenate(inset1_all)})    
    
    #Save
    if respError_var == 'RespError':
        filename_tmp = 'Stats/MeanAbsoluteErrors_Studies_'  + str(len(studies)) + '.csv'
        filename = Path(path_results / filename_tmp)
        inset1_df.to_csv(filename)
    
    #Retrieve statistics from JASP (done manually for the sake of time, 
    #shown are Holm-corrected p-values of post-hoc tests)
    if len(studies) == 2:
        plotStats_inset1 = dict({'alpha': np.array((.0001)), #evBound vs. menRot
                          'x1': [0],
                          'x2': [1], 
                          'y': [np.max([np.max(inset1_all[0]), np.max(inset1_all[1])])+np.max([scipy.stats.sem(inset1_all[0]), scipy.stats.sem(inset1_all[1])])],
                          'h': [0.1],
                          'vertLines': [True], 
                          'rotation': [0]})
        
    elif len(studies) == 3:
        plotStats_inset1 = dict({'alpha': np.array((.001, .028)), #evBound vs. funcStates, evBound vs. menRot, funcStates vs. menRot
                          'x1': [0, 1],
                          'x2': [2, 2], 
                          'y': [np.max([np.max(inset1_all[0]), np.max(inset1_all[2])])+1*np.max([scipy.stats.sem(inset1_all[0]), scipy.stats.sem(inset1_all[2])]),
                                np.max([np.max(inset1_all[1]), np.max(inset1_all[2])])+3*np.max([scipy.stats.sem(inset1_all[1]), scipy.stats.sem(inset1_all[2])])],
                          'h': [0.1, 0.1],
                          'vertLines': [True, True], 
                          'rotation': [0, 0]})
        
    #Inset 2
    inset2_df = pd.DataFrame({'Study': np.repeat(curr_studies, dim(subjects)),
                              'Subject': np.concatenate(subjects),
                              'Precision': np.concatenate(inset2_all)})    
    
    #Save
    if respError_var == 'RespError':
        filename_tmp = 'Stats/Precision_Studies_'  + str(len(studies)) + '.csv'
        filename = Path(path_results / filename_tmp)
        inset2_df.to_csv(filename)
    
    #Retrieve statistics from JASP (done manually for the sake of time, 
    #shown are Holm-corrected p-values of post-hoc tests)
    if len(studies) == 2:
        plotStats_inset2 = dict({'alpha': np.array((.0001)), #evBound vs. menRot
                          'x1': [0],
                          'x2': [1], 
                          'y': [np.max([np.max(inset2_all[0]), np.max(inset2_all[1])])+np.max([scipy.stats.sem(inset2_all[0]), scipy.stats.sem(inset2_all[1])])],
                          'h': [0.1],
                          'vertLines': [True], 
                          'rotation': [0]})
        
    elif len(studies) == 3:
        plotStats_inset2 = dict({'alpha': np.array((.0001, .002)), #evBound vs. funcStates, evBound vs. menRot, funcStates vs. menRot
                          'x1': [0, 1],
                          'x2': [2, 2], 
                          'y': [np.max([np.max(inset2_all[0]), np.max(inset2_all[2])])+1*np.max([scipy.stats.sem(inset2_all[0]), scipy.stats.sem(inset2_all[2])]),
                                np.max([np.max(inset2_all[1]), np.max(inset2_all[2])])+3.5*np.max([scipy.stats.sem(inset2_all[1]), scipy.stats.sem(inset2_all[2])])],
                          'h': [0.1, 0.1],
                          'vertLines': [True, True], 
                          'rotation': [0, 0]})
        
    #Plot
    if len(studies) == 2:
        leg_labels = ['Study 1', 'Study 2']
        col = ['#b11226', '#0061B5']
    elif len(studies) == 3:
        leg_labels = ['Study 1', 'Study 2', 'Study 3']
        col = ['#F4A460', '#b11226', '#0061B5']
    
    ax = plotLine(x=np.linspace(int(np.mean(angle_bins[0:2])), int(np.mean(angle_bins[-2:])), 
                                len(angle_bins)-1), 
                  y=dat2plot_all, my_title='Objective performance', col=col,
                  xticks=[-90, -60, -30, 0, 30, 60, 90], xticklabels=None, xlabel='Adjustment error (in deg)', xlim_split=None,
                  yticks=[0, .1, .2, .3, .4], yticklabels=['0', '0.1', '0.2', '0.3', '0.4'], ylabel='Probability',
                  ylim_bottom=None, ylim_top=None,
                  leg_labels=leg_labels, figsize=(6, 4), axisBreak=0, axisXBreak=0, p_val=None, plotStats=None,
                  factor_x=0, factor_y=0)
     
    #Insert arrows
    ax[0].annotate(s='', xy=(-20, 0.005), xytext=(0.005, 0.005), 
                    font='Arial', fontsize=12-4, color='dimgray',
                    horizontalalignment='center', arrowprops=dict(color='dimgray', arrowstyle='->')) #CCW
    
    ax[0].annotate(s='  Ccw  ', xy=(-20, 0.005), xytext=(0.007, 0.007), 
                    font='Arial', fontsize=12-6, color='dimgray',
                    horizontalalignment='right') #CCW
    
    ax[0].annotate(s='', xy=(20, 0.005), xytext=(0.005, 0.005), 
                    font='Arial', fontsize=12-4, color='dimgray',
                    horizontalalignment='center', arrowprops=dict(color='dimgray', arrowstyle='->')) #CW
    
    ax[0].annotate(s='  Cw  ', xy=(20, 0.005), xytext=(0.007, 0.007), 
                    font='Arial', fontsize=12-6, color='dimgray',
                    horizontalalignment='left') #CCW
    
    #Create inset axis
    ax_inset = ax[0].inset_axes([0.09, .15, .25, .5], facecolor='white')
    ax_inset.patch.set_alpha(0)
    
    ax_inset2 = ax[0].inset_axes([.75, .15, .25, .5], facecolor='white')
    ax_inset2.patch.set_alpha(0)
    
    if len(studies) == 2:
        x = [0, 1]
        
        inset1_all[1] = np.append(inset1_all[1], np.nan)
        inset1_all = np.asarray(inset1_all)
        
        inset2_all[1] = np.append(inset2_all[1], np.nan)
        inset2_all = np.asarray(inset2_all)
        
        yticks_inset1 = [4, 12]
        yticks_inset2 = [5, 19]
        
        col = ['#b11226', '#0061B5']
        xticklabels = ['S1', 'S2']
    elif len(studies) == 3:
        x = [0, 1, 2]
        
        inset1_all[0] = np.append(inset1_all[0], np.nan)
        inset1_all[2] = np.append(inset1_all[2], [np.nan, np.nan])
        inset1_all = np.asarray(inset1_all)
        
        yticks_inset1 = [4, 14]
        #yticks_inset2 = [5, 21]
        yticks_inset2 = [5, 19]
        
        inset2_all[0] = np.append(inset2_all[0], np.nan)
        inset2_all[2] = np.append(inset2_all[2], [np.nan, np.nan])
        inset2_all = np.asarray(inset2_all)
        
        col = ['#F4A460', '#b11226', '#0061B5']
        xticklabels = ['S1', 'S2', 'S3']
    
    ax_inset1 = plotBox(x=x, y=inset1_all, 
               my_title='Mean absolute error', col=np.repeat(col, 1), 
               xticks=x, xticklabels=xticklabels, xlabel=None,
               yticks=yticks_inset1, yticklabels=None, ylabel='Degrees', 
               ylim_bottom=None, ylim_top=None,
               axisBreak=0, figsize=None, p_val=None, plotStats=plotStats_inset1, 
               factor_x=None, factor_y=0, ax=ax_inset)

    ax_inset2 = plotBox(x=x, y=inset2_all, 
               my_title='Dispersion', col=np.repeat(col, 1), 
               xticks=x, xticklabels=xticklabels, xlabel=None,
               yticks=yticks_inset2, yticklabels=None, ylabel='Degrees', 
               ylim_bottom=None, ylim_top=None,
               axisBreak=0, figsize=None, p_val=None, plotStats=plotStats_inset2, 
               factor_x=None, factor_y=0, ax=ax_inset2)
    
    #Fix ylabel positions
    ax_inset1[0].yaxis.labelpad = .001
    ax_inset2[0].yaxis.labelpad = .001
    
    #Save
    if respError_var == 'RespError':
        filename_tmp = 'Figures/ErrorDist_Studies_' + str(len(studies)) + '.svg'
        format_tmp = 'svg'
        
        filename = Path(path_results / filename_tmp)
        saveFig(plt.gcf(), filename, format=format_tmp)
    
    return