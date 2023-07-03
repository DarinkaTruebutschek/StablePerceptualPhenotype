#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 15:18:54 2022

@author: darinka

#Purpose: All functions necessary to perform SD analyses.
#Author: Darinka Truebutschek
#Date created: 23/11/2022
#Date last modified: 23/11/2022
#Python version: 3.7.1

"""

import numpy as np


### Basic functions ###

#Compute moving averages
def computeMovingAverage(data=None, bin_width=20):
    """
    :param: data: single-subject data which should be smoothed
    :param: bin_width: how large should the smoothing kernel be (in degrees)

    """
    
    import pycircstat #toolbox for circular statistics
    import pandas as pd
    
    data = data.copy(deep=True)
    
    #First, etraxt relevant data (i.e., stimulus orientation differences & response errors)
    x = data.Delta_angle_norm.values
    y = data.Resp_error_demeaned.values
    y[np.isnan(x)] = np.nan
    x[np.isnan(y)] = np.nan
    
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    #Pad the data (to be able to smooth even those data at the 'edges' of the circle)
    x_padded = x-180
    x_padded = np.append(x_padded, x)
    x_padded = np.append(x_padded, x+180)
    
    y_padded = np.tile(y, 3)
    
    #Smooth the data (aka, take the mean of the data) within a given bin
    data_smoothed = np.zeros(181)
    
    for bini in range(0, 181):
        range_tmp = (np.array((1, bin_width)) - np.floor(bin_width/2)-1-90+(bini))
        data_smoothed[bini]=np.rad2deg(pycircstat.mean(np.deg2rad(y_padded[(x_padded >= range_tmp[0]) & (x_padded <= range_tmp[1])])))
    
    data_smoothed = np.mod(data_smoothed+90, 180)-90
    return data_smoothed

#Smooth standard deviation
def smooth_circData(x=None, y=None, bin_width=20):
    """
    :param: x: circular range to consider
    :param: y: circular data to be smoothed
    :param: bin_width: how large should the smoothing kernel be (in degrees)

    """
        
    import pycircstat #toolbox for circular statistics
    
    #Pad the data (to be able to smooth even those data at the 'edges' of the circle)
    x_padded = x-180
    x_padded = np.append(x_padded, x)
    x_padded = np.append(x_padded, x+180)
    
    y_padded = np.tile(y, 3)
    
    #Smooth the data (aka, take the mean of the data) within a given bin
    data_smoothed = np.zeros(181)
    
    for bini in range(0, 181):
        range_tmp = (np.array((1, bin_width)) - np.floor(bin_width/2)-1-90+(bini))
        data_smoothed[bini]=np.rad2deg(pycircstat.mean(np.deg2rad(y_padded[(x_padded >= range_tmp[0]) & (x_padded <= range_tmp[1])])))
    
    data_smoothed = np.mod(data_smoothed+90, 180)-90
    
    return data_smoothed


### Plotting ###
def plotSD(x, y, yerr, fit, my_title, col, xticks, xticklabels, xlabel, xlim_split,
            yticks, yticklabels, ylabel, ylim_bottom, ylim_top,
            axisBreak, axisXBreak, figsize, p_val, plotStats, factor_x, factor_y, collapseSubs, label, label_fits, my_legend):
    """
    :param x: angles 
    :param y: error
    :param yerr: errorbar
    :param fit: fitted DoG to plot
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
    :param axisBreak: should there be a break in the axis or not
    :param axisXBreak: should there be a break in the x axis or not
    :param figsize: size of figure
    :param p_val: p values to plot
    :param plotStats: will the fit be shown with solid (if significant) or dashed (if not) line
    :param factor_x: by how many percent to extend x-axis
    :param factor_y: by how many percent to extend y-axis
    :param collapseSubs: do we need to plot SEM or not?
    :param label: label for legend
    :param label_fits: label for legend
    :param my_legend: show legend or not
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
    
    if my_legend:
        if label == None:
            label = ['data']
            label_fits = ['fit']

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
    
    #Plot
    for ax_i, axis in enumerate(ax):
        
        #First, plot actual data
        for linei, _ in enumerate(np.arange(y.ndim)):
            if y.ndim > 1: #aka, we are plotting more than one line
                y_tmp = y[linei, :]
                yerr_tmp = yerr[linei, :]
                col_tmp = col[linei]
                if len(label_fits) < 4:
                    fit_tmp = fit[linei, :]
                    plotStats_tmp = plotStats[linei]
                if linei == 0:
                    fit_study = np.squeeze(fit[0])
                    plotStats_study = plotStats[0]
                    label_fits_study = label_fits[0:2]
                    col_study = col[0:2]
                    alpha_study = [1, 1]
                else:
                    fit_study = np.squeeze(fit[1])
                    plotStats_study = plotStats[1]
                    label_fits_study = label_fits[2:]
                    col_study = col[2:]
                    alpha_study = [.7, .7]
            else:
                y_tmp = y
                yerr_tmp = yerr
                col_tmp = col
                fit_tmp = fit
                plotStats_tmp = plotStats
            ax[ax_i] = sns.lineplot(x=x, y=y_tmp, color=col_tmp, linewidth=.8, alpha=.8, ax=ax[ax_i], zorder=2, label=label[linei]) 
        
            #Then, plot sem around the group mean
            if collapseSubs == 'pooled':
                plt.fill_between(x=x, y1=y_tmp-yerr_tmp, y2=y_tmp+yerr_tmp, 
                                 color=col_tmp, alpha=.2, zorder=3, label='_nolegend_')
        
            #Third, plot fit
            if len(label_fits) == 4:
                for modeli, _ in enumerate(np.arange(fit_study.ndim)):
                    ax[ax_i] = sns.lineplot(x=x, y=fit_study[modeli, :], color=col_study[modeli], linewidth=4, 
                                    linestyle=plotStats_study[modeli], ax=ax[ax_i], zorder=4, label=label_fits_study[modeli], 
                                    alpha=alpha_study[modeli])
            else:
                ax[ax_i] = sns.lineplot(x=x, y=fit_tmp, color=col_tmp, linewidth=4, 
                                        linestyle=plotStats_tmp, ax=ax[ax_i], zorder=4, label=label_fits[linei])
        
        #Plot markers 
        ax[ax_i].hlines(0, -100, 100, colors='dimgray', linestyles='dotted', linewidth=.8, zorder=1)
        ax[ax_i].vlines(0, -9, 9, colors='dimgray', linestyles='dotted', linewidth=.8, zorder=1)
        
        #Adjust ylims specifically if we plot p-values (to account for the added space)
        factor = np.max(yticks)*factor_y 
        ax[ax_i].set_ylim((np.min(yticks)-factor, np.max(yticks)+factor))
        
        factor = np.max(xticks)*factor_x 
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
        
        #Add axis indicators to show cw-ccw
        ax[ax_i].annotate(s='CW', xy=(5, ax[ax_i].get_ylim()[0]), 
                        font=rc['font.family'], fontsize=rc['font.size']+6, color='dimgray', alpha=.2,
                        horizontalalignment='left') #CW
        ax[ax_i].annotate(s='CCW', xy=(-5, ax[ax_i].get_ylim()[0]), 
                        font=rc['font.family'], fontsize=rc['font.size']+6, color='dimgray', alpha=.2,
                        horizontalalignment='right') #CCW
        ax[ax_i].annotate(s='CW', xy=(-80, ax[ax_i].get_ylim()[1]), 
                        font=rc['font.family'], fontsize=rc['font.size']+6, color='dimgray', alpha=.2,
                        horizontalalignment='right', verticalalignment='top', rotation=90, zorder=1) #CW
        ax[ax_i].annotate(s='CCW', xy=(-80, ax[ax_i].get_ylim()[0]), 
                        font=rc['font.family'], fontsize=rc['font.size']+6, color='dimgray', alpha=.2,
                        horizontalalignment='right', verticalalignment='bottom', rotation=90, zorder=1) #CCW
                
        #Prettify axes
        ax[ax_i] = pretty_plot(ax[ax_i])
        
        #Legend
        if my_legend:
            ax[0].legend(prop={'family': rc['font.family'], 'size': rc['font.size']-4})
        
        #Set title
        if axisXBreak == 0:
            ax[0].set_title(my_title, fontdict={'fontfamily': rc['font.family'], 'fontsize': rc['font.size']-1, 
                                            'fontweight': 'bold'})
        else:
            plt.suptitle(my_title, y=.9, verticalalignment='bottom',
                          fontfamily=rc['font.family'], fontsize=rc['font.size']-1, fontweight='bold')
    
    #Insert axis break if wanted
    if axisBreak:
        if (axisXBreak==0) | (axisXBreak==1):
            breakAxis(ax=ax, ylim_b=(ax[0].get_ylim()[0], ylim_bottom), ylim_t=(ylim_top, ax[0].get_ylim()[1]))
                    
    if axisXBreak:
        if (axisBreak==0) | (axisBreak==1):
            breakXAxis(ax=ax, xlim_b=(ax[0].get_xlim()[0], xlim_split[0][1]), xlim_t=(xlim_split[1][0], ax[0].get_xlim()[1]))
      
    return ax

### Run analysis ###
def run_Analysis_SD(data, currentStudy, model, bins, bin_width, collapseSubs, 
                    dog_fittingSteps, stats_n_permutations, savename, 
                    path_results, rerun_fit, rerun_perms, rerun_bootstrapp):
    """
    :param data: pandas dataFrame with only those trials that will be used in the SD analysis
    :param currentStudy: which study are we currently looking at
    :param model: which model (i.e., DoG, Clifford, etc) to use to fit the data
    :param bins: how to bin x-axis (i.e., angles)
    :param bin_width: how to bin SD data
    :param collapseSubs: how to treat individual subjects (singleSub | groupMean | pooled)
    :param dog_fittingSteps: how many steps to consider when fitting dog (default: 200)
    :param stats_n_permutations: how many permutations to run (default: 10,000)
    :param savename: specifics of analysis (i.e., which session, how many permutations, etc)
    :param path_results: where to save the data
    :param rerun_fit: should we rerun the model fit or simply load from file?
    :param rerun_perms: should we rerun the permutations or simply load them from file?
    :param rerun_bootstrapp: should we rerun the bootstrapping or simply load them from file?
    """
    
    #Imports
    import os.path
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import scipy.stats as scipy
    
    from pathlib import Path
    
    from JoV_Analysis_basicFuncs import analysis_dog, analysis_clifford, analysis_dvm
    from JoV_Analysis_stats import (perform_permute_SD, perform_bootstrapping_SD, 
                                    perform_permute_SD_clifford, perform_bootstrapping_SD_clifford,
                                    perform_permute_SD_dvm, perform_bootstrapping_SD_dvm)
    from JoV_Analysis_SD import computeMovingAverage, plotSD
    from JoV_Analysis_basicFitting import dog, clifford, dvm
    from JoV_Analysis_basicPlottingFuncs import saveFig
        
    #Initialize important variables
    data_smoothed = np.zeros((len(np.unique(data.Subject)), 181))
    data_smoothed_permuted = np.zeros((stats_n_permutations, len(np.unique(data.Subject)), 181))
    
    if model == 'DoG':
        model_params = np.zeros((1, 3)) #store best fitting parameters
        gof = np.zeros((1, 5)) #store measures of goodness of fit
        perms = np.zeros((stats_n_permutations, 6)) #store permutation distributions
        bootstrappingCoefs = np.zeros((stats_n_permutations, 3))
        significance = np.zeros((1, 1))
    elif model == 'Clifford':
        model_params = np.zeros((1, 4)) #store best fitting parameters
        perms = np.zeros((stats_n_permutations, 5)) #store permutation distributions
        bootstrappingCoefs = np.zeros((stats_n_permutations, 4))
        significance = np.zeros((1, 1))
    elif model == 'DvM':
        model_params = np.zeros((1, 3)) #store best fitting parameters
        gof = np.zeros((1, 2)) #store measures of goodness of fit
        perms = np.zeros((stats_n_permutations, 6)) #store permutation distributions
        bootstrappingCoefs = np.zeros((stats_n_permutations, 6))
        significance = np.zeros((1, 1))
    
    '''
    ====================================================
    Smooth data (for plotting)
    ====================================================
    '''
    #Extract relevant data
    if collapseSubs == 'pooled':

        #Compute moving averages (for plotting, so this will be done seperately for each subject)
        for subi, sub in enumerate(np.unique(data.Subject)):
            print('Smoothing data for subject: ' + sub)
            data_smoothed[subi, :] = computeMovingAverage(data[data.Subject==sub], bin_width=bin_width)
            
    elif collapseSubs == 'singleSub':
        data_smoothed = computeMovingAverage(data, bin_width=bin_width)

    '''
    ====================================================
    Compute Model fit & save it
    ====================================================
    '''
    print('Fitting model')
    
    #Determine filenames
    filename_model = 'Group/Fits/Study_' + currentStudy + '_' + model + '_bestParams_' + collapseSubs + '_'  + savename + '.csv'
    
    if (model == 'DoG') | (model == 'DvM'):
        filename_gof = 'Group/Fits/Study_' + currentStudy + '_' + model + '_GoF_' + collapseSubs + '_'  + savename + '.csv'
    
    #Fit
    if (collapseSubs == 'pooled') | (collapseSubs == 'singleSub'):
        
        #Load pre-existing data
        if rerun_fit == 0:
            model_params = pd.read_csv(path_results / filename_model, index_col=0).to_numpy()
            
            if (model == 'DoG') | (model == 'DvM'):
                gof = pd.read_csv(path_results / filename_gof, index_col=0).to_numpy()
        else:
            if model == 'DoG':
                model_params, gof = analysis_dog(data=data, dat2fit=collapseSubs,
                                                  fittingSteps=dog_fittingSteps)
                
                #Save
                pd.DataFrame(model_params, columns=['Amplitude', 'Width', 'MinCost']).to_csv(path_results / filename_model)
                pd.DataFrame(gof, columns=['SSE', 'RSquared', ' ', ' ', ' ']).to_csv(path_results / filename_gof)
            elif model == 'Clifford':
                model_params = analysis_clifford(data=data, dat2fit=collapseSubs,
                                                  fittingSteps=dog_fittingSteps)

                #Save
                pd.DataFrame(model_params, columns=['Centering param', 'Scaling param', 'Sign', 'MinCost']).to_csv(path_results / filename_model)
            elif model == 'DvM':
                model_params, gof = analysis_dvm(data=data, dat2fit=collapseSubs,
                                                  fittingSteps=dog_fittingSteps)
                
                #Save
                pd.DataFrame(model_params, columns=['Amplitude', 'Kappa', 'MinCost']).to_csv(path_results / filename_model)
                pd.DataFrame(gof, columns=['SSE', 'RSquared']).to_csv(path_results / filename_gof)
                
    '''
    ====================================================
    Permute model fits
    ====================================================
    '''
    print('Permuting model fits')
    
    #Determine filenames
    filename_perms = 'Group/Perms/Perms_Study_' + currentStudy + '_' + model + '_' + collapseSubs + '_'  + savename + '.npy'
    filename_perms_angles = 'Group/Perms/Perms_Angles_Study_' + currentStudy + '_' + model + '_' + collapseSubs + '_'  + savename + '.npy'
    filename_perms_respErrors = 'Group/Perms/Perms_RespErrors_Study_' + currentStudy + '_' + model + '_' + collapseSubs + '_'  + savename + '.npy'
    filename_perms_dataSmoothed = 'Group/Perms/Perms_dataSmoothed_Study_' + currentStudy + '_' + model + '_' + collapseSubs + '_'  + savename + '.npy'

    #Permute fit
    if (collapseSubs == 'pooled') | (collapseSubs == 'singleSub'):
        
        #Load pre-existing data
        if rerun_perms == 0:
            perms = np.load(path_results / filename_perms)
            permuted_angles = np.load(path_results / filename_perms_angles)
            permuted_respErrors = np.load(path_results / filename_perms_respErrors)
            data_smoothed_permuted = np.load(path_results / filename_perms_dataSmoothed)
        else:
            #Prep data for permutation (i.e., taking care of additional nans from outliers)
            data_perms = data.copy(deep=True)
            data_perms = data[(~np.isnan(data.Delta_angle_norm))]
            data_perms = data[(~np.isnan(data.Resp_error_demeaned))]
            
            if model == 'DoG':
                perms, permuted_angles, permuted_respErrors = perform_permute_SD(data=data_perms, dat2fit=collapseSubs, 
                                                            n_permutations=stats_n_permutations, bestParams_actualFit=None, 
                                                            fittingSteps=dog_fittingSteps)
            
                # #Smooth permuted angles and respErrors (for plotting only)
                # for permi, _ in enumerate(np.arange(stats_n_permutations)):
                #     print('Smoothing permutations : ' + str(permi))
                #     df_tmp = pd.DataFrame(np.array([permuted_angles[permi, :], permuted_respErrors[permi, :]]).T, 
                #                                                        columns=['Delta_angle_norm', 'Resp_error_demeaned'])
                #     #Insert subject variable & permutation variable
                #     df_tmp.insert(0, 'Subject', data_perms.Subject.values)
                
                #     #Compute moving averages (for plotting, so this will be done seperately for each subject)
                #     tmp_all = []
                #     for subi, sub in enumerate(np.unique(data.Subject)):
                #         tmp = computeMovingAverage(data=df_tmp[df_tmp.Subject==sub], bin_width=bin_width)
                #         tmp_all.append(tmp)
                    
                #     data_smoothed_permuted[permi, :, :] = tmp_all

            elif model == 'Clifford':
                perms, permuted_angles, permuted_respErrors = perform_permute_SD_clifford(data=data_perms, dat2fit=collapseSubs, 
                                                                n_permutations=stats_n_permutations, bestParams_actualFit=None, 
                                                                fittingSteps=dog_fittingSteps)
                
                # #Smooth permuted angles and respErrors (for plotting only)
                # for permi, _ in enumerate(np.arange(stats_n_permutations)):
                #     df_tmp = pd.DataFrame(np.array([permuted_angles[permi, :], permuted_respErrors[permi, :]]).T, 
                #                                                        columns=['Delta_angle_norm', 'Resp_error_demeaned'])
                
                #     #Insert subject variable & permutation variable
                #     df_tmp.insert(0, 'Subject', data_perms.Subject.values)
                
                #     #Compute moving averages (for plotting, so this will be done seperately for each subject)
                #     tmp_all = []
                #     for subi, sub in enumerate(np.unique(data.Subject)):
                #         tmp = computeMovingAverage(data=df_tmp[df_tmp.Subject==sub], bin_width=bin_width)
                #         tmp_all.append(tmp)
                    
                #     data_smoothed_permuted[permi, :, :] = tmp_all
            elif model == 'DvM':
                perms, permuted_angles, permuted_respErrors = perform_permute_SD_dvm(data=data_perms, dat2fit=collapseSubs, 
                                                                n_permutations=stats_n_permutations, bestParams_actualFit=None, 
                                                                fittingSteps=dog_fittingSteps)
                
            #Smooth permuted angles and respErrors (for plotting only)
            for permi, _ in enumerate(np.arange(stats_n_permutations)):
                df_tmp = pd.DataFrame(np.array([permuted_angles[permi, :], permuted_respErrors[permi, :]]).T, 
                                                                   columns=['Delta_angle_norm', 'Resp_error_demeaned'])
            
                #Insert subject variable & permutation variable
                df_tmp.insert(0, 'Subject', data_perms.Subject.values)
            
                #Compute moving averages (for plotting, so this will be done seperately for each subject)
                tmp_all = []
                for subi, sub in enumerate(np.unique(data.Subject)):
                    tmp = computeMovingAverage(data=df_tmp[df_tmp.Subject==sub], bin_width=bin_width)
                    tmp_all.append(tmp)
                
                data_smoothed_permuted[permi, :, :] = tmp_all
                
                # #Quick and dirty plot
                # fit_perm = dvm(np.deg2rad(np.linspace(-90, 90, 181)), perms[permi, 0], perms[permi, 1], 0)
                # fit_perm = np.rad2deg(fit_perm)
                # plt.figure()
                # plt.plot(np.mean(data_smoothed_permuted[permi, :, :], axis=0))
                # plt.plot(fit_perm)
                
            #Save
            np.save(path_results / filename_perms, perms)
            np.save(path_results / filename_perms_angles, permuted_angles)
            np.save(path_results / filename_perms_respErrors, permuted_respErrors)
            np.save(path_results / filename_perms_dataSmoothed, data_smoothed_permuted)
                
    '''
    ====================================================
    Bootstrapp confidence intervals
    ====================================================
    '''
    print('Bootstrapping confidence intervals')
    
    #Determine filenames
    filename_bootstrapp = 'Group/Bootstrapp/Bootstrapp_Study_' + currentStudy + '_' + model + '_' + collapseSubs + '_'  + savename + '.npy'
    
    if (collapseSubs == 'pooled') | (collapseSubs == 'singleSub'):
        
        #Load pre-existing data
        if rerun_bootstrapp == 0:
            bootstrappingCoefs = np.load(path_results / filename_bootstrapp)
        else:
            if model == 'DoG':
                bootstrappingCoefs = perform_bootstrapping_SD(data=data, dat2fit=collapseSubs, 
                                                            n_permutations=stats_n_permutations,  
                                                            fittingSteps=dog_fittingSteps)
            elif model == 'Clifford':
                bootstrappingCoefs = perform_bootstrapping_SD_clifford(data=data, dat2fit=collapseSubs, 
                                                                n_permutations=stats_n_permutations,  
                                                                fittingSteps=dog_fittingSteps)
            elif model == 'DvM':
                bootstrappingCoefs = perform_bootstrapping_SD_dvm(data=data, dat2fit=collapseSubs, 
                                                                n_permutations=stats_n_permutations,  
                                                                fittingSteps=dog_fittingSteps)

            #Save
            np.save(path_results / filename_bootstrapp, bootstrappingCoefs)
        
    '''
    ====================================================
    Plot smoothed data overlaid with fit
    ====================================================
    '''
    #Determine whether or not fitted peak is significant & do a quick and dirty plot
    if model == 'DoG':
        if np.squeeze(model_params)[0] > 0:
            significance = (np.sum(perms[:, 0] >= np.squeeze(model_params)[0])) / np.shape(perms)[0] #amplitude
        elif np.squeeze(model_params)[0] < 0:
            significance = (np.sum(perms[:, 0] <= np.squeeze(model_params)[0])) / np.shape(perms)[0] #amplitude
        significance_sse = (np.sum(perms[:, 4] <= np.squeeze(gof)[0])) / np.shape(perms)[0] #amplitude
        significance_rsq = (np.sum(perms[:, 5] >= np.squeeze(gof)[1])) / np.shape(perms)[0] #amplitude
        print('Significance of fitted peak is: ' +  str(significance))
        print('SSE significance of fitted peak is: ' +  str(significance_sse))
        print('RSq significance of fitted peak is: ' +  str(significance_rsq))
        
        #Compute fit
        fit = dog(np.linspace(-90, 90, 181), np.squeeze(model_params)[0], np.squeeze(model_params)[1])
    elif model == 'Clifford':
        
        #Fit
        fit = np.squeeze(model_params)[2] * clifford(np.deg2rad(np.linspace(-90, 90, 181)), np.squeeze(model_params)[0], np.squeeze(model_params)[1])
        peak2peak = np.rad2deg(np.squeeze(model_params)[2] * (fit.max()-fit.min()))
        fit= np.rad2deg(fit)
        
        #Significance
        significance = (np.sum(perms[:, 3] >= peak2peak)) / np.shape(perms)[0]
    
        print('Significance of fitted peak is: ' +  str(significance))
    elif model == 'DvM':
        
        #Compute actual fit and actual peak2peak
        fit = dvm(np.deg2rad(np.linspace(-90, 90, 181)), np.squeeze(model_params)[0], np.squeeze(model_params)[1], 0)
        fit = np.rad2deg(fit)
        peak2peak = np.sign(np.squeeze(model_params)[0]) * (fit.max()-fit.min())
        peak2peak = peak2peak / 2
        
        peak2peak_perms = perms[:, 2]
        peak2peak_perms = peak2peak_perms / 2
        
        if np.squeeze(model_params)[0] > 0:
            significance = (np.sum(peak2peak_perms >= peak2peak)) / np.shape(perms)[0] #amplitude
        elif np.squeeze(model_params)[0] < 0:
            significance = (np.sum(peak2peak_perms <= peak2peak)) / np.shape(perms)[0] #amplitude
        significance_sse = (np.sum(perms[:, 4] <= np.squeeze(gof)[0])) / np.shape(perms)[0] #amplitude
        significance_rsq = (np.sum(perms[:, 5] >= np.squeeze(gof)[1])) / np.shape(perms)[0] #amplitude
        print('Significance of fitted peak is: ' +  str(significance))
        print('SSE significance of fitted peak is: ' +  str(significance_sse))
        print('RSq significance of fitted peak is: ' +  str(significance_rsq))
        
    if significance < .05:
        plotStats = 'solid'
    elif (significance >= .05) & (significance <= .1):
        plotStats = 'dashed'
    else:
        plotStats = 'dotted'
    
    #Amplitude
    plt.figure()
    if model != 'DvM':
        plt.hist(perms[:, 0], 100) #actual distribution of permutations
        plt.plot([np.squeeze(model_params)[0], np.squeeze(model_params)[0]], [0, 45], '-r', linewidth=2, 
                  label='p =  ' + str(significance))
    else:
        plt.hist(peak2peak_perms, 100)
        plt.plot([peak2peak, peak2peak], [0, 45], '-r', linewidth=2, 
              label='p =  ' + str(significance))
    plt.xlabel('Permutation SD Amplitude')
    plt.ylabel('Permutation Samples')
    plt.title('Permuting amplitudes')
    plt.legend()
    
    #SSE
    plt.figure()
    plt.hist(perms[:, 4], 100) #actual distribution of permutations
    plt.plot([np.squeeze(gof)[0], np.squeeze(gof)[0]], [0, 45], '-r', linewidth=2, 
              label='p =  ' + str(significance_sse))
    plt.xlabel('Permutation SSE')
    plt.ylabel('Permutation Samples')
    plt.title('Permuting SSE')
    plt.legend()
    
    #RSq
    plt.figure()
    plt.hist(perms[:, 5], 100) #actual distribution of permutations
    plt.plot([np.squeeze(gof)[1], np.squeeze(gof)[1]], [0, 45], '-r', linewidth=2, 
              label='p =  ' + str(significance_rsq))
    plt.xlabel('Permutation RSq')
    plt.ylabel('Permutation Samples')
    plt.title('Permuting RSq')
    plt.legend()
    
    #Determine figure parameters
    if currentStudy == 'evBound':
        col = '#F4A460'
        my_title = 'Study 1'
    elif currentStudy == 'funcStates':
        col = '#b11226'
        my_title = 'Study 2'
    elif currentStudy == 'menRot':
        col = '#0061B5'
        my_title = 'Study 3'
    
    xlabel = 'Previous - current stimulus orientation (in deg)'
    ylabel = 'Response error on current trial (in deg)'
    
    #PlotSD
    if collapseSubs == 'pooled':
        ax = plotSD(x=np.linspace(-90, 90, 181), y=np.mean(data_smoothed, axis=0), 
                    yerr=scipy.sem(data_smoothed, axis=0), fit=fit,
                    my_title=my_title, col=col, 
                    xticks=[-90, -60, -30, 0, 30, 60, 90], xticklabels=None, xlabel=xlabel, xlim_split=False,
                    yticks=[-3.25, 0, 3.25], yticklabels=None, ylabel=ylabel, ylim_bottom=False, ylim_top=False,
                    axisBreak=0, axisXBreak=0, figsize=(4, 4), p_val=significance, plotStats=plotStats, 
                    factor_x=0.05, factor_y=0.05, collapseSubs=collapseSubs, label=None, label_fits=None, my_legend=True)
    elif collapseSubs == 'singleSub':
        ax = plotSD(x=np.linspace(-90, 90, 181), y=data_smoothed, 
                    yerr=None, fit=fit,
                    my_title=my_title + '- Subject ' + np.unique(data.Subject.values)[0], col=col, 
                    xticks=[-90, -60, -30, 0, 30, 60, 90], xticklabels=None, xlabel=xlabel, xlim_split=False,
                    yticks=[np.round(np.min(data_smoothed), decimals=1), 0, np.round(np.max(data_smoothed), decimals=1)], yticklabels=None, ylabel=ylabel, ylim_bottom=False, ylim_top=False,
                    axisBreak=0, axisXBreak=0, figsize=(4, 4), p_val=significance, plotStats=plotStats, 
                    factor_x=0.05, factor_y=0.05, collapseSubs=collapseSubs, label=None, label_fits=None, my_legend=True)
    
    #Save
    filename_tmp = 'Figures/PooledSD_Study_' + currentStudy + '_' + model + '_' + collapseSubs + '_'  + savename + '.svg'
    format_tmp = 'svg'

    filename = Path(path_results / filename_tmp)
    saveFig(plt.gcf(), filename, format=format_tmp)

    '''
    ====================================================
    Check bootstrapped distributions
    ====================================================
    '''
    if model == 'DoG':
        plt.figure()
        plt.hist(bootstrappingCoefs[:, 0], 100) #actual distribution of permutations
        plt.xlabel('Bootstrapping Amplitude')
        plt.ylabel('Bootstrapping Samples')
    elif model == 'Clifford':
        plt.figure()
        plt.hist(bootstrappingCoefs[:, 3], 100) #actual distribution of permutations
        plt.xlabel('Bootstrapping Peak2peak')
        plt.ylabel('Bootstrapping Samples')
    elif model == 'DvM':
        peak2peak_bootstrapp = bootstrappingCoefs[:, 2]
        peak2peak_bootstrapp = peak2peak_bootstrapp / 2
        plt.figure()
        plt.hist(peak2peak_bootstrapp, 100) #actual distribution of permutations
        plt.xlabel('Bootstrapping Peak2peak')
        plt.ylabel('Bootstrapping Samples')
    return

### Run model-free analysis ###
def run_Analysis_modelfreeSD(data, stats_n_permutations, 
                             rerun_fit, rerun_perms, savename, 
                             path_results):
    """
    :param data: pandas dataFrame with only those trials that will be used in the SD analysis
    :param stats_n_permutations: how many times to run the permutation
    :param rerun_fit: recompute fit?
    :param rerun_perms: rerun permutations?
    :param savename: specifics of analysis (i.e., which session, how many permutations, etc)
    :param path_results: where to save the data

    """

    #Imports
    import os.path
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import scipy.stats as scipy
    
    from pathlib import Path
    import pycircstat
    
    from JoV_Analysis_stats import perform_permute_modelFree
    from JoV_Analysis_basicPlottingFuncs import saveFig
    
    from JoV_Analysis_basicPlottingFuncs import saveFig
            
    '''
    ====================================================
    Initialize important variables
    ====================================================
    '''
    meanError_cw = np.zeros((len(np.unique(data.Subject)), 3)) #subjects x 3 sessions
    meanError_ccw = np.zeros((len(np.unique(data.Subject)), 3)) #subjects x 3 sessions
    
    permutations_bothSess = np.zeros((len(np.unique(data.Subject)), stats_n_permutations, 3))
    permutations_Sess1 = np.zeros((len(np.unique(data.Subject)), stats_n_permutations, 3))
    permutations_Sess2 = np.zeros((len(np.unique(data.Subject)), stats_n_permutations, 3))
    
    '''
    ====================================================
    Compute model-free measure: 1/subject
    ====================================================
    '''
    for subi, sub in enumerate(np.unique(data.Subject)):
        
        #Across both sessions
        tmp = data[(data.Subject==sub) & (data.incl_trialsCW==1)].Resp_error_demeaned.values
        tmp = tmp[~np.isnan(tmp)] #exclude remaining outlier trials
        meanError_cw[subi, 0] = np.rad2deg(pycircstat.mean(np.deg2rad(tmp)))
        
        tmp = data[(data.Subject==sub) & (data.incl_trialsCCW==1)].Resp_error_demeaned.values
        tmp = tmp[~np.isnan(tmp)] #exclude remaining outlier trials
        meanError_ccw[subi, 0] = np.rad2deg(pycircstat.mean(np.deg2rad(tmp)))
        
        #Session 1
        tmp = data[(data.Subject==sub) & (data.incl_trialsCW==1) & (data.Session==0)].Resp_error_demeaned.values
        tmp = tmp[~np.isnan(tmp)] #exclude remaining outlier trials
        meanError_cw[subi, 1] = np.rad2deg(pycircstat.mean(np.deg2rad(tmp)))
        
        tmp = data[(data.Subject==sub) & (data.incl_trialsCCW==1) & (data.Session==0)].Resp_error_demeaned.values
        tmp = tmp[~np.isnan(tmp)] #exclude remaining outlier trials
        meanError_ccw[subi, 1] = np.rad2deg(pycircstat.mean(np.deg2rad(tmp)))
        
        #Session 2
        tmp = data[(data.Subject==sub) & (data.incl_trialsCW==1) & (data.Session==1)].Resp_error_demeaned.values
        tmp = tmp[~np.isnan(tmp)] #exclude remaining outlier trials
        meanError_cw[subi, 2] = np.rad2deg(pycircstat.mean(np.deg2rad(tmp)))
        
        tmp = data[(data.Subject==sub) & (data.incl_trialsCCW==1) & (data.Session==1)].Resp_error_demeaned.values
        tmp = tmp[~np.isnan(tmp)] #exclude remaining outlier trials
        meanError_ccw[subi, 2] = np.rad2deg(pycircstat.mean(np.deg2rad(tmp)))
    
    #Bring into normal space
    meanError_cw = np.mod(meanError_cw+90, 180)-90
    meanError_ccw = np.mod(meanError_ccw+90, 180)-90
    
    #Take the difference
    modelfree_SD = np.subtract(meanError_cw, meanError_ccw)
    
    #Convert into a pandas dataframe for saving
    df = pd.DataFrame(data=modelfree_SD, columns=['Ampl_comb', 'Ampl_Sess1', 'Ampl_Sess2'])
    df.insert(0, 'Subject', np.unique(data.Subject))
    
    #Save
    filename = 'Stats/SD_' + savename + '.csv'
    df.to_csv(path_results / filename)
    
    '''
    ====================================================
    Print stats
    ====================================================
    '''
    
    print(scipy.ttest_1samp(modelfree_SD[:, 0], popmean=0, alternative='two-sided'))
    print(scipy.ttest_1samp(modelfree_SD[:, 1], popmean=0, alternative='two-sided'))
    print(scipy.ttest_1samp(modelfree_SD[:, 2], popmean=0, alternative='two-sided'))
    
    '''
    ====================================================
    Permute model-free measure
    ====================================================
    '''
    for subi, sub in enumerate(np.unique(data.Subject)):
        
        #Combined across both sessions
        print('Permuting model free across both sessions for subject: ' + sub)
        tmp = perform_permute_modelFree(data=data[data.Subject==sub], n_permutations=stats_n_permutations)
        permutations_bothSess[subi, :] = tmp
        
        #Session 1
        print('Permuting model free for Session 1 for subject: ' + sub)
        tmp = perform_permute_modelFree(data=data[(data.Subject==sub) & (data.Session==0)], n_permutations=stats_n_permutations)
        permutations_Sess1[subi, :] = tmp
        
        #Session 2
        print('Permuting model free for Session 2 for subject: ' + sub)
        tmp = perform_permute_modelFree(data=data[(data.Subject==sub) & (data.Session==1)], n_permutations=stats_n_permutations)
        permutations_Sess2[subi, :] = tmp
        
    #Save (to be sure)
    filename = 'Group/Perms/SD_BothSess_' + savename + '.npy'
    np.save(path_results / filename, permutations_bothSess)
    
    filename = 'Group/Perms/SD_Sess1_' + savename + '.npy'
    np.save(path_results / filename, permutations_Sess1)
    
    filename = 'Group/Perms/SD_Sess2_' + savename + '.npy'
    np.save(path_results / filename, permutations_Sess2)
    
    '''
    ====================================================
    Combine it into 1 big dataframe
    ====================================================
    '''
    
    df_perms = pd.DataFrame(data=np.reshape(permutations_bothSess, (len(np.unique(data.Subject))*stats_n_permutations, 3)), 
                                             columns = ['ErrorCW_comb', 'ErrorCCW_comb', 'Ampl_comb'])
    
    df_perms_Sess1 = pd.DataFrame(data=np.reshape(permutations_Sess1, (len(np.unique(data.Subject))*stats_n_permutations, 3)), 
                                             columns = ['ErrorCW_Sess1', 'ErrorCCW_Sess1', 'Ampl_Sess1'])
    df_perms_Sess2 = pd.DataFrame(data=np.reshape(permutations_Sess2, (len(np.unique(data.Subject))*stats_n_permutations, 3)), 
                                             columns = ['ErrorCW_Sess2', 'ErrorCCW_Sess2', 'Ampl_Sess2'])
    
    df_perms = df_perms.join(df_perms_Sess1)
    df_perms = df_perms.join(df_perms_Sess2)
    
    df_perms.insert(0, 'Study', np.repeat(np.unique(data.Study), len(df_perms)))
    df_perms.insert(1, 'Subject', np.repeat(np.unique(data.Subject), stats_n_permutations))
    df_perms.insert(2, 'Permutation', np.tile(np.linspace(0, stats_n_permutations-1, stats_n_permutations), len(np.unique(data.Subject))))

    #Save
    filename = 'Group/Perms/SD_All_' + savename + '.csv'
    df_perms.to_csv(path_results / filename)
    
    return

### Assess FWHM of DOG fits ###
def run_getFWHM(currentStudy, model, collapseSubs, savename, path_results):
    
    """
    :param currentStudy: which study are we talking about
    :param model: which model was fit
    :param collapseSubs: pooled | singleSub
    :param savename: which analysis file to load
    :param path_results: where to save the data

    """
    
    #Imports
    import os.path
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import scipy.stats as scipy
    
    from pathlib import Path
    import pycircstat
    
    from JoV_Analysis_basicFitting import dog, clifford, dvm
    from JoV_Analysis_basicPlottingFuncs import saveFig
    
    '''
    ====================================================
    Load pooled group fits
    ====================================================
    '''

    filename_model = ('Group/Fits/Study_' + currentStudy + '_' + model + '_bestParams_' + collapseSubs 
                        + '_'  + savename + '.csv')
    Fits = pd.read_csv(path_results / filename_model, index_col=0)
    
    '''
    ====================================================
    Compute and plot model 
    ====================================================
    '''
    if model == 'DoG':
        fit = dog(np.linspace(-90, 90, 181), Fits.Amplitude.values, Fits.Width.values)
        plt.plot(np.linspace(-90, 90, 181), fit)
    else:
        fit = dvm(np.deg2rad(np.linspace(-90, 90, 181)), Fits.Amplitude.values, Fits.Kappa.values, 0)
        fit = np.rad2deg(fit)
        plt.plot(np.linspace(-90, 90, 181), fit)

    '''
    ====================================================
    Compute FWHM
    ====================================================
    '''
    
    #Needed definitions
    def lin_interp(x, y, i, half):
        return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

    def half_max_x(x, y):
        half = max(y)/2.
        #half = min(y)
        signs = np.sign(np.add(y, -half))
        zero_crossings = (signs[0:-2] != signs[1:-1])
        zero_crossings_i = np.where(zero_crossings)[0]
        return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]
    
    #Data
    x=np.linspace(-90, 90, 181)
    
    #Find the two crossing points
    hmx = half_max_x(x, fit)
    
    #Print the answer
    fwhm = hmx[1] - hmx[0]
    print("FWHM:{:.3f}".format(fwhm))

    #Plot for quality check
    half = max(fit)/2.0
    plt.figure()
    plt.plot(x, fit)
    plt.plot(hmx, [half, half])
    #plt.show()
    
    '''
    ====================================================
    Compute broader width
    ====================================================
    '''
    if model == 'DoG':
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx
    
        percentCover = 0.998
        x_intercept = np.sqrt(-np.log(1-percentCover))/Fits.Width
    
        #Plot for quality check
        plt.figure()
        plt.plot(x, fit)
        ind_start =  int(np.squeeze(np.where(x==0)))
        ind_stop = find_nearest(x, x_intercept.values)
        plt.fill_between(x[ind_start:ind_stop+1], fit[ind_start:ind_stop+1])
    
        #Print the answer
        print("Bounds:{:.3f}".format(float(x_intercept.values)))
    
    return

### Run analysis with a moving average ###
def run_Analysis_SD_movingAvg(data, currentStudy, model, bins, bin_width, collapseSubs, 
                    dog_fittingSteps, stats_n_permutations, savename, 
                    path_results, rerun_fit, rerun_perms, rerun_bootstrapp):
    """
    :param data: pandas dataFrame with only those trials that will be used in the SD analysis
    :param currentStudy: which study are we currently looking at
    :param model: which model (i.e., DoG, Clifford, etc) to use to fit the data
    :param bins: how to bin x-axis (i.e., angles)
    :param bin_width: how to bin SD data
    :param collapseSubs: how to treat individual subjects (singleSub | groupMean | pooled)
    :param dog_fittingSteps: how many steps to consider when fitting dog (default: 200)
    :param stats_n_permutations: how many permutations to run (default: 10,000)
    :param savename: specifics of analysis (i.e., which session, how many permutations, etc)
    :param path_results: where to save the data
    :param rerun_fit: should we rerun the model fit or simply load from file?
    :param rerun_perms: should we rerun the permutations or simply load them from file?
    :param rerun_bootstrapp: should we rerun the bootstrapping or simply load them from file?
    """
    
    #Imports
    import os.path
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import scipy.stats as scipy
    
    from pathlib import Path
    
    from JoV_Analysis_basicFuncs import analysis_dog, analysis_clifford, analysis_dvm
    from JoV_Analysis_stats import (perform_permute_SD, perform_bootstrapping_SD, 
                                    perform_permute_SD_clifford, perform_bootstrapping_SD_clifford,
                                    perform_permute_SD_dvm, perform_bootstrapping_SD_dvm)
    from JoV_Analysis_SD import computeMovingAverage, plotSD
    from JoV_Analysis_basicFitting import dog, clifford, dvm
    from JoV_Analysis_basicPlottingFuncs import saveFig
        
    #Initialize important variables
    data_smoothed_tmp = np.zeros((len(np.unique(data.Subject)), 181))
    data_smoothed_permuted = np.zeros((stats_n_permutations, len(np.unique(data.Subject)), 181))
    
    if model == 'DoG':
        model_params = np.zeros((1, 3)) #store best fitting parameters
        gof = np.zeros((1, 5)) #store measures of goodness of fit
        perms = np.zeros((stats_n_permutations, 6)) #store permutation distributions
        bootstrappingCoefs = np.zeros((stats_n_permutations, 3))
        significance = np.zeros((1, 1))
    elif model == 'Clifford':
        model_params = np.zeros((1, 4)) #store best fitting parameters
        perms = np.zeros((stats_n_permutations, 5)) #store permutation distributions
        bootstrappingCoefs = np.zeros((stats_n_permutations, 4))
        significance = np.zeros((1, 1))
    elif model == 'DvM':
        model_params = np.zeros((1, 3)) #store best fitting parameters
        gof = np.zeros((1, 2)) #store measures of goodness of fit
        perms = np.zeros((stats_n_permutations, 6)) #store permutation distributions
        bootstrappingCoefs = np.zeros((stats_n_permutations, 6))
        significance = np.zeros((1, 1))
    
    '''
    ====================================================
    Smooth data 
    ====================================================
    '''
    #Extract relevant data
    if collapseSubs == 'pooled':

        #Compute moving averages (for plotting, so this will be done seperately for each subject)
        for subi, sub in enumerate(np.unique(data.Subject)):
            print('Smoothing data for subject: ' + sub)
            data_smoothed_tmp[subi, :] = computeMovingAverage(data[data.Subject==sub], bin_width=bin_width)
            
    elif collapseSubs == 'singleSub':
        data_smoothed_tmp = computeMovingAverage(data, bin_width=bin_width)
        data_smoothed = pd.DataFrame(columns=['Subject', 'Resp_error_demeaned', 'Delta_angle_norm'])
        data_smoothed.Resp_error_demeaned = data_smoothed_tmp
        data_smoothed.Delta_angle_norm = np.linspace(-90, 90, 181)
        data_smoothed.Subject = np.tile(data.Subject.values[0], len(data_smoothed))

    '''
    ====================================================
    Compute Model fit & save it
    ====================================================
    '''
    print('Fitting model')
    
    #Determine filenames
    filename_model = 'Group/Fits/Study_' + currentStudy + '_' + model + '_bestParams_' + collapseSubs + '_'  + savename + '.csv'
    
    if (model == 'DoG') | (model == 'DvM'):
        filename_gof = 'Group/Fits/Study_' + currentStudy + '_' + model + '_GoF_' + collapseSubs + '_'  + savename + '.csv'
    
    #Fit
    if (collapseSubs == 'pooled') | (collapseSubs == 'singleSub'):
        
        #Load pre-existing data
        if rerun_fit == 0:
            model_params = pd.read_csv(path_results / filename_model, index_col=0).to_numpy()
            
            if (model == 'DoG') | (model == 'DvM'):
                gof = pd.read_csv(path_results / filename_gof, index_col=0).to_numpy()
        else:
            if model == 'DoG':
                model_params, gof = analysis_dog(data=data_smoothed, dat2fit=collapseSubs,
                                                  fittingSteps=dog_fittingSteps)
                
                #Save
                pd.DataFrame(model_params, columns=['Amplitude', 'Width', 'MinCost']).to_csv(path_results / filename_model)
                pd.DataFrame(gof, columns=['SSE', 'RSquared', ' ', ' ', ' ']).to_csv(path_results / filename_gof)
            elif model == 'Clifford':
                model_params = analysis_clifford(data=data_smoothed, dat2fit=collapseSubs,
                                                  fittingSteps=dog_fittingSteps)

                #Save
                pd.DataFrame(model_params, columns=['Centering param', 'Scaling param', 'Sign', 'MinCost']).to_csv(path_results / filename_model)
            elif model == 'DvM':
                model_params, gof = analysis_dvm(data=data_smoothed, dat2fit=collapseSubs,
                                                  fittingSteps=dog_fittingSteps)
                
                #Save
                pd.DataFrame(model_params, columns=['Amplitude', 'Kappa', 'MinCost']).to_csv(path_results / filename_model)
                pd.DataFrame(gof, columns=['SSE', 'RSquared']).to_csv(path_results / filename_gof)
                
    '''
    ====================================================
    Permute model fits
    ====================================================
    '''
    print('Permuting model fits')
    
    #Determine filenames
    filename_perms = 'Group/Perms/Perms_Study_' + currentStudy + '_' + model + '_' + collapseSubs + '_'  + savename + '.npy'
    filename_perms_angles = 'Group/Perms/Perms_Angles_Study_' + currentStudy + '_' + model + '_' + collapseSubs + '_'  + savename + '.npy'
    filename_perms_respErrors = 'Group/Perms/Perms_RespErrors_Study_' + currentStudy + '_' + model + '_' + collapseSubs + '_'  + savename + '.npy'
    #filename_perms_dataSmoothed = 'Group/Perms/Perms_dataSmoothed_Study_' + currentStudy + '_' + model + '_' + collapseSubs + '_'  + savename + '.npy'

    #Permute fit
    if (collapseSubs == 'pooled') | (collapseSubs == 'singleSub'):
        
        #Load pre-existing data
        if rerun_perms == 0:
            perms = np.load(path_results / filename_perms)
            permuted_angles = np.load(path_results / filename_perms_angles)
            permuted_respErrors = np.load(path_results / filename_perms_respErrors)
            #data_smoothed_permuted = np.load(path_results / filename_perms_dataSmoothed)
        else:
            #Prep data for permutation (i.e., taking care of additional nans from outliers)
            data_perms = data_smoothed.copy(deep=True)
            data_perms = data_smoothed[(~np.isnan(data_smoothed.Delta_angle_norm))]
            data_perms = data_smoothed[(~np.isnan(data_smoothed.Resp_error_demeaned))]
            
            if model == 'DoG':
                perms, permuted_angles, permuted_respErrors = perform_permute_SD(data=data_perms, dat2fit=collapseSubs, 
                                                            n_permutations=stats_n_permutations, bestParams_actualFit=None, 
                                                            fittingSteps=dog_fittingSteps)
            
                # #Smooth permuted angles and respErrors (for plotting only)
                # for permi, _ in enumerate(np.arange(stats_n_permutations)):
                #     print('Smoothing permutations : ' + str(permi))
                #     df_tmp = pd.DataFrame(np.array([permuted_angles[permi, :], permuted_respErrors[permi, :]]).T, 
                #                                                        columns=['Delta_angle_norm', 'Resp_error_demeaned'])
                #     #Insert subject variable & permutation variable
                #     df_tmp.insert(0, 'Subject', data_perms.Subject.values)
                
                #     #Compute moving averages (for plotting, so this will be done seperately for each subject)
                #     tmp_all = []
                #     for subi, sub in enumerate(np.unique(data.Subject)):
                #         tmp = computeMovingAverage(data=df_tmp[df_tmp.Subject==sub], bin_width=bin_width)
                #         tmp_all.append(tmp)
                    
                #     data_smoothed_permuted[permi, :, :] = tmp_all

            elif model == 'Clifford':
                perms, permuted_angles, permuted_respErrors = perform_permute_SD_clifford(data=data_perms, dat2fit=collapseSubs, 
                                                                n_permutations=stats_n_permutations, bestParams_actualFit=None, 
                                                                fittingSteps=dog_fittingSteps)
                
                # #Smooth permuted angles and respErrors (for plotting only)
                # for permi, _ in enumerate(np.arange(stats_n_permutations)):
                #     df_tmp = pd.DataFrame(np.array([permuted_angles[permi, :], permuted_respErrors[permi, :]]).T, 
                #                                                        columns=['Delta_angle_norm', 'Resp_error_demeaned'])
                
                #     #Insert subject variable & permutation variable
                #     df_tmp.insert(0, 'Subject', data_perms.Subject.values)
                
                #     #Compute moving averages (for plotting, so this will be done seperately for each subject)
                #     tmp_all = []
                #     for subi, sub in enumerate(np.unique(data.Subject)):
                #         tmp = computeMovingAverage(data=df_tmp[df_tmp.Subject==sub], bin_width=bin_width)
                #         tmp_all.append(tmp)
                    
                #     data_smoothed_permuted[permi, :, :] = tmp_all
            elif model == 'DvM':
                perms, permuted_angles, permuted_respErrors = perform_permute_SD_dvm(data=data_perms, dat2fit=collapseSubs, 
                                                                n_permutations=stats_n_permutations, bestParams_actualFit=None, 
                                                                fittingSteps=dog_fittingSteps)
                
            # #Smooth permuted angles and respErrors (for plotting only)
            # for permi, _ in enumerate(np.arange(stats_n_permutations)):
            #     df_tmp = pd.DataFrame(np.array([permuted_angles[permi, :], permuted_respErrors[permi, :]]).T, 
            #                                                        columns=['Delta_angle_norm', 'Resp_error_demeaned'])
            
            #     #Insert subject variable & permutation variable
            #     df_tmp.insert(0, 'Subject', data_perms.Subject.values)
            
            #     #Compute moving averages (for plotting, so this will be done seperately for each subject)
            #     tmp_all = []
            #     for subi, sub in enumerate(np.unique(data.Subject)):
            #         tmp = computeMovingAverage(data=df_tmp[df_tmp.Subject==sub], bin_width=bin_width)
            #         tmp_all.append(tmp)
                
            #     data_smoothed_permuted[permi, :, :] = tmp_all
                
                # #Quick and dirty plot
                # fit_perm = dvm(np.deg2rad(np.linspace(-90, 90, 181)), perms[permi, 0], perms[permi, 1], 0)
                # fit_perm = np.rad2deg(fit_perm)
                # plt.figure()
                # plt.plot(np.mean(data_smoothed_permuted[permi, :, :], axis=0))
                # plt.plot(fit_perm)
                
            #Save
            np.save(path_results / filename_perms, perms)
            np.save(path_results / filename_perms_angles, permuted_angles)
            np.save(path_results / filename_perms_respErrors, permuted_respErrors)
            #np.save(path_results / filename_perms_dataSmoothed, data_smoothed_permuted)
                
    '''
    ====================================================
    Bootstrapp confidence intervals
    ====================================================
    '''
    print('Bootstrapping confidence intervals')
    
    #Determine filenames
    filename_bootstrapp = 'Group/Bootstrapp/Bootstrapp_Study_' + currentStudy + '_' + model + '_' + collapseSubs + '_'  + savename + '.npy'
    
    if (collapseSubs == 'pooled') | (collapseSubs == 'singleSub'):
        
        #Load pre-existing data
        if rerun_bootstrapp == 0:
            bootstrappingCoefs = np.load(path_results / filename_bootstrapp)
        else:
            if model == 'DoG':
                bootstrappingCoefs = perform_bootstrapping_SD(data=data, dat2fit=collapseSubs, 
                                                            n_permutations=stats_n_permutations,  
                                                            fittingSteps=dog_fittingSteps)
            elif model == 'Clifford':
                bootstrappingCoefs = perform_bootstrapping_SD_clifford(data=data, dat2fit=collapseSubs, 
                                                                n_permutations=stats_n_permutations,  
                                                                fittingSteps=dog_fittingSteps)
            elif model == 'DvM':
                bootstrappingCoefs = perform_bootstrapping_SD_dvm(data=data_smoothed, dat2fit=collapseSubs, 
                                                                n_permutations=stats_n_permutations,  
                                                                fittingSteps=dog_fittingSteps)

            #Save
            np.save(path_results / filename_bootstrapp, bootstrappingCoefs)
        
    '''
    ====================================================
    Plot smoothed data overlaid with fit
    ====================================================
    '''
    #Determine whether or not fitted peak is significant & do a quick and dirty plot
    if model == 'DoG':
        if np.squeeze(model_params)[0] > 0:
            significance = (np.sum(perms[:, 0] >= np.squeeze(model_params)[0])) / np.shape(perms)[0] #amplitude
        elif np.squeeze(model_params)[0] < 0:
            significance = (np.sum(perms[:, 0] <= np.squeeze(model_params)[0])) / np.shape(perms)[0] #amplitude
        significance_sse = (np.sum(perms[:, 4] <= np.squeeze(gof)[0])) / np.shape(perms)[0] #amplitude
        significance_rsq = (np.sum(perms[:, 5] >= np.squeeze(gof)[1])) / np.shape(perms)[0] #amplitude
        print('Significance of fitted peak is: ' +  str(significance))
        print('SSE significance of fitted peak is: ' +  str(significance_sse))
        print('RSq significance of fitted peak is: ' +  str(significance_rsq))
        
        #Compute fit
        fit = dog(np.linspace(-90, 90, 181), np.squeeze(model_params)[0], np.squeeze(model_params)[1])
    elif model == 'Clifford':
        
        #Fit
        fit = np.squeeze(model_params)[2] * clifford(np.deg2rad(np.linspace(-90, 90, 181)), np.squeeze(model_params)[0], np.squeeze(model_params)[1])
        peak2peak = np.rad2deg(np.squeeze(model_params)[2] * (fit.max()-fit.min()))
        fit= np.rad2deg(fit)
        
        #Significance
        significance = (np.sum(perms[:, 3] >= peak2peak)) / np.shape(perms)[0]
    
        print('Significance of fitted peak is: ' +  str(significance))
    elif model == 'DvM':
        
        #Compute actual fit and actual peak2peak
        fit = dvm(np.deg2rad(np.linspace(-90, 90, 181)), np.squeeze(model_params)[0], np.squeeze(model_params)[1], 0)
        fit = np.rad2deg(fit)
        peak2peak = np.sign(np.squeeze(model_params)[0]) * (fit.max()-fit.min())
        peak2peak = peak2peak / 2
        
        peak2peak_perms = perms[:, 2]
        peak2peak_perms = peak2peak_perms / 2
        
        if np.squeeze(model_params)[0] > 0:
            significance = (np.sum(peak2peak_perms >= peak2peak)) / np.shape(perms)[0] #amplitude
        elif np.squeeze(model_params)[0] < 0:
            significance = (np.sum(peak2peak_perms <= peak2peak)) / np.shape(perms)[0] #amplitude
        significance_sse = (np.sum(perms[:, 4] <= np.squeeze(gof)[0])) / np.shape(perms)[0] #amplitude
        significance_rsq = (np.sum(perms[:, 5] >= np.squeeze(gof)[1])) / np.shape(perms)[0] #amplitude
        print('Significance of fitted peak is: ' +  str(significance))
        print('SSE significance of fitted peak is: ' +  str(significance_sse))
        print('RSq significance of fitted peak is: ' +  str(significance_rsq))
        
    if significance < .05:
        plotStats = 'solid'
    elif (significance >= .05) & (significance <= .1):
        plotStats = 'dashed'
    else:
        plotStats = 'dotted'
    
    #Amplitude
    plt.figure()
    if model != 'DvM':
        plt.hist(perms[:, 0], 100) #actual distribution of permutations
        plt.plot([np.squeeze(model_params)[0], np.squeeze(model_params)[0]], [0, 45], '-r', linewidth=2, 
                  label='p =  ' + str(significance))
    else:
        plt.hist(peak2peak_perms, 100)
        plt.plot([peak2peak, peak2peak], [0, 45], '-r', linewidth=2, 
              label='p =  ' + str(significance))
    plt.xlabel('Permutation SD Amplitude')
    plt.ylabel('Permutation Samples')
    plt.title('Permuting amplitudes')
    plt.legend()
    
    #SSE
    plt.figure()
    plt.hist(perms[:, 4], 100) #actual distribution of permutations
    plt.plot([np.squeeze(gof)[0], np.squeeze(gof)[0]], [0, 45], '-r', linewidth=2, 
              label='p =  ' + str(significance_sse))
    plt.xlabel('Permutation SSE')
    plt.ylabel('Permutation Samples')
    plt.title('Permuting SSE')
    plt.legend()
    
    #RSq
    plt.figure()
    plt.hist(perms[:, 5], 100) #actual distribution of permutations
    plt.plot([np.squeeze(gof)[1], np.squeeze(gof)[1]], [0, 45], '-r', linewidth=2, 
              label='p =  ' + str(significance_rsq))
    plt.xlabel('Permutation RSq')
    plt.ylabel('Permutation Samples')
    plt.title('Permuting RSq')
    plt.legend()
    
    #Determine figure parameters
    if currentStudy == 'evBound':
        col = '#F4A460'
        my_title = 'Study 1'
    elif currentStudy == 'funcStates':
        col = '#b11226'
        my_title = 'Study 2'
    elif currentStudy == 'menRot':
        col = '#0061B5'
        my_title = 'Study 3'
    
    xlabel = 'Previous - current stimulus orientation (in deg)'
    ylabel = 'Response error on current trial (in deg)'
    
    #PlotSD
    if collapseSubs == 'pooled':
        ax = plotSD(x=np.linspace(-90, 90, 181), y=np.mean(data_smoothed, axis=0), 
                    yerr=scipy.sem(data_smoothed, axis=0), fit=fit,
                    my_title=my_title, col=col, 
                    xticks=[-90, -60, -30, 0, 30, 60, 90], xticklabels=None, xlabel=xlabel, xlim_split=False,
                    yticks=[-3.25, 0, 3.25], yticklabels=None, ylabel=ylabel, ylim_bottom=False, ylim_top=False,
                    axisBreak=0, axisXBreak=0, figsize=(4, 4), p_val=significance, plotStats=plotStats, 
                    factor_x=0.05, factor_y=0.05, collapseSubs=collapseSubs, label=None, label_fits=None, my_legend=True)
    elif collapseSubs == 'singleSub':
        ax = plotSD(x=np.linspace(-90, 90, 181), y=data_smoothed.Resp_error_demeaned.values, 
                    yerr=None, fit=fit,
                    my_title=my_title + '- Subject ' + np.unique(data.Subject.values)[0], col=col, 
                    xticks=[-90, -60, -30, 0, 30, 60, 90], xticklabels=None, xlabel=xlabel, xlim_split=False,
                    yticks=[np.round(np.min(data_smoothed.Resp_error_demeaned.values), decimals=1), 0, np.round(np.max(data_smoothed.Resp_error_demeaned.values), decimals=1)], yticklabels=None, ylabel=ylabel, ylim_bottom=False, ylim_top=False,
                    axisBreak=0, axisXBreak=0, figsize=(4, 4), p_val=significance, plotStats=plotStats, 
                    factor_x=0.05, factor_y=0.05, collapseSubs=collapseSubs, label=None, label_fits=None, my_legend=True)
    
    #Save
    filename_tmp = 'Figures/PooledSD_Study_' + currentStudy + '_' + model + '_' + collapseSubs + '_'  + savename + '.svg'
    format_tmp = 'svg'

    filename = Path(path_results / filename_tmp)
    saveFig(plt.gcf(), filename, format=format_tmp)

    '''
    ====================================================
    Check bootstrapped distributions
    ====================================================
    '''
    if model == 'DoG':
        plt.figure()
        plt.hist(bootstrappingCoefs[:, 0], 100) #actual distribution of permutations
        plt.xlabel('Bootstrapping Amplitude')
        plt.ylabel('Bootstrapping Samples')
    elif model == 'Clifford':
        plt.figure()
        plt.hist(bootstrappingCoefs[:, 3], 100) #actual distribution of permutations
        plt.xlabel('Bootstrapping Peak2peak')
        plt.ylabel('Bootstrapping Samples')
    elif model == 'DvM':
        peak2peak_bootstrapp = bootstrappingCoefs[:, 2]
        peak2peak_bootstrapp = peak2peak_bootstrapp / 2
        plt.figure()
        plt.hist(peak2peak_bootstrapp, 100) #actual distribution of permutations
        plt.xlabel('Bootstrapping Peak2peak')
        plt.ylabel('Bootstrapping Samples')
    return

### Run model-free analysis ###
def run_Analysis_modelfreeSD(data, stats_n_permutations, 
                             rerun_fit, rerun_perms, savename, 
                             path_results):
    """
    :param data: pandas dataFrame with only those trials that will be used in the SD analysis
    :param stats_n_permutations: how many times to run the permutation
    :param rerun_fit: recompute fit?
    :param rerun_perms: rerun permutations?
    :param savename: specifics of analysis (i.e., which session, how many permutations, etc)
    :param path_results: where to save the data

    """

    #Imports
    import os.path
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import scipy.stats as scipy
    
    from pathlib import Path
    import pycircstat
    
    from JoV_Analysis_stats import perform_permute_modelFree
    from JoV_Analysis_basicPlottingFuncs import saveFig
    
    from JoV_Analysis_basicPlottingFuncs import saveFig
            
    '''
    ====================================================
    Initialize important variables
    ====================================================
    '''
    meanError_cw = np.zeros((len(np.unique(data.Subject)), 3)) #subjects x 3 sessions
    meanError_ccw = np.zeros((len(np.unique(data.Subject)), 3)) #subjects x 3 sessions
    
    permutations_bothSess = np.zeros((len(np.unique(data.Subject)), stats_n_permutations, 3))
    permutations_Sess1 = np.zeros((len(np.unique(data.Subject)), stats_n_permutations, 3))
    permutations_Sess2 = np.zeros((len(np.unique(data.Subject)), stats_n_permutations, 3))
    
    '''
    ====================================================
    Compute model-free measure: 1/subject
    ====================================================
    '''
    for subi, sub in enumerate(np.unique(data.Subject)):
        
        #Across both sessions
        tmp = data[(data.Subject==sub) & (data.incl_trialsCW==1)].Resp_error_demeaned.values
        tmp = tmp[~np.isnan(tmp)] #exclude remaining outlier trials
        meanError_cw[subi, 0] = np.rad2deg(pycircstat.mean(np.deg2rad(tmp)))
        
        tmp = data[(data.Subject==sub) & (data.incl_trialsCCW==1)].Resp_error_demeaned.values
        tmp = tmp[~np.isnan(tmp)] #exclude remaining outlier trials
        meanError_ccw[subi, 0] = np.rad2deg(pycircstat.mean(np.deg2rad(tmp)))
        
        #Session 1
        tmp = data[(data.Subject==sub) & (data.incl_trialsCW==1) & (data.Session==0)].Resp_error_demeaned.values
        tmp = tmp[~np.isnan(tmp)] #exclude remaining outlier trials
        meanError_cw[subi, 1] = np.rad2deg(pycircstat.mean(np.deg2rad(tmp)))
        
        tmp = data[(data.Subject==sub) & (data.incl_trialsCCW==1) & (data.Session==0)].Resp_error_demeaned.values
        tmp = tmp[~np.isnan(tmp)] #exclude remaining outlier trials
        meanError_ccw[subi, 1] = np.rad2deg(pycircstat.mean(np.deg2rad(tmp)))
        
        #Session 2
        tmp = data[(data.Subject==sub) & (data.incl_trialsCW==1) & (data.Session==1)].Resp_error_demeaned.values
        tmp = tmp[~np.isnan(tmp)] #exclude remaining outlier trials
        meanError_cw[subi, 2] = np.rad2deg(pycircstat.mean(np.deg2rad(tmp)))
        
        tmp = data[(data.Subject==sub) & (data.incl_trialsCCW==1) & (data.Session==1)].Resp_error_demeaned.values
        tmp = tmp[~np.isnan(tmp)] #exclude remaining outlier trials
        meanError_ccw[subi, 2] = np.rad2deg(pycircstat.mean(np.deg2rad(tmp)))
    
    #Bring into normal space
    meanError_cw = np.mod(meanError_cw+90, 180)-90
    meanError_ccw = np.mod(meanError_ccw+90, 180)-90
    
    #Take the difference
    modelfree_SD = np.subtract(meanError_cw, meanError_ccw)
    
    #Convert into a pandas dataframe for saving
    df = pd.DataFrame(data=modelfree_SD, columns=['Ampl_comb', 'Ampl_Sess1', 'Ampl_Sess2'])
    df.insert(0, 'Subject', np.unique(data.Subject))
    
    #Save
    filename = 'Stats/SD_' + savename + '.csv'
    df.to_csv(path_results / filename)
    
    '''
    ====================================================
    Print stats
    ====================================================
    '''
    
    print(scipy.ttest_1samp(modelfree_SD[:, 0], popmean=0, alternative='two-sided'))
    print(scipy.ttest_1samp(modelfree_SD[:, 1], popmean=0, alternative='two-sided'))
    print(scipy.ttest_1samp(modelfree_SD[:, 2], popmean=0, alternative='two-sided'))
    
    '''
    ====================================================
    Permute model-free measure
    ====================================================
    '''
    for subi, sub in enumerate(np.unique(data.Subject)):
        
        #Combined across both sessions
        print('Permuting model free across both sessions for subject: ' + sub)
        tmp = perform_permute_modelFree(data=data[data.Subject==sub], n_permutations=stats_n_permutations)
        permutations_bothSess[subi, :] = tmp
        
        #Session 1
        print('Permuting model free for Session 1 for subject: ' + sub)
        tmp = perform_permute_modelFree(data=data[(data.Subject==sub) & (data.Session==0)], n_permutations=stats_n_permutations)
        permutations_Sess1[subi, :] = tmp
        
        #Session 2
        print('Permuting model free for Session 2 for subject: ' + sub)
        tmp = perform_permute_modelFree(data=data[(data.Subject==sub) & (data.Session==1)], n_permutations=stats_n_permutations)
        permutations_Sess2[subi, :] = tmp
        
    #Save (to be sure)
    filename = 'Group/Perms/SD_BothSess_' + savename + '.npy'
    np.save(path_results / filename, permutations_bothSess)
    
    filename = 'Group/Perms/SD_Sess1_' + savename + '.npy'
    np.save(path_results / filename, permutations_Sess1)
    
    filename = 'Group/Perms/SD_Sess2_' + savename + '.npy'
    np.save(path_results / filename, permutations_Sess2)
    
    '''
    ====================================================
    Combine it into 1 big dataframe
    ====================================================
    '''
    
    df_perms = pd.DataFrame(data=np.reshape(permutations_bothSess, (len(np.unique(data.Subject))*stats_n_permutations, 3)), 
                                             columns = ['ErrorCW_comb', 'ErrorCCW_comb', 'Ampl_comb'])
    
    df_perms_Sess1 = pd.DataFrame(data=np.reshape(permutations_Sess1, (len(np.unique(data.Subject))*stats_n_permutations, 3)), 
                                             columns = ['ErrorCW_Sess1', 'ErrorCCW_Sess1', 'Ampl_Sess1'])
    df_perms_Sess2 = pd.DataFrame(data=np.reshape(permutations_Sess2, (len(np.unique(data.Subject))*stats_n_permutations, 3)), 
                                             columns = ['ErrorCW_Sess2', 'ErrorCCW_Sess2', 'Ampl_Sess2'])
    
    df_perms = df_perms.join(df_perms_Sess1)
    df_perms = df_perms.join(df_perms_Sess2)
    
    df_perms.insert(0, 'Study', np.repeat(np.unique(data.Study), len(df_perms)))
    df_perms.insert(1, 'Subject', np.repeat(np.unique(data.Subject), stats_n_permutations))
    df_perms.insert(2, 'Permutation', np.tile(np.linspace(0, stats_n_permutations-1, stats_n_permutations), len(np.unique(data.Subject))))

    #Save
    filename = 'Group/Perms/SD_All_' + savename + '.csv'
    df_perms.to_csv(path_results / filename)
    
    return


#plt.plot(np.mean(data_smoothed, axis=0))
#plt.plot(fit)
