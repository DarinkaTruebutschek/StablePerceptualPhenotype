#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 15:55:37 2022

@author: darinka

#Purpose: All functions necessary to plot SD analyses from multiple studies.
#Author: Darinka Truebutschek
#Date created: 25/11/2022
#Date last modified: 25/11/2022
#Python version: 3.7.1

"""

def plotBar(x, y, my_title, col, xticks, xticklabels, xlabel,
            yticks, yticklabels, ylabel, ylim_bottom, ylim_top,
            axisBreak, figsize, p_val, plotStats, factor_x, factor_y, ax, bootstrapp):
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
    :param bootstrapp: are we plotting bootstrapped SD?
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

        #Errorbars if wanted
        if (bootstrapp is not None) & (np.max(x) < 3.5): #when only plotting two bars
            #err = np.std(bootstrapp[:, :, 0], axis=1) #this is for the DoG fit
            err = np.std(bootstrapp[:, :, 2] / 2, axis=1) #had only computed peak2peak
            print('Bootstrapping standard deviation: ' + str(err))
            ax[ax_i].bar(x=x, height=y, color=col, yerr=err, ecolor=col, linewidth=2)
        else:
            idx = [0, 2, 1, 3] #S1_Dog, S1_DvM, S2_Dog, S2_DvM
            ax[ax_i].bar(x=x, height=np.reshape(y, (4, ))[idx], color=col, yerr=np.reshape(bootstrapp, (4, ))[idx], ecolor=col, linewidth=2)

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
            
        #Significance
        if p_val is not None:
            n_comps = np.sum(plotStats['alpha']<.05)
            
            for compi, comps in enumerate(np.arange(n_comps)):
                alpha = plotStats['alpha'][compi]
                    
                add_stats(ax=ax[0], alpha=alpha, x1=plotStats['x1'][compi], 
                          x2=plotStats['x2'][compi], y=plotStats['y'][compi], h=plotStats['h'][compi], 
                          vertLines=plotStats['vertLines'][compi], rotation=plotStats['rotation'][compi])
            
        #Insert axis break if wanted
        if axisBreak:
            breakAxis(ax=ax, ylim_b=(ax[0].get_ylim()[0], ylim_bottom), ylim_t=(ylim_top, ax[0].get_ylim()[1]))

    return ax

def run_plot_SD(data, sess2plot, model, collapseSubs, stats_n_permutations, my_sig,
                savename, bin_width, path_results):
    """
    :param data: raw data to be plotted
    :param sess2plot: which session to plot
    :param model: which model was used to fit SD
    :param collapseSubs: how was SD computed
    :param stats_n_permutations: how many permutations had been run
    :param my_sig: which distribution to use for significance (i.e., amplitude vs. model fit)
    :param savename: initial saveNames used 
    :param bin_width: which bin width to use for smoothing
    :param path_results: where to plot the data
    """
    
    #Imports
    import os.path
    
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.stats as scipy
    
    from pathlib import Path
    
    from JoV_Analysis_basicFuncs import dim
    
    from JoV_Analysis_SD import computeMovingAverage, plotSD
    from JoV_Analysis_basicFitting import dog, clifford, dvm
    from JoV_Analysis_PlotSD import plotBar
    from JoV_Analysis_basicPlottingFuncs import saveFig
    
    '''
    ====================================================
    Determine important variables
    ====================================================
    '''
    currentStudies = np.unique(data.Study)
    
    currentSubs = []
    for studi, study in enumerate(currentStudies):
        tmp = np.unique(data.Subject[data.Study==study])
        currentSubs.append(tmp)
    
    data_smoothed = np.zeros((len(currentStudies), np.max(dim(currentSubs)), 181))
    data_smoothed[:] = np.nan   
    
    '''
    ====================================================
    Smooth the raw data (for plotting)
    ====================================================
    '''
    for studi, study in enumerate(currentStudies):
        for subi, sub in enumerate(np.unique(data.Subject[data.Study==study])):
            print('Smoothing data for subject: ' + sub)
            data_smoothed[studi, subi, :] = computeMovingAverage(data[(data.Study==study) & (data.Subject==sub)], 
                                                                 bin_width=bin_width)
    
    '''
    ====================================================
    Load all of the other data necessary for plotting
    ====================================================
    '''
    if model == 'DoG':
        Fits = pd.DataFrame(columns=['Study', 'Amplitude', 'Width', 'MinCost', 'SSE', 'RSquared'])
        Perms = np.zeros((len(currentStudies), stats_n_permutations, 6))
        Bootstrapps = np.zeros((len(currentStudies), stats_n_permutations, 3))
        
        for studi, study in enumerate(currentStudies):
            
            #Fits
            filename_model = ('Group/Fits/Study_' + study + '_' + model + '_bestParams_' 
                                + collapseSubs + '_'  + savename[studi] + '.csv')
            filename_gof = ('Group/Fits/Study_' + study + '_' + model + '_GoF_' 
                                + collapseSubs + '_'  + savename[studi] + '.csv')
            
            tmp = pd.read_csv(path_results / filename_model, index_col=0)
            tmp_gof = pd.read_csv(path_results / filename_gof, index_col=0)
            
            tmp.insert(0, 'Study', study)
            tmp.insert(4, 'SSE', tmp_gof.SSE.values)
            tmp.insert(5, 'RSquared', tmp_gof.RSquared.values)

            Fits = pd.concat((Fits, tmp))

            #Permutations
            filename_perms = ('Group/Perms/Perms_Study_' + study + '_' + model + '_' 
                              + collapseSubs + '_'  + savename[studi] + '.npy')
            Perms[studi, :, :] = np.load(path_results / filename_perms)
            
            #Bootstrapps
            filename_bootstrapp = ('Group/Bootstrapp/Bootstrapp_Study_' + study + '_' + model + '_' 
                                   + collapseSubs + '_'  + savename[studi] + '.npy')
            
            Bootstrapps[studi, :, :] = np.load(path_results / filename_bootstrapp)
            
            #Save
            filename_groupFits = ('Stats/GroupFits_Study_' + study + '_' + model + '_' 
                              + collapseSubs + '_'  + savename[studi] + '.csv')
            Fits.to_csv(path_results / filename_groupFits)
    elif model == 'DvM':
        Fits = pd.DataFrame(columns=['Study', 'Amplitude', 'Kappa', 'MinCost', 'SSE', 'RSquared', 'Peak2peak'])
        Perms = np.zeros((len(currentStudies), stats_n_permutations, 6))
        Bootstrapps = np.zeros((len(currentStudies), stats_n_permutations, 6))
        
        for studi, study in enumerate(currentStudies):

            #Fits DvM
            filename_model = ('Group/Fits/Study_' + study + '_DvM_bestParams_' 
                                + collapseSubs + '_'  + savename[studi] + '.csv')
            filename_gof = ('Group/Fits/Study_' + study + '_DvM_GoF_' 
                                + collapseSubs + '_'  + savename[studi] + '.csv')
            
            tmp = pd.read_csv(path_results / filename_model, index_col=0)
            tmp_gof = pd.read_csv(path_results / filename_gof, index_col=0)
            
            tmp.insert(0, 'Study', study)
            tmp.insert(4, 'SSE', tmp_gof.SSE.values)
            tmp.insert(5, 'RSquared', tmp_gof.RSquared.values)
            
            #Compute half peak2peak
            fit = dvm(np.deg2rad(np.linspace(-90, 90, 181)), tmp.Amplitude.values[0], tmp.Kappa.values[0], 0)
            fit = np.rad2deg(fit)
            peak2peak = np.sign(tmp.Amplitude.values[0]) * (fit.max()-fit.min())
            peak2peak = peak2peak / 2
            
            tmp.insert(6, 'Peak2peak', peak2peak)

            Fits = pd.concat((Fits, tmp))
            
            #Permutations
            filename_perms = ('Group/Perms/Perms_Study_' + study + '_DvM_' 
                              + collapseSubs + '_'  + savename[studi] + '.npy')
            Perms[studi, :, :] = np.load(path_results / filename_perms)
            
            #Bootstrapps
            filename_bootstrapp = ('Group/Bootstrapp/Bootstrapp_Study_' + study + '_DvM_' 
                                   + collapseSubs + '_'  + savename[studi] + '.npy')
            
            Bootstrapps[studi, :, :] = np.load(path_results / filename_bootstrapp)
            
            #Save
            filename_groupFits = ('Stats/GroupFits_Study_' + study + '_DvM_' 
                              + collapseSubs + '_'  + savename[studi] + '.csv')
            Fits.to_csv(path_results / filename_groupFits)
    elif model == 'DoG&DvM':
        Fits_dog = pd.DataFrame(columns=['Study', 'Amplitude', 'Width', 'MinCost', 'SSE', 'RSquared'])
        Perms_dog = np.zeros((len(currentStudies), stats_n_permutations, 6))
        Bootstrapps_dog = np.zeros((len(currentStudies), stats_n_permutations, 3))
        Fits_dvm = pd.DataFrame(columns=['Study', 'Amplitude', 'Kappa', 'MinCost', 'SSE', 'RSquared', 'Peak2peak'])
        Perms_dvm = np.zeros((len(currentStudies), stats_n_permutations, 6))
        Bootstrapps_dvm = np.zeros((len(currentStudies), stats_n_permutations, 6))
        
        for studi, study in enumerate(currentStudies):
            
            #Fits DoG
            filename_model_dog = ('Group/Fits/Study_' + study + '_DoG_bestParams_' 
                                + collapseSubs + '_'  + savename[studi] + '.csv')
            filename_gof_dog = ('Group/Fits/Study_' + study + '_DoG_GoF_' 
                                + collapseSubs + '_'  + savename[studi] + '.csv')
            
            tmp = pd.read_csv(path_results / filename_model_dog, index_col=0)
            tmp_gof = pd.read_csv(path_results / filename_gof_dog, index_col=0)
            
            tmp.insert(0, 'Study', study)
            tmp.insert(4, 'SSE', tmp_gof.SSE.values)
            tmp.insert(5, 'RSquared', tmp_gof.RSquared.values)

            Fits_dog = pd.concat((Fits_dog, tmp))
            
            #Fits DvM
            filename_model_dvm = ('Group/Fits/Study_' + study + '_DvM_bestParams_' 
                                + collapseSubs + '_'  + savename[studi] + '.csv')
            filename_gof_dvm = ('Group/Fits/Study_' + study + '_DvM_GoF_' 
                                + collapseSubs + '_'  + savename[studi] + '.csv')
            
            tmp = pd.read_csv(path_results / filename_model_dvm, index_col=0)
            tmp_gof = pd.read_csv(path_results / filename_gof_dvm, index_col=0)
            
            tmp.insert(0, 'Study', study)
            tmp.insert(4, 'SSE', tmp_gof.SSE.values)
            tmp.insert(5, 'RSquared', tmp_gof.RSquared.values)
            
            #Compute half peak2peak
            fit = dvm(np.deg2rad(np.linspace(-90, 90, 181)), tmp.Amplitude.values[0], tmp.Kappa.values[0], 0)
            fit = np.rad2deg(fit)
            peak2peak = np.sign(tmp.Amplitude.values[0]) * (fit.max()-fit.min())
            peak2peak = peak2peak / 2
            
            tmp.insert(6, 'Peak2peak', peak2peak)

            Fits_dvm = pd.concat((Fits_dvm, tmp))
            
            #Permutations
            filename_perms_dog = ('Group/Perms/Perms_Study_' + study + '_DoG_' 
                              + collapseSubs + '_'  + savename[studi] + '.npy')
            Perms_dog[studi, :, :] = np.load(path_results / filename_perms_dog)
            
            filename_perms_dvm = ('Group/Perms/Perms_Study_' + study + '_DvM_' 
                              + collapseSubs + '_'  + savename[studi] + '.npy')
            Perms_dvm[studi, :, :] = np.load(path_results / filename_perms_dvm)
            
            #Bootstrapps
            filename_bootstrapp_dog = ('Group/Bootstrapp/Bootstrapp_Study_' + study + '_DoG_' 
                                   + collapseSubs + '_'  + savename[studi] + '.npy')
            
            Bootstrapps_dog[studi, :, :] = np.load(path_results / filename_bootstrapp_dog)
            
            filename_bootstrapp_dvm = ('Group/Bootstrapp/Bootstrapp_Study_' + study + '_DvM_' 
                                   + collapseSubs + '_'  + savename[studi] + '.npy')
            
            Bootstrapps_dvm[studi, :, :] = np.load(path_results / filename_bootstrapp_dvm)
            
            #Save
            filename_groupFits_dog = ('Stats/GroupFits_Study_' + study + '_DoG_' 
                              + collapseSubs + '_'  + savename[studi] + '.csv')
            Fits_dog.to_csv(path_results / filename_groupFits_dog)

            filename_groupFits_dvm = ('Stats/GroupFits_Study_' + study + '_DvM_' 
                              + collapseSubs + '_'  + savename[studi] + '.csv')
            Fits_dvm.to_csv(path_results / filename_groupFits_dvm)

    '''
    ====================================================
    Determine statistical significance & compute fits
    ====================================================
    '''
    if model == 'DoG':
        significance = np.zeros((len(currentStudies)))
        plotStats = []
        model_fits = np.zeros((len(currentStudies), len(np.linspace(-90, 90, 181))))
    
        for studi, study in enumerate(currentStudies):
            if my_sig == 'amplitude':
                significance[studi] = (np.sum(Perms[studi, :, 0] >= Fits.Amplitude[Fits.Study==study].values)) / np.shape(Perms)[1]
            elif my_sig == 'rsquared':
               significance[studi] = (np.sum(Perms[studi, :, 0] >= Fits.Amplitude[Fits.Study==study].values)) / np.shape(Perms)[1] 
               print('Amplitude significance for study ' + study + ': ' + str(significance))
               
               significance[studi] = (np.sum(Perms[studi, :, 5] >= Fits.RSquared[Fits.Study==study].values)) / np.shape(Perms)[1]
               print('RSquared significance for study ' + study + ': ' + str(significance))
               
            model_fits[studi, :] = dog(np.linspace(-90, 90, 181), Fits.Amplitude[Fits.Study==study].values, Fits.Width[Fits.Study==study].values)

            
            if significance[studi] < .05:
                tmp = 'solid'
            elif (significance[studi] >= .05) & (significance[studi] <= .1):
                tmp = 'dashed'
            else:
                tmp = 'dotted'
        
            plotStats.append(tmp)
    elif model == 'DvM':
        significance = np.zeros((len(currentStudies)))
        plotStats = []
        model_fits = np.zeros((len(currentStudies), len(np.linspace(-90, 90, 181))))
        
        for studi, study in enumerate(currentStudies):
            peak2peak_perms = Perms[studi, :, 2]
            peak2peak_perms = peak2peak_perms / 2
            
            if my_sig == 'amplitude':
                significance[studi] = (np.sum(peak2peak_perms >= Fits.Peak2peak[Fits.Study==study].values)) / np.shape(Perms)[1]
                print('Amplitude significance for study ' + study + ': ' + str(significance))
            elif my_sig == 'rsquared':
               significance[studi] = (np.sum(peak2peak_perms >= Fits.Peak2peak[Fits.Study==study].values)) / np.shape(Perms)[1]
               print('Amplitude significance for study ' + study + ': ' + str(significance))
               
               significance[studi] = (np.sum(Perms[studi, :, 5] >= Fits.RSquared[Fits.Study==study].values)) / np.shape(Perms)[1]
               print('RSquared significance for study ' + study + ': ' + str(significance))
               
            model_fits[studi, :] = dvm(np.deg2rad(np.linspace(-90, 90, 181)), Fits.Amplitude[Fits.Study==study].values, Fits.Kappa[Fits.Study==study].values, 0)
            model_fits[studi, :] = np.rad2deg(model_fits[studi, :])

            if significance[studi] < .05:
                tmp = 'solid'
            elif (significance[studi] >= .05) & (significance[studi] <= .1):
                tmp = 'dashed'
            else:
                tmp = 'dotted'  
                
            plotStats.append(tmp)
    elif model == 'DoG&DvM':
        significance_dog = np.zeros((len(currentStudies)))
        plotStats_dog = []
        model_fits_dog = np.zeros((len(currentStudies), len(np.linspace(-90, 90, 181))))
        
        significance_dvm = np.zeros((len(currentStudies)))
        plotStats_dvm = []
        model_fits_dvm = np.zeros((len(currentStudies), len(np.linspace(-90, 90, 181))))
        
        for studi, study in enumerate(currentStudies):
            peak2peak_perms = Perms_dvm[studi, :, 2]
            peak2peak_perms = peak2peak_perms / 2
            
            if my_sig == 'amplitude':
                significance_dog[studi] = (np.sum(Perms_dog[studi, :, 0] >= Fits_dog.Amplitude[Fits_dog.Study==study].values)) / np.shape(Perms_dog)[1]
                
                significance_dvm[studi] = (np.sum(peak2peak_perms >= Fits_dvm.Peak2peak[Fits_dvm.Study==study].values)) / np.shape(Perms_dvm)[1]
            elif my_sig == 'rsquared':
               significance_dog[studi] = (np.sum(Perms_dog[studi, :, 0] >= Fits_dog.Amplitude[Fits_dog.Study==study].values)) / np.shape(Perms_dog)[1] 
               print('Amplitude significance for study ' + study + ': ' + str(significance_dog))
               
               significance_dog[studi] = (np.sum(Perms_dog[studi, :, 5] >= Fits_dog.RSquared[Fits_dog.Study==study].values)) / np.shape(Perms_dog)[1]
               print('RSquared significance for study ' + study + ': ' + str(significance_dog))
               
               significance_dvm[studi] = (np.sum(peak2peak_perms >= Fits_dvm.Peak2peak[Fits_dvm.Study==study].values)) / np.shape(Perms_dvm)[1]
               print('Amplitude significance for study ' + study + ': ' + str(significance_dvm))
               
               significance_dvm[studi] = (np.sum(Perms_dvm[studi, :, 5] >= Fits_dvm.RSquared[Fits_dvm.Study==study].values)) / np.shape(Perms_dvm)[1]
               print('RSquared significance for study ' + study + ': ' + str(significance_dvm))
               
            model_fits_dog[studi, :] = dog(np.linspace(-90, 90, 181), Fits_dog.Amplitude[Fits_dog.Study==study].values, Fits_dog.Width[Fits_dog.Study==study].values)
            
            model_fits_dvm[studi, :] = dvm(np.deg2rad(np.linspace(-90, 90, 181)), Fits_dvm.Amplitude[Fits_dvm.Study==study].values, Fits_dvm.Kappa[Fits_dvm.Study==study].values, 0)
            model_fits_dvm[studi, :] = np.rad2deg(model_fits_dvm[studi, :])

            
            if significance_dog[studi] < .05:
                tmp = 'solid'
            elif (significance_dog[studi] >= .05) & (significance_dog[studi] <= .1):
                tmp = 'dashed'
            else:
                tmp = 'dotted'
        
            plotStats_dog.append(tmp)
            
            if significance_dvm[studi] < .05:
                tmp = 'solid'
            elif (significance_dvm[studi] >= .05) & (significance_dvm[studi] <= .1):
                tmp = 'dashed'
            else:
                tmp = 'dotted'
            
            plotStats_dvm.append(tmp)
        
    '''
    ====================================================
    Define important figure parameters
    ====================================================
    '''
    if model != 'DoG&DvM':
        if sess2plot == 'all':
            if savename[0] != 'perms_1000_evBound_con':
                my_title = 'Serial dependence pooled across subjects'
                xlabel = 'Previous-current stimulus orientation (in deg)'
                yticks = [-3.25, 0, 3.25]
                yticks_inset = [0, 3]
            else:
                my_title = 'Control analysis pooled across subjects'
                xlabel = 'Future-current stimulus orientation (in deg)'
                yticks = [-2.5, 0, 2.5]
                yticks_inset = [-1.25, 1.25]
            figsize = (8, 4)
            factor_x = 0
            my_legend = True
            labels =  ['Study1 - data', 'Study2 - data']
            fit_labels = ['Study1 - fit', 'Study2 - fit']
        elif sess2plot == 'Session1':
            xlabel = 'Previous-current stimulus orientation (in deg)'
            my_title = 'Session 1'
            figsize = (4, 3)
            yticks = [-4.5, 0, 4.5]
            factor_x = .05
            my_legend = False
            labels =  ['_nolegend_', '_nolegend_']
            fit_labels = ['_nolegend_', '_nolegend_']
            
            yticks_inset = [0, 4.5]
        elif sess2plot == 'Session2':
            xlabel = 'Previous-current stimulus orientation (in deg)'
            my_title = 'Session 2'
            figsize = (4, 3)
            yticks = [-4.5, 0, 4.5]
            factor_x = .05
            my_legend = False
            labels =  ['_nolegend_', '_nolegend_']
            fit_labels = ['_nolegend_', '_nolegend_']
            
            yticks_inset = [0, 4.5]
        
        col = ['#b11226', '#0061B5']
        ylabel = 'Response error on current trial (in deg)'
    else:
        if sess2plot == 'all':
            if savename[0] != 'perms_1000_evBound_con':
                my_title = 'Comparing DoG vs. DvM fits for serial dependence pooled across subjects'
                xlabel = 'Previous-current stimulus orientation (in deg)'
                yticks = [-3.25, 0, 3.25]
                yticks_inset = [0, 3.2]
            else:
                my_title = 'Control analysis pooled across subjects'
                xlabel = 'Future-current stimulus orientation (in deg)'
                yticks = [-2.5, 0, 2.5]
                yticks_inset = [-1.25, 1.25]
            figsize = (9, 4)
            factor_x = 0
            my_legend = True
            labels =  ['Study1 - data', 'Study2 - data']
            fit_labels = ['Study1 - DoG fit', 'Study2 - DoG fit', 'Study1 - DvM fit', 'Study2 - DvM fit']
        elif sess2plot == 'Session1':
            xlabel = 'Previous-current stimulus orientation (in deg)'
            my_title = 'Session 1'
            figsize = (4, 3)
            yticks = [-4.5, 0, 4.5]
            factor_x = .05
            my_legend = False
            labels =  ['_nolegend_', '_nolegend_']
            fit_labels = ['_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_']
            
            yticks_inset = [0, 4.5]
        elif sess2plot == 'Session2':
            xlabel = 'Previous-current stimulus orientation (in deg)'
            my_title = 'Session 2'
            figsize = (4, 3)
            yticks = [-4.5, 0, 4.5]
            factor_x = .05
            my_legend = False
            labels =  ['_nolegend_', '_nolegend_']
            fit_labels = ['_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_']
            
            yticks_inset = [0, 4.5]
        
        col = ['#b11226', '#0061B5', '#6c0b17', '#003869']
        ylabel = 'Response error on current trial (in deg)'
        
    
    '''
    ====================================================
    Plot
    ====================================================
    '''
    if model != 'DoG&DvM':
        ax = plotSD(x=np.linspace(-90, 90, 181), y=np.nanmean(data_smoothed, axis=1), 
                 yerr=scipy.sem(data_smoothed, axis=1, nan_policy='omit'), fit=model_fits,
                 my_title=my_title, col=col, 
                 xticks=[-90, -60, -30, 0, 30, 60, 90], xticklabels=None, xlabel=xlabel, xlim_split=False,
                 yticks=yticks, yticklabels=None, ylabel=ylabel, ylim_bottom=False, ylim_top=False,
                 axisBreak=0, axisXBreak=0, figsize=figsize, p_val=significance, plotStats=plotStats, 
                 factor_x=factor_x, factor_y=0.05, collapseSubs=collapseSubs, label=labels,
                 label_fits = fit_labels, my_legend=my_legend)
        
        #Plot inset
        if sess2plot == 'all':
            #ax_inset = ax[0].inset_axes([0.8, .06, .2, .2], facecolor='white')
            ax_inset = ax[0].inset_axes([0.175, .78, .3, .15], facecolor='darkgray')
            ax_inset.patch.set_alpha(.25)
        else:
            ax_inset = ax[0].inset_axes([0.225, .775, .25, .15], facecolor='darkgray')
            ax_inset.patch.set_alpha(.25)
        
        #Prepare plotStats
        if model == 'DoG':
            plotStats_inset = dict({'alpha': significance,
                                    'x1': [1, 2],
                                    'x2': [1, 2], 
                                    'y': [Fits.Amplitude.values[0], Fits.Amplitude.values[1]],
                                    'h': [0.1, 0.1],
                                    'vertLines': [False, False], 
                                    'rotation': [0, 0]})
            
            ax_inset = plotBar(x=[1, 2], y=Fits.Amplitude.values, my_title='SD amplitude',
                               col=col, xticks=[1, 2], xticklabels=['S1', 'S2'], xlabel=None,
                               yticks=yticks_inset, yticklabels=None, ylabel=None, ylim_bottom=None, ylim_top=None,
                               axisBreak=0, figsize=None, p_val=significance, plotStats=plotStats_inset,
                               factor_x=None, factor_y=0, ax=ax_inset, bootstrapp=Bootstrapps)
        else:
            plotStats_inset = dict({'alpha': significance,
                                    'x1': [1, 2],
                                    'x2': [1, 2], 
                                    'y': [Fits.Peak2peak.values[0], Fits.Peak2peak.values[1]],
                                    'h': [0.1, 0.1],
                                    'vertLines': [False, False], 
                                    'rotation': [0, 0]})
        
            ax_inset = plotBar(x=[1, 2], y=Fits.Peak2peak.values, my_title='SD amplitude',
                           col=col, xticks=[1, 2], xticklabels=['S1', 'S2'], xlabel=None,
                           yticks=yticks_inset, yticklabels=None, ylabel=None, ylim_bottom=None, ylim_top=None,
                           axisBreak=0, figsize=None, p_val=significance, plotStats=plotStats_inset,
                           factor_x=None, factor_y=0, ax=ax_inset, bootstrapp=Bootstrapps)
       
        if savename[0] != 'perms_1000_evBound_con':
            if model == 'DoG':
                filename_tmp = 'Figures/PooledSD_Session_' + sess2plot + '.svg'
            else:
                filename_tmp = 'Figures/PooledSD_DvM_Session_' + sess2plot + '.svg'
        else:
            if model == 'DoG':
                filename_tmp = 'Figures/PooledSD_con_Session_' + sess2plot + '.svg'
            else:
                filename_tmp = 'Figures/PooledSD_DvM_con_Session_' + sess2plot + '.svg'
        format_tmp = 'svg'
        
        filename = Path(path_results / filename_tmp)
        saveFig(plt.gcf(), filename, format=format_tmp)    
    else:
        ax = plotSD(x=np.linspace(-90, 90, 181), y=np.nanmean(data_smoothed, axis=1), 
                 yerr=scipy.sem(data_smoothed, axis=1, nan_policy='omit'), fit=[model_fits_dog, model_fits_dvm],
                 my_title=my_title, col=col, 
                 xticks=[-90, -60, -30, 0, 30, 60, 90], xticklabels=None, xlabel=xlabel, xlim_split=False,
                 yticks=yticks, yticklabels=None, ylabel=ylabel, ylim_bottom=False, ylim_top=False,
                 axisBreak=0, axisXBreak=0, figsize=figsize, p_val=[significance_dog, significance_dvm], plotStats=[plotStats_dog, plotStats_dvm], 
                 factor_x=factor_x, factor_y=0.05, collapseSubs=collapseSubs, label=labels,
                 label_fits = fit_labels, my_legend=my_legend)
        
        #Reorder legend items
        if sess2plot == 'all':
            handles, labels_tmp = ax[0].get_legend_handles_labels()
            order = [0, 1, 4, 3, 2, 5]
            ax[0].legend([handles[idx] for idx in order], [labels_tmp[idx] for idx in order], prop={'family': 'Arial', 'size': 8})

        #Plot inset
        if sess2plot == 'all':
            #ax_inset = ax[0].inset_axes([0.8, .06, .2, .2], facecolor='white')
            ax_inset = ax[0].inset_axes([0.175, .78, .3, .15], facecolor='darkgray')
            ax_inset.patch.set_alpha(.25)
        else:
            ax_inset = ax[0].inset_axes([0.225, .775, .25, .15], facecolor='darkgray')
            ax_inset.patch.set_alpha(.25)
        
        #Prepare plotStats
        reorder_idx = [0, 2, 1, 3]
        
        plotStats_inset = dict({'alpha': np.reshape(np.squeeze([significance_dog, significance_dvm]), (4,))[reorder_idx],
                          'x1': [1, 1.8, 3, 3.8],
                          'x2': [1, 1.8, 3, 3.8], 
                          'y': [Fits_dog.Amplitude.values[0], Fits_dvm.Peak2peak.values[0], Fits_dog.Amplitude.values[1], Fits_dvm.Peak2peak.values[1]],
                          'h': [0.1, 0.1, 0.1, 0.1],
                          'vertLines': [False, False, False, False], 
                          'rotation': [0, 0, 0, 0]})
        
        #Prepare bootstrapps
        err_dog = np.std(Bootstrapps_dog[:, :, 0], axis=1)
        print('Bootstrapping standard deviation DoG: ' + str(err_dog))
        err_dvm = np.std(Bootstrapps_dvm[:, :, 2], axis=1)
        err_dvm = err_dvm / 2 #to get half-peak
        print('Bootstrapping standard deviation DvM: ' + str(err_dvm))
        
        bootstrapp = [err_dog, err_dvm]
        
        ax_inset = plotBar(x=[1, 1.8, 3, 3.8], y=np.squeeze([Fits_dog.Amplitude.values, Fits_dvm.Peak2peak.values]), 
                           my_title='SD amplitude',
                           col=[col[idx] for idx in reorder_idx], xticks=[1.4, 3.4], xticklabels=['S1', 'S2'], xlabel=None,
                           yticks=yticks_inset, yticklabels=None, ylabel=None, ylim_bottom=None, ylim_top=None,
                           axisBreak=0, figsize=None, p_val=significance_dog, plotStats=plotStats_inset,
                           factor_x=None, factor_y=0, ax=ax_inset, bootstrapp=np.squeeze(bootstrapp))

        if savename[0] != 'perms_1000_evBound_con':
            filename_tmp = 'Figures/PooledSD_Session_comparison_DoG_vs_DvM_' + sess2plot + '.svg'
        else:
            filename_tmp = 'Figures/PooledSD_con_Session_comparison_DoG_vs_DvM' + sess2plot + '.svg'
        format_tmp = 'svg'
        
        filename = Path(path_results / filename_tmp)
        saveFig(plt.gcf(), filename, format=format_tmp)  

    return