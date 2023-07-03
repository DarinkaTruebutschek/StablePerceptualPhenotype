#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 15:55:37 2022

@author: darinka

#Purpose: All functions necessary to plot single-subject SD analyses from multiple studies.
#Author: Darinka Truebutschek
#Date created: 28/11/2022
#Date last modified: 28/11/2022
#Python version: 3.7.1

"""

import numpy as np

### Plotting ###
def plotHist(dat2plot, column, my_title, col, xticks, xticklabels, xlabel, xlim_split,
            yticks, yticklabels, ylabel, ylim_bottom, ylim_top,
            axisBreak, axisXBreak, figsize, factor_x, factor_y):
    """
    :param dat2plot: pd dataframe to be plotted
    :param column: which column to plot
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
    if yticklabels == None:
        yticklabels = yticks

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
        
        if column == 'FWHM':
            bins = np.linspace(10, 80, 8)
        else:
            #bins = np.linspace(-3, 10, 14)
            bins = np.linspace(-3, 9, 13)
            #bins = np.histogram_bin_edges(dat2plot[column][~np.isnan(dat2plot[column])])
        bin_w = (max(bins) - min(bins)) / (len(bins)-1)
        
        #First, plot the entire data
        weights = np.ones_like(dat2plot[column][~np.isnan(dat2plot[column])]) / len(dat2plot[column][~np.isnan(dat2plot[column])])
        plt.hist(dat2plot[column][~np.isnan(dat2plot[column])], density=False, weights=weights, 
                 bins=bins, color='darkgray', rwidth=.75, label='Full group')
        
        #Then, plot only the significant data overlaid in a different color
        if (column == 'Amplitude'):
            weights2 = np.ones_like(dat2plot[column][dat2plot.Sig_RSquared <= .155]) / len(dat2plot[column][dat2plot.Sig_RSquared <= .155])
            plt.hist(dat2plot[column][dat2plot.Sig_RSquared <= .155], density=False, weights=weights2, 
                     bins=bins, color=col, alpha=.85, rwidth=.975, label='Subjects with significant model fit')        
        elif (column == 'Peak2peak') | (column == 'FWHM'):
            ind1_tmp = ~np.isnan(dat2plot[column]).values
            ind2_tmp = dat2plot.Sig_Amp <= .05
            dat2plot_tmp = dat2plot[column][(ind1_tmp) & (ind2_tmp.values)]
            weights2 = np.ones_like(dat2plot_tmp) / len(dat2plot_tmp)
            plt.hist(dat2plot_tmp, density=False, weights=weights2, 
                     bins=bins, color=col, alpha=.85, rwidth=.975, label='Subjects with significant model fit')
        
        #Add markers if we are looking at amplitude
        if (column == 'Amplitude') | (column == 'Peak2peak'):
            plt.axvline(0, 0, 1., linestyle='dotted', color='dimgray') #marks 0 line
            
            ax[0].annotate('Repulsion', (-0.25, 0.29),
                            font='Arial', fontsize=12, color='dimgray',
                            horizontalalignment='right')
            ax[0].annotate('Attraction', (0.25, 0.29),
                            font='Arial', fontsize=12, color='dimgray',
                            horizontalalignment='left')
            
        #Add axes ticks
        xticks = np.arange(min(bins)+bin_w/2, max(bins), bin_w)
        xticks = np.round(xticks, decimals=1)
        ax[ax_i].set_xticks(xticks)
        
        #Adjust ylims specifically if we plot p-values (to account for the added space)
        factor = np.max(yticks)*factor_y 
        ax[ax_i].set_ylim((np.min(yticks)-factor, np.max(yticks)+factor))
        
        factor = np.max(xticks)*factor_x 
        ax[ax_i].set_xlim((np.min(xticks)-factor, np.max(xticks)+factor))
        
        #Set labels
        xticklabels = xticks
        ax[ax_i].set_xticklabels(xticklabels, font=rc['font.family'], fontsize=rc['font.size']-2)
        if xlabel:
            if axisXBreak == 0:
                ax[ax_i].set_xlabel(xlabel, font=rc['font.family'], fontsize=rc['font.size']-2)
            else:
                fig.supxlabel(xlabel, y= .05, verticalalignment='top', fontfamily=rc['font.family'], 
                              fontsize=rc['font.size']+4, color='dimgray')
        
        ax[ax_i].set_yticks(yticks)
        ax[ax_i].set_yticklabels(yticklabels, font=rc['font.family'], fontsize=rc['font.size']-2)
        if ylabel:
            if axisBreak == 0:
                ax[ax_i].set_ylabel(ylabel, font=rc['font.family'], fontsize=rc['font.size']-2)
            else:
                ax[0].set_ylabel(ylabel, font=rc['font.family'], fontsize=rc['font.size']-2)
                
        #Prettify axes
        ax[ax_i] = pretty_plot(ax[ax_i])
        
        #Insert legend
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

def run_plot_SD_VariabilityInterSub(currentStudies, model, collapseSubs, stats_n_permutations, my_sig,
                           savename, bin_width, path_results):
    
    """
    :param currentStudies: which data to include
    :param model: which model was used to fit SD
    :param collapseSubs: how was SD computed
    :param stats_n_permutations: how many permutations had been run
    :param my_sig: which type of significance to plot
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
    
    from JoV_Analysis_basicFitting import dog, clifford, dvm
    from JoV_Analysis_PlotSD_VariabilityInterSub import plotHist
    from JoV_Analysis_basicPlottingFuncs import saveFig
           
    '''
    ====================================================
    Load df containing all the necessary information to be plotted
    ====================================================
    '''
    if model == 'DoG':
        Fits = pd.DataFrame(columns=['Study', 'Subject', 'Amplitude', 'Width', 
                                     'MinCost', 'SSE', 'RSquared'])
    elif (model == 'DvM') | (model == 'DvM_movAvg'):
        Fits = pd.DataFrame(columns=['Study', 'Amplitude', 'Kappa', 
                                     'MinCost', 'SSE', 'RSquared', 'Peak2peak', 'Subject'])
    for studi, study in enumerate(currentStudies):
        filename = ('Stats/Study_' + study + '_' + model + '_combinedFits_' +
                      savename[studi] +'.csv')
        Fits_tmp = pd.read_csv(path_results / filename, index_col=0)
        
        if model == 'DoG':
            Fits_tmp.insert(0, 'Study', np.repeat(study, len(Fits_tmp)))
        elif (model == 'DvM') | (model == 'DvM_movAvg'):
            Fits_tmp.Study = np.repeat(study, len(Fits_tmp))
        
        Fits = pd.concat((Fits, Fits_tmp))
    
    '''
    ====================================================
    Load all of the permutation data to be able to compute significance
    ====================================================
    '''
    Perms = np.zeros((len(Fits), stats_n_permutations, 6))
    for subi, sub in enumerate(Fits.Subject):
        study = Fits.iloc[subi].Study
        
        if study == 'evBound':
            savename_tmp = 'perms_1000_evBound'
        elif study == 'menRot':
            savename_tmp = 'perms_1000_menRot'
        
        if sub < 10:
            sub_tmp = '00' + str(sub)
        elif sub >= 10:
            sub_tmp = '0' + str(sub)
            
        
        #Permutations
        if model == 'DoG':
            filename_perms = ('Group/Perms/Perms_Study_' + study + '_' + model + '_' 
                              + collapseSubs + '_'  + savename_tmp + '_Subject_' +  sub_tmp + '.npy')
        else:
            filename_perms = ('Group/Perms/Perms_Study_' + study + '_DvM_' 
                              + collapseSubs + '_'  + savename_tmp + '_movAvg_Subject_' +  sub_tmp + '.npy')
        Perms[subi, :, :] = np.load(path_results / filename_perms)
    
    #Compute peak to peak
    if (model == 'DvM') | (model == 'DvM_movAvg'):
        Perms_peak2peak = Perms[:, :, 2] 
        Perms_peak2peak = Perms_peak2peak / 2
            
            
    '''
    ====================================================
    Determine FWHM for each individual subject
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
        
        if len(zero_crossings_i) == 2:
            return [lin_interp(x, y, zero_crossings_i[0], half),
                    lin_interp(x, y, zero_crossings_i[1], half)]
        else:
            return [np.nan, np.nan]
    
    #Data
    x=np.linspace(-90, 90, 181)
    fwhm_group = []
    
    for subi, sub in enumerate(Fits.Subject):
        
        #Compute fit
        if model == 'DoG':
            fit = dog(x, Fits.iloc[subi].Amplitude, Fits.iloc[subi].Width)
        else:
            fit = dvm(np.deg2rad(x), Fits.iloc[subi].Amplitude, Fits.iloc[subi].Kappa, 0)
            fit = np.rad2deg(fit)
    
        #Find the two crossing points
        hmx = half_max_x(x, fit)
        
        if ~np.isnan([hmx[0]]):
            #Compute and append the answer
            fwhm = hmx[1] - hmx[0]
            fwhm_group.append(fwhm)

            # #Plot for quality check
            # half = max(fit)/2.0
            # plt.figure()
            # plt.plot(x, fit)
            # plt.plot(hmx, [half, half])
        else:
            fwhm = np.nan
            fwhm_group.append(fwhm)
    
    Fits.insert(7, 'FWHM', fwhm_group)

    '''
    ====================================================
    Determine statistical significance 
    ====================================================
    '''
    significance = np.zeros((len(Fits)))
    plotStats = []
    
    for subi, sub in enumerate(Fits.Subject):
        if model == 'DoG':
            significance[subi] = np.sum(Perms[subi, :, 5] >= Fits.iloc[subi].RSquared) / np.shape(Perms)[1]
        else:
            significance[subi] = np.sum(np.abs(Perms_peak2peak[subi, :]) >= np.abs(Fits.iloc[subi].Peak2peak)) / np.shape(Perms)[1]
    
    if model == 'DoG':
        Fits.insert(8, 'Sig_RSquared', significance)
    
        #Save
        filename = 'Stats/BothStudies_DoG_combinedFits_perm_1000_withSig.csv'
    else:
        Fits.insert(8, 'Sig_Amp', significance)
    
        #Save
        filename = 'Stats/BothStudies_DvM_combinedFits_perm_1000_withSig.csv'
    Fits.to_csv(path_results / filename)

    '''
    ====================================================
    Plot distribution of amplitudes
    ====================================================
    '''
    
    if model == 'DoG':
        column = 'Amplitude'
        my_title ='Distribution of amplitude parameters'
        xlabel = 'Amplitude parameter (in deg)'
        filename_tmp = 'Figures/Histogramm_SD_Amplitudes_' + my_sig + '.svg'
    else:
        column = 'Peak2peak'
        my_title ='Distribution of amplitudes'
        xlabel = 'Amplitude (in deg)'
        filename_tmp = 'Figures/Histogramm_SD_DvM_Amplitudes_' + my_sig + '.svg'
        
    ax = plotHist(dat2plot=Fits, column=column, 
                  my_title=my_title, col='#702963',
                  xticks=None, xticklabels=None, xlabel=xlabel, xlim_split=False,
                  yticks=[0, .1, .2, .3], yticklabels=None, ylabel='Probability', ylim_bottom=False, ylim_top=False,
                  figsize=(8, 4), axisBreak=0, axisXBreak=0, factor_x=0.075, factor_y=0)
    
    format_tmp = 'svg'
    
    filename = Path(path_results / filename_tmp)
    saveFig(plt.gcf(), filename, format=format_tmp)    
    
    '''
    ====================================================
    Plot distribution of FWHM
    ====================================================
    '''

    ax = plotHist(dat2plot=Fits, column='FWHM', 
                  my_title='Distribution of FWHM', col='#702963',
                  xticks=None, xticklabels=None, xlabel='FWHM (in deg)', xlim_split=False,
                  yticks=[0, .1, .2, .3], yticklabels=None, ylabel='Probability', ylim_bottom=False, ylim_top=False,
                  figsize=(8, 4), axisBreak=0, axisXBreak=0, factor_x=0.1, factor_y=0)
    
    if model == 'DoG':
        filename_tmp = 'Figures/Histogramm_SD_FWHM_' + my_sig + '.svg'
    else:
        filename_tmp = 'Figures/Histogramm_SD_DvM_FWHM_' + my_sig + '.svg'
    format_tmp = 'svg'
    
    filename = Path(path_results / filename_tmp)
    saveFig(plt.gcf(), filename, format=format_tmp)    
    
    return

