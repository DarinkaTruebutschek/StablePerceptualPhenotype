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
def plotScatter(dat2plot, column, my_title, col, xlabel, xlim_split,
            ylabel, ylim_bottom, ylim_top,
            axisBreak, axisXBreak, figsize, factor_x, factor_y):
    """
    :param dat2plot: pd dataframe to be plotted
    :param column: which column to plot
    :param my_title: title for the entire plot
    :param col: color to be used for plotting
    :param xlabel: label to be used for x-axis
    :param xlim_split: how to split x data 
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
        
        ax[ax_i].scatter(dat2plot[column][dat2plot.Session==0], dat2plot[column][dat2plot.Session==1],
                         s=50, marker='o', c=col, zorder=1)
                
        #Add axes ticks
        my_xlim = np.round(np.max(np.abs([np.min(dat2plot[column].values), 
                   np.max(dat2plot[column].values)])))
        
        if (column == 'Amplitude') | (column == 'Peak2peak'):
            #xticks = np.linspace(-my_xlim, my_xlim, int(my_xlim/2))
            
            #Specific to model-free approach
            if np.min(my_xlim) > 10.5:
                xticks = [-5, -2.5, 0, 2.5, 5, 7.5, 10, 12.5]
            else:
                xticks = [-3, 0, 3, 6, 9]
        elif column == 'FWHM':
            xticks = [0, 20, 40, 60, 80]
        else:
            xticks = np.linspace(0, my_xlim, int(my_xlim/8))
        xticks = np.round(xticks, decimals=1)
        ax[ax_i].set_xticks(xticks)
        
        yticks = xticks
        
        #Specific to model-free approach
        if (column == 'Amplitude') & (np.min(my_xlim) > 10.5):
            yticks = [-4, -2, 0, 2, 4, 6, 8, 10]
            
        ax[ax_i].set_yticks(yticks)
        
        #Add the identity line
        ax[ax_i].plot([np.min(xticks), np.max(xticks)], [np.min(xticks), np.max(xticks)],
                      color='dimgray', linestyle='dotted', zorder=0) # identity line
        
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
        
        yticklabels = yticks
        ax[ax_i].set_yticks(yticks)
        ax[ax_i].set_yticklabels(yticklabels, font=rc['font.family'], fontsize=rc['font.size']-2)
        if ylabel:
            if axisBreak == 0:
                ax[ax_i].set_ylabel(ylabel, font=rc['font.family'], fontsize=rc['font.size']-2)
            else:
                ax[0].set_ylabel(ylabel, font=rc['font.family'], fontsize=rc['font.size']-2)
                
        #Prettify axes
        ax[ax_i] = pretty_plot(ax[ax_i])
        
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

### Plotting ###
def plotScatter_significance(col, figsize):
    
    """
    :param col: which color to use
    :param figsize: size of figure
    """

    import matplotlib.pyplot as plt
    import scipy.stats as scipy
    import seaborn as sns
    
    from JoV_Analysis_basicPlottingFuncs import basicFigProps, add_stats, breakAxis, breakXAxis, pretty_plot
    
    #Set style
    rc = basicFigProps()
    plt.rcParams.update(rc)
    
    #Hard-coded variables (as retrieved from Jasp)
    nSubs_included = [18, 12, 9]
    
    Amp_PearsonR_values = [0.542, 0.845, 0.910]
    Amp_SpearmanRho_values = [0.591, 0.685, 0.717]
    Amp_PVal_pearson = [.020, 0.0005469, 0.0006619]
    Amp_PVal_spearman = [.011, .017, .037]

    Width_PearsonR_values = [.160, -0.043, -0.518]
    Width_SpearmanRho_values = [.084, -0.123, -0.557]
    Width_PVal_pearson = [.526, .894, .153]
    Width_PVal_spearman = [.740, .703, .119]

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=rc['savefig.dpi'])
    ax = np.array(ax, ndmin=1) #necessary step to make axis object iterable 

    #Plot
    for ax_i, axis in enumerate(ax):
        
        #Plot Pearson R as a function of size
        ax[ax_i].scatter([1, 2, 3], Amp_PVal_pearson, 
                         s=50, marker='o', c=col, label='Pearson - amplitude')
        
        #Plot Pearson R for Width as a function of size
        ax[ax_i].scatter([1, 2, 3], Width_PVal_pearson, 
                         s=50, marker='*', c=col, label='Pearson - FWHM')
        
        #Plot Spearman Rho as a function of size
        ax[ax_i].scatter([1, 2, 3], Amp_PVal_spearman, alpha=1, 
                         s=50, marker='o', c='#381532', label='Spearman - amplitude')
        
        #Plot Pearson R for Width as a function of size
        ax[ax_i].scatter([1, 2, 3], Width_PVal_spearman, alpha=1,
                         s=50, marker='*', c='#381532', label='Spearman - FWHM')
        
        #Add statistical significance line
        plt.hlines(.05, np.min(ax[ax_i].get_xlim()), np.max(ax[ax_i].get_xlim()),
                   color='dimgray', linestyle='dotted')
        
        #Add axes ticks
        xticks = [1, 2, 3]
        xticklabels = ['< .155', '< .1', '< .05']
        ax[ax_i].set_xticks(xticks)
        ax[ax_i].set_xticklabels(xticklabels, font=rc['font.family'], fontsize=rc['font.size']-2)
        ax[ax_i].set_xlabel('Inclusion cutoff p-value', font=rc['font.family'], fontsize=rc['font.size']-2)
        
        yticks = [0, .05, .25, .5, .75, 1]
        yticklabels = ['0', '.05', '.25', '.5', '.75', '1']
        ax[ax_i].set_yticks(yticks)
        ax[ax_i].set_yticklabels(yticklabels, font=rc['font.family'], fontsize=rc['font.size']-2)
        ax[ax_i].set_ylabel('Correlation p-value', font=rc['font.family'], fontsize=rc['font.size']-2)
        
        #Insert legend
        plt.legend(prop={'family': rc['font.family'], 'size': rc['font.size']-4})
        
        #Prettify axes
        ax[ax_i] = pretty_plot(ax[ax_i])
        
        #Set title
        ax[ax_i].set_title('Between-session correlation significance', fontdict={'fontfamily': rc['font.family'], 'fontsize': rc['font.size']-1,
                                                                                 'fontweight': 'bold'})

    return ax

def run_plot_SD_ConsistencyIntraSub(currentStudies, model, collapseSubs, stats_n_permutations, my_sig,
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
    from JoV_Analysis_PlotSD_plotCorrelation import plotScatter, plotScatter_significance
    from JoV_Analysis_basicPlottingFuncs import saveFig
           
    '''
    ====================================================
    Load df containing all the necessary information to be plotted
    ====================================================
    '''
    
    if model == 'DoG':
        Fits = pd.read_csv(path_results / 'Group/MeasuresForCorr_raw.csv')
    elif model == 'DvM':
        Fits = pd.read_csv(path_results / 'Group/MeasuresForCorr_DvM_raw.csv')
    elif model == 'DvM_movAvg':
        Fits = pd.read_csv(path_results / 'Group/MeasuresForCorr_DvM_movAvg_raw.csv')
        
    '''
    ====================================================
    Load all of the permutation data to be able to compute significance
    ====================================================
    '''
    Perms = np.zeros((len(Fits), stats_n_permutations, 6))
    for subi, sub in enumerate(Fits.Subject):
        study = Fits.iloc[subi].Study
        
        if study == 'evBound':
            if Fits.iloc[subi].Session == 0:
                if model != 'DvM_movAvg':
                    savename_tmp = 'perms_1000_evBound_Session1'
                else:
                    savename_tmp = 'perms_1000_evBound_Session1_movAvg'
            else:
                if model != 'DvM_movAvg':
                    savename_tmp = 'perms_1000_evBound_Session2'
                else:
                    savename_tmp = 'perms_1000_evBound_Session2_movAvg'
        elif study == 'menRot':
            if Fits.iloc[subi].Session == 0:
                if model != 'DvM_movAvg':
                    savename_tmp = 'perms_1000_menRot_Session1'
                else:
                    savename_tmp = 'perms_1000_menRot_Session1_movAvg'
            else:
                if model != 'DvM_movAvg':
                    savename_tmp = 'perms_1000_menRot_Session2'
                else:
                    savename_tmp = 'perms_1000_menRot_Session2_movAvg'
        
        if sub < 10:
            sub_tmp = '00' + str(sub)
        elif sub >= 10:
            sub_tmp = '0' + str(sub)
            
        
        #Permutations
        if model != 'DvM_movAvg':
            filename_perms = ('Group/Perms/Perms_Study_' + study + '_' + model + '_' 
                              + collapseSubs + '_'  + savename_tmp + '_Subject_' +  sub_tmp + '.npy')
        else:
            filename_perms = ('Group/Perms/Perms_Study_' + study + '_DvM_' 
                              + collapseSubs + '_'  + savename_tmp + '_Subject_' +  sub_tmp + '.npy')
        print(filename_perms)
        Perms[subi, :, :] = np.load(path_results / filename_perms)
            
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
    if model == 'DoG':
        significance_rsquared = np.zeros((len(Fits)))
        plotStats_rsquared = []
    
        for subi, sub in enumerate(Fits.Subject):
            significance_rsquared[subi] = np.sum(Perms[subi, :, 5] >= Fits.iloc[subi].RSquared) / np.shape(Perms)[1]
    
        Fits.insert(8, 'Sig_RSquared', significance_rsquared)
    
        #Save
        filename = 'Stats/BothStudies_DoG_seperateFits_perm_1000_withSig.csv'
        Fits.to_csv(path_results / filename)
    else:
        significance_amplitude = np.zeros((len(Fits)))
        plotStats_amplitude = []
        
        for subi, sub in enumerate(Fits.Subject):
            peak2peak_perms = Perms[subi, :, 2]
            peak2peak_perms = peak2peak_perms / 2
            
            significance_amplitude[subi] = ((np.sum(np.abs(peak2peak_perms) >= np.abs(Fits.iloc[subi].Peak2peak)) / np.shape(Perms)[1]))
        
        Fits.insert(8, 'Sig_Amplitude', significance_amplitude)


    '''
    ====================================================
    Plot scatter plots for amplitude (with those subjects 
    with p < .155 for both sessions)
    ====================================================
    '''
    if model == 'DoG':
        sig_threshold = .155
        sig_measure = 'Sig_RSquared'
        column = 'Amplitude'
    else:
        sig_threshold = .05
        sig_measure = 'Sig_Amplitude'
        column = 'Peak2peak'
        
    incl_subs = np.zeros(len(Fits))
    
    for subi, sub in enumerate(Fits.Subject):
        study = Fits.iloc[subi].Study
        session = Fits.iloc[subi].Session

        if session == 0:
            if (Fits.iloc[subi][sig_measure] <= sig_threshold) & (Fits[sig_measure][(Fits.Study==study) & 
                                                                      (Fits.Subject==sub) & 
                                                                      (Fits.Session==1)].values <= sig_threshold):
                incl_subs[subi] = 1 
        elif session == 1:
            if (Fits.iloc[subi][sig_measure] <= sig_threshold) & (Fits[sig_measure][(Fits.Study==study) & 
                                                                      (Fits.Subject==sub) & 
                                                                      (Fits.Session==0)].values <= sig_threshold):
                incl_subs[subi] = 1 
    
    dat2plot = Fits.loc[incl_subs==1, :]

    ax = plotScatter(dat2plot=dat2plot, column=column, 
                  my_title='Correlation of amplitude', col='#702963',
                  xlabel='SD magnitude session 1 (in deg)', xlim_split=False,
                  ylabel='SD magnitude session 2 (in deg)', ylim_bottom=False, ylim_top=False,
                  figsize=(4, 4), axisBreak=0, axisXBreak=0, factor_x=0.2, factor_y=.1)
    
    if model == 'DoG':
        filename_tmp = 'Figures/Correlation_SD_Amplitudes_' + my_sig + '.svg'
    elif (model == 'DvM') | (model == 'DvM_movAvg'):
        filename_tmp = 'Figures/Correlation_SD_DvM_Amplitudes_' + my_sig + '.svg'    
    format_tmp = 'svg'
    
    filename = Path(path_results / filename_tmp)
    saveFig(plt.gcf(), filename, format=format_tmp)    
    
    '''
    ====================================================
    Plot scatter plots for FWHM (with those subjects 
    with p < .155 for both sessions)
    ====================================================
    '''
    incl_subs = np.zeros(len(Fits))
    
    if model == 'DoG':
        sig_threshold = .155
        sig_measure = 'Sig_RSquared'
        column = 'FWHM'
    else:
        sig_threshold = .05
        sig_measure = 'Sig_Amplitude'
        column = 'FWHM'
    
    incl_subs = np.zeros(len(Fits))
    
    for subi, sub in enumerate(Fits.Subject):
        study = Fits.iloc[subi].Study
        session = Fits.iloc[subi].Session

        if session == 0:
            if (Fits.iloc[subi][sig_measure] <= sig_threshold) & (Fits[sig_measure][(Fits.Study==study) & 
                                                                      (Fits.Subject==sub) & 
                                                                      (Fits.Session==1)].values <= sig_threshold):
                incl_subs[subi] = 1 
        elif session == 1:
            if (Fits.iloc[subi][sig_measure] <= sig_threshold) & (Fits[sig_measure][(Fits.Study==study) & 
                                                                      (Fits.Subject==sub) & 
                                                                      (Fits.Session==0)].values <= sig_threshold):
                incl_subs[subi] = 1 
    
    dat2plot = Fits.loc[incl_subs==1, :]
    #dat2plot = dat2plot.loc[~np.isnan(dat2plot.FWHM), :]

    ax = plotScatter(dat2plot=dat2plot, column=column, 
                  my_title='Correlation of FWHM', col='#702963',
                  xlabel='SD width session 1 (in deg)', xlim_split=False,
                  ylabel='SD width session 2 (in deg)', ylim_bottom=False, ylim_top=False,
                  figsize=(4, 4), axisBreak=0, axisXBreak=0, factor_x=0.05, factor_y=0.05)
    
    if model == 'DoG':
        filename_tmp = 'Figures/Correlation_SD_FWHM_' + my_sig + '.svg'
    else:
        filename_tmp = 'Figures/Correlation_SD_DvM_FWHM_' + my_sig + '.svg'
    format_tmp = 'svg'
    
    filename = Path(path_results / filename_tmp)
    saveFig(plt.gcf(), filename, format=format_tmp)    
    
    '''
    ====================================================
    Quick scatter to highlight stability of results as a 
    function of subjects included
    ====================================================
    '''
    if model == 'DoG':
        ax = plotScatter_significance(col='#702963', figsize=(5, 3.5))
    
        filename_tmp = 'Figures/Invariance_correlation_' + my_sig + '.svg'
        format_tmp = 'svg'
    
        filename = Path(path_results / filename_tmp)
        saveFig(plt.gcf(), filename, format=format_tmp)  
    
    '''
    ====================================================
    Is this effect driven primarily by subjects with a large 
    amplitude?
    ====================================================
    '''
    #First, compute the angle, by which each subject's data deviates from the
    #identity line (this is a measure that is not driven by the size of the amplitudes to begin with)
    angles = []
    significance_included = []
    amp_Sess1 = []
    amp_Sess2 = []
    
    for _, study in enumerate(np.unique(Fits.Study)):
        for subi, sub in enumerate(np.unique(Fits.Subject[(Fits.Study==study) & (Fits.Session==0)])):
            
            #Compute slope
            if model == 'DoG':
                x = Fits.Amplitude.values[(Fits.Study==study) & (Fits.Session==0) & 
                                          (Fits.Subject==sub)]
                y = Fits.Amplitude.values[(Fits.Study==study) & (Fits.Session==1) & 
                                          (Fits.Subject==sub)]
            else:
                x = Fits.Peak2peak.values[(Fits.Study==study) & (Fits.Session==0) & 
                                          (Fits.Subject==sub)]
                y = Fits.Peak2peak.values[(Fits.Study==study) & (Fits.Session==1) & 
                                          (Fits.Subject==sub)]
            
            slope = np.deg2rad(y)/np.deg2rad(x)
            
            angle_tmp = np.rad2deg(np.arctan(slope))
            
            angles.append(angle_tmp[0])
            
            #Retrieve significance
            if model == 'DoG':
                if ((Fits.Sig_RSquared.values[(Fits.Study==study) & (Fits.Session==0) & 
                                              (Fits.Subject==sub)] < .155) & (Fits.Sig_RSquared.values[(Fits.Study==study) & (Fits.Session==1) & 
                                                                                                       (Fits.Subject==sub)] < .155)):
                    significance_tmp = 1
                else:
                    significance_tmp  = 0
            else:
                if ((Fits.Sig_Amplitude.values[(Fits.Study==study) & (Fits.Session==0) & 
                                              (Fits.Subject==sub)] < .05) & (Fits.Sig_Amplitude.values[(Fits.Study==study) & (Fits.Session==1) & 
                                                                                                       (Fits.Subject==sub)] < .05)):
                    significance_tmp = 1
                else:
                    significance_tmp  = 0
            
            significance_included.append(significance_tmp)
            
            #Retrieve amplitudes
            if model == 'DoG':
                amp_tmp1 = Fits.Amplitude.values[(Fits.Study==study) & (Fits.Session==0) & 
                                                 (Fits.Subject==sub)]
                amp_tmp2 = Fits.Amplitude.values[(Fits.Study==study) & (Fits.Session==1) & 
                                                 (Fits.Subject==sub)]
            else:
                amp_tmp1 = Fits.Peak2peak.values[(Fits.Study==study) & (Fits.Session==0) & 
                                                 (Fits.Subject==sub)]
                amp_tmp2 = Fits.Peak2peak.values[(Fits.Study==study) & (Fits.Session==1) & 
                                                 (Fits.Subject==sub)]
            
            amp_Sess1.append(amp_tmp1[0])
            amp_Sess2.append(amp_tmp2[0])
    
    angles = np.array(angles)
    significance_included = np.array(significance_included)
    amp_Sess1 = np.array(amp_Sess1)
    amp_Sess2 = np.array(amp_Sess2)
    
    #Then, compute mean amplitude in preparation for median split
    tmp1 = np.abs(amp_Sess1)
    tmp2 = np.abs(amp_Sess2)
    meanAmp = np.mean([tmp1, tmp2], axis=0)
    
    #Perform median split
    angles = angles[significance_included == 1]
    meanAmp = meanAmp[significance_included == 1]
    
    sortIndex = np.argsort(np.abs(meanAmp))
    meanAmp_sorted = meanAmp[sortIndex]
    angles_sorted = angles[sortIndex]
    
    if model == 'DoG':
        scipy.ttest_1samp(angles_sorted[0:9], 45)
        scipy.ttest_1samp(angles_sorted[9:], 45)
    else:
        scipy.ttest_1samp(angles_sorted[0:13], 45)
        scipy.ttest_1samp(angles_sorted[14:], 45)
    
    #Save as pd dataframe for import to JASP
    medianSplit = pd.DataFrame(np.transpose([angles_sorted, meanAmp_sorted]), columns=['SortedAngles', 'SortedMeanAmplitudes'])
    group = np.zeros(len(medianSplit))
    if model == 'DoG':
        group[9:] = 1
    else:
        group[14:] = 1
    
    medianSplit.insert(2, 'Group', group)
    
    if model == 'DoG':
        medianSplit.to_csv(path_results / 'Stats/MedianSplit.csv')
    else:
        medianSplit.to_csv(path_results / 'Stats/MedianSplit_DvM.csv')
    return

def run_plot_SD_ConsistencyIntraSub_modelFree(currentStudies, bounds,
                           savename, path_results):
    
    """
    :param currentStudies: which data to include
    :param bounds: which bounds were used for 
    :param savename: initial saveNames used 
    :param path_results: where to plot the data
    """
    
    #Imports
    import os.path
    
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.stats as scipy
    
    from pathlib import Path
    
    from JoV_Analysis_basicFitting import dog, clifford
    from JoV_Analysis_PlotSD_plotCorrelation import plotScatter, plotScatter_significance
    from JoV_Analysis_PlotSD_singleSubs import plotAmplitude_singleSubs
    from JoV_Analysis_basicPlottingFuncs import saveFig
           
    '''
    ====================================================
    Load df containing all the necessary information to be plotted
    ====================================================
    '''
    
    if bounds == 'specific':
        Fits_evBound = pd.read_csv(path_results / 'Stats/SD_modelfree_subjSpec_perms_1000_evBound.csv')
        Fits_menRot = pd.read_csv(path_results / 'Stats/SD_modelfree_subjSpec_perms_1000_menRot.csv')
        filename_tmp = path_results / 'Group/Perms/SD_BothSess_modelfree_subjSpec_perms_1000_evBound.npy'
        Perms_evBound = np.load(filename_tmp)
        filename_tmp = path_results / 'Group/Perms/SD_BothSess_modelfree_subjSpec_perms_1000_menRot.npy'
        Perms_menRot = np.load(filename_tmp)
    elif bounds == '45':
        Fits_evBound = pd.read_csv(path_results / 'Stats/SD_modelfree_45_perms_1000_evBound.csv')
        Fits_menRot = pd.read_csv(path_results / 'Stats/SD_modelfree_45_perms_1000_menRot.csv')
    elif bounds == '60':
        Fits_evBound = pd.read_csv(path_results / 'Stats/SD_modelfree_60_perms_1000_evBound.csv')
        Fits_menRot = pd.read_csv(path_results / 'Stats/SD_modelfree_60_perms_1000_menRot.csv')
        
    '''
    ====================================================
    Normalize within each study
    ====================================================
    '''
    def normalize(x, min_x, max_x):
        return (x - min_x) / (max_x - min_x)
    
    x_evBound = np.squeeze([Fits_evBound.Ampl_Sess1.values, Fits_evBound.Ampl_Sess2.values])
    min_evBound = np.min(x_evBound)
    max_evBound = np.max(x_evBound)
    
    x_menRot = np.squeeze([Fits_menRot.Ampl_Sess1.values, Fits_menRot.Ampl_Sess2.values])
    min_menRot = np.min(x_menRot)
    max_menRot = np.max(x_menRot)

    norm_Sess1_evBound = normalize(Fits_evBound.Ampl_Sess1.values, min_evBound, max_evBound)
    norm_Sess2_evBound = normalize(Fits_evBound.Ampl_Sess2.values, min_evBound, max_evBound)
    
    norm_Sess1_menRot = normalize(Fits_menRot.Ampl_Sess1.values, min_menRot, max_menRot)
    norm_Sess2_menRot = normalize(Fits_menRot.Ampl_Sess2.values, min_menRot, max_menRot)
    
    #Add
    Fits_evBound.insert(0, 'Study', np.repeat('evBound', len(Fits_evBound)))
    Fits_evBound.insert(6, 'Ampl_Sess1_norm', norm_Sess1_evBound)
    Fits_evBound.insert(7, 'Ampl_Sess2_norm', norm_Sess2_evBound)
    
    Fits_menRot.insert(0, 'Study', np.repeat('menRot', len(Fits_menRot)))
    Fits_menRot.insert(6, 'Ampl_Sess1_norm', norm_Sess1_menRot)
    Fits_menRot.insert(7, 'Ampl_Sess2_norm', norm_Sess2_menRot)
    
    #Combine
    Fits = pd.concat([Fits_evBound, Fits_menRot])
    
    '''
    ====================================================
    Quick correlation check
    ====================================================
    '''
    print('Whole group: ' + str(scipy.pearsonr(Fits.Ampl_Sess1, Fits.Ampl_Sess2)))
    print('Whole group norm: ' + str(scipy.pearsonr(Fits.Ampl_Sess1_norm, Fits.Ampl_Sess2_norm)))
    
    print(scipy.spearmanr(Fits.Ampl_Sess1, Fits.Ampl_Sess2))
    print(scipy.spearmanr(Fits.Ampl_Sess1_norm, Fits.Ampl_Sess2_norm))
    
    #EvBound
    print('EvBound: ' + str(scipy.pearsonr(Fits.Ampl_Sess1[Fits.Study=='evBound'], Fits.Ampl_Sess2[Fits.Study=='evBound'])))
    print('EvBound norm: ' + str(scipy.pearsonr(Fits.Ampl_Sess1_norm[Fits.Study=='evBound'], Fits.Ampl_Sess2_norm[Fits.Study=='evBound'])))
    
    print(scipy.spearmanr(Fits.Ampl_Sess1[Fits.Study=='evBound'], Fits.Ampl_Sess2[Fits.Study=='evBound']))
    print(scipy.spearmanr(Fits.Ampl_Sess1_norm[Fits.Study=='evBound'], Fits.Ampl_Sess2_norm[Fits.Study=='evBound']))
    
    #MenRot
    print('MenRot: ' + str(scipy.pearsonr(Fits.Ampl_Sess1[Fits.Study=='menRot'], Fits.Ampl_Sess2[Fits.Study=='menRot'])))
    print('MenRot norm: ' + str(scipy.pearsonr(Fits.Ampl_Sess1_norm[Fits.Study=='menRot'], Fits.Ampl_Sess2_norm[Fits.Study=='menRot'])))
    
    print(scipy.spearmanr(Fits.Ampl_Sess1[Fits.Study=='menRot'], Fits.Ampl_Sess2[Fits.Study=='menRot']))
    print(scipy.spearmanr(Fits.Ampl_Sess1_norm[Fits.Study=='menRot'], Fits.Ampl_Sess2_norm[Fits.Study=='menRot']))
    
    '''
    ====================================================
    Plot scatter plots 
    ====================================================
    '''
    
    #Bring data into long format
    Fits_long = pd.DataFrame(columns=['Study', 'Subject', 'Session', 'Amplitude', 'Amplitude_norm'])
    
    study_long = np.repeat(Fits.Study.values, 2)
    
    subject_long = np.tile(Fits.Subject.values[Fits.Study=='evBound'], 2)
    subject_long2 = np.tile(Fits.Subject.values[Fits.Study=='menRot'], 2)
    subject_all = np.hstack((subject_long, subject_long2))
    
    session_long = np.repeat(np.array((0, 1)), 21)
    session_long2 = np.repeat(np.array((0, 1)), 20)
    session_all = np.hstack((session_long, session_long2))
    
    amp_long = np.hstack((Fits.Ampl_Sess1.values[(Fits.Study=='evBound')], Fits.Ampl_Sess2.values[(Fits.Study=='evBound')],
                          Fits.Ampl_Sess1.values[(Fits.Study=='menRot')], Fits.Ampl_Sess2.values[(Fits.Study=='menRot')]))
    amp_long_norm = np.hstack((Fits.Ampl_Sess1_norm.values[(Fits.Study=='evBound')], Fits.Ampl_Sess2_norm.values[(Fits.Study=='evBound')],
                          Fits.Ampl_Sess1_norm.values[(Fits.Study=='menRot')], Fits.Ampl_Sess2_norm.values[(Fits.Study=='menRot')]))
    
    Fits_long.Study = study_long
    Fits_long.Subject = subject_all
    Fits_long.Session = session_all
    Fits_long.Amplitude = amp_long
    Fits_long.Amplitude_norm = amp_long_norm

    ax = plotScatter(dat2plot=Fits_long, column='Amplitude', 
                  my_title='Correlation of model-free amplitude', col='#702963',
                  xlabel='SD magnitude session 1 (in deg)', xlim_split=False,
                  ylabel='SD magnitude session 2 (in deg)', ylim_bottom=False, ylim_top=False,
                  figsize=(8, 4), axisBreak=0, axisXBreak=0, factor_x=0.075, factor_y=0.075)
    
    filename_tmp = 'Figures/Correlation_SD_Amplitudes_modelFree_' + bounds + '.svg'
    format_tmp = 'svg'
    
    filename = Path(path_results / filename_tmp)
    saveFig(plt.gcf(), filename, format=format_tmp)    
    
    '''
    ====================================================
    Is this effect driven primarily by subjects with a large 
    amplitude?
    ====================================================
    '''
    #First, compute the angle, by which each subject's data deviates from the
    #identity line (this is a measure that is not driven by the size of the amplitudes to begin with)
    angles = []
    amp_Sess1 = []
    amp_Sess2 = []
    
    for _, study in enumerate(np.unique(Fits_long.Study)):
        for subi, sub in enumerate(np.unique(Fits_long.Subject[(Fits_long.Study==study) & (Fits_long.Session==0)])):
            
            #Compute slope
            x = Fits_long.Amplitude.values[(Fits_long.Study==study) & (Fits_long.Session==0) & 
                                    (Fits_long.Subject==sub)]
            y = Fits_long.Amplitude.values[(Fits_long.Study==study) & (Fits_long.Session==1) & 
                                    (Fits_long.Subject==sub)]
            
            slope = np.deg2rad(y)/np.deg2rad(x)
            
            angle_tmp = np.rad2deg(np.arctan(slope))
            
            angles.append(angle_tmp[0])
            
            #Retrieve amplitudes
            amp_tmp1 = Fits_long.Amplitude.values[(Fits_long.Study==study) & (Fits_long.Session==0) & 
                                    (Fits_long.Subject==sub)]
            amp_tmp2 = Fits_long.Amplitude.values[(Fits_long.Study==study) & (Fits_long.Session==1) & 
                                    (Fits_long.Subject==sub)]
            
            amp_Sess1.append(amp_tmp1[0])
            amp_Sess2.append(amp_tmp2[0])
    
    angles = np.array(angles)
    amp_Sess1 = np.array(amp_Sess1)
    amp_Sess2 = np.array(amp_Sess2)
    
    #Then, compute mean amplitude in preparation for median split
    tmp1 = np.abs(amp_Sess1)
    tmp2 = np.abs(amp_Sess2)
    meanAmp = np.mean([tmp1, tmp2], axis=0)
    
    #Perform median split
    sortIndex = np.argsort(np.abs(meanAmp))
    meanAmp_sorted = meanAmp[sortIndex]
    angles_sorted = angles[sortIndex]
    
    scipy.ttest_1samp(angles_sorted[0:21], 45)
    scipy.ttest_1samp(angles_sorted[21:], 45)
    
    #Save as pd dataframe for import to JASP
    medianSplit = pd.DataFrame(np.transpose([angles_sorted, meanAmp_sorted]), columns=['SortedAngles', 'SortedMeanAmplitudes'])
    group = np.zeros(len(medianSplit))
    group[21:] = 1
    
    medianSplit.insert(2, 'Group', group)
    
    filename = 'Stats/MedianSplit_modelFree_' + bounds + '.csv'
    medianSplit.to_csv(path_results / filename)
    
    '''
    ====================================================
    Assess significance
    ====================================================
    '''
    sig_evBound = []
    
    for subi, sub in enumerate(np.unique(Fits.Subject[Fits.Study=='evBound'])):
        tmp = np.sum(np.abs(Perms_evBound[subi, :, 2]) >= np.abs(Fits.Ampl_comb.values[(Fits.Study=='evBound') &
                                                                               (Fits.Subject==sub)]))
        tmp = tmp / 1000
        
        sig_evBound.append(tmp)
    
    sig_menRot = []
    
    for subi, sub in enumerate(np.unique(Fits.Subject[Fits.Study=='menRot'])):
        tmp = np.sum(np.abs(Perms_menRot[subi, :, 2]) >= np.abs(Fits.Ampl_comb.values[(Fits.Study=='menRot') &
                                                                               (Fits.Subject==sub)]))
        tmp = tmp / 1000
        
        sig_menRot.append(tmp)
        
    sig_evBound = np.array(sig_evBound)
    sig_menRot = np.array(sig_menRot)
    sig2plot = np.concatenate((sig_evBound, sig_menRot))
    
    '''
    ====================================================
    Plot the subject specific amplitudes across the two sessions
    ====================================================
    '''

    x = np.arange(len(Fits))+1
    y = Fits.Ampl_comb.values
    
    #Sort all necessary arrays
    sort_ind = np.argsort(y)
    y_sorted = y[sort_ind]
    sig2plot_sorted = sig2plot[sort_ind]
    
    #Determine color
    col = np.repeat('darkgray', len(Fits))
    
    my_title = 'Single-subject model-free SD amplitudes in Study 1 and 2'
    col[np.squeeze(sig2plot_sorted <= .05)] = ['#702963']
    col[(np.squeeze(sig2plot_sorted > .05)) & (
        (np.squeeze(sig2plot_sorted <= .155)))] = ['#eed1e9']
    xticklabels = ['', '', '', '', '5', '', '', '', '', '10',
                   '', '', '', '', '15', '', '', '', '', '20', 
                   '', '', '', '', '25', '', '', '', '', '30', 
                   '', '', '', '', '35', '', '', '', '', '40', 
                   '']
    yticks = [-3.5, 0, 10]

    plotAmplitude_singleSubs(x=x, y=y_sorted, my_title=my_title, col=col, xticks=x, 
                             xticklabels=xticklabels, xlabel='Subject number', xlim_split=False,
                             yticks=yticks, yticklabels=yticks, ylabel='Degrees', 
                             ylim_bottom=False, ylim_top=False,
                             axisBreak=0, axisXBreak=0, figsize=(16, 4), factor_x=0.025, factor_y=0)
    

    filename_tmp = 'Figures/SingleSubject_modelfreeSD_Amplitudes_' + bounds + '.svg'
    format_tmp = 'svg'
    
    filename = Path(path_results / filename_tmp)
    saveFig(plt.gcf(), filename, format=format_tmp)    
    
    return

