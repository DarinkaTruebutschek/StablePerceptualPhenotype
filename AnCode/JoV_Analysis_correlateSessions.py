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
def plotAmplitude_singleSubs(x, y, my_title, col, xticks, xticklabels, xlabel, xlim_split,
            yticks, yticklabels, ylabel, ylim_bottom, ylim_top,
            axisBreak, axisXBreak, figsize, factor_x, factor_y):
    """
    :param x: angles 
    :param y: error
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
    
    from SerDep_EvBound_ProbeDelay_Analysis_basicPlottingFuncs import basicFigProps, add_stats, breakAxis, breakXAxis, pretty_plot
    
    #Set style
    rc = basicFigProps()
    plt.rcParams.update(rc)
    
    #Determine necessary variables
    if xticklabels == None:
        xticklabels = xticks
    
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
        
        #First, plot actual data
        ax[ax_i].bar(x=x, height=y, color=col)
        
        #Plot markers 
        ax[ax_i].hlines(0, -5, 35, colors='dimgray', linestyles='dotted', linewidth=.8, zorder=1)
        
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

def plot_singleSubjectFits(x, y, fit, my_title, col_dat, col_fit, 
                           linewidth_dat, linewidth_fit, alpha_dat, alpha_fit, xticks, xticklabels, xlabel, 
            yticks, yticklabels, ylabel, figsize, figtitle, sig, sig_sse, sig_rsquared):
    """
    :param x: angles 
    :param y: errors
    :param fit: individual subject DOG fits
    :param my_title: title for the entire plot
    :param col_dat: color for the actual data
    :param col_fit: color to be used for plotting fits
    :param linewidth_dat: line thickness for data
    :param linewidth_fit: line thickness for fit
    :param alpha_dat: alpha for data
    :param alpha_fit: alpha for fit
    :param xticks: xticks to be used
    :param xticklabels: xticklabels to be used (if None, xticks == xticklabels)
    :param xlabel: label to be used for x-axis
    :param yticks: yticks to be used
    :param yticklabels: yticklabels to be used (if None, yticks == yticklabels)
    :param ylabel: label to be used for y-axis   
    :param figsize: size of figure
    :param figtitle: title for the entire figure
    :param sig: how to plot significance
    :param sig_sse: signifcance SSE
    :param sig_rsquared: significance rsquared

    """
    
    import matplotlib.pyplot as plt
    import scipy.stats as scipy
    import seaborn as sns
    
    from matplotlib.path import Path
    from matplotlib.patches import BoxStyle
    
    
    from JoV_Analysis_basicPlottingFuncs import basicFigProps, add_stats, breakAxis, breakXAxis, pretty_plot
    
    #Defined to be able to modify title
    class ExtendedTextBox(BoxStyle._Base):
        """
        An Extended Text Box that expands to the axes limits 
                            if set in the middle of the axes
        """
    
        def __init__(self, pad=0.3, width=500.):
            """
            width: 
                width of the textbox. 
                Use `ax.get_window_extent().width` 
                       to get the width of the axes.
            pad: 
                amount of padding (in vertical direction only)
            """
            self.width=width
            self.pad = pad
            super(ExtendedTextBox, self).__init__()
    
        def transmute(self, x0, y0, width, height, mutation_size):
            """
            x0 and y0 are the lower left corner of original text box
            They are set automatically by matplotlib
            """
            # padding
            pad = mutation_size * self.pad
    
            # we add the padding only to the box height
            height = height + 2.*pad
            # boundary of the padded box
            y0 = y0 - pad
            y1 = y0 + height
            _x0 = x0
            x0 = _x0 +width /2. - self.width/2.
            x1 = _x0 +width /2. + self.width/2.
    
            cp = [(x0, y0),
                  (x1, y0), (x1, y1), (x0, y1),
                  (x0, y0)]
    
            com = [Path.MOVETO,
                   Path.LINETO, Path.LINETO, Path.LINETO,
                   Path.CLOSEPOLY]
    
            path = Path(cp, com)
    
            return path
    
    #Set style
    rc = basicFigProps()
    plt.rcParams.update(rc)
    
    #Register custom style
    BoxStyle._style_list['ext'] = ExtendedTextBox
    
    #Determine necessary variables
    if xticklabels == None:
        xticklabels = xticks
    
    #Determine necessary variables
    if yticklabels == None:
        yticklabels = yticks
    
    #First, determine the number of subplots needed
    n_subs = np.shape(y)[0]
    
    if n_subs == 21:
        n_cols = 7
        n_rows = 3
    elif n_subs == 20:
        n_cols = 7
        n_rows = 3
    
    #Initialize figure
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize, 
                           sharex=True, sharey=True, dpi=rc['savefig.dpi'])
    ax = ax.flatten()
    
    if n_subs == 20:
        
        #Duplicate last entry in order to avoid issue with suptitle
        y_tmp = np.concatenate([y, [y[19]]])
        y = y_tmp
        
        fit_tmp = np.concatenate([fit, [fit[19]]])
        fit = fit_tmp
        
        sig_tmp = np.concatenate((sig, np.expand_dims(sig[19], axis=0)))
        sig = sig_tmp
        
        sig_tmp = np.concatenate((sig_sse, np.expand_dims(sig_sse[19], axis=0)))
        sig_sse = sig_tmp
        
        sig_tmp = np.concatenate((sig_rsquared, np.expand_dims(sig_rsquared[19], axis=0)))
        sig_rsquared = sig_tmp
        
        #ax[20].set_axis_off()
        #ax = ax[0:-1]
    
    #Plot
    for ax_i, axis in enumerate(ax):
        
        #First, plot actual data: 2 lines per plot
        ax[ax_i].plot(x, y[ax_i, 0, :], linewidth=linewidth_dat, alpha=alpha_dat, color=col_dat[0])
        
        if np.shape(y)[1] > 1:
            ax[ax_i].plot(x, y[ax_i, 1, :], linewidth=linewidth_dat, alpha=alpha_dat, color=col_dat[1])
        
        #Plot markers
        ax[ax_i].hlines(0, -100, 100, colors='dimgray', linestyles='dotted', linewidth=.8, zorder=1)
        ax[ax_i].vlines(0, -9, 9, colors='dimgray', linestyles='dotted', linewidth=.8, zorder=1)
        
        #Then, plot fit (and determine significance)
        linestyle_session1_tmp = sig_rsquared[:, 0]
        
        if np.shape(y)[1] > 1:
            linestyle_session2_tmp = sig_rsquared[:, 1]
        
        linestyle_session1 = []
        linestyle_session2 = []
        
        for sigi, subsig in enumerate(linestyle_session1_tmp):
            if subsig <= .05:
                tmp = 'solid'
            elif (subsig >.05) & (subsig <= .1):
                tmp = 'dashed'
            else:
                tmp = 'dotted'
            
            linestyle_session1.append(tmp)
        
        if np.shape(y)[1] > 1:
            for sigi, subsig in enumerate(linestyle_session2_tmp):
                if subsig <= .05:
                    tmp = 'solid'
                elif (subsig >.05) & (subsig <= .1):
                    tmp = 'dashed'
                else:
                    tmp = 'dotted'
                
                linestyle_session2.append(tmp)
        
        ax[ax_i].plot(x, fit[ax_i, 0, :], linewidth=linewidth_fit, linestyle=linestyle_session1[ax_i], alpha=alpha_fit, color=col_dat[0])
        
        if np.shape(y)[1] > 1:
            ax[ax_i].plot(x, fit[ax_i, 1, :], linewidth=linewidth_fit, linestyle=linestyle_session2[ax_i], alpha=alpha_fit, color=col_dat[1])
        
        #Add actual significance values
        if n_subs == 21:
            y1 = -5.5
            y2 = -4.5
            y3 = -3.5
        else:
            y1 = -12.5
            y2 = -11.5
            y3 = -10.5
        text_sig = 'p_ampl: ' + str(sig[ax_i][0])
        ax[ax_i].text(-90, y1, text_sig, font=rc['font.family'], fontsize=rc['font.size']-2)
        
        text_sig = 'p_sse: ' + str(sig_sse[ax_i][0])
        ax[ax_i].text(-90, y2, text_sig, font=rc['font.family'], fontsize=rc['font.size']-2)
        
        text_sig = 'p_r2: ' + str(sig_rsquared[ax_i][0])
        ax[ax_i].text(-90, y3, text_sig, font=rc['font.family'], fontsize=rc['font.size']-2)
        
        # #Last, plot significance as shaded area if wanted
        # if sig is not False:
        #     subs = np.unique(sig.Subject)
        #     subSig_tmp = sig[sig.Subject==subs[ax_i]]
        #     subSig = subSig_tmp[subSig_tmp.P_value < .05]
        #     subSig_marg = (subSig_tmp[(subSig_tmp.P_value >= .05) & (subSig_tmp.P_value < .1)])
            
        #     for sigi, _ in enumerate(np.arange(len(subSig))):
        #         ax[ax_i].axvspan(subSig['Angular distance'].values[sigi], subSig['Angular distance'].values[sigi], alpha=0.5, color=[105/255, 105/255, 105/255], edgecolor=None) #Block1
        #     for sigi, _ in enumerate(np.arange(len(subSig_marg))):
        #         ax[ax_i].axvspan(subSig_marg['Angular distance'].values[sigi], subSig_marg['Angular distance'].values[sigi], alpha=0.5, color=[255/255, 141/255, 133/255], edgecolor=None) #Block1
        
        #Set ylim
        ax[ax_i].set_ylim((yticks[0], yticks[-1]))        
        
        #Add axes ticks, labels, etc
        ax[ax_i].set_xticks(xticks)
        ax[ax_i].set_xticklabels(xticklabels, font=rc['font.family'], fontsize=rc['font.size']-2)
        
        ax[ax_i].set_yticks(yticks)
        ax[ax_i].set_yticklabels(yticklabels, font=rc['font.family'], fontsize=rc['font.size']-2)
        
        #Prettify axes
        if (n_subs == 21) | (n_subs == 20) :
            relevantAx = [0, 7, 14, 15, 16, 17, 18, 19, 20]
            if (ax_i in relevantAx):
                ax[ax_i] = pretty_plot(ax[ax_i])
                
                if ax_i in [0, 7]:
                    for side in ['top', 'right', 'bottom']:
                        ax[ax_i].spines[side].set_visible(False)
                    ax[ax_i].get_xaxis().set_visible(False)
                elif ax_i in [15, 16, 17, 18, 19, 20]:
                    for side in ['top', 'right', 'left']:
                        ax[ax_i].spines[side].set_visible(False)
                    ax[ax_i].get_yaxis().set_visible(False)
            else:
                for side in ['top', 'right', 'bottom', 'left']:
                    ax[ax_i].spines[side].set_visible(False)
                ax[ax_i].get_xaxis().set_visible(False)
                ax[ax_i].get_yaxis().set_visible(False)
        
        #Set title
        title = ax[ax_i].set_title(my_title[ax_i], fontdict={'fontfamily': rc['font.family'], 'fontsize': rc['font.size']-1, 
                                        'fontweight': 'bold'}, backgroundcolor='darkgray')
        
        #Set the box style of the title text box to custom box
        bb = title.get_bbox_patch()
        
        #Use the axes' width as width of the text box
        bb.set_boxstyle("ext", pad=0.4, width=ax[ax_i].get_window_extent().width)
        
    #Add axis-labels spanning all subplots
    fig.supxlabel(xlabel, y=0.1, verticalalignment='top', fontfamily=rc['font.family'], 
                  fontsize=rc['font.size']+4, color='dimgray')
    fig.supylabel(ylabel, x=0.1, fontfamily=rc['font.family'], 
                  fontsize=rc['font.size']+4, color='dimgray')
    fig.suptitle(figtitle,
                 y=.925, verticalalignment='bottom', fontfamily=rc['font.family'], 
                 fontsize=rc['font.size']+6, fontweight='bold')
    return ax

def plotHist(x, y, my_title, col, xticks, xticklabels, xlabel, xlim_split,
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

#Correlation between individual subjects
def run_correlateSessions(data, model, collapseSubs, stats_n_permutations,
                           savename, bin_width, path_results):
    
    """
    :param data: raw data to be plotted
    :param sess2plot: which session to plot
    :param model: which model was used to fit SD
    :param collapseSubs: how was SD computed
    :param stats_n_permutations: how many permutations had been run
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
    
    from JoV_Analysis_SD import computeMovingAverage
    from JoV_Analysis_basicFitting import dog, clifford
    from JoV_Analysis_PlotSD_singleSubs import plot_singleSubjectFits
    from JoV_Analysis_basicPlottingFuncs import saveFig
    
    '''
    ====================================================
    Determine important variables
    ====================================================
    '''
    currentStudies = ['evBound', 'menRot']
    currentSessions = [0, 1]
    currentSubs = [np.unique(data[0].Subject),
                   np.unique(data[2].Subject)]
    
    #inclSubs_evBound = ['005', '006', '007', '008', '009', '010', 
                        #'012', '013', '015', '017', '020', '021', 
                        #'025']
    
   # inclSubs_menRot = ['008', '009', '017', '019', '020', '021', 
                       # '024', '025', '027', '031', '032', '035']
    
    inclSubs_evBound = ['005', '006', '007', '010', '012', 
                        '015', '018', '020', '025']
    
    inclSubs_menRot = ['007', '008', '009', '017', '019', 
                       '024', '025', '027', '032', '035']
            
    '''
    ====================================================
    Load all of the other data necessary
    ====================================================
    '''
    
    if model == 'DoG':
        if len(savename) > 1:
            Fits = pd.DataFrame(columns=['Study', 'Subject', 'Session', 'Amplitude', 'Width', 'MinCost', 'SSE', 'RSquared', 'inclCorr'])
        else:
            Fits = pd.DataFrame(columns=['Study', 'Subject', 'Amplitude', 'Width', 'MinCost', 'SSE', 'RSquared', 'inclCorr'])
    
    for studi, study in enumerate(currentStudies):
        for sessi, sess in enumerate(currentSessions):
            tmp_sess = pd.DataFrame(columns=['Study', 'Subject', 'Session', 'Amplitude', 'Width', 'MinCost', 'SSE', 'RSquared', 'inclCorr'])
            for subi, sub in enumerate(currentSubs[studi]):
                
                #Fits
                filename_model = ('Group/Fits/Study_' + study + '_' + model + '_bestParams_' 
                                    + collapseSubs + '_perms_1000_'  + study + '_Session' + str(sess+1) + '_Subject_' +  sub + '.csv')
                
                tmp = pd.read_csv(path_results / filename_model, index_col=0)
                tmp.insert(0, 'Study', study)
                tmp.insert(1, 'Subject', sub)
                tmp.insert(2, 'Session', sess)
                
                #GoF
                filename_gof = ('Group/Fits/Study_' + study + '_' + model + '_GoF_' 
                                    + collapseSubs + '_perms_1000_' + study + '_Session' + str(sess+1) + '_Subject_' +  sub + '.csv')
                
                tmp_gof = pd.read_csv(path_results / filename_gof, index_col=0)
                tmp.insert(6, 'SSE', tmp_gof.SSE)
                tmp.insert(7, 'RSquared', tmp_gof.RSquared)
                
                #Check whether or not subject is included
                if study == 'evBound':
                    if sub in inclSubs_evBound:
                        included = 1
                    else:
                        included = 0
                elif study == 'menRot':
                    if sub in inclSubs_menRot:
                        included = 1
                    else:
                        included = 0
                
                tmp.insert(8, 'inclCorr', included)
                
                tmp_sess = pd.concat((tmp_sess, tmp))
            
            Fits = pd.concat((Fits, tmp_sess))
            
    #Save fits
    Fits.to_csv(path_results / 'Group/MeasuresForCorr_raw.csv')
    
    '''
    ====================================================
    Normalize within each study
    ====================================================
    '''
    fits2norm = Fits.Amplitude.values
    
    def normalize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    
    fits2norm[np.where((Fits.Study == 'evBound'))] = normalize(fits2norm[np.where((Fits.Study == 'evBound'))])
    fits2norm[np.where((Fits.Study == 'menRot'))] = normalize(fits2norm[np.where((Fits.Study == 'menRot'))])
    
    Fits.Amplitude = fits2norm
    
    #Save fits
    Fits.to_csv(path_results / 'Group/MeasuresForCorr_normalized.csv')
    
    '''
    ====================================================
    Plot
    ====================================================
    '''
    plt.scatter(Fits.Amplitude[(Fits.Study=='evBound') & (Fits.Session==0) & (Fits.inclCorr==1)], 
                Fits.Amplitude[(Fits.Study=='evBound') & (Fits.Session==1) & (Fits.inclCorr==1)])
    
    # corrCoeff_evBound, corrSig_evBound = scipy.stats.pearsonr(Fits.Amplitude[(Fits.Study=='evBound') & (Fits.Session==0) & (Fits.inclCorr==1)],
    #                                           Fits.Amplitude[(Fits.Study=='evBound') & (Fits.Session==1) & (Fits.inclCorr==1)])
    
    # spearman_corrCoeff_evBound, spearman_corrSig_evBound = scipy.stats.spearmanr(Fits.Amplitude[(Fits.Study=='evBound') & (Fits.Session==0) & (Fits.inclCorr==1)],
    #                                           Fits.Amplitude[(Fits.Study=='evBound') & (Fits.Session==1) & (Fits.inclCorr==1)])
    
    
    
    plt.scatter(Fits.Amplitude[(Fits.Study=='menRot') & (Fits.Session==0) & (Fits.inclCorr==1)], 
                Fits.Amplitude[(Fits.Study=='menRot') & (Fits.Session==1) & (Fits.inclCorr==1)])
    
    corrCoeff_menRot, corrSig_menRot = scipy.stats.pearsonr(Fits.Amplitude[(Fits.Study=='menRot') & (Fits.Session==0) & (Fits.inclCorr==1)],
                                              Fits.Amplitude[(Fits.Study=='menRot') & (Fits.Session==1) & (Fits.inclCorr==1)])
    
    
    plt.scatter(Fits.Amplitude[(Fits.Session==0) & (Fits.inclCorr==1)], 
                Fits.Amplitude[(Fits.Session==1) & (Fits.inclCorr==1)])
    
    corrCoeff, corrSig = scipy.stats.pearsonr(Fits.Amplitude[(Fits.Session==0) & (Fits.inclCorr==1)],
                                              Fits.Amplitude[(Fits.Session==1) & (Fits.inclCorr==1)])
    
    
    
    plt.scatter(Fits.Amplitude[(Fits.Study=='evBound') & (Fits.Session==0)], 
                Fits.Amplitude[(Fits.Study=='evBound') & (Fits.Session==1)])
    plt.scatter(Fits.Amplitude[(Fits.Study=='menRot') & (Fits.Session==0)], 
                Fits.Amplitude[(Fits.Study=='menRot') & (Fits.Session==1)])
    
    '''
    ====================================================
    Plot histograms
    ====================================================
    '''
    
    # plt.hist(Fits.Amplitude[(Fits.Study=='evBound')], bins=25)
    # plt.hist(Fits.Amplitude[(Fits.Study=='evBound') & (Fits.Session==0)], bins=25)
    # plt.hist(Fits.Amplitude[(Fits.Study=='evBound') & (Fits.Session==1)], bins=25)
    
    
    # dat2plot = pd.melt(Fits, id_vars=['Study', 'Session'], value_vars='Amplitude', value_name='Amplitude')
    # sns.histplot(data=dat2plot, x='Amplitude', hue='Session', multiple='stack')
    
    #EvBound
    dat2plot = pd.melt(Fits[(Fits.Study=='evBound') & (Fits.inclCorr==1)], id_vars=['Study', 'Session'], value_vars='Amplitude', value_name='Amplitude')
    sns.histplot(data=dat2plot, x='Amplitude', hue='Session', multiple='layer')
    
    #EvBound
    dat2plot = pd.melt(Fits[(Fits.Study=='menRot') & (Fits.inclCorr==1)], id_vars=['Study', 'Session'], value_vars='Amplitude', value_name='Amplitude')
    sns.histplot(data=dat2plot, x='Amplitude', hue='Session', multiple='layer')
        
 
    return


# ### Scraps ###
# corrVals = np.zeros((20))
# corrSig = np.zeros((20))

# for subi, sub in enumerate(currentSubs[0]):
#     corrVals[subi], corrSig[subi] = scipy.stats.pearsonr(test[subi, 0, :], test[subi, 1, :])
    
# #Plot
# plt.scatter(np.linspace(1, 20, 20), corrVals, c='dimgray')
# plt.scatter(np.linspace(1, 20, 20)[corrSig < .05], corrVals[corrSig < .05], c='red')
# plt.xlabel('Subject')
# plt.ylabel('Pearson Correlation Coefficient Session 1 and 2')