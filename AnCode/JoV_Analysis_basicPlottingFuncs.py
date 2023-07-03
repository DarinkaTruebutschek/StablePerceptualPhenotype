#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Purpose: This file contains basic plotting functions 
#Date created: 09/29/2021
#Date last modified: 09/29/2021
#Python version: 3.9

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sio

'''
====================================================
Basic plotting functions
====================================================
'''
#Determine basic figure parameters
def basicFigProps():
    from JoV_Analysis_cfg import text_font, text_size

    rc = {'figure.figsize': (10,5) ,
          'axes.facecolor': 'white',
          'axes.spines.left': True,
          'axes.spines.right': False,
          'axes.spines.bottom': True,
          'axes.spines.top': False,
          'axes.grid' : False,
          'grid.color': '.8',
          'font.family': text_font,
          'font.size' : text_size,
          'savefig.dpi': 300}

    return rc

# Draw statistical significance
def add_stats(ax=None, alpha=.5, x1=None, x2=None,  y=None, h=None, text='*', vertLines=True, rotation=0):
    """
    Annotate plot with p-values.
    
    :param ax: which axis to plot this in
    :param alpha: significance level
    :param x1: x-position where to start bracket
    :param x2: x-position where to end bracket
    :param y: y-position at which to draw bracket/text
    :param h: offset between y position and start of text
    :param text: which text to show
    :param vertLines: draw vertical lines yes or no?
    :param rotation: how to rotate the text
    """
   
    from JoV_Analysis_basicPlottingFuncs import basicFigProps
    
    #Set style
    rc = basicFigProps()
    
    if ax is None:
        ax = plt.gca()
    
    #Draw vertical lines
    if vertLines:
        if rotation == 0:
            ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], linewidth=1, color='dimgray', label='_nolegend_')
        else:
            #ax.plot([y, y+h, y+h, y], [x1, x1, x2, x2], linewidth=1, color='dimgray')
            ax.plot([y-h, y, y, y-h], [x1, x1, x2, x2], linewidth=1, color='dimgray', label='_nolegend_')
   
    #Draw text
    if (alpha <= .05) & (alpha > .01):
        text = '*'
    elif (alpha <= .01) & (alpha > .001):
        text = '**'
    elif alpha <= .001:
        text = '***'
   
    if vertLines:
        if rotation == 0:
            ax.text((x1+x2)*.5, y-h, s=text, ha='center', va='bottom', color='dimgray', fontname=rc['font.family'], fontsize=rc['font.size']-3,
                rotation=rotation, label='_nolegend_')
        else:
            #ax.text((y+h), (x1+x2)*.5, s=text, ha='left', va='center', color='dimgray', fontname=rc['font.family'], fontsize=rc['font.size']-3,
                #rotation=rotation)
            ax.text(y+h, (x1+x2)*.5, s=text, ha='left', va='center', color='dimgray', fontname=rc['font.family'], fontsize=rc['font.size']-3,
                rotation=rotation, label='_nolegend_')
    else:
        ax.text(x1, y+h, s=text, ha='center', va='bottom', color='dimgray', fontname=rc['font.family'], fontsize=rc['font.size']-3,
                rotation=rotation, label='_nolegend_')
       
    return 

#Insert axis breaks on y axis
def breakAxis(ax=None, ylim_b=None, ylim_t=None):
    
    """
    :param ax: axis object to plot in
    :param ylim_bottom = ylim to be applied to bottom part
    :param ylim_top = ylim to be applied to top part

    
    """
    
    #Initialize axis object if need be & determine how many different axes there are
    if ax is None:
        ax = plt.gca()
    else:
        axis_of_interest = np.shape(ax)[0]

    if axis_of_interest == 2:
        #Set ylims
        ax[1].set_ylim(ylim_b)
        ax[0].set_ylim(ylim_t)
        
        #Hide the spines between ax1 and ax2
        ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) #turns off ticks
        ax[0].spines['bottom'].set_visible(False) #turns off spine
        
        #Plot slanted lines
        d = .5  #proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='dimgray', mec='dimgray', mew=1, clip_on=False)
        ax[0].plot([0], transform=ax[0].transAxes, **kwargs)
        ax[1].plot([1], transform=ax[1].transAxes, **kwargs)
    elif axis_of_interest == 4:
        #Set ylims
        ax[3].set_ylim(ylim_b)
        ax[2].set_ylim(ylim_b)
        ax[1].set_ylim(ylim_t)
        ax[0].set_ylim(ylim_t)
        
        #Hide the spines between ax1 and ax2
        ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) #turns off ticks
        ax[0].spines['bottom'].set_visible(False) #turns off spine
        
        ax[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) #turns off ticks
        ax[1].spines['bottom'].set_visible(False) #turns off spine
        ax[1].tick_params(axis='y', which='both', left=False, top=False, labelleft=False) #turns off ticks
        ax[1].spines['left'].set_visible(False) #turns off spine
        
        ax[3].tick_params(axis='y', which='both', left=False, top=False, labelleft=False) #turns off ticks
        ax[3].spines['left'].set_visible(False) #turns off spine
        
        #Plot slanted lines
        d = .5  #proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='dimgray', mec='dimgray', mew=1, clip_on=False)
        ax[0].plot([0], transform=ax[0].transAxes, **kwargs)
        ax[2].plot([1], transform=ax[2].transAxes, **kwargs)
        #ax[2].plot([np.nan, 0], transform=ax[2].transAxes, **kwargs)
        #ax[3].plot([0], transform=ax[3].transAxes, **kwargs)
    
    return ax

#Insert axis breaks on x axis
def breakXAxis(ax=None, xlim_b=None, xlim_t=None):
    
    """
    :param ax: axis object to plot in
    :param xlim_bottom = xlim to be applied to bottom part
    :param xlim_top = xlim to be applied to top part

    
    """
    
    #Initialize axis object if need be
    if ax is None:
        ax = plt.gca()
    else:
        axis_of_interest = np.shape(ax)[0]
    
    if axis_of_interest == 2:
        #Set ylims
        ax[0].set_xlim(xlim_b)
        ax[1].set_xlim(xlim_t)
        
        #Hide the spines between ax1 and ax2
        ax[1].tick_params(axis='y', which='both', left=False, top=False, labelleft=False) #turns off ticks
        ax[1].spines['left'].set_visible(False) #turns off spine
        
        #Plot slanted lines
        d = .5  #proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='dimgray', mec='dimgray', mew=1, clip_on=False)
        ax[0].plot([np.nan, 0], transform=ax[0].transAxes, **kwargs)
        ax[1].plot([0], transform=ax[1].transAxes, **kwargs)
    elif axis_of_interest == 4:
        #Set xlims
        ax[3].set_xlim(xlim_t)
        ax[2].set_xlim(xlim_b)
        ax[1].set_xlim(xlim_t)
        ax[0].set_xlim(xlim_b)
        
        #Plot slanted lines
        d = .5  #proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='dimgray', mec='dimgray', mew=1, clip_on=False)
        #ax[0].plot([0], transform=ax[0].transAxes, **kwargs)
        #ax[2].plot([1], transform=ax[2].transAxes, **kwargs)
        ax[2].plot([np.nan, 0], transform=ax[2].transAxes, **kwargs)
        ax[3].plot([0], transform=ax[3].transAxes, **kwargs)
        
    return ax

#Define spacing of ticks and labels
def _set_ticks(ax=None, ticks=None, axisTicks='x', minorTicks=False):
    """
    :param ax: axis object to plot in
    :param ticks: original ticks
    :param axisTicks: which axis to get the labels for
    :param minorTicks: include additional unlabeled ticks between labeled ticks?
    
    """
    
    #Initialize axis object if need be
    if ax is None:
        ax = plt.gca()
    
    #Select the new ticks and ticklabels 
    if ticks is None:
        if axisTicks == 'x':
            ticks = ax.get_xticks()
        else:
            ticks = ax.get_yticks()
        if minorTicks:
            tickmarks = np.linspace(np.min(ticks), np.max(ticks), 3)
            ticklabels = (["%.2f" % np.round(ticki, 2) if ticki in tickmarks else '' for ticki in ticks[0 :]])
        else:
            tickmarks = np.linspace(np.min(ticks), np.max(ticks), 3)
            ticks = tickmarks
            ticklabels = (["%.2f" % np.round(ticki, 2) for ticki in ticks[0 :]])
    else:
        ticklabels = ([ticki for ticki in ticks[0 :]])
    
    return ticks, ticklabels

#Plot pretty axes
def pretty_plot(ax=None):
    """
    :param ax: axis object to plot in
    
    """
    
    #Initialize axis object if needed
    if ax is None:
        plt.gca()
    
    #Set color of axis, labels, and spines    
    ax.tick_params(colors='dimgray')
    ax.xaxis.label.set_color('dimgray')
    ax.yaxis.label.set_color('dimgray')
    
    try:
        ax.zaxis.label.set_color('dimgray')
    except AttributeError:
        pass
    try:
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    except ValueError:
        pass
    
    ax.spines['left'].set_color('dimgray')
    ax.spines['bottom'].set_color('dimgray')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    return ax

#Save
def saveFig(fig=None, filename='TestFigure.png', format=format):
    """
    :param fig: figure to save
    :param filename: path to save
    :param format: which format to save file as
    
    """
    
    #Initialize figure if needed
    if fig is None:
        fig = plt.gcf()
    
    #Save
    plt.savefig(filename, format=format, dpi=300, bbox_inches='tight')

    
'''
====================================================
High-level plotting functions
====================================================
'''

# Pretty bar plot
def pretty_bar(ax=None, data=None,  indPoints=True, xticks=None, yticks=None, xlabel=None, ylabel=None,
               figTitle=None, textFont='Arial', textSize=12, textWeight='normal', color=None, alpha=1):
    """
    :param ax: which axis object to plot into
    :param data: which data to plot the boxplot for
    :param indPoints: overlay individual data points or not?
    :param xticks: which xticks and labels
    :param yticks: which yticks and labels
    :param xlabel: label of x-axis
    :param ylabel: label of y-axis
    :param title: title of the figure
    :param textFont: font to use for any text
    :param textSize: size of font
    :param textWeight: italics, normal, or bold?
    :param color: color to use for plot
    :param alpha: alpha to use for plot

    """
    # Import needed functions
    from JoV_Analysis_basicPlottingFuncs import pretty_plot

    # Initiaize random number generator
    rng = np.random.default_rng()

    # Determine how many boxplots will need to be drawn
    if not np.shape(data):
        n_plots = 1
    else:
        n_plots = np.shape(data)[1]

    # Initialize axis if needed
    if ax is None:
        ax = plt.gca()

    # Initialize x position of individual data points
    if indPoints:
        x = []

    # Plot
    for ploti in np.arange(n_plots):
         y = np.mean(data, axis=0)
         y_err = sio.stats.sem(data, axis=0)

         plt.bar(x=xticks[ploti], height=y[ploti], yerr=y_err[ploti], width=1, 
                 facecolor='w', edgecolor=color[ploti], alpha=alpha[ploti], linewidth=4,
                 ecolor=color[ploti], error_kw={'elinewidth': 3})
         
         # Add individual data points
         if indPoints:
            jitter = rng.integers(low=-5, high=5, size=np.shape(data)[0])
            jitter = jitter/100
            jitter = np.expand_dims(jitter, 1)

            x_tmp = np.tile(xticks[ploti], (np.shape(data)[0], 1))+jitter
            x.append(x_tmp)

            ax = plt.scatter(
                x=x_tmp, y=data[:, ploti], color=color[ploti], s=48, edgecolors='k', alpha=alpha[ploti], zorder=2)

    # Set category/x-labels
    ax = plt.gca()
    if len(xlabel) != len(xticks):
        tmp = np.array_split(xticks, len(xlabel))
        
        xticks_tmp = []
        for ploti in np.arange(len(tmp)):
            xticks_tmp1 = np.mean(tmp[ploti])
            xticks_tmp.append(xticks_tmp1)
        
        ax.set_xticks(xticks_tmp)
        ax.set_xticklabels(xlabel, fontname=textFont,
                       fontsize=textSize-2, fontweight=textWeight)
    else:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabel, fontname=textFont,
                      fontsize=textSize-2, fontweight=textWeight)
        
    # Set yticks/yticklabels
    if yticks is None:
        yticks, yticklabels = _set_ticks(
            ax=ax, ticks=None, axisTicks='y', minorTicks=False)
    else:
        yticks, yticklabels = _set_ticks(
            ax=ax, ticks=yticks, axisTicks='y', minorTicks=True)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontname=textFont,
                       fontsize=textSize-2, fontweight=textWeight)

    # Set axis limits
    ax.set_ylim((np.min(yticks), np.max(yticks)))

    # Prettify labels
    ax.set_ylabel(ylabel, fontname=textFont,
                  fontsize=textSize, fontweight=textWeight)

    # Set title
    ax.set_title(figTitle, fontname=textFont,
                 fontsize=textSize+2, fontweight='bold')

    # Prettify axes
    ax = pretty_plot(ax)
    
    return ax
    
# Pretty boxplot plot
def pretty_box(ax=None, data=None,  indPoints=True, yticks=None, xlabel=None, ylabel=None,
               figTitle=None, textFont='Arial', textSize=12, textWeight='normal', color=None, alpha=1, linewidth=1):
    """
    :param ax: which axis object to plot into
    :param data: which data to plot the boxplot for
    :param indPoints: overlay individual data points or not?
    :param yticks: which yticks and labels
    :param xlabel: label of x-axis
    :param ylabel: label of y-axis
    :param title: title of the figure
    :param textFont: font to use for any text
    :param textSize: size of font
    :param textWeight: italics, normal, or bold?
    :param color: color to use for plot
    :param alpha: alpha to use for plot
    :param linewidth: how thick the outline of the boxplot should be

    """

    # Import needed functions
    from JoV_basicPlottingFuncs import pretty_plot

    # Initiaize random number generator
    rng = np.random.default_rng()

    # Determine how many boxplots will need to be drawn
    n_plots = np.shape(data)[1]

    # Initialize axis if needed
    if ax is None:
        ax = plt.gca()

    # Initialize x position of individual data points
    if indPoints:
        x = []

    # Plot
    for ploti in np.arange(n_plots):
        # Regular boxplot with median and no outliers shown
        boxprops = dict(linestyle='-', linewidth=linewidth,
                        color=color[ploti], alpha=alpha[ploti])
        whiskerprops = dict(linewidth=1, color=color[ploti])
        medianprops = dict(linestyle='-', linewidth=2,
                           color=color[ploti], alpha=alpha[ploti])

        ax = plt.boxplot(x=data[:, ploti], notch=False, positions=[ploti+1], boxprops=boxprops,
                         whiskerprops=whiskerprops, medianprops=medianprops, showcaps=False, showfliers=False, zorder=1)

        # Add individual data points
        if indPoints:
            jitter = rng.integers(low=-5, high=5, size=np.shape(data)[0])
            jitter = jitter/100
            jitter = np.expand_dims(jitter, 1)

            x_tmp = np.tile(ploti+1, (np.shape(data)[0], 1))+jitter
            x.append(x_tmp)

            ax = plt.scatter(
                x=x_tmp, y=data[:, ploti], color=color[ploti], edgecolors='k', alpha=alpha[ploti], zorder=2)

        # Connect these points
        if (ploti == n_plots-1):
            if indPoints:
                x = np.squeeze(np.asarray(x))
                ax = plt.plot(x, np.transpose(data), linewidth=.5,
                          color='dimgray', zorder=1)

    # Set category/x-labels
    ax = plt.gca()
    ax.set_xticklabels(xlabel, fontname=textFont,
                       fontsize=textSize-2, fontweight=textWeight)

    # Set yticks/yticklabels
    if yticks is None:
        yticks, yticklabels = _set_ticks(
            ax=ax, ticks=None, axisTicks='y', minorTicks=False)
    else:
        yticks, yticklabels = _set_ticks(
            ax=ax, ticks=yticks, axisTicks='y', minorTicks=True)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontname=textFont,
                       fontsize=textSize-2, fontweight=textWeight)

    # Set axis limits
    ax.set_ylim((np.min(yticks), np.max(yticks)))

    # Prettify labels
    ax.set_ylabel(ylabel, fontname=textFont,
                  fontsize=textSize, fontweight=textWeight)

    # Set title
    ax.set_title(figTitle, fontname=textFont,
                 fontsize=textSize+2, fontweight='bold')

    # Prettify axes
    ax = pretty_plot(ax)

    return ax

# Pretty histogram plot
def pretty_hist(ax=None, x=None, y=None, yerr=None, xticks=None, yticks=None, xlabel=None, ylabel=None,
                figTitle=None, textFont='Arial', textSize=12, textWeight='normal', color=None, alpha=1, width=1):
    """
    :param ax: which axis object to plot into
    :param x: which data to plot on x axis
    :param y: which data to plot on y axis
    :param yerr: errorbar to plot (sem)
    :param xticks: which xticks and labels
    :param yticks: which yticks and labels
    :param xlabel: label of x-axis
    :param ylabel: label of y-axis
    :param title: title of the figure
    :param textFont: font to use for any text
    :param textSize: size of font
    :param textWeight: italics, normal, or bold?
    :param color: color to use for plot
    :param alpha: alpha to use for plot
    :param width: width of bar (in units of x axis)

    """

    # Import needed functions
    from JoV_basicPlottingFuncs import pretty_plot

    x = np.asarray(x)
    y = np.asarray(y)

    # Initialize axis if needed
    if ax is None:
        ax = plt.gca()

    # Plot the data
    ax.bar(x=x, height=y, yerr=yerr, width=width, color=color, alpha=alpha, align='edge',
           error_kw=dict(elinewidth=1, ecolor='dimgray', capsize=0))

    # Distribute x/y-ticks and -labels
    if xticks is None:
        xticks, xticklabels = _set_ticks(
            ax=ax, ticks=None, axisTicks='x', minorTicks=False)
    else:
        xticks, xticklabels = _set_ticks(
            ax=ax, ticks=xticks, axisTicks='x', minorTicks=True)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontname=textFont,
                       fontsize=textSize-2, fontweight=textWeight)

    if yticks is None:
        yticks, yticklabels = _set_ticks(
            ax=ax, ticks=None, axisTicks='y', minorTicks=False)
    else:
        yticks, yticklabels = _set_ticks(
            ax=ax, ticks=yticks, axisTicks='y', minorTicks=True)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontname=textFont,
                       fontsize=textSize-2, fontweight=textWeight)

    # Set axis limits
    ax.set_xlim((np.min(xticks), np.max(xticks)))
    ax.set_ylim((np.min(yticks), np.max(yticks)))

    # Prettify labels
    ax.set_xlabel(xlabel, fontname=textFont,
                  fontsize=textSize, fontweight=textWeight)
    ax.set_ylabel(ylabel, fontname=textFont,
                  fontsize=textSize, fontweight=textWeight)

    # Set title
    ax.set_title(figTitle, fontname=textFont,
                 fontsize=textSize+2, fontweight='bold')

    # Prettify axes
    ax = pretty_plot(ax)

    return ax

# Pretty line plot
def pretty_line(ax=None, x=None, y=None, yerr=None, xticks=None, yticks=None, xlabel=None, ylabel=None,
                figTitle=None, textFont='Arial', textSize=12, textWeight='normal', color=None, alpha=None, 
                linestyle='-', linewidth=2, legend=0, leglabel='None'):
    """
    :param ax: which axis object to plot into
    :param x: which data to plot on x axis
    :param y: which data to plot on y axis
    :param yerr: errorbar to plot (sem)
    :param xticks: which xticks and labels
    :param yticks: which yticks and labels
    :param xlabel: label of x-axis
    :param ylabel: label of y-axis
    :param title: title of the figure
    :param textFont: font to use for any text
    :param textSize: size of font
    :param textWeight: italics, normal, or bold?
    :param color: color to use for plot
    :param alpha: alpha to use for colors
    :param linestyle: which linestyle to use for a given line
    :param linewidth: how thick should the line be
    :param legend: display legend or not
    :param leglabel: label for legend

    """

    # Import needed functions
    from JoV_Analysis_basicPlottingFuncs import pretty_plot

    x = np.asarray(x)
    y = np.asarray(y)

    # Initialize axis if needed
    if ax is None:
        ax = plt.gca()

    # Plot the data
    ax.plot(x, y, color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle)

    # Plot the standard error
    if yerr is not None:
        plt.fill_between(x, y-yerr, y+yerr, color=color, alpha=.1)

    # Distribute x/y-ticks and -labels
    if xticks is None:
        xticks, xticklabels = _set_ticks(
            ax=ax, ticks=None, axisTicks='x', minorTicks=False)
    else:
        xticks, xticklabels = _set_ticks(
            ax=ax, ticks=xticks, axisTicks='x', minorTicks=True)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontname=textFont,
                       fontsize=textSize-2, fontweight=textWeight)

    if yticks is None:
        yticks, yticklabels = _set_ticks(
            ax=ax, ticks=None, axisTicks='y', minorTicks=False)
    else:
        yticks, yticklabels = _set_ticks(
            ax=ax, ticks=yticks, axisTicks='y', minorTicks=True)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontname=textFont,
                       fontsize=textSize-2, fontweight=textWeight)

    # Set axis limits
    ax.set_xlim((np.min(xticks), np.max(xticks)))
    ax.set_ylim((np.min(yticks), np.max(yticks)))
    
    #Insert legend
    if legend:
        plt.legend(labels=leglabel)

    # Prettify labels
    ax.set_xlabel(xlabel, fontname=textFont,
                  fontsize=textSize, fontweight=textWeight)
    ax.set_ylabel(ylabel, fontname=textFont,
                  fontsize=textSize, fontweight=textWeight)

    # Set title
    ax.set_title(figTitle, fontname=textFont,
                 fontsize=textSize+2, fontweight='bold')
    # Prettify axes
    ax = pretty_plot(ax)

    return ax

#Pretty scatter plot
def pretty_scatter(ax=None, x=None, y=None, xticks=None, yticks=None, xlabel=None, ylabel=None, figTitle=None,
                   textFont='Arial', textSize=12, textWeight='normal',
                   color=None, alpha=1):

    """
    :param ax: which axis object to plot into
    :param x: which data to plot on x axis
    :param y: which data to plot on y axis
    :param xticks: which xticks and labels
    :param yticks: which yticks and labels
    :param xlabel: label of x-axis
    :param ylabel: label of y-axis
    :param title: title of the figure
    :param textFont: font to use for any text
    :param textSize: size of font
    :param textWeight: italics, normal, or bold?
    :param color: color to use for plot
    :param alpha: which alpha level to use for the colors

    """
    
    #Determine how many different scatters should be drawn
    if y.ndim == 3:
        n_plots = 2 #plot 2 conditions seperately (conds x subs x y-data)
    
    # Initialize axis if needed
    if ax is None:
        ax = plt.gca()

    # Plot the data
    for ploti in np.arange(n_plots):
        ax.scatter(x=np.tile(x, (np.shape(y)[1], 1)), y=y[ploti, :, :], color=color[ploti], s=48, edgecolors=color[ploti], alpha=alpha[ploti], label = '_nolegend_') #individual data points
        ax.scatter(x=x, y=np.mean(y[ploti, :, :], axis=0), color=color[ploti], s=65, edgecolors='k', alpha=1) #group mean
    
    # Distribute x/y-ticks and -labels
    if xticks is None:
        xticks, xticklabels = _set_ticks(
            ax=ax, ticks=None, axisTicks='x', minorTicks=False)
    else:
        xticks, xticklabels = _set_ticks(
            ax=ax, ticks=xticks, axisTicks='x', minorTicks=True)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontname=textFont,
                       fontsize=textSize-2, fontweight=textWeight)

    if yticks is None:
        yticks, yticklabels = _set_ticks(
            ax=ax, ticks=None, axisTicks='y', minorTicks=False)
    else:
        yticks, yticklabels = _set_ticks(
            ax=ax, ticks=yticks, axisTicks='y', minorTicks=True)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontname=textFont,
                       fontsize=textSize-2, fontweight=textWeight)

    # Set axis limits
    ax.set_xlim((np.min(xticks), np.max(xticks)))
    ax.set_ylim((np.min(yticks), np.max(yticks)))
    
    #Insert legend
    #if legend:
        #plt.legend(labels=leglabel)

    # Prettify labels
    ax.set_xlabel(xlabel, fontname=textFont,
                  fontsize=textSize, fontweight=textWeight)
    ax.set_ylabel(ylabel, fontname=textFont,
                  fontsize=textSize, fontweight=textWeight)

    # Set title
    ax.set_title(figTitle, fontname=textFont,
                 fontsize=textSize+2, fontweight='bold')
    # Prettify axes
    ax = pretty_plot(ax)
    
    return ax

# Pretty seaborn plot
def pretty_seaborn(ax=None, plotType=None, data=None, x=None, y=None,
                   xticks=None, yticks=None, xlabel=None, ylabel=None, figTitle=None,
                   textFont='Arial', textSize=12, textWeight='normal',
                   color=None, bins=None):
    """
    :param ax: which axis object to plot into
    :param plotType: which type of seaborn plot to plot
    :param data: which data to plot
    :param x: which data to plot on x axis
    :param y: which data to plot on y axis
    :param xticks: which xticks and labels
    :param yticks: which yticks and labels
    :param xlabel: label of x-axis
    :param ylabel: label of y-axis
    :param title: title of the figure
    :param textFont: font to use for any text
    :param textSize: size of font
    :param textWeight: italics, normal, or bold?
    :param color: color to use for plot
    :param bins: in case of histogram, how to bin the data

    """

    # Import needed functions
    from JoV_Analysis_basicPlottingFuncs import pretty_plot

    data = np.asarray(data)

    # Initialize axis if needed
    if ax is None:
        ax = plt.gca()

    # Plot the data as a function of the specific seaborn plot type
    if plotType == 'histplot':
        if ylabel == 'Probability':
            stat = 'probability'
        ax = sns.histplot(data=data, bins=bins, stat=stat,
                          color=color, edgecolor='k', alpha=1, linewidth=1)
    elif plotType == 'lineplot':
        ax = sns.lineplot(x=x, y=y, color=color, linewidth=2)
    elif plotType == 'barplot':
        ax = sns.barplot(x=x, y=y, color=color, saturation=1)

    # Distribute x/y-ticks and -labels
    if xticks is None:
        xticks, xticklabels = _set_ticks(
            ax=ax, ticks=None, axisTicks='x', minorTicks=False)
    else:
        xticks, xticklabels = _set_ticks(
            ax=ax, ticks=xticks, axisTicks='x', minorTicks=True)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontname=textFont,
                       fontsize=textSize-2, fontweight=textWeight)

    if yticks is None:
        yticks, yticklabels = _set_ticks(
            ax=ax, ticks=None, axisTicks='y', minorTicks=False)
    else:
        yticks, yticklabels = _set_ticks(
            ax=ax, ticks=yticks, axisTicks='y', minorTicks=True)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontname=textFont,
                       fontsize=textSize-2, fontweight=textWeight)

    # Set axis limits
    ax.set_xlim((np.min(xticks), np.max(xticks)))
    ax.set_ylim((np.min(yticks), np.max(yticks)))

    # Prettify labels
    ax.set_xlabel(xlabel, fontname=textFont,
                  fontsize=textSize, fontweight=textWeight)
    ax.set_ylabel(ylabel, fontname=textFont,
                  fontsize=textSize, fontweight=textWeight)

    # Set title
    ax.set_title(figTitle, fontname=textFont,
                 fontsize=textSize+2, fontweight='bold')
    # Prettify axes
    ax = pretty_plot(ax)

    return ax


# Pretty multicomponent plot
def pretty_multiComp(ax=None, plotType_main='line', data_main=None, xticks_main=None, yticks_main=None,
                     xlabel_main=None, ylabel_main=None, figTitle_main=None, textFont='Arial', textSize=12,
                     textWeight='normal', color=None, alpha=None, x_line=None, linestyle_line='-', linewidth_line=1,
                     legend_line=0, leglabel_line=None, data_inset=None, plotType_inset='bar', xticks_inset=None,
                     yticks_inset=None, xlabel_inset=None, ylabel_inset=None, positions_inset=None, figTitle_inset=None):
    """
    :param ax: which axis object to plot into
    :param plotType_main: which type of plot should the main plot be
    :param data_main: which data to plot in the main plot
    :param xticks_main: which xticks and labels for main plot
    :param yticks_main: which yticks and labels for main plot
    :param xlabel_main: label of x-axis for main plot
    :param ylabel_main: label of y-axis for main plot
    :param title_main: title of the figure for main plot
    :param textFont: font to use for any text
    :param textSize: size of font
    :param textWeight: italics, normal, or bold?
    :param color: color to use for plot
    :param alpha: which alpha level to use for the colors
    :param line_x: x data for line plot
    :param linestyle_line: which linestyle to use for a given line
    :param linewidth_line: how thick should a given line be
    :param legend_line: display legend for line plot or not
    :param leglabel_line: labels to use for the legend
    :param data_inset: data to plot as inset
    :param plotType_inset: which type of plot should be the inset
    :param xticks_inset: which xticks and labels for inset
    :param yticks_inset: which yticks and labels for inset
    :param xlabel_inset: label of x-axis for inset
    :param ylabel_inset: label of y-axis for inset
    :param positions_inset: where to position the insets 
    :param figTitle_inset: which title for the insets
    
    """

    #Import needed functions
    from JoV_Analysis_basicPlottingFuncs import pretty_plot, pretty_line, pretty_bar

    #Initialize axis if needed
    if ax is None:
        ax = plt.gca()

    #Determine number of plots needed
    n_plots = np.shape(data_main)[1]

    #Retrieve data for main plot and plot
    for ploti in np.arange(n_plots):
        if plotType_main == 'line':
            y = np.mean(data_main, axis=0)[ploti]
            yerr = sio.stats.sem(data_main)[ploti]

            ax = pretty_line(ax=None, x=x_line, y=y, yerr=yerr, xticks=xticks_main, yticks=yticks_main, xlabel=xlabel_main, ylabel=ylabel_main,
                        figTitle=figTitle_main, textFont='Arial', textSize=12, textWeight='normal', color=color[ploti],
                        alpha=alpha[ploti], linestyle=linestyle_line[ploti], linewidth=linewidth_line, legend=0, leglabel=leglabel_line[ploti])
    
    #Add legend 
    if legend_line:
        ax.legend(leglabel_line, loc='upper left', prop={'family': 'Arial', 'weight': 'normal', 'size': textSize-4}, frameon=1)
    
    #Adjust ylim
    ax = plt.gca()
    ax.set_ylim((yticks_main[0]-.001, yticks_main[-1]))
    
    #Plot insets if wanted
    if data_inset is not None:
        
        #Determine number of insets to be plotted
        n_insets =  np.shape(data_inset)[0]
        
        #Determine the position of the individual insets
        positions = positions_inset
        
        #Plot the different insets
        for inseti in np.arange(n_insets):
            if plotType_inset[inseti] == 'bar':
                a = plt.axes(positions[inseti])
                pretty_bar(ax=a, data=data_inset[inseti],  indPoints=True, xticks=xticks_inset[inseti], yticks=yticks_inset[inseti], xlabel=xlabel_inset[inseti], ylabel=ylabel_inset[inseti],
                           figTitle=figTitle_inset[inseti], textFont='Arial', textSize=8, textWeight='normal', color=color, alpha=alpha)
    
    return ax
