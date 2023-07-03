#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Purpose: This script contains several basic functions to fit data with models, perform bootstrapping and permutation testing
#Author: Darinka Truebutschek
#Date created: 21/02/2022
#Date last modified: 21/02/2022
#Python version: 3.7.1

import numpy as np
import scipy.stats as sio

'''
====================================================
Simple function definitions
====================================================
'''

### Derivative of Gaussian function ###
def dog(x, a, w):
    
    """
    :param x: in the context of SD, the relative position of the past trial
    :param a: in the context of SD, the amplitude of the curve, the parameter to be estimated
    :param w: in the context of SD, the width of the curve, treated here as a free parameter
    
    """ 
    # #Check whether input is in radians and, if not, convert
    # if np.max(np.unique(x)) > 10:
    #     print('Converting input to radians')
    #     x = np.deg2rad(x)
    
    c = np.sqrt(2) / np.exp(-0.5) #constant associated with the DOG function
    
    return x * a * w * c * np.exp(-(w*x)**2)

### Clifford tilt model function ###
def clifford(x, c, s):
    
    """
    :param x: in the context of SD, the relative position of the past trial (in radians)
    :param c: c parameter
    :param s: s parameter
    
    """
    
    #Check whether input is in radians and, if not, convert
    if np.max(np.unique(x)) > 10:
        print('Converting input to radians')
        x = np.deg2rad(x)
        c = np.deg2rad(c)
        
    theta_ad = np.arcsin((np.sin(x)) / np.sqrt(((s*np.cos(x) - c)) ** 2 +
                                               (np.sin(x)) ** 2))
    test = s * np.cos(x) - c < 0
    theta_ad[test] = np.pi - theta_ad[test]
    
    return np.mod(theta_ad - x + np.pi, 2 * np.pi) - np.pi

### Derivative of modified van Mises function
def dvm(x, a, kappa, mu=0):
    
    """
    :param x: in the context of SD, the relative position of the past trial
    :param a: in the context of SD, the amplitude of the curve, parameter to be estimated
    :param kappa: in the context of SD, the width of the curve, treated as a free parameter
    :param mu: symmetry axis of the van Mises derivative, aka, in the context of SD, 0
    
    """
    import numpy as np
    from scipy.special import i0 as modBessel
    
    exp = np.exp(kappa * np.cos(x-mu))
    
    #return -((a * kappa * np.sin(x-mu) * exp) / (2 * np.pi * modBessel(kappa)))
    return ((a * kappa * np.sin(x-mu) * exp) / (2 * np.pi * modBessel(kappa)))
    #return -((a * kappa * np.sin(x-mu) * exp)) #Bessel-function not solvable for high kappas needed for resolution
  
'''
====================================================
Fitting procedures
====================================================
'''

### Fit DOG ###
def fit_dog(y, x, fittingSteps):
    
    """
    :param y: observations to be fit, in the context of SD, response errors
    :param x: predictor, in the context of SD, relative angular distance
    :param fittingSteps: how many iterations of the fitting procedure to perform
    
    """ 
    
    from scipy.optimize import least_squares
    
    from JoV_Analysis_basicFitting import dog, compute_SSE, compute_rSquared
    
    #We need to minimize the residuals (aka, data-model)
    def _solver(params):
        a, w = params
        
        return y-dog(x, a, w)
    
    #Initilize bookkeeping variables
    gof = np.zeros((5)) #to store measures of goodness of fit
    
    #Range of plausible values to be tried for amplitude parameter
    #min_a = -np.pi
    #max_a = np.pi
    min_a = -10
    max_a = 10
    
    #Range of values to be tried for width parameter
    min_w = 0.02 #a la Fritsche 2017
    max_w = 0.2 #a la Fritsche
    
    min_cost = np.inf

    for _ in range(fittingSteps):
        
        #Determine random starting positions of parameters within specified range
        params_0 = [np.random.rand() * (max_a - min_a) + min_a,
                    np.random.rand() * (max_w - min_w) + min_w] 
                                    
        try:
            result = least_squares(_solver, params_0, bounds=([min_a, min_w], [max_a, max_w]))
        except ValueError:
            continue
        
        #Check whether the residual error is smaller than the previous one
        if result['cost'] < min_cost:
            best_params, min_cost, y_res = result['x'], result['cost'], result.fun
            
            #Compute goodness of fit measures
            y_pred = dog(x, best_params[0], best_params[1])
            gof[0] = compute_SSE(y_pred, y) #SSE
            gof[1] = compute_rSquared(y_pred, y)
    try:
        return best_params[0], best_params[1], min_cost, gof
    except UnboundLocalError:
        return np.nan, np.nan, min_cost, gof

### Fit Clifford model ###
def fit_clifford(y, x, fittingSteps=200):
    
    """
    :param y: observations to be fit, in the context of SD, residual response errors (in radians)
    :param x: predictor, in the context of SD, relative angular distance (in radians)
    :param fittingSteps: how many iterations of the fitting procedure to perform
    
    """ 
        
    from scipy.optimize import least_squares
    
    #Check whether input is in radians and, if not, convert
    if np.max(np.unique(x)) > 10:
        print('Converting input to radians')
        x = np.deg2rad(x)
        y = np.deg2rad(y)
    
    def _solver(params):
        c, s, m = params
        m = np.sign(m)
        return y - m * clifford(x, c, s)
    
    min_c = 0.
    max_c = 1.
    
    min_s = 0.
    max_s = 1.
    
    min_m = -1.
    max_m = 1.
    
    min_cost = np.inf
    
    for _ in range(fittingSteps):
        
        #Determine random starting positions of parameters within specified range
        params_0 = [np.random.rand() * (max_c - min_c) + min_c,
                    np.random.rand() * (max_s - min_s) + min_s,
                    np.random.rand() * (max_m - min_m + min_m)]
        
        try:
            result = least_squares(_solver, params_0,
                                   bounds = ([min_c, min_s, min_m],
                                             [max_c, max_s, max_m]))
        except ValueError:
            continue
        if result['cost'] < min_cost:
            best_params, min_cost = result['x'], result['cost']
    
    try:
        return best_params[0], best_params[1], np.sign(best_params[2]), min_cost
    except UnboundLocalError:
        return np.nan, np.nan, np.nan, min_cost

### Fit derivative of van mises model ###
def fit_dvm(y, x, fittingSteps=200):
    
    """
    :param y: observations to be fit, in the context of SD, residual response errors (in radians)
    :param x: predictor, in the context of SD, relative angular distance (in radians)
    :param fittingSteps: how many iterations of the fitting procedure to perform
    
    """ 
        
    from scipy.optimize import least_squares
    from JoV_Analysis_basicFitting import dvm, compute_SSE, compute_rSquared
    
    #Check whether input is in radians and, if not, convert
    # if np.max(np.unique(x)) > 10:
    #     print('Converting input to radians')
    #     x = np.deg2rad(x)
    #     y = np.deg2rad(y)
    
    def _solver(params):
        a, kappa = params
        return y - dvm(x, a, kappa, mu=0)
    
    #Initilize bookkeeping variables
    gof = np.zeros((2)) #to store measures of goodness of fit
    
    min_a = -15
    max_a = 15
    
    min_kappa = 0 #this would be equivalent to a uniform distribution
    max_kappa = 200 #700 #a bit more inclusive than Fritsche 2017
    
    min_cost = np.inf
    
    for _ in range(fittingSteps):
        
        #Determine random starting positions of parameters within specified range
        params_0 = [np.random.rand() * (max_a - min_a) + min_a,
                    np.random.rand() * (max_kappa - min_kappa) + min_kappa]
        #print(params_0)
        
        try:
            result = least_squares(_solver, params_0,
                                   bounds = ([min_a, min_kappa],
                                             [max_a, max_kappa]))
        except ValueError:
            continue
        if result['cost'] < min_cost:
            best_params, min_cost = result['x'], result['cost']
            
            #Compute goodness of fit measures
            y_pred = dvm(x, best_params[0], best_params[1], mu=0)
            gof[0] = compute_SSE(y_pred, y) #SSE
            gof[1] = compute_rSquared(y_pred, y)
            
    try:
        return best_params[0], best_params[1], min_cost, gof
    except UnboundLocalError:
        return np.nan, np.nan, np.nan, min_cost, gof
            
'''
====================================================
Goodness of fit measures
====================================================
'''

### SSE (Sum of squares due to error, the smaller, the better) ###
def compute_SSE(y_pred, y):    
    
    """
    :param y_pred: predicted values
    :param y: actual values

    """ 

    return np.sum((y-y_pred)**2)

### R-squared (range: 0-1) ###
def compute_rSquared(y_pred, y):
    
    """
    :param y_pred: predicted values
    :param y: actual values

    """ 
    
    #Compute SSR, the total sum of squares
    #ssr = np.sum((y_pred-np.sum(y_pred))**2)
    
    #Compute SST, sum of squares about the mean
    #sst = np.sum((y-np.sum(y_pred))**2)
    
    #Compute SSR, the sum of square of the residuals
    ssr = np.sum((y-y_pred)**2)
    
    #Compute SST, the sum of squares about the mean
    sst = np.sum((y-y.mean())**2)
    
    return 1-(ssr/sst)


