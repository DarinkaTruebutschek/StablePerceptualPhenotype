#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Purpose: This script contains several basic statistic functions
#Author: Darinka Truebutschek
#Date created: 21/01/2022
#Date last modified: 21/01/2022
#Python version: 3.7.1

import mne
import numpy as np
import scipy.stats as sio
import statsmodels.api as sm

from mne.stats import permutation_t_test
from statsmodels.formula.api import ols

### Paired t-test ###
def paired_ttest(data=None, n_permutations=1000, tail=0, n_jobs=1, verbose=True):
    
    """
    :param data: numpy array with 1 columns corresponding to the differerence in conditions to be tested 
    (as the test is always performed against 0)
    :param n_permutations: how many iterations to perform the test over
    :param tail: two-sided (0), upper-sided (1), or lower-sided (-1) test
    :param n_jobs: how many processors to recruit for this computation
    :param verbose: how much detail should the output provide
    
    """ 
    
    T_obs, p_values, _ = permutation_t_test(X=data, n_permutations=n_permutations, tail=tail, n_jobs=n_jobs, verbose=verbose)
    
    return T_obs, p_values

### Permutation test to assess significance of SD at the group-level ###
def perform_permute_SD(data=None, dat2fit='pooled', n_permutations=10, bestParams_actualFit=None, fittingSteps=1):
    
    """
    :param data: data for which to perform the permutation analysis
    :param dat2fit: for what type of data to perform permutation analysis - pooled data | group mean | single-subject
    :param n_permutations: how many different iterations of the permutation procedure to run
    :param bestParams_actualFit: actual parameters obtained by the fitted distribution to be compared against
    :param fittingSteps: how many iterations of the fitting procedure to perform
    
    """ 
    
    #Imports
    import matplotlib.pyplot as plt
    import numpy as np
    import pycircstat
    
    from joblib import Parallel, delayed #to allow for parallelization across permutation loops
    
    from JoV_Analysis_basicFitting import dog, fit_dog
        
    #Determine important variables for further usage
    subs = np.unique(data.Subject)
    n_subs = len(subs)
    
    #Initialize permutation distribution 
    permutation_distribution = np.zeros((n_permutations, 6)) #column 1: amplitude parameter, column 2: width parameter, column3: peak2peak
    #column 4: min_cost, column5: sse, column6=r²
    permuted_angles = np.zeros((n_permutations, np.sum(~np.isnan(data.Delta_angle_norm.values))))
    permuted_respErrors = np.zeros((n_permutations, np.sum(~np.isnan(data.Delta_angle_norm.values))))
    
    #Define permutation
    for i, _ in enumerate(np.arange(n_permutations)):
        print('Permutation ' + str(i))
        
        #Randomly shuffle delta_angle label for each participant
        perm_data = data.copy(deep=True)

        shuffledLabels = []
        for subi, sub in enumerate(subs):
            shuffledLabels_tmp = perm_data.Delta_angle_norm.values[perm_data.Subject==sub]
            np.random.shuffle(shuffledLabels_tmp)
            
            shuffledLabels.append(shuffledLabels_tmp)
        flat_shuffledLabels = [item for sublist in shuffledLabels for item in sublist]
        perm_data.Delta_angle_norm = np.array(flat_shuffledLabels)
        
        #Prepare data for fitting procedure
        if dat2fit == 'pooled':
            x = perm_data.Delta_angle_norm.values
            y = perm_data.Resp_error_demeaned.values
        elif dat2fit == 'singleSub':
            x = perm_data.Delta_angle_norm.values
            y = perm_data.Resp_error_demeaned.values
        
        #Fit
        a, w, min_cost, gof = fit_dog(y, x, fittingSteps)
        
        #Compute peak-to-peak
        fit = dog(np.linspace(-90, 90, 181), a, w)
        peak2peak = np.sign(a) * (fit.max()-fit.min())
        
        permutation_distribution[i, :] = [a, w, peak2peak, min_cost, gof[0], gof[1]]
        
        permuted_angles[i, :] = x
        permuted_respErrors[i, :] = y
        
    return permutation_distribution, permuted_angles, permuted_respErrors
        
### Bootstrapping to assess standard deviation of SD effect ###
def perform_bootstrapping_SD(data=None, dat2fit='pooled', n_permutations=10, fittingSteps=1):
    
    """
    :param data: data for which to perform the bootstrapping analysis
    :param dat2fit: for what type of data to perform permutation analysis - pooled data | group mean | single-subject
    :param n_permutations: how many different iterations of the bootstrapping procedure to run
    :param fittingSteps: how many iterations of the fitting procedure to perform
    
    """ 
    
    from JoV_Analysis_basicFitting import dog, fit_dog
    
    #Initialize bootstrapping distribution 
    bootstrappingCoef_distribution = np.zeros((n_permutations, 3)) #column 1: amplitude parameter, column 2: width parameter, column3: peak2peak
    
    #Determine important variables for further usage
    subs = np.unique(data.Subject)
    n_subs = len(subs)
    
    if n_subs > 1:
        #Randomly select subjects (with replacement from all subjects) & compute the desired parameters
        for i, _ in enumerate(np.arange(n_permutations)):
            print('Bootstrapping ' + str(i))
            
            sample_ind = np.squeeze(np.random.randint(low=0, high=n_subs, size=[n_subs, 1])) #which subjects to include in the current fit
            bootstrapp_subs = np.unique(data.Subject)[sample_ind]
            
            #Select data for fitting 
            x = []
            y = []
            if dat2fit == 'pooled':
                for subi, sub in enumerate(bootstrapp_subs):
                    
                    x_tmp = data.Delta_angle_norm.values[data.Subject==sub]
                    y_tmp = data.Resp_error_demeaned.values[data.Subject==sub]
                    
                    #y_tmp = y_tmp[~np.isnan(x_tmp)]
                    #x_tmp = x_tmp[~np.isnan(x_tmp)]
                    
                    #Take care of remaining nans designating outlier trials
                    x_tmp = x_tmp[~np.isnan(y_tmp)]
                    y_tmp = y_tmp[~np.isnan(y_tmp)]
                    
                    x.append(x_tmp)
                    y.append(y_tmp)
                
                x_flat = np.asarray([item for sublist in x for item in sublist])
                y_flat = np.asarray([item for sublist in y for item in sublist])
                
                x = x_flat
                y = y_flat
            
                #Fit
                a, w, min_cost, gof = fit_dog(y, x, fittingSteps)
            
                #Compute peak-to-peak
                fit = dog(np.linspace(-90, 90, 181), a, w)
                peak2peak = np.sign(a) * (fit.max()-fit.min())
            
                bootstrappingCoef_distribution[i, :] = [a, w, peak2peak]
    else:
        #Randomly select trials (with replacement) & compute the desired parameters
        for i, _ in enumerate(np.arange(n_permutations)):
            print('Permutation ' + str(i))
            
            x_tmp = data.Delta_angle_norm.values
            y_tmp = data.Resp_error_demeaned.values
            
            #Take care of remaining nans designating outlier trials
            x = x_tmp[~np.isnan(y_tmp)]
            y = y_tmp[~np.isnan(y_tmp)]
            
            sample_ind = np.squeeze(np.random.randint(low=0, high=len(x), size=[len(x), 1]))
            
            x_bootstrapp = x[sample_ind]
            y_bootstrapp = y[sample_ind]
            
            #Fit
            a, w, min_cost, gof = fit_dog(y_bootstrapp, x_bootstrapp, fittingSteps)
        
            #Compute peak-to-peak
            fit = dog(np.linspace(-90, 90, 181), a, w)
            peak2peak = np.sign(a) * (fit.max()-fit.min())
        
            bootstrappingCoef_distribution[i, :] = [a, w, peak2peak]

    return bootstrappingCoef_distribution

### Permutation test to assess significance of SD at the group-level ###
def perform_permute_SD_clifford(data=None, dat2fit='pooled', n_permutations=10, bestParams_actualFit=None, fittingSteps=1):
    
    """
    :param data: data for which to perform the permutation analysis
    :param dat2fit: for what type of data to perform permutation analysis - pooled data | group mean | single-subject
    :param n_permutations: how many different iterations of the permutation procedure to run
    :param bestParams_actualFit: actual parameters obtained by the fitted distribution to be compared against
    :param fittingSteps: how many iterations of the fitting procedure to perform
    
    """ 
    
    #Imports
    import matplotlib.pyplot as plt
    import numpy as np
    import pycircstat
    
    from joblib import Parallel, delayed #to allow for parallelization across permutation loops
    
    from JoV_Analysis_basicFitting import clifford, fit_clifford
        
    #Determine important variables for further usage
    subs = np.unique(data.Subject)
    n_subs = len(subs)

    #Initialize permutation distribution 
    permutation_distribution = np.zeros((n_permutations, 5)) #c, s, sign, peak2peak, cost
    permuted_angles = np.zeros((n_permutations, np.sum(~np.isnan(data.Delta_angle_norm.values))))
    permuted_respErrors = np.zeros((n_permutations, np.sum(~np.isnan(data.Delta_angle_norm.values))))
    
    #Define permutation
    for i, _ in enumerate(np.arange(n_permutations)):
        print('Permutation ' + str(i))
        
        #Randomly shuffle delta_angle label for each participant
        perm_data = data.copy(deep=True)
        
        shuffledLabels = []
        for subi, sub in enumerate(subs):
            shuffledLabels_tmp = perm_data.Delta_angle_norm.values[perm_data.Subject==sub]
            np.random.shuffle(shuffledLabels_tmp)
            
            shuffledLabels.append(shuffledLabels_tmp)
        flat_shuffledLabels = [item for sublist in shuffledLabels for item in sublist]
        perm_data.Delta_angle_norm = np.array(flat_shuffledLabels)
        
        #Prepare data for fitting procedure
        if dat2fit == 'pooled':
            x = perm_data.Delta_angle_norm.values
            y = perm_data.Resp_error_demeaned.values
        elif dat2fit == 'singleSub':
            x = perm_data.Delta_angle_norm.values
            y = perm_data.Resp_error_demeaned.values
        
        x = np.deg2rad(x)
        y = np.deg2rad(y)
        
        #Fit
        c, s, sign, min_cost = fit_clifford(y, x, fittingSteps)
        
        #Compute peak-to-peak
        fit = sign * clifford(np.deg2rad(np.linspace(-90, 90, 181)), c, s)
        peak2peak = sign * (fit.max()-fit.min())
        
        permutation_distribution[i, :] = [c, s, sign, peak2peak, min_cost]
        
        permuted_angles[i, :] = np.rad2deg(x)
        permuted_respErrors[i, :] = np.rad2deg(y)
                
    return permutation_distribution, permuted_angles, permuted_respErrors

### Bootstrapping to assess standard deviation of SD effect ###
def perform_bootstrapping_SD_clifford(data=None, dat2fit='pooled', n_permutations=10, fittingSteps=1):
    
    """
    :param data: data for which to perform the bootstrapping analysis
    :param dat2fit: for what type of data to perform permutation analysis - pooled data | group mean | single-subject
    :param n_permutations: how many different iterations of the bootstrapping procedure to run
    :param fittingSteps: how many iterations of the fitting procedure to perform
    
    """ 
    
    from JoV_Analysis_basicFitting import clifford, fit_clifford
    
    #Initialize bootstrapping distribution 
    bootstrappingCoef_distribution = np.zeros((n_permutations, 4)) #c, s, sign, peak2peak
    
    #Determine important variables for further usage
    subs = np.unique(data.Subject)
    n_subs = len(subs)
    
    if n_subs > 1:
        #Randomly select subjects (with replacement from all subjects) & compute the desired parameters
        for i, _ in enumerate(np.arange(n_permutations)):
            print('Bootstrapping ' + str(i))
            
            sample_ind = np.squeeze(np.random.randint(low=0, high=n_subs, size=[n_subs, 1])) #which subjects to include in the current fit
            bootstrapp_subs = np.unique(data.Subject)[sample_ind]
            
            #Select data for fitting 
            x = []
            y = []
            if dat2fit == 'pooled':
                for subi, sub in enumerate(bootstrapp_subs):
                    
                    x_tmp = data.Delta_angle_norm.values[data.Subject==sub]
                    y_tmp = data.Resp_error_demeaned.values[data.Subject==sub]
                    
                    #y_tmp = y_tmp[~np.isnan(x_tmp)]
                    #x_tmp = x_tmp[~np.isnan(x_tmp)]
                    
                    #Take care of remaining nans designating outlier trials
                    x_tmp = x_tmp[~np.isnan(y_tmp)]
                    y_tmp = y_tmp[~np.isnan(y_tmp)]
                    
                    x.append(x_tmp)
                    y.append(y_tmp)
                
                x_flat = np.asarray([item for sublist in x for item in sublist])
                y_flat = np.asarray([item for sublist in y for item in sublist])
                
                x = np.deg2rad(x_flat)
                y = np.deg2rad(y_flat)
            
                #Fit
                c, s, sign, min_cost = fit_clifford(y, x, fittingSteps)
            
                #Compute peak-to-peak
                fit = sign * clifford(np.deg2rad(np.linspace(-90, 90, 181)), c, s)
                peak2peak = sign * (fit.max()-fit.min())
            
                bootstrappingCoef_distribution[i, :] = [c, s, sign, peak2peak]
    else:
        #Randomly select trials (with replacement) & compute the desired parameters
        for i, _ in enumerate(np.arange(n_permutations)):
            print('Permutation ' + str(i))
            
            x_tmp = data.Delta_angle_norm.values
            y_tmp = data.Resp_error_demeaned.values
            #y_tmp = y_tmp[~np.isnan(x_tmp)]
            #x_tmp = x_tmp[~np.isnan(x_tmp)]
                    
            #Take care of remaining nans designating outlier trials
            x_tmp = x_tmp[~np.isnan(y_tmp)]
            y_tmp = y_tmp[~np.isnan(y_tmp)]
            
            sample_ind = np.squeeze(np.random.randint(low=0, high=len(x), size=[len(x), 1]))
            
            x_bootstrapp = x[sample_ind]
            y_bootstrapp = y[sample_ind]
            
            #Fit
            c, s, sign, min_cost = fit_clifford(np.deg2rad(y_bootstrapp), np.deg2rad(x_bootstrapp), fittingSteps)
        
            #Compute peak-to-peak
            fit = sign * clifford(np.deg2rad(np.linspace(-90, 90, 181)), c, s)
            peak2peak = sign * (fit.max()-fit.min())
        
            bootstrappingCoef_distribution[i, :] = [c, s, sign, peak2peak]

    return bootstrappingCoef_distribution

### Permutation test to assess significance of SD at the group-level ###
def perform_permute_modelFree(data=None, n_permutations=10):
    
    """
    :param data: data for which to perform the permutation analysis
    :param n_permutations: how many different iterations of the permutation procedure to run

    """ 
    
    #Imports
    import numpy as np
    import pycircstat
    
    from joblib import Parallel, delayed #to allow for parallelization across permutation loops
    
    #First, take care of trials outside of bounds considered for SD as well as 
    #remaining nan values (aka, outliers)
    data = data[(~np.isnan(data.incl_trialsCW)) & (~np.isnan(data.Resp_error_demeaned))]

    #Initialize permutation distribution 
    permutation_distribution = np.zeros((n_permutations, 3)) #column 1: clockwise mean, 
    #column 2: counter-clockwise mean, column 3: difference

    #Define permutation
    for i, _ in enumerate(np.arange(n_permutations)):
        print('Permutation ' + str(i))
        
        #Randomly shuffle category to which a given trial belongs for each subject
        perm_data = data.copy(deep=True)

        shuffledLabels = []
        shuffledLabels_tmp = perm_data.incl_trialsCW.values
        np.random.shuffle(shuffledLabels_tmp)
            
        shuffledLabels.append(shuffledLabels_tmp)
       
        perm_data.incl_trialsCW = shuffledLabels[0]
        perm_data.incl_trialsCCW = shuffledLabels[0] 
        perm_data.incl_trialsCCW[perm_data.incl_trialsCCW == 1] = 2
        perm_data.incl_trialsCCW[perm_data.incl_trialsCCW == 0] = 1
        perm_data.incl_trialsCCW[perm_data.incl_trialsCCW == 2] = 0
        
        #Prepare data for fitting procedure
        tmp = perm_data[(perm_data.incl_trialsCW==1)].Resp_error_demeaned.values
        meanError_cw = np.rad2deg(pycircstat.mean(np.deg2rad(tmp)))
        
        tmp = perm_data[(perm_data.incl_trialsCCW==1)].Resp_error_demeaned.values
        meanError_ccw = np.rad2deg(pycircstat.mean(np.deg2rad(tmp)))
        
        #Bring into normal space
        meanError_cw = np.mod(meanError_cw+90, 180)-90
        meanError_ccw = np.mod(meanError_ccw+90, 180)-90
        
        modelFree_SD = meanError_cw - meanError_ccw
        
        #Save
        permutation_distribution[i, :] = [meanError_cw, meanError_ccw, modelFree_SD]
        
    return permutation_distribution

### Permutation test to assess significance of SD at the group-level ###
def perform_permute_SD_dvm(data=None, dat2fit='pooled', n_permutations=10, bestParams_actualFit=None, fittingSteps=1):
    
    """
    :param data: data for which to perform the permutation analysis
    :param dat2fit: for what type of data to perform permutation analysis - pooled data | group mean | single-subject
    :param n_permutations: how many different iterations of the permutation procedure to run
    :param bestParams_actualFit: actual parameters obtained by the fitted distribution to be compared against
    :param fittingSteps: how many iterations of the fitting procedure to perform
    
    """ 
    
    #Imports
    import matplotlib.pyplot as plt
    import numpy as np
    import pycircstat
    
    from joblib import Parallel, delayed #to allow for parallelization across permutation loops
    
    from JoV_Analysis_basicFitting import dvm, fit_dvm
        
    #Determine important variables for further usage
    subs = np.unique(data.Subject)
    n_subs = len(subs)

    #Initialize permutation distribution 
    permutation_distribution = np.zeros((n_permutations, 6)) #a, kappa, peak2peak, cost, gof
    permuted_angles = np.zeros((n_permutations, np.sum(~np.isnan(data.Delta_angle_norm.values))))
    permuted_respErrors = np.zeros((n_permutations, np.sum(~np.isnan(data.Delta_angle_norm.values))))
    
    #Define permutation
    for i, _ in enumerate(np.arange(n_permutations)):
        print('Permutation ' + str(i))
        
        #Randomly shuffle delta_angle label for each participant
        perm_data = data.copy(deep=True)
        
        shuffledLabels = []
        for subi, sub in enumerate(subs):
            shuffledLabels_tmp = perm_data.Delta_angle_norm.values[perm_data.Subject==sub]
            np.random.shuffle(shuffledLabels_tmp)
            
            shuffledLabels.append(shuffledLabels_tmp)
        flat_shuffledLabels = [item for sublist in shuffledLabels for item in sublist]
        perm_data.Delta_angle_norm = np.array(flat_shuffledLabels)
        
        #Prepare data for fitting procedure
        if dat2fit == 'pooled':
            x = perm_data.Delta_angle_norm.values
            y = perm_data.Resp_error_demeaned.values
        elif dat2fit == 'singleSub':
            x = perm_data.Delta_angle_norm.values
            y = perm_data.Resp_error_demeaned.values
        
        x_rad = np.deg2rad(x)
        y_rad = np.deg2rad(y)
        
        #Fit
        a, kappa, min_cost, gof = fit_dvm(y_rad, x_rad, fittingSteps)
       
        #Compute peak-to-peak
        fit = dvm(np.deg2rad(np.linspace(-90, 90, 181)), a, kappa, 0)
        fit = np.rad2deg(fit)
        peak2peak = np.sign(a) * (fit.max()-fit.min())
        
        permutation_distribution[i, :] = [a, kappa, peak2peak, min_cost, gof[0], gof[1]]
        
        permuted_angles[i, :] = x
        permuted_respErrors[i, :] = y
                
    return permutation_distribution, permuted_angles, permuted_respErrors

### Bootstrapping to assess standard deviation of SD effect ###
def perform_bootstrapping_SD_dvm(data=None, dat2fit='pooled', n_permutations=10, fittingSteps=1):
    
    """
    :param data: data for which to perform the bootstrapping analysis
    :param dat2fit: for what type of data to perform permutation analysis - pooled data | group mean | single-subject
    :param n_permutations: how many different iterations of the bootstrapping procedure to run
    :param fittingSteps: how many iterations of the fitting procedure to perform
    
    """ 
    
    from JoV_Analysis_basicFitting import dvm, fit_dvm
    
    #Initialize bootstrapping distribution 
    bootstrappingCoef_distribution = np.zeros((n_permutations, 6)) #column 1: amplitude parameter, column 2: kappa parameter, column3: peak2peak
    
    #Determine important variables for further usage
    subs = np.unique(data.Subject)
    n_subs = len(subs)
    
    if n_subs > 1:
        #Randomly select subjects (with replacement from all subjects) & compute the desired parameters
        for i, _ in enumerate(np.arange(n_permutations)):
            print('Bootstrapping ' + str(i))
            
            sample_ind = np.squeeze(np.random.randint(low=0, high=n_subs, size=[n_subs, 1])) #which subjects to include in the current fit
            bootstrapp_subs = np.unique(data.Subject)[sample_ind]
            
            #Select data for fitting 
            x = []
            y = []
            if dat2fit == 'pooled':
                for subi, sub in enumerate(bootstrapp_subs):
                    
                    x_tmp = data.Delta_angle_norm.values[data.Subject==sub]
                    y_tmp = data.Resp_error_demeaned.values[data.Subject==sub]
                    
                    #y_tmp = y_tmp[~np.isnan(x_tmp)]
                    #x_tmp = x_tmp[~np.isnan(x_tmp)]
                    
                    #Take care of remaining nans designating outlier trials
                    x_tmp = x_tmp[~np.isnan(y_tmp)]
                    y_tmp = y_tmp[~np.isnan(y_tmp)]
                    
                    x.append(x_tmp)
                    y.append(y_tmp)
                
                x_flat = np.asarray([item for sublist in x for item in sublist])
                y_flat = np.asarray([item for sublist in y for item in sublist])
                
                x = x_flat
                y = y_flat
                
                x_rad = np.deg2rad(x)
                y_rad = np.deg2rad(y)
            
                #Fit
                a, kappa, min_cost, gof = fit_dvm(y_rad, x_rad, fittingSteps)
            
                #Compute peak-to-peak
                fit = dvm(np.deg2rad(np.linspace(-90, 90, 181)), a, kappa, 0)
                fit = np.rad2deg(fit)
                peak2peak = np.sign(a) * (fit.max()-fit.min())
            
                bootstrappingCoef_distribution[i, :] = [a, kappa, peak2peak, min_cost, gof[0], gof[1]]
    else:
        #Randomly select trials (with replacement) & compute the desired parameters
        for i, _ in enumerate(np.arange(n_permutations)):
            print('Permutation ' + str(i))
            
            x_tmp = data.Delta_angle_norm.values
            y_tmp = data.Resp_error_demeaned.values
            
            #Take care of remaining nans designating outlier trials
            x = x_tmp[~np.isnan(y_tmp)]
            y = y_tmp[~np.isnan(y_tmp)]
            
            sample_ind = np.squeeze(np.random.randint(low=0, high=len(x), size=[len(x), 1]))
            
            x_bootstrapp = x[sample_ind]
            y_bootstrapp = y[sample_ind]
            
            x_rad = np.deg2rad(x_bootstrapp)
            y_rad = np.deg2rad(y_bootstrapp)
            
            #Fit
            a, kappa, min_cost, gof = fit_dvm(y_rad, x_rad, fittingSteps)
        
            #Compute peak-to-peak
            fit = dvm(np.deg2rad(np.linspace(-90, 90, 181)), a, kappa, 0)
            fit = np.rad2deg(fit)
            peak2peak = np.sign(a) * (fit.max()-fit.min())
        
            bootstrappingCoef_distribution[i, :] = [a, kappa, peak2peak, min_cost, gof[0], gof[1]]

    return bootstrappingCoef_distribution