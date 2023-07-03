#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Purpose: This file contains basic functions needed to analyze an experiment run in psychopy
#Date created: 22/11/2022
#Date last modified: 22/11/2022
#Python version: 3.7.1

import numpy as np

'''
====================================================
Basic information extraction
====================================================
'''

# Get all dimensions of a nested list
def dim(a):
    """
    :param a: input object
    
    """
    if not type(a) == list:
        return []
    
    sizes = []
    for i, _ in enumerate(np.arange(len(a))):
        sizes.append(len(a[i]))
    
    return sizes

#Compute response error
def computeRespError(data):
    """
    :param data: group data 
    """
    
    from JoV_Analysis_basicFuncs import getAngularDistance
    
    #First, add the correct response
    corrResp = np.empty(len(data))
    corrResp[:] = np.nan
    
    corrResp[data.Study=='funcStates'] = data.Mem_angle[data.Study=='funcStates'].values
    
    corrResp[(data.Study=='evBound') & (data.Rot_angle==-60)] = np.mod(data.Mem_angle[(data.Study=='evBound') & (data.Rot_angle==-60)].values - 60, 180) 
    corrResp[(data.Study=='evBound') & (data.Rot_angle==0)] = data.Mem_angle[(data.Study=='evBound') & (data.Rot_angle==0)].values
    corrResp[(data.Study=='evBound') & (data.Rot_angle==60)] = np.mod(data.Mem_angle[(data.Study=='evBound') & (data.Rot_angle==60)].values + 60, 180) 
    
    corrResp[(data.Study=='menRot') & (data.Rot_angle==-60)] = np.mod(data.Mem_angle[(data.Study=='menRot') & (data.Rot_angle==-60)].values - 60, 180) 
    corrResp[(data.Study=='menRot') & (data.Rot_angle==0)] = data.Mem_angle[(data.Study=='menRot') & (data.Rot_angle==0)].values
    corrResp[(data.Study=='menRot') & (data.Rot_angle==60)] = np.mod(data.Mem_angle[(data.Study=='menRot') & (data.Rot_angle==60)].values + 60, 180) 
    
    data.corrResp = corrResp
    
    #Next, normlize the response angles if need be
    respAngles = data.Resp_angle.values
    respAngles[~np.isnan(respAngles)] = np.mod(respAngles[~np.isnan(respAngles)], 180)
    
    data.Resp_angle = respAngles
    
    #Compute the response error (i.e., )
    respError = getAngularDistance(data.corrResp.values, data.Resp_angle.values)
    
    data.insert(64, 'RespError', respError)
    
    return data

#Compute angular distance 
def getAngularDistance(angle1, angle2):
    """
    :param angle1: reference angle (i.e., the angle to which the second one is compared to)
    :param angle2: comparison angle
    
    """
    #Convert input to array
    angle1 = np.asarray(angle1)
    angle2 = np.asarray(angle2)
    
    #Find smallest distance between the two angles
    angularDistance = np.zeros_like(angle1)
    angularDistance[~np.isnan(angle2)] = np.mod(angle1[~np.isnan(angle2)]-angle2[~np.isnan(angle2)]+90, 180)-90
    angularDistance[np.isnan(angle2)] = np.nan
    angularDistance = angularDistance * -1 #clockwise error = +, counter-clockwise error = -
    
    return angularDistance

#Compute circular error
def getCircularError(errorIn, directionality=1):
    """
    :errorIn: which data to compute the circular mean/std for
    :directionality: compute absolute errors (0) or directional errors (1)

    """
    
    #Import any needed toolboxes
    import pycircstat #toolbox for circular statistics
    
    #Compute the circular mean
    if directionality:
        error = np.deg2rad(np.asarray(errorIn)) #convert from degrees to radians
    else:
        error = np.deg2rad(np.asarray(np.abs(errorIn))) #convert ABSOLUTE ERROR from degrees to radians
    circ_mean = np.rad2deg(pycircstat.mean(error[~np.isnan(error)]))
    
    #Compute the circular std and var
    circ_std = np.rad2deg(pycircstat.std(error[~np.isnan(error)]))
    circ_var = np.rad2deg(pycircstat.var(error[~np.isnan(error)]))
    
    return circ_mean, circ_std, circ_var

#Retrieve timestamps for stimuli 
def getTimestamps(data, label, expPart):
    """
    :param data: which data to retrieve the timestamps from
    :param label: which entry to look for in the data
    :expPart: which part of the session (i.e., training vs. experiment to consider)

    """
    
    tmp = np.where(data.iloc[:, 2] == 'Added new global key event: globalQuit')
    
    if expPart == 'main':
        data_label = data.iloc[tmp[0][-1]: -1, 0][data.iloc[tmp[0][-1]: -1, 2] == label]

    #Return function at that level if the label was not found in the data
    if data_label.size==0:
        return None, None

    #Then, extract only the associated values of the pd
    data_label = data_label.to_numpy()
    data_label = data_label.astype(float)
    
    #Check where training session ends (i.e., point at which difference in timestamps < 0) & discard
    if np.min(np.diff(data_label)) < 0:
        start_exp = np.where(np.diff(data_label) == np.min(np.diff(data_label)))
        data_label = data_label[int(start_exp[0])+1 : -1]
        
        print('Discarding part of the logfile corresponding to the training session')

    #Last, extract onset & offset times 
    delta_time = [np.diff(data_label) > 1]
    onsets = data_label[1 :][delta_time]
    onsets = np.insert(onsets, 0, data_label[0])
    offsets = data_label[0 : -1][delta_time]
    offsets = np.append(offsets, data_label[-1])

    return onsets, offsets

def insertMiniblocks(data):
    """
    :param data: pandas dataframe into which to insert "miniblock" column (i.e., continuous 
                                                                           trial segments without breaks)

    """
    miniblocki = 1
    miniblocks = np.zeros(len(data))
    breakpoints = np.squeeze(np.where(data.Break.values == 1))
    
    for breaki in breakpoints:
        miniblocks[breaki+1:breakpoints[0]+breaki+2] = miniblocki
        miniblocki=miniblocki+1
    
    data.insert(2, 'Miniblocks', miniblocks)
    return data

#Remove outliers
def removeOutliers(data, cutoff=3, qualCheck=True, study='Test', subject='Pilot'):
    """
    :param: data: single-subject from which to remove outliers
    :param cutoff: cutoff used for outlier detection
    :param qualCheck: plot the results?
    :param study: which study are we plotting the results for?
    :param subject: which subjects are we plotting the results for?
    
    """
    
    #Import any needed toolboxes
    import pycircstat #toolbox for circular statistics
    
    from JoV_Analysis_basicFuncs import getCircularError

    #Compute the mean circular error across all of the subject's data 
    mean_error, std_error, _ = getCircularError(data.RespError.values, directionality=1)
    mean_error = np.mod(mean_error+90, 180)-90
    print('MEAN RESPONSE ERROR: ' + str(mean_error))
    print('STANDARD DEVIATION: ' + str(std_error))

    #Remove trials with errors larger than 3 sds of error distribution
    cutoff = cutoff*std_error
    
    data['Resp_error_clean'] = data.RespError.values #essentially the same data but with outliers removed
    data_clean = data.copy(deep=True)

    if (mean_error+cutoff < 90) & (mean_error-cutoff > -90):
        if True in np.unique((data_clean.Resp_error_clean > np.mod(mean_error+cutoff, 90)) | 
            (data_clean.Resp_error_clean < np.mod(mean_error-cutoff, -90))):
            data_clean.Resp_error_clean[(data_clean.Resp_error_clean > np.mod(mean_error+cutoff, 90)) | 
            (data_clean.Resp_error_clean <  np.mod(mean_error-cutoff, -90))] = np.nan
    else:
        if True in np.unique((data_clean.Resp_error_clean > mean_error+cutoff) | 
            (data_clean.Resp_error_clean < mean_error-cutoff)):
            data_clean.Resp_error_clean[(data_clean.Resp_error_clean > mean_error+cutoff) | 
            (data_clean.Resp_error_clean <  mean_error-cutoff)] = np.nan   
    
    data_removed = data.copy(deep=True)
    data_removed = data[np.isnan(data_clean.Resp_error_clean)]
    
    #Quick plot to check success of outlier removal
    if qualCheck:
        import matplotlib.pyplot as plt
        
        fig = plt.figure()

        plt.scatter(data_clean.Mem_angle-90, data_clean.Resp_angle-90, color='y') #clean data
        plt.scatter(data_removed.Mem_angle-90, data_removed.Resp_angle-90, color='r') #clean data
        plt.plot(np.linspace(-90, 90, 1000), np.linspace(-90, 90, 1000), color='k', alpha=.8) #fit for no-rotation trials

        ax = plt.gca()
        ax.set_xlabel('Stimulus orientation')
        ax.set_ylabel('Response orientation')
        
        plt.title('Responses for study ' + study + ' subject ' + subject)
        
    return mean_error, std_error, data_clean

'''
====================================================
Serial dependence - DOG fitting analysis pipeline
====================================================
'''

### Fit the DOG to either single-subject or group-level data ###
def analysis_dog(data=None, dat2fit='pooled', fittingSteps=200):
    """
    :param data: which data to use for fitting procedure, can be single subject or pooled data from all subjects
    :param dat2fit: what type of data to fit (can be single, subject, group mean, or pooled data)
    :param fittingSteps: how many iterations of the fitting procedure to perform
    
    """ 
    
    from JoV_Analysis_basicFitting import fit_dog
    
    #Get DOG fit
    if dat2fit == 'pooled':
        loopCount = 1 #how many different parameter estimates to expect
        
        x = data.Delta_angle_norm.values
        y = data.Resp_error_demeaned.values
        #y = y[~np.isnan(x)] #remaining nans are outlier trials
        #x = x[~np.isnan(x)]
        
        #Take care of remaining nans designating outlier trials
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]
        
    elif dat2fit == 'group_mean':
        loopCount = 1
        
        y = data #mean data
        x = np.linspace(-90, 90, 181)
        
    elif dat2fit == 'singleSub':
        subs = np.unique(data.Subject)
        loopCount = len(subs)
        
        #Initialize variables
        x = data.Delta_angle_norm.values
        y = data.Resp_error_demeaned.values
        
        #Take care of remaining nans designating outlier trials
        sub = data.Subject[~np.isnan(y)]
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]

    #Initialize parameters
    dog_params = np.zeros((loopCount, 3))
    gof = np.zeros((loopCount, 5))
    
    for loopi, _ in enumerate(np.arange(loopCount)):
        if loopCount == 1:
            a, w, min_cost, gof[loopi, :] = fit_dog(y, x, fittingSteps)
        else:
            a, w, min_cost, gof[loopi, :] = fit_dog(y[sub==subs[loopi]], x[sub==subs[loopi]], fittingSteps)
        dog_params[loopi, :] = [a, w, min_cost]
    
    return dog_params, gof

'''
====================================================
Serial dependence - Clifford fitting analysis pipeline
====================================================
'''

### Fit the Clifford model to either single-subject or group-level data ###
def analysis_clifford(data=None, dat2fit='pooled', fittingSteps=200):
    """
    :param data: which data to use for fitting procedure, can be single subject or pooled data from all subjects
    :param dat2fit: what type of data to fit (can be single, subject, group mean, or pooled data)
    :param fittingSteps: how many iterations of the fitting procedure to perform
    
    """ 
    
    from JoV_Analysis_basicFitting import fit_clifford
    
    #Get Clifford fit
    if dat2fit == 'pooled':
        loopCount = 1 #how many different parameter estimates to expect
        
        x = data.Delta_angle_norm.values
        y = data.Resp_error_demeaned.values
        #y = y[~np.isnan(x)] #remaining nans are outlier trials
        #x = x[~np.isnan(x)]
        
        #Take care of remaining nans designating outlier trials
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]
        
        y = np.deg2rad(y)
        x = np.deg2rad(x)
        
    elif dat2fit == 'group_mean':
        loopCount = 1
        
        y = data #mean data
        x = np.linspace(-90, 90, 181)
        
        y = np.deg2rad(y)
        x = np.deg2rad(x)
        
    elif dat2fit == 'singleSub':
        subs = np.unique(data.Subject)
        loopCount = len(subs)
        
        #Initialize variables
        x = data.Delta_angle_norm.values
        y = data.Resp_error_demeaned.values
        sub = data.Subject[~np.isnan(x)]
        y = y[~np.isnan(x)]
        x = x[~np.isnan(x)]
        
        y = np.deg2rad(y)
        x = np.deg2rad(x)

    #Initialize parameters
    clifford_params = np.zeros((loopCount, 4))
    
    for loopi, _ in enumerate(np.arange(loopCount)):
        if loopCount == 1:
            c, s, sign, min_cost = fit_clifford(y, x, fittingSteps)
        else:
            c, s, sign, min_cost = fit_clifford(y[sub==subs[loopi]], x[sub==subs[loopi]], fittingSteps)
        clifford_params[loopi, :] = [c, s, sign, min_cost]
    
    return clifford_params

'''
====================================================
Serial dependence - DVM fitting analysis pipeline
====================================================
'''

### Fit the DVM to either single-subject or group-level data ###
def analysis_dvm(data=None, dat2fit='pooled', fittingSteps=200):
    """
    :param data: which data to use for fitting procedure, can be single subject or pooled data from all subjects
    :param dat2fit: what type of data to fit (can be single, subject, group mean, or pooled data)
    :param fittingSteps: how many iterations of the fitting procedure to perform
    
    """ 
    
    from JoV_Analysis_basicFitting import fit_dvm
    
    #Get DOG fit
    if dat2fit == 'pooled':
        loopCount = 1 #how many different parameter estimates to expect
        
        x = data.Delta_angle_norm.values
        y = data.Resp_error_demeaned.values
        #y = y[~np.isnan(x)] #remaining nans are outlier trials
        #x = x[~np.isnan(x)]
        
        #Take care of remaining nans designating outlier trials
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]
        
    elif dat2fit == 'group_mean':
        loopCount = 1
        
        y = data #mean data
        x = np.linspace(-90, 90, 181)
        
    elif dat2fit == 'singleSub':
        subs = np.unique(data.Subject)
        loopCount = len(subs)
        
        #Initialize variables
        x = data.Delta_angle_norm.values
        y = data.Resp_error_demeaned.values
        
        #Take care of remaining nans designating outlier trials
        sub = data.Subject[~np.isnan(y)]
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]
    
    #Convert to radians
    x_rad = np.deg2rad(x)
    y_rad = np.deg2rad(y)

    #Initialize parameters
    dvm_params = np.zeros((loopCount, 3))
    gof = np.zeros((loopCount, 2))
    
    for loopi, _ in enumerate(np.arange(loopCount)):
        if loopCount == 1:
            a, kappa, min_cost, gof[loopi, :] = fit_dvm(y_rad, x_rad, fittingSteps)
        else:
            a, kappa, min_cost, gof[loopi, :] = fit_dvm(y_rad[sub==subs[loopi]], x_rad[sub==subs[loopi]], fittingSteps)
        dvm_params[loopi, :] = [a, kappa, min_cost]
    
    return dvm_params, gof