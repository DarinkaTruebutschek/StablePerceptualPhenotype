#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:03:28 2022

@author: darinka
"""

'''
====================================================
Import needed tools
====================================================
'''

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from colorama import Fore, Back, init, Style
from copy import deepcopy
from numpy.random import default_rng

from pathlib import Path

from JoV_Analysis_cfg import *
from JoV_Analysis_loadData import loadData
from JoV_Analysis_runDemographics import runDemographics
from JoV_Analysis_objectivePerformance import run_Analysis_objectivePerformance
from JoV_Analysis_basicFuncs import computeRespError, getAngularDistance, removeOutliers
from JoV_Analysis_Preprocessing import run_removeMeanError
from JoV_Analysis_SD import run_Analysis_SD, run_Analysis_modelfreeSD, run_getFWHM, run_Analysis_SD_movingAvg
from JoV_Analysis_PlotSD import run_plot_SD
from JoV_Analysis_PlotSD_singleSubs import run_plot_SD_singleSubs
from JoV_Analysis_controls import run_control_dependenceR2
from JoV_Analysis_PlotSD_VariabilityInterSub import run_plot_SD_VariabilityInterSub
from JoV_Analysis_PlotSD_plotCorrelation import run_plot_SD_ConsistencyIntraSub, run_plot_SD_ConsistencyIntraSub_modelFree

init(autoreset=True)

'''
====================================================
Combine data from all three experiments in 1 big dataframe
====================================================
'''

data_all = loadData(studies)

'''
====================================================
Get demographics info
====================================================
'''

runDemographics(data_all[data_all.Study != 'funcStates'])

'''
====================================================
Trial selection: Determine specifically which trials
we will look at in the SD analyses
====================================================
'''

#FuncStates: all trials
#EvBound: all no-rotation repetition trials (i.e., excluding switch trials)
#MenRot: all no-rotation trials (excluding those following the 2AFC-task)

trialSelection = np.zeros(len(data_all))

for triali, trial in enumerate(np.arange(len(trialSelection))):
    
    if (triali > 0): #start with second trial
        #Make sure we are in the same study, session, miniblock, and subject
        if ((data_all.Study[triali] == data_all.Study[triali-1]) &
            (data_all.Subject[triali] == data_all.Subject[triali-1]) &
            (data_all.Session[triali] == data_all.Session[triali-1]) &
            (data_all.Miniblocks[triali] == data_all.Miniblocks[triali-1]) &
            (data_all.incl_JoV[triali] == 1)):
            
            #FuncStates    
            if data_all.Study[triali] == 'funcStates':
                trialSelection[triali] = 1
            
            #EvBound
            elif data_all.Study[triali] == 'evBound':
                if ((data_all.EventPos[triali] > 0) &
                    (data_all.Rot_angle[triali] == 0) & (data_all.Rot_angle[triali-1] == 0) &
                    (data_all.Resp_req[triali] == 1 ) & (data_all.Resp_req[triali-1] == 1)):
                        trialSelection[triali] = 1
            #MenRot
            elif data_all.Study[triali] == 'menRot':
                if ((data_all.Rot_angle[triali] == 0) & (data_all.Rot_angle[triali-1] == 0) &
                    (data_all.AFC[triali] == 0) & (data_all.AFC[triali-1] == 0)):
                        trialSelection[triali] = 1
                
data_all.insert(63, 'incl_trials', trialSelection)

'''
====================================================
Controll analysis: Trial selection for control analysis: Resp error on current & previous 
trial less than or larger than 10 deg
====================================================
'''
#Compute response error
respError = computeRespError(data_all) 
respError = respError.RespError.values

trialSelection_green = np.zeros(len(data_all))
trialSelection_notGreen = np.zeros(len(data_all))

for triali, trial in enumerate(np.arange(len(trialSelection))):
    
    #Make sure we are only looking at trials that have already been included
    if (triali > 0):
        if (data_all.incl_trials[triali] == 1):
            if (np.abs(respError[triali]) < 10) & (np.abs(respError[triali-1]) < 10):
                trialSelection_green[trial] = 1
                trialSelection_notGreen[triali] = 0
            elif (np.abs(respError[triali]) < 10) & (np.abs(respError[triali-1]) > 10):
                trialSelection_green[triali] = 0
                trialSelection_notGreen[triali] = 1
            else:
                trialSelection_green[triali] = 0
                trialSelection_notGreen[triali] = 0
                
data_all = data_all.drop(['RespError'], axis=1)

data_all.insert(64, 'incl_trialsGreen', trialSelection_green)
data_all.insert(65, 'incl_trialsNotGreen', trialSelection_notGreen)

'''
====================================================
Determine the average number of trials included per
participant and study
====================================================
'''

n_trials_evBound = []

for subi, sub in enumerate(np.unique(data_all.Subject[data_all.Study=='evBound'])):
    tmp = len(data_all[(data_all.Study=='evBound') & (data_all.incl_trials==1) & 
                       (data_all.Subject==sub)])
    n_trials_evBound.append(tmp)

n_trials_menRot = []

for subi, sub in enumerate(np.unique(data_all.Subject[(data_all.Study=='menRot') & 
                                                      (data_all.incl_JoV==1)])):
    tmp = len(data_all[(data_all.Study=='menRot') & (data_all.incl_trials==1) & 
                       (data_all.Subject==sub) & (data_all.incl_JoV==1)])
    n_trials_menRot.append(tmp)

print(str(np.mean(n_trials_menRot)))
print(str(np.std(n_trials_menRot)))
   
'''
====================================================
Assess raw objective performance on reproduction task
====================================================
'''

angle_bins = np.linspace(-90, 90, 19)
respError_var = 'RespError'

#For all 3 studies
run_Analysis_objectivePerformance(deepcopy(data_all), studies, respError_var, angle_bins, path_results)

#For my studies
run_Analysis_objectivePerformance(deepcopy(data_all), ['evBound', 'menRot'], respError_var, angle_bins, path_results)

'''
====================================================
Compute delta angle
====================================================
'''

#Compute response error
data_all = computeRespError(data_all) 

#Get angular distances
currOri = data_all.Mem_angle.values
prevOri = deepcopy(currOri)
prevOri = np.insert(prevOri, 0, np.nan)

deltaAngle = np.mod(prevOri[:-1]-currOri+90, 180)-90

data_all.insert(65, 'Delta_angle_norm', deltaAngle)

'''
====================================================
Trial selection model-free analysis: Categorize trials
as either being clockwise or counter-clockwise and select
data for further steps
====================================================
'''
trialSelection_ccw = np.zeros(len(data_all))
trialSelection_cw = np.zeros(len(data_all))

for triali, trial in enumerate(np.arange(len(trialSelection_ccw))):
    if data_all.Study[triali] == 'evBound':
        lowerBound = bounds_modelFree_evBound[0]
        upperBound = bounds_modelFree_evBound[1]
    elif data_all.Study[triali] == 'menRot':
        lowerBound = bounds_modelFree_menRot[0]
        upperBound = bounds_modelFree_menRot[1]
    else:
        lowerBound = bounds_modelFree[0]
        upperBound = bounds_modelFree[1]
    if (data_all.Delta_angle_norm[triali] < 0) & (data_all.Delta_angle_norm[triali] > lowerBound):
        trialSelection_ccw[triali] = 1
        trialSelection_cw[triali] = 0
    elif (data_all.Delta_angle_norm[triali] > 0) & (data_all.Delta_angle_norm[triali] < upperBound):
        trialSelection_ccw[triali] = 0
        trialSelection_cw[triali] = 1
    else:
        trialSelection_ccw[triali] = np.nan
        trialSelection_cw[triali] = np.nan

data_all.insert(66, 'incl_trialsCW', trialSelection_cw)   
data_all.insert(67, 'incl_trialsCCW', trialSelection_ccw)     
    
selData = data_all[(data_all.incl_trials==1) & (data_all.incl_JoV==1)]
selData_green = data_all[(data_all.incl_trialsGreen==1) & (data_all.incl_JoV==1)]
selData_notGreen = data_all[(data_all.incl_trialsNotGreen==1) & (data_all.incl_JoV==1)]

# '''
# ====================================================
# Select data (i.e., trials & subjects) for control analysis
# ====================================================
# '''

# #Get angular distances
# currOri = data_all.Mem_angle.values
# futureOri = deepcopy(currOri)
# futureOri = futureOri[1:]

# futureOri = np.concatenate((futureOri, np.expand_dims(futureOri[0], axis=0)))
# futureOri[-1] = np.nan

# deltaAngle_con = np.mod(futureOri-currOri+90, 180)-90

# data_all.Delta_angle_norm = deltaAngle_con

# selData_con = data_all[(data_all.incl_trials==1) & (data_all.incl_JoV==1)]


'''
====================================================
Preprocess data in prep for SD analyses:
Remove outliers and correct for mean shift - done 
separately for the each session
====================================================
'''   

#First, remove all outliers (done separately within each subject and each session)
data_clean = []

for studi, study in enumerate(studies):
    for subi, sub in enumerate(np.unique(selData.Subject[selData.Study==study])):
        
        data_tmp = []
        
        for sessi, sess in enumerate(np.unique(selData.Session)):
            _, _, tmp = removeOutliers(data=selData[(selData.Study==study) & (selData.Subject==sub) 
                                                          & (selData.Session==sessi)], 
                                            cutoff=3, qualCheck=False, study=study, subject=sub)
            data_tmp.append(tmp)
        
        data_tmp = pd.concat(data_tmp)
        data_clean.append(data_tmp)
    
data_clean = pd.concat(data_clean)

#Determine how many trials were removed on average
for studi, study in enumerate(np.unique(data_clean.Study)):
    n_trials_removed = []
    for subi, sub in enumerate(np.unique(data_clean.Subject)):
        tmp = len(np.where(data_clean.Resp_error_clean[(data_clean.Study==study) & 
                                                   (data_clean.Subject==sub)] != 
                       data_clean.RespError[(data_clean.Study==study) & 
                                                                  (data_clean.Subject==sub)])[0])
        
        n_trials_removed.append(tmp)
    print(study)
    print(str(np.mean(n_trials_removed)))
    print(str(np.std(n_trials_removed)))
        
#Then, demean the data 
data_demeaned = []

for studi, study in enumerate(studies):
    
    data_tmp_study = []
    
    for subi, sub in enumerate(np.unique(data_clean.Subject[data_clean.Study==study])):
    
        data_tmp = []
    
        for sessi, sess in enumerate(np.unique(data_clean.Session)):
            tmp = run_removeMeanError(data_clean[(data_clean.Study==study) & (data_clean.Subject==sub) 
                                                 & (data_clean.Session==sessi)])
        
            data_tmp.append(tmp)
            
        data_tmp = pd.concat(data_tmp)  
        data_tmp_study.append(data_tmp)
    
    data_tmp_study = pd.concat(data_tmp_study)
    data_demeaned.append(data_tmp_study)

data_demeaned = pd.concat(data_demeaned)

'''
====================================================
Preprocess data in prep for SD analyses:
Remove outliers and correct for mean shift - done 
separately for the each session (control analysis green)
====================================================
'''   

#First, remove all outliers (done separately within each subject and each session)
data_clean_green = []

for studi, study in enumerate(studies):
    for subi, sub in enumerate(np.unique(selData_green.Subject[selData_green.Study==study])):
        
        data_tmp = []
        
        for sessi, sess in enumerate(np.unique(selData_green.Session)):
            _, _, tmp = removeOutliers(data=selData_green[(selData_green.Study==study) & (selData_green.Subject==sub) 
                                                          & (selData_green.Session==sessi)], 
                                            cutoff=3, qualCheck=False, study=study, subject=sub)
            data_tmp.append(tmp)
        
        data_tmp = pd.concat(data_tmp)
        data_clean_green.append(data_tmp)
    
data_clean_green = pd.concat(data_clean_green)

#Determine how many trials were removed on average
for studi, study in enumerate(np.unique(data_clean_green.Study)):
    n_trials_removed = []
    for subi, sub in enumerate(np.unique(data_clean_green.Subject)):
        tmp = len(np.where(data_clean_green.Resp_error_clean[(data_clean_green.Study==study) & 
                                                   (data_clean_green.Subject==sub)] != 
                       data_clean_green.RespError[(data_clean_green.Study==study) & 
                                                                  (data_clean_green.Subject==sub)])[0])
        
        n_trials_removed.append(tmp)
    print(study)
    print(str(np.mean(n_trials_removed)))
    print(str(np.std(n_trials_removed)))
        
#Then, demean the data 
data_demeaned_green = []

for studi, study in enumerate(studies):
    
    data_tmp_study = []
    
    for subi, sub in enumerate(np.unique(data_clean_green.Subject[data_clean_green.Study==study])):
    
        data_tmp = []
    
        for sessi, sess in enumerate(np.unique(data_clean_green.Session)):
            tmp = run_removeMeanError(data_clean_green[(data_clean_green.Study==study) & (data_clean_green.Subject==sub) 
                                                 & (data_clean_green.Session==sessi)])
        
            data_tmp.append(tmp)
            
        data_tmp = pd.concat(data_tmp)  
        data_tmp_study.append(data_tmp)
    
    data_tmp_study = pd.concat(data_tmp_study)
    data_demeaned_green.append(data_tmp_study)

data_demeaned_green = pd.concat(data_demeaned_green)

'''
====================================================
Preprocess data in prep for SD analyses:
Remove outliers and correct for mean shift - done 
separately for the each session (control analysis not green)
====================================================
'''   

#First, remove all outliers (done separately within each subject and each session)
data_clean_notGreen = []

for studi, study in enumerate(studies):
    for subi, sub in enumerate(np.unique(selData_notGreen.Subject[selData_notGreen.Study==study])):
        
        data_tmp = []
        
        for sessi, sess in enumerate(np.unique(selData_notGreen.Session)):
            _, _, tmp = removeOutliers(data=selData_notGreen[(selData_notGreen.Study==study) & (selData_notGreen.Subject==sub) 
                                                          & (selData_notGreen.Session==sessi)], 
                                            cutoff=3, qualCheck=False, study=study, subject=sub)
            data_tmp.append(tmp)
        
        data_tmp = pd.concat(data_tmp)
        data_clean_notGreen.append(data_tmp)
    
data_clean_notGreen = pd.concat(data_clean_notGreen)

#Determine how many trials were removed on average
for studi, study in enumerate(np.unique(data_clean_notGreen.Study)):
    n_trials_removed = []
    for subi, sub in enumerate(np.unique(data_clean_notGreen.Subject)):
        tmp = len(np.where(data_clean_notGreen.Resp_error_clean[(data_clean_notGreen.Study==study) & 
                                                   (data_clean_notGreen.Subject==sub)] != 
                       data_clean_notGreen.RespError[(data_clean_notGreen.Study==study) & 
                                                                  (data_clean_notGreen.Subject==sub)])[0])
        
        n_trials_removed.append(tmp)
    print(study)
    print(str(np.mean(n_trials_removed)))
    print(str(np.std(n_trials_removed)))
        
#Then, demean the data 
data_demeaned_notGreen = []

for studi, study in enumerate(studies):
    
    data_tmp_study = []
    
    for subi, sub in enumerate(np.unique(data_clean_notGreen.Subject[data_clean_notGreen.Study==study])):
    
        data_tmp = []
    
        for sessi, sess in enumerate(np.unique(data_clean_notGreen.Session)):
            tmp = run_removeMeanError(data_clean_notGreen[(data_clean_notGreen.Study==study) & (data_clean_notGreen.Subject==sub) 
                                                 & (data_clean_notGreen.Session==sessi)])
        
            data_tmp.append(tmp)
            
        data_tmp = pd.concat(data_tmp)  
        data_tmp_study.append(data_tmp)
    
    data_tmp_study = pd.concat(data_tmp_study)
    data_demeaned_notGreen.append(data_tmp_study)

data_demeaned_notGreen = pd.concat(data_demeaned_notGreen)

'''
====================================================
Preprocess data in prep for SD analyses:
Remove outliers and correct for mean shift - done 
separately for the each session
====================================================
'''   

# #First, remove all outliers (done separately within each subject and each session)
# data_clean_con = []

# for studi, study in enumerate(studies):
#     for subi, sub in enumerate(np.unique(selData_con.Subject[selData_con.Study==study])):
        
#         data_tmp = []
        
#         for sessi, sess in enumerate(np.unique(selData_con.Session)):
#             _, _, tmp = removeOutliers(data=selData_con[(selData_con.Study==study) & (selData_con.Subject==sub) 
#                                                           & (selData_con.Session==sessi)], 
#                                             cutoff=3, qualCheck=False, study=study, subject=sub)
#             data_tmp.append(tmp)
        
#         data_tmp = pd.concat(data_tmp)
#         data_clean_con.append(data_tmp)
    
# data_clean_con = pd.concat(data_clean_con)

# #Then, demean the data 
# data_demeaned_con = []

# for studi, study in enumerate(studies):
    
#     data_tmp_study = []
    
#     for subi, sub in enumerate(np.unique(data_clean_con.Subject[data_clean_con.Study==study])):
    
#         data_tmp = []
    
#         for sessi, sess in enumerate(np.unique(data_clean_con.Session)):
#             tmp = run_removeMeanError(data_clean_con[(data_clean_con.Study==study) & (data_clean_con.Subject==sub) 
#                                                   & (data_clean_con.Session==sessi)])
        
#             data_tmp.append(tmp)
            
#         data_tmp = pd.concat(data_tmp)  
#         data_tmp_study.append(data_tmp)
    
#     data_tmp_study = pd.concat(data_tmp_study)
#     data_demeaned_con.append(data_tmp_study)

# data_demeaned_con = pd.concat(data_demeaned_con)

'''
====================================================
Check remaining error distributions
====================================================
'''

angle_bins = np.linspace(-90, 90, 19)
respError_var = 'Resp_error_demeaned'

#For all 3 studies
run_Analysis_objectivePerformance(deepcopy(data_demeaned), studies, respError_var, angle_bins, path_results)

#For my studies
run_Analysis_objectivePerformance(deepcopy(data_demeaned), ['evBound', 'menRot'], respError_var, angle_bins, path_results)     

# '''
# ====================================================
# Control SD analysis across both sessions
# ====================================================
# '''   
# del selData_con

# currentStudies = ['evBound', 'funcStates', 'menRot']

# selData_con = [data_demeaned_con[(data_demeaned_con.Study=='evBound')],
#             data_demeaned_con[(data_demeaned_con.Study=='funcStates')],
#             data_demeaned_con[(data_demeaned_con.Study=='menRot')]]

# saveNames = ['perms_1000_evBound_con',
#               'perms_1000_funcStates_con',
#               'perms_1000_menRot_con']

# for condi, _ in enumerate(np.arange(len(selData_con))):
    
#     print('Computing SD for study ' + currentStudies[condi])
    
#     #First, we will look at the pooled data (i.e., pretending there is only a single subject)
#     run_Analysis_SD(data=selData_con[condi], currentStudy=currentStudies[condi], model='DoG', bins=bins, bin_width=bin_width, 
#                 collapseSubs='pooled', dog_fittingSteps=dog_fittingSteps, stats_n_permutations=10,
#                 savename=saveNames[condi], path_results=path_results, rerun_fit=1, rerun_perms=1, rerun_bootstrapp=1)
    
# '''
# ====================================================
# Plot the results from the control analyses in the same figure
# ====================================================
# '''
# #Across both sessions   
# saveNames = ['perms_1000_evBound_con',
#               'perms_1000_menRot_con']

# run_plot_SD(data=data_demeaned_con[data_demeaned_con.Study != 'funcStates'], sess2plot='all', model='DoG', collapseSubs='pooled', 
#             stats_n_permutations=1000, my_sig='rsquared', savename=saveNames, bin_width=bin_width, path_results=path_results)

# '''
# ====================================================
# Compute FWHM for control analysis
# ====================================================
# '''  
# currentStudies = ['evBound', 'menRot']

# saveNames = ['perms_1000_evBound_con',
#              'perms_1000_menRot_con']

# for condi, _ in enumerate(np.arange(len(currentStudies))):
#     run_getFWHM(currentStudy=currentStudies[condi], model='DoG', collapseSubs='pooled', 
#                 savename=saveNames[condi], path_results=path_results)
    
'''
====================================================
Classic SD analysis across both sessions
====================================================
'''   
del selData

#currentStudies = ['evBound', 'funcStates', 'menRot']

#selData = [data_demeaned[(data_demeaned.Study == 'evBound')],
           #data_demeaned[(data_demeaned.Study == 'funcStates')],
           #data_demeaned[(data_demeaned.Study == 'menRot')]]

#saveNames = ['perms_1000_evBound',
             #'perms_1000_funcStates',
             #'perms_1000_menRot']
             
currentStudies = ['evBound', 'menRot']

selData = [data_demeaned[(data_demeaned.Study == 'evBound')],
           data_demeaned[(data_demeaned.Study == 'menRot')]]

saveNames = ['perms_1000_evBound',
             'perms_1000_menRot']

for condi, _ in enumerate(np.arange(len(selData))):
    
    print('Computing SD for study ' + currentStudies[condi])
    
    #First, we will look at the pooled data (i.e., pretending there is only a single subject)
    run_Analysis_SD(data=selData[condi], currentStudy=currentStudies[condi], model='DvM', bins=bins, bin_width=bin_width, 
                collapseSubs='pooled', dog_fittingSteps=dog_fittingSteps, stats_n_permutations=1000,
                savename=saveNames[condi], path_results=path_results, rerun_fit=0, rerun_perms=0, rerun_bootstrapp=0)
    
'''
====================================================
Classic SD analysis across both sessions: Green
====================================================
'''   
del selData

#currentStudies = ['evBound', 'funcStates', 'menRot']

#selData = [data_demeaned[(data_demeaned.Study == 'evBound')],
           #data_demeaned[(data_demeaned.Study == 'funcStates')],
           #data_demeaned[(data_demeaned.Study == 'menRot')]]

#saveNames = ['perms_1000_evBound',
             #'perms_1000_funcStates',
             #'perms_1000_menRot']
             
currentStudies = ['evBound', 'menRot']

selData = [data_demeaned_green[(data_demeaned_green.Study == 'evBound')],
           data_demeaned_green[(data_demeaned_green.Study == 'menRot')]]

saveNames = ['perms_1000_evBound_green',
             'perms_1000_menRot_green']

for condi, _ in enumerate(np.arange(len(selData))):
    
    print('Computing SD for study ' + currentStudies[condi])
    
    #First, we will look at the pooled data (i.e., pretending there is only a single subject)
    run_Analysis_SD(data=selData[condi], currentStudy=currentStudies[condi], model='DoG', bins=bins, bin_width=bin_width, 
                collapseSubs='pooled', dog_fittingSteps=dog_fittingSteps, stats_n_permutations=1000,
                savename=saveNames[condi], path_results=path_results, rerun_fit=1, rerun_perms=1, rerun_bootstrapp=1)

'''
====================================================
Classic SD analysis across both sessions: Not green
====================================================
'''   
del selData

#currentStudies = ['evBound', 'funcStates', 'menRot']

#selData = [data_demeaned[(data_demeaned.Study == 'evBound')],
           #data_demeaned[(data_demeaned.Study == 'funcStates')],
           #data_demeaned[(data_demeaned.Study == 'menRot')]]

#saveNames = ['perms_1000_evBound',
             #'perms_1000_funcStates',
             #'perms_1000_menRot']
             
currentStudies = ['evBound', 'menRot']

selData = [data_demeaned_notGreen[(data_demeaned_notGreen.Study == 'evBound')],
           data_demeaned_notGreen[(data_demeaned_notGreen.Study == 'menRot')]]

saveNames = ['perms_1000_evBound_notGreen_prevOnly',
             'perms_1000_menRot_notGreen_prevOnly']

for condi, _ in enumerate(np.arange(len(selData))):
    
    print('Computing SD for study ' + currentStudies[condi])
    
    #First, we will look at the pooled data (i.e., pretending there is only a single subject)
    run_Analysis_SD(data=selData[condi], currentStudy=currentStudies[condi], model='DoG', bins=bins, bin_width=bin_width, 
                collapseSubs='pooled', dog_fittingSteps=dog_fittingSteps, stats_n_permutations=1000,
                savename=saveNames[condi], path_results=path_results, rerun_fit=1, rerun_perms=1, rerun_bootstrapp=1)


'''
====================================================
Classic SD analysis across both sessions: 10,000 permutations 
====================================================
'''   
del selData

#currentStudies = ['evBound', 'funcStates', 'menRot']

#selData = [data_demeaned[(data_demeaned.Study == 'evBound')],
           #data_demeaned[(data_demeaned.Study == 'funcStates')],
           #data_demeaned[(data_demeaned.Study == 'menRot')]]

#saveNames = ['perms_1000_evBound',
             #'perms_1000_funcStates',
             #'perms_1000_menRot']
             
currentStudies = ['evBound', 'menRot']

selData = [data_demeaned[(data_demeaned.Study == 'evBound')],
           data_demeaned[(data_demeaned.Study == 'menRot')]]

saveNames = ['perms_10000_evBound',
             'perms_10000_menRot']

for condi, _ in enumerate(np.arange(len(selData))):
    
    print('Computing SD for study ' + currentStudies[condi])
    
    #First, we will look at the pooled data (i.e., pretending there is only a single subject)
    run_Analysis_SD(data=selData[condi], currentStudy=currentStudies[condi], model='DoG', bins=bins, bin_width=bin_width, 
                collapseSubs='pooled', dog_fittingSteps=dog_fittingSteps, stats_n_permutations=10000,
                savename=saveNames[condi], path_results=path_results, rerun_fit=0, rerun_perms=0, rerun_bootstrapp=0)



'''
====================================================
Compute FWHM to inform bounds for model-free analysis
====================================================
'''  
currentStudies = ['evBound', 'menRot']

#Across both sessions
saveNames = ['perms_1000_evBound',
             'perms_1000_menRot']

for condi, _ in enumerate(np.arange(len(currentStudies))):
    run_getFWHM(currentStudy=currentStudies[condi], model='DvM', collapseSubs='pooled', 
                savename=saveNames[condi], path_results=path_results)
    
#Session 1
saveNames = ['perms_1000_evBound_Session1',
              'perms_1000_menRot_Session1']

for condi, _ in enumerate(np.arange(len(currentStudies))):
    run_getFWHM(currentStudy=currentStudies[condi], model='DvM', collapseSubs='pooled', 
                savename=saveNames[condi], path_results=path_results)

#Session 2
saveNames = ['perms_1000_evBound_Session2',
              'perms_1000_menRot_Session2']

for condi, _ in enumerate(np.arange(len(currentStudies))):
    run_getFWHM(currentStudy=currentStudies[condi], model='DvM', collapseSubs='pooled', 
                savename=saveNames[condi], path_results=path_results)

'''
====================================================
Classic SD analysis separately for both sessions
====================================================
'''   
del selData

currentStudies = ['evBound', 'evBound', 'menRot', 'menRot']

selData = [data_demeaned[(data_demeaned.Study=='evBound') & (data_demeaned.Session==0)],
           data_demeaned[(data_demeaned.Study=='evBound') & (data_demeaned.Session==1)],
           data_demeaned[(data_demeaned.Study=='menRot') & (data_demeaned.Session==0)],
           data_demeaned[(data_demeaned.Study=='menRot') & (data_demeaned.Session==1)]]

saveNames = ['perms_1000_evBound_Session1', 'perms_1000_evBound_Session2',
             'perms_1000_menRot_Session1', 'perms_1000_menRot_Session2']

for condi, _ in enumerate(np.arange(len(selData))):
    
    print('Computing SD for study ' + currentStudies[condi])
    
    #First, we will look at the pooled data (i.e., pretending there is only a single subject)
    run_Analysis_SD(data=selData[condi], currentStudy=currentStudies[condi], model='DvM', bins=bins, bin_width=bin_width, 
                collapseSubs='pooled', dog_fittingSteps=dog_fittingSteps, stats_n_permutations=1000,
                savename=saveNames[condi], path_results=path_results, rerun_fit=1, rerun_perms=1, rerun_bootstrapp=1)

'''
====================================================
Plot the above results in the same figure
====================================================
'''
#Across both sessions   
saveNames = ['perms_1000_evBound',
             'perms_1000_menRot']

run_plot_SD(data=data_demeaned[data_demeaned.Study != 'funcStates'], sess2plot='all', model='DvM', collapseSubs='pooled', 
            stats_n_permutations=1000, my_sig='amplitude', savename=saveNames, bin_width=bin_width, path_results=path_results)

#Session1
saveNames = ['perms_1000_evBound_Session1',
             'perms_1000_menRot_Session1']

run_plot_SD(data=data_demeaned[(data_demeaned.Study != 'funcStates') & (data_demeaned.Session==0)], sess2plot='Session1', model='DvM', collapseSubs='pooled', 
            stats_n_permutations=1000, my_sig='amplitude', savename=saveNames, bin_width=bin_width, path_results=path_results)

#Session2
saveNames = ['perms_1000_evBound_Session2',
             'perms_1000_menRot_Session2']

run_plot_SD(data=data_demeaned[(data_demeaned.Study != 'funcStates') & (data_demeaned.Session==1)], sess2plot='Session2', model='DvM', collapseSubs='pooled', 
            stats_n_permutations=1000, my_sig='amplitude', savename=saveNames, bin_width=bin_width, path_results=path_results)

'''
====================================================
Control analysis: Plot DoG vs DvM fit in same figure
====================================================
'''
#Across both sessions   
saveNames = ['perms_1000_evBound',
             'perms_1000_menRot']

run_plot_SD(data=data_demeaned[data_demeaned.Study != 'funcStates'], sess2plot='all', model='DoG&DvM', collapseSubs='pooled', 
            stats_n_permutations=1000, my_sig='amplitude', savename=saveNames, bin_width=bin_width, path_results=path_results)

#Session1
saveNames = ['perms_1000_evBound_Session1',
             'perms_1000_menRot_Session1']

run_plot_SD(data=data_demeaned[(data_demeaned.Study != 'funcStates') & (data_demeaned.Session==0)], sess2plot='Session1', model='DoG&DvM', collapseSubs='pooled', 
            stats_n_permutations=1000, my_sig='amplitude', savename=saveNames, bin_width=bin_width, path_results=path_results)

#Session2
saveNames = ['perms_1000_evBound_Session2',
             'perms_1000_menRot_Session2']

run_plot_SD(data=data_demeaned[(data_demeaned.Study != 'funcStates') & (data_demeaned.Session==1)], sess2plot='Session2', model='DoG&DvM', collapseSubs='pooled', 
            stats_n_permutations=1000, my_sig='amplitude', savename=saveNames, bin_width=bin_width, path_results=path_results)

'''
====================================================
Control analysis: How does model fit (as measured by R2)
depend on trial number
====================================================
'''
del selData

selData = [data_demeaned[(data_demeaned.Study=='evBound')],
           data_demeaned[(data_demeaned.Study=='menRot')]]

run_control_dependenceR2(data=selData, path_results=path_results)

'''
====================================================
Classic SD analysis for each subject across the two sessions
====================================================
'''   
del selData

currentStudies = ['evBound', 'menRot']

selData = [data_demeaned[(data_demeaned.Study=='evBound')],
           data_demeaned[(data_demeaned.Study=='menRot')]]

saveNames = ['perms_1000_evBound', 'perms_1000_menRot']

for condi, _ in enumerate(np.arange(len(selData))):
    
    #Determine number of subjects
    subs_tmp = np.unique(selData[condi].Subject)
    
    for subi, sub in enumerate(subs_tmp):
        saveNames_tmp = saveNames[condi] + '_Subject_' + sub
        print('Computing SD for study ' + currentStudies[condi] + ' subject ' + sub)
    
        run_Analysis_SD(data=selData[condi][selData[condi].Subject==sub], currentStudy=currentStudies[condi], model='DvM', bins=bins, bin_width=bin_width+5, 
                    collapseSubs='singleSub', dog_fittingSteps=dog_fittingSteps, stats_n_permutations=1000,
                    savename=saveNames_tmp, path_results=path_results, rerun_fit=1, rerun_perms=1, rerun_bootstrapp=1)


'''
====================================================
Classic SD analysis for each subject seperately for both sessions
====================================================
'''   
del selData

currentStudies = ['evBound', 'evBound', 'menRot', 'menRot']

selData = [data_demeaned[(data_demeaned.Study=='evBound') & (data_demeaned.Session==0)],
           data_demeaned[(data_demeaned.Study=='evBound') & (data_demeaned.Session==1)],
           data_demeaned[(data_demeaned.Study=='menRot') & (data_demeaned.Session==0)],
           data_demeaned[(data_demeaned.Study=='menRot') & (data_demeaned.Session==1)]]

saveNames = ['perms_1000_evBound_Session1', 'perms_1000_evBound_Session2',
             'perms_1000_menRot_Session1', 'perms_1000_menRot_Session2']

for condi, _ in enumerate(np.arange(len(selData))):
    
    #Determine number of subjects
    subs_tmp = np.unique(selData[condi].Subject)
    
    for subi, sub in enumerate(subs_tmp):
        saveNames_tmp = saveNames[condi] + '_Subject_' + sub
        print('Computing SD for study ' + currentStudies[condi] + ' subject ' + sub)
    
        run_Analysis_SD(data=selData[condi][selData[condi].Subject==sub], currentStudy=currentStudies[condi], model='DvM', bins=bins, bin_width=bin_width+5, 
                    collapseSubs='singleSub', dog_fittingSteps=dog_fittingSteps, stats_n_permutations=1000,
                    savename=saveNames_tmp, path_results=path_results, rerun_fit=1, rerun_perms=1, rerun_bootstrapp=1)

'''
====================================================
Moving-average SD analysis for each subject across the two sessions
====================================================
'''   
del selData

currentStudies = ['evBound', 'menRot']

selData = [data_demeaned[(data_demeaned.Study=='evBound')],
           data_demeaned[(data_demeaned.Study=='menRot')]]

saveNames = ['perms_1000_evBound_movAvg', 'perms_1000_menRot_movAvg']

for condi, _ in enumerate(np.arange(len(selData))):
    
    #Determine number of subjects
    subs_tmp = np.unique(selData[condi].Subject)
    
    for subi, sub in enumerate(subs_tmp):
        saveNames_tmp = saveNames[condi] + '_Subject_' + sub
        print('Computing SD for study ' + currentStudies[condi] + ' subject ' + sub)
    
        run_Analysis_SD_movingAvg(data=selData[condi][selData[condi].Subject==sub], currentStudy=currentStudies[condi], model='DvM', bins=bins, bin_width=bin_width+5, 
                    collapseSubs='singleSub', dog_fittingSteps=dog_fittingSteps, stats_n_permutations=1000,
                    savename=saveNames_tmp, path_results=path_results, rerun_fit=1, rerun_perms=1, rerun_bootstrapp=1)

'''
====================================================
Moving-average SD analysis for each subject seperately for both sessions
====================================================
'''   
del selData

currentStudies = ['evBound', 'evBound', 'menRot', 'menRot']

selData = [data_demeaned[(data_demeaned.Study=='evBound') & (data_demeaned.Session==0)],
           data_demeaned[(data_demeaned.Study=='evBound') & (data_demeaned.Session==1)],
           data_demeaned[(data_demeaned.Study=='menRot') & (data_demeaned.Session==0)],
           data_demeaned[(data_demeaned.Study=='menRot') & (data_demeaned.Session==1)]]

saveNames = ['perms_1000_evBound_Session1_movAvg', 'perms_1000_evBound_Session2_movAvg',
             'perms_1000_menRot_Session1_movAvg', 'perms_1000_menRot_Session2_movAvg']

for condi, _ in enumerate(np.arange(len(selData))):
    
    #Determine number of subjects
    subs_tmp = np.unique(selData[condi].Subject)
    
    for subi, sub in enumerate(subs_tmp):
        saveNames_tmp = saveNames[condi] + '_Subject_' + sub
        print('Computing SD for study ' + currentStudies[condi] + ' subject ' + sub)
    
        run_Analysis_SD_movingAvg(data=selData[condi][selData[condi].Subject==sub], currentStudy=currentStudies[condi], model='DvM', bins=bins, bin_width=bin_width, 
                    collapseSubs='singleSub', dog_fittingSteps=dog_fittingSteps, stats_n_permutations=1000,
                    savename=saveNames_tmp, path_results=path_results, rerun_fit=1, rerun_perms=1, rerun_bootstrapp=1)

'''
====================================================
Plot the single-subject SD results in the same figure: both sessions combined
====================================================
'''

#EvBound
del selData

selData = data_demeaned[(data_demeaned.Study=='evBound')]
           
saveNames = ['perms_1000_evBound_movAvg']

#run_plot_SD_singleSubs(data=selData, model='DvM_movAvg', collapseSubs='singleSub', stats_n_permutations=1000,
                           #my_sig='rsquared', savename=saveNames, bin_width=bin_width+5, path_results=path_results)
run_plot_SD_singleSubs(data=selData, model='DvM_movAvg', collapseSubs='singleSub', stats_n_permutations=1000,
                           my_sig='amplitude', savename=saveNames, bin_width=bin_width+5, path_results=path_results)

#MenRot
del selData

selData = data_demeaned[(data_demeaned.Study=='menRot')]
           
saveNames = ['perms_1000_menRot_movAvg']

run_plot_SD_singleSubs(data=selData, model='DvM_movAvg', collapseSubs='singleSub', stats_n_permutations=1000,
                           my_sig='amplitude', savename=saveNames, bin_width=bin_width+5, path_results=path_results)

'''
====================================================
Assess the variability of the effect for both studies combined
====================================================
'''

currentStudy = ['evBound', 'menRot']

saveNames = ['perms_1000_evBound_movAvg', 'perms_1000_menRot_movAvg']

run_plot_SD_VariabilityInterSub(currentStudies=currentStudy, model='DvM_movAvg', collapseSubs='singleSub', stats_n_permutations=1000,
                           my_sig='amplitude', savename=saveNames, bin_width=bin_width+5, path_results=path_results)


'''
====================================================
Plot the single-subject SD results in the same figure
====================================================
'''

#EvBound
del selData

selData = data_demeaned[(data_demeaned.Study=='evBound')]
           
saveNames = ['perms_1000_evBound_Session1_movAvg', 'perms_1000_evBound_Session2_movAvg']

run_plot_SD_singleSubs(data=selData, model='DvM_movAvg', collapseSubs='singleSub', stats_n_permutations=1000,
                           my_sig='amplitude', savename=saveNames, bin_width=bin_width+10, path_results=path_results)

#MenRot
del selData

selData = data_demeaned[(data_demeaned.Study=='menRot')]
           
saveNames = ['perms_1000_menRot_Session1_movAvg', 'perms_1000_menRot_Session2_movAvg']

run_plot_SD_singleSubs(data=selData, model='DvM_movAvg', collapseSubs='singleSub', stats_n_permutations=1000,
                           my_sig='amplitude', savename=saveNames, bin_width=bin_width+10, path_results=path_results)

'''
====================================================
Assess the correlation between the two sessions
====================================================
'''

currentStudy = ['evBound', 'menRot']

saveNames = ['perms_1000_evBound_movAvg', 'perms_1000_menRot_movAvg']

run_plot_SD_ConsistencyIntraSub(currentStudies=currentStudy, model='DvM_movAvg', collapseSubs='singleSub', stats_n_permutations=1000,
                           my_sig='amplitude', savename=saveNames, bin_width=bin_width+5, path_results=path_results)

'''
====================================================
Model-free SD analysis across both sessions as well as
seperately for the 2 sessions
====================================================
'''   
del selData

currentStudies = ['evBound', 'menRot']

selData = [data_demeaned[(data_demeaned.Study=='evBound')],
           data_demeaned[(data_demeaned.Study=='menRot')]]

if bounds_modelFree_menRot[0] == -86:
    saveNames = ['modelfree_subjSpec_perms_1000_evBound',
                 'modelfree_subjSpec_perms_1000_menRot']
elif bounds_modelFree_menRot[0] == -60:
    saveNames = ['modelfree_60_perms_1000_evBound',
                 'modelfree_60_perms_1000_menRot']
else:
    saveNames = ['modelfree_45_perms_1000_evBound',
                 'modelfree_45_perms_1000_menRot']

for condi, _ in enumerate(np.arange(len(selData))):
    
    print('Computing SD for study ' + currentStudies[condi])
    
    #First, we will look at the pooled data (i.e., pretending there is only a single subject)
    run_Analysis_modelfreeSD(data=selData[condi], 
                             stats_n_permutations=1000, rerun_fit=0, rerun_perms=0, 
                             savename=saveNames[condi], path_results=path_results)

'''
====================================================
Assess the correlation between the two sessions: model-free
====================================================
'''

currentStudy = ['evBound', 'menRot']

saveNames = ['perms_1000_evBound_modelFree', 'perms_1000_menRot_modelFree']

run_plot_SD_ConsistencyIntraSub_modelFree(currentStudies=currentStudy, bounds='specific', 
                                          savename=saveNames, path_results=path_results)

run_plot_SD_ConsistencyIntraSub_modelFree(currentStudies=currentStudy, bounds='45', 
                                          savename=saveNames, path_results=path_results)

run_plot_SD_ConsistencyIntraSub_modelFree(currentStudies=currentStudy, bounds='60', 
                                          savename=saveNames, path_results=path_results)













'''
====================================================
Assess correlation between session 1 and session 2
====================================================
'''
del selData

selData = [data_demeaned[(data_demeaned.Study=='evBound') & (data_demeaned.Session==0)],
           data_demeaned[(data_demeaned.Study=='evBound') & (data_demeaned.Session==1)],
           data_demeaned[(data_demeaned.Study=='menRot') & (data_demeaned.Session==0)],
           data_demeaned[(data_demeaned.Study=='menRot') & (data_demeaned.Session==1)]]

saveNames = ['perms_1000_evBound_Session1', 'perms_1000_evBound_Session2',
             'perms_1000_menRot_Session1', 'perms_1000_menRot_Session2']

           

run_plot_SD_singleSubs(data=selData, model='DoG', collapseSubs='singleSub', stats_n_permutations=1000,
                           savename=saveNames, bin_width=bin_width+5, path_results=path_results)