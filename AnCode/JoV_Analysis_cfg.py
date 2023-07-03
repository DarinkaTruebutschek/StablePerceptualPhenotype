#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Purpose: Config file to set/store all important parameters
#Author: Darinka Truebutschek
#Date created: 22/11/2022
#Date last modified: 22/11/2022
#Python version: 3.7.1

import numpy as np

from pathlib import Path

### Define important subject-specific variable variables ###
studies = ['funcStates', 'evBound', 'menRot']

#Func states
ListSubjects_funcStates = ['008', '009', '010', '011', '012', '013', '014', '015', 
                    '016', '017', '018', '019', '020', '021', '022', '023',
                    '024', '025', '026', '027', '029', '030', '031']
ListAge_funcStates = [24, 24, 26, 27, 27, 25, 26, 22, 24, 24, 26, 30, 25, 33, 29, 23, 24, 24, 21, 24, 29, 50, 20]
ListGender_funcStates = [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0]
ListSessionDistance_funcStates = [1, 2, 7, 4, 2, 3, 4, 1, 7, 4, 4, 1, 5, 3, 5, 3, 2, 3, 7, 3, 9, 1, 7]

#EventBound
ListSubjects_evBound = ['005', '006', '007', '008', '009', '010', '011', '012', '013', '014', 
                '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025']
ListRotationDirection_evBound = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 
                      0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 
                      0] #rotation direction of first session: 0=clockwise, 1=counter-clockwise
ListAge_evBound = [29, 25, 27, 21, 40, 26, 30, 22, 57, 33, 20, 23, 34, 24, 23, 33, 24, 26, 25, 21, 38]
ListGender_evBound = [1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0]
ListSessionDistance_evBound = [1, 1, 1, 1, 5, 28, 7, 2, 2, 7, 5, 2, 2, 7, 3, 2, 2, 3, 2, 2, 2]
ListExperimenter_Session1_evBound = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1]
ListExperimenter_Session2_evBound = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1]

#MenRot
ListSubjects_menRot = ['007', '008', '009', '010',
            '011', '012', '013', '016', '017', '018', '019', '020', 
            '021', '022', '023', '024', '025', '026', '027', '029', '030', 
            '031', '032', '033', '034', '035']
ListRespMapping_menRot = [0, 1, 0, 1, 
                   0, 1, 0, 1, 0, 1, 0, 1, 
                   0, 1, 0, 1, 0, 1, 0, 0, 1, 
                   0, 1, 0, 1, 0]
ListAge_menRot = [25, 28, 24, 39, 21, 32, 40, 26, 35, 
           27, 35, 32, 24, 38, 42, 32, 34, 45, 40, 24, 34, 27, 47, 29, 
           26, 38]    
ListGender_menRot = [1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1,
              1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0] #0=men, 1=women
ListSessionDistance_menRot = [3, 2, 1, 15, 11, 13, 30, 1, 2, 
                       1, 1, 5, 1, 2, 4, 5, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1]#in days


#Computer
computer = 'linux'

### Paths definitions ###
if computer == 'mac':
	path_main = Path('/Users/darinka/Documents/Docs/Postdoc_Lucia/Chrome/SerDep_FuncStates/') #main path to experimental paradigm
elif computer == 'win':
    path_main = Path('P:/2021-0295-CHROME/gitHub_final/Chrome_final/SerDep_FuncStates/')
    #path_main = Path('C:/Users/darinka.truebutschek/Desktop/gitHub_final/Chrome_final/SerDep_EvBound/')
elif computer == 'linux':
    path_main = Path('/home/darinka/Documents/Chrome_final/JoV/')
    path_main_funcStates = Path('/home/darinka/Documents/Chrome_final/SerDep_FuncStates/')
    path_main_evBound = Path('/home/darinka/Documents/Chrome_final/SerDep_EvBound/')
    path_main_menRot = Path('/home/darinka/Documents/Chrome_final/SerDep_EvBound_ProbeDelay/')

path_anScripts = path_main / 'AnCode/' #analysis code
path_rawDat_funcStates = path_main_funcStates / 'Results/Group/' #raw data (should be in BIDS format)
path_rawDat_evBound = path_main_evBound / 'Results_final/Group/' #raw data (should be in BIDS format)
path_rawDat_menRot = path_main_menRot / 'Results/Group/' #raw data (should be in BIDS format)
path_results = path_main / 'Results/'
path_sims = path_main / 'Simulations/' #simulated behavioral data

### Text fonts ###
text_font = 'Arial'
text_size = 12

### Response error computation ###
normalize_respAngles = 1 #normalize all angular responses to fall into the same space? (i.e., 0 - 180 deg)
normalize_rotAngles = 1 ##normalize all counter-clockwise rotations to fall into the same space as clockwise rotations? 

### Data analysis ###
analysis_collapsed = 0 #do analysis collapsed across rotation conditions yes (1) or no (0)

### Outlier removal ###
outlier_collapsed = 0 #remove outliers collapsed across all data (1) or not (0)
outlier_cutoff = 3 #how many standard deviations above/below the circular mean to consider an outlier

### Response error analysis ###
bins = np.linspace(-90, 90, 37)

### Serial dependence analysis ###
bin_width = 20 #how large should the bin in degrees be over which one computes the serial dependence?
dog_fittingSteps = 200
dog_n_permuts = 10 #how many permutations to run to assess significance of DOG 

bounds_modelFree = [-45, 45] #
bounds_modelFree_evBound = [-54, 54]
bounds_modelFree_menRot = [-86, 86]

#bounds_modelFree_evBound = [-45, 45]
#bounds_modelFree_menRot = [-45, 45]

### Stats ###
stats_posthoc = 1 #should paired t-tests be used (for initial evaluation) or results of post-hoc tests (as identified with JASP)
stats_alpha = .05 #p-level at which to consider a test significant
stats_tail = 0 #lower-sided (-1), two-sided (0), or upper-sided (1) test
stats_n_permutations = 1000 #how many permutations to run
stats_n_jobs = -1 #how many processors to use (in parallel)
    
### Simulations ###
n_blocks = 4
n_trials_per_block = 288
n_trials = n_blocks * n_trials_per_block
n_breaks = 4

n_permutations = 100

SD_process = 'WM-response' #perceptual: SD arises at the perceptul level, that is, subsequenct stimuli are already perceived in a biased manner     
SD_win = [-50, 50] #window (in degrees), in which to expect maximal SD

noise_noRot_sd = 75
noise_rot_sd = 15

