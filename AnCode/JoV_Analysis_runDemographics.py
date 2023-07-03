#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:53:14 2022

@author: darinka

Purpose: Get demographics
"""

def runDemographics(data):
    
    '''
    param data: group data to compute demographics for
    
    '''
    
    import numpy as np
    
    from itertools import compress
    
    from JoV_Analysis_cfg import (studies, ListAge_funcStates, ListGender_funcStates, ListAge_evBound,
                                  ListGender_evBound, ListAge_menRot, ListGender_menRot, ListSessionDistance_funcStates,
                                  ListSessionDistance_evBound, ListSessionDistance_menRot)
    
    
    ListAge = []
    ListGender = []
    ListSession = []
    
    for studi, study in enumerate(studies):
        
        subs = np.unique(data.Subject[data.Study==study])
        
        if study == 'funcStates':
            subs_incl = np.unique(data.Subject[(data.Study==study) & (data.incl_JoV==1)]) 
            age = ListAge_funcStates
            gender = ListGender_funcStates
            session = ListSessionDistance_funcStates
        elif study == 'evBound':
            subs_incl = np.unique(data.Subject[(data.Study==study) & (data.incl_JoV==1)]) 
            age = ListAge_evBound
            gender = ListGender_evBound
            session = ListSessionDistance_evBound
        elif study == 'menRot':
            subs_incl = np.unique(data.Subject[(data.Study==study) & (data.incl_JoV==1)]) 
            age = ListAge_menRot
            gender = ListGender_menRot
            session = ListSessionDistance_menRot
            
        subs_logical = [1 if subject in subs_incl else 0 for subject in subs]
        
        ListAge_tmp = list(compress(age, subs_logical))
        ListGender_tmp = list(compress(gender, subs_logical))
        ListSession_tmp = list(compress(session, subs_logical))
        
        ListAge_tmp = np.array(ListAge_tmp)
        ListGender_tmp = np.array(ListGender_tmp)
        ListSession_tmp = np.array(ListSession_tmp)
        
        #Print relevant variables
        print('Mean age for study ' + study + ' was: ' + str(np.mean(ListAge_tmp)))
        print('Std age for study ' + study + ' was: ' + str(np.std(ListAge_tmp)))
        
        print('Number of women for study ' + study + ' was: ' + str(np.sum(ListGender_tmp)))
        
        #Append
        ListAge.append(ListAge_tmp)
        ListGender.append(ListGender_tmp)
        ListSession.append(ListSession_tmp)
    
    #Overall demographics
    print('Overall mean age was: ' + str(np.mean(np.concatenate(ListAge))))
    print('Overall std age was: ' + str(np.std(np.concatenate(ListAge))))
    
    print('Overall number of women was: ' + str(np.sum(np.concatenate(ListGender))))
    
    print('Overall range of sessions was: ' + str(np.min(np.concatenate(ListSession))) + ' ' +
          str(np.max(np.concatenate(ListSession))))
    print('Overall mean session distance was: ' + str(np.mean(np.concatenate(ListSession))))
    print('Overall std session distance was: ' + str(np.std(np.concatenate(ListSession))))
    
    return