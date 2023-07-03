#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:10:24 2022

@author: darinka

Purpose: All functions necessary to load the data.
"""

def loadData(studies):
    """
    :param studies: list of all studies to load

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    from pathlib import Path
    
    from JoV_Analysis_cfg import path_rawDat_funcStates, path_rawDat_evBound, path_rawDat_menRot
    
    data = []
    
    for studi, study in enumerate(studies):

        print('Loading data for study ' + study)
        
        ### Load group data ###
        if study == 'funcStates':
            data_tmp = pd.read_pickle(path_rawDat_funcStates / 'Group_data_excl.pkl')
            
            #Drop some columns
            data_tmp = data_tmp.drop(columns=['Templ_index', 'Short_Break', 'Long_Break', 'miniblock_start', 
                                              'miniblock_end', 'memory_probed', 'mblock_no', 'Unnamed: 0.1', 
                                              'accuracyJud'])
            
            #Rename same columns to harmonize
            data_tmp = data_tmp.rename(columns={'MemItem_bin': 'Mem_angle_bin', 'MemItem_angle': 'Mem_angle', 
                                     'MemItem_deltangle': 'Delta_angle', 'MemItem_Repr': 'Resp_angle', 
                                     'MemItem_RT': 'Resp_rt', 'incl_accuracyJud': 'incl_JoV'})
        elif study == 'evBound':
            data_tmp = pd.read_pickle(path_rawDat_evBound / 'Group_data.pkl')
            
            #Add subject inclusion column
            incl_JoV = np.ones(len(data_tmp))
            data_tmp.insert(19, 'incl_JoV', incl_JoV)
            
        elif study == 'menRot':
            data_tmp = pd.read_pickle(path_rawDat_menRot / 'Group_data_excl.pkl')
            
            #Drop data from pilot task
            index=np.squeeze(np.where(data_tmp.TaskVersion==0))
            
            data_tmp = data_tmp[11880 :]
            
            #Drop some columns
            data_tmp = data_tmp.drop(columns=['TaskVersion', 'incl_dprime', 'incl_70', 'incl_60', 'incl_chance', 'dprime_2AFC', 
                                              'accuracyRot', 'subjExp'])
            
            #Rename same columns to harmonize
            data_tmp = data_tmp.rename(columns={'incl_65': 'incl_JoV'})
        
        ### Insert study column ###
        data_tmp.insert(0, 'Study', study)
        
        ### Append ###
        data.append(data_tmp)
        
        ### Print columns ###
        print(data_tmp.columns)
    
    #Concatenate
    data_df = pd.concat([data[0], data[1],data[2]], axis=0, ignore_index=True)

    return data_df