#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Purpose: All functions necessary to preprocess the data in preparation for the main analysis.
#Author: Darinka Truebutschek
#Date created: 20/06/2022
#Date last modified: 20/06/2022
#Python version: 3.7.1

import numpy as np

### Run preprocessing ###
def run_removeMeanError(data_in):
    """
    :param data_in: pandas dataFrame of a single subject

    """
    import copy
    
    from JoV_Analysis_basicFuncs import getCircularError
    
    #Compute mean Error
    meanError, _, _ = getCircularError(data_in.Resp_error_clean, directionality=1)
    meanError = np.mod(meanError+90, 180)-90
    print(meanError)
    
    #Demean data
    meanError_demeaned = copy.deepcopy(data_in.Resp_error_clean.values)
    meanError_demeaned = meanError_demeaned-meanError

    data_out = data_in
    data_out['Resp_error_demeaned'] = meanError_demeaned

    return data_out

