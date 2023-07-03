#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:00:32 2023

@author: darinka
"""
import matplotlib.pyplot as plt

import numpy as np

amp = 10000

kappa = 4
test_ang = np.linspace(-90, 90, 181)
test_fig = dvm(np.deg2rad(test_ang), np.deg2rad(amp), kappa, 0)

plt.plot(test_fig)
plt.title(['Amplitude: ' + str(amp) + ', Kappa: ' +  str(kappa)])

plt.plot(test_fig)
plt.title(['Amplitude: ' + str(amp) + ', Kappa: ' +  str(kappa)])