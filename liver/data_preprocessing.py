#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 19:49:39 2023

@author: dliu
"""

import pandas as pd
import matplotlib.pyplot as plt
from pynomo.nomographer import *

path = '../data/胰腺癌术后肝转移191例.xlsx'
data = pd.read_excel(path)
# print(data.columns)

data.rename(columns=dict(data.iloc[0,:]), inplace=True)
data = data[1:]

### target
# data['tas'] #'胰腺癌术后发生肝转移间隔时间(月)'

variables = ['tas', 'status', 'sex', 'sctr', 'stra', 'age65']
train_data = data[variables]

from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(train_data, duration_col='tas', event_col='status')
cph.print_summary()  # access the individual results using cph.summary



# from pynomo.nomographer import *
 
# N_params={
#         'u_min':1.0,
#         'u_max':10.0,
#         'function':lambda u:u,
#         'title':'u',
#         }
 
# block_params={
#               'block_type':'type_8',
#               'f_params':N_params,
#               'width':5.0,
#               'height':15.0,
#                       }

# main_params={
#               'filename':'ex_axes_1.pdf',
#               'paper_height':15.0,
#               'paper_width':5.0,
#               'block_params':[block_params],
#               'transformations':[('scale paper',)]
#               }
 
# Nomographer(main_params)