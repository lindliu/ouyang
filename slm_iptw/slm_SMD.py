# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 15:54:29 2022

@author: 海上的月光
"""
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from psm import *


def get_SMD(data, status, covariate):
    """standardized mean difference"""
    
    # data[covariate] = data[covariate].fillna(data[covariate].median())
    
    data_T = data[data[status]==1]
    data_C = data[data[status]==0]
    
    T_mean = data_T[covariate].mean()
    C_mean = data_C[covariate].mean()
    print(f'{covariate}: {T_mean:.2f}, {C_mean:.2f}')
    T_std = data_T[covariate].std()
    C_std = data_C[covariate].std()
    
    SMD = (T_mean-C_mean)/((T_std**2+C_std**2)/2)**.5
    SMD = round(abs(SMD), 2)
    
    return SMD


def get_SMD_weight(data, status, covariate, weight):
    """
    weighted standardized mean difference
    https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy 
    """

    data[covariate] = data[covariate].fillna(data[covariate].median())
    
    data_T = data[data[status]==1]
    data_C = data[data[status]==0]
    
    weight_T = weight[data[status]==1]
    weight_C = weight[data[status]==0]
    
    T_mean = np.average(data_T[covariate], weights=weight_T) 
    C_mean = np.average(data_C[covariate], weights=weight_C) 
    # print(f'{covariate}: {T_mean:.2f}, {C_mean:.2f}')
    # T_std = weight_T.sum()/(weight_T.sum()**2 - (weight_T**2).sum()) * (weight_T*(data_T[covariate]-T_mean)**2).sum()
    # C_std = weight_C.sum()/(weight_C.sum()**2 - (weight_C**2).sum()) * (weight_C*(data_C[covariate]-C_mean)**2).sum()
    T_std = np.average((data_T[covariate]-T_mean)**2, weights=weight_T) ** .5
    C_std = np.average((data_C[covariate]-C_mean)**2, weights=weight_C) ** .5
    
    # T_std = data_T[covariate].std()
    # C_std = data_C[covariate].std()
    
    SMD = (T_mean-C_mean)/((T_std**2+C_std**2)/2)**.5
    SMD = round(abs(SMD), 2)
    
    return SMD

if __name__=="__main__":
    
    status_name = 'sln1'
    data_file = '1369mapc'

    df = pd.read_excel(f"./data_sln/{data_file}.xlsx")
    df_selected = pd.read_excel(f"./data_sln/{data_file}_selected.xlsx")
    df_iptw = pd.read_excel(f"./data_sln/{data_file}_iptw.xlsx")
    

    #############################################
    ########## SMD to original data #############
    #############################################
    covariate_list = ['age0', 'kps', 'gender', 'pcsize5', 'pcloca','liverm','peritoneum',\
                      'lungm','bonem','spleenm', 'pain','wl10','ca199500', 'cea10','alb35',\
                      'alt60', 'tb24','alp130','ldh250','wbc10','hb120', 'bsc', 'resected', 'sctr' ]

    SMD_orig, SMD_psm, SMD_iptw = [], [], []
    for covariate in covariate_list:
        SMD_orig.append(get_SMD(df, status_name, covariate))
        SMD_psm.append(get_SMD(df_selected, status_name, covariate))
        SMD_iptw.append(get_SMD_weight(df_iptw, status_name, covariate, df_iptw['iptw']))
        
        
        
        