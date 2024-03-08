# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:02:42 2022

@author: 海上的月光
"""



import glob
import os
from psm import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__=="__main__":

    data_file = '1369mapc'
    df = pd.read_excel(f"./data_sln/{data_file}.xlsx")
    
    ##### IPTW #####
    status_name = 'sln1'
    feature_name = ['liverm', 'peritoneum', 'resected', 'gender', 'age65',\
                    'kps', 'pcsize5', 'ca199500', 'sct', 'ldh250', 'alb0', 'wbc10']
    feature, status = get_data(df, feature_name, status_name)
    
    ### get propensity score
    propensity_score = pre_logit(feature, status)
    
    ps_T = propensity_score[status==1]
    ps_C = propensity_score[status==0]
    
    ### get weight
    weight = np.zeros_like(propensity_score)
    
    # ## ATT weight
    # weight[status==1] = 1
    # weight[status==0] = ps_C/(1-ps_C)
    
    # ## ATE weight
    # weight[status==1] = 1./(ps_T)
    # weight[status==0] = 1./(1-ps_C)
    
    ## Stabilized ATE weight
    p = .044
    weight[status==1] = p/(ps_T)
    weight[status==0] = (1-p)/(1-ps_C)
    
    ### delete extrem data
    index_sort = np.argsort(weight)
    delete_num = int(index_sort.shape[0]*0.001)
    delete_idx = np.r_[index_sort[:delete_num],index_sort[-delete_num:]]
    weight = np.delete(weight, delete_idx)
    df_iptw = df.drop(index=delete_idx).reset_index()
    
    df_weight = pd.DataFrame(weight, columns=['iptw'])
    df_iptw = pd.concat([df_iptw, df_weight], axis=1)
    
    n_with = df_iptw[df_iptw[status_name]==1]['iptw'].sum()
    n_without = df_iptw[df_iptw[status_name]==0]['iptw'].sum()
    print(f'n with: {n_with:.2f}, n without: {n_without:.2f}')
    
    save_data(df_iptw, f"./data_sln/{data_file}_iptw.xlsx")
