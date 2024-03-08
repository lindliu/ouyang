# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 11:13:47 2022

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

    
    # ##### data preparation #####
    # data_file = '1369mapc'
    # df_org = pd.read_stata(f'./data_sln/{data_file}.dta')
    # df_org = df_org.drop(['_st', '_d', '_t', '_t0'], axis=1)
    
    
    # bsc = np.zeros([df_org.shape[0]])
    # mm = np.logical_and.reduce((df_org['surgery']==0, df_org['sct']==0, df_org['radiation']==0, df_org['tace']==0))
    # bsc[mm] = 1
    # df_bsc = pd.DataFrame(bsc, columns=['bsc'])
    
    # df_org = pd.concat([df_org, df_bsc], axis=1)
    
    # df_org.loc[df_org['pcloca']>1, 'pcloca']=2
    # df_org.loc[df_org['pcloca']<=1, 'pcloca']=1
    # save_data(df_org, f"./data_sln/{data_file}.xlsx")
    # #############################
    
    status_name = 'sln1'
    data_file = '1369mapc'
    
    df = pd.read_excel(f"./data_sln/{data_file}.xlsx")
    df_selected = pd.read_excel(f"./data_sln/{data_file}_selected.xlsx")
    df_iptw = pd.read_excel(f"./data_sln/{data_file}_iptw.xlsx")

    

    name_covariat = {'liverm':'Liver metastases', 'peritoneum':'Peritoneal metastases',\
                     'resected':'Resection of PDAC', 'gender':'Sex', 'age65':'Age', \
                     'kps':'KPS', 'pcsize5':'Pancreatic tumor size', 'ca199500':'CA 19-9', \
                     'sct':'SCT', 'ldh250':'LDH', 'alb0':'Albumin', 'wbc10':'WBC'
                     }
    SMD_orig, SMD_psm, SMD_iptw = [], [], []
    for covariate in name_covariat.keys():
        SMD_orig.append(get_SMD(df, status_name, covariate))
        SMD_psm.append(get_SMD(df_selected, status_name, covariate))
        SMD_iptw.append(get_SMD_weight(df_iptw, status_name, covariate, df_iptw['iptw']))

    SMD_orig = np.array(SMD_orig)
    SMD_psm = np.array(SMD_psm)
    SMD_iptw = np.array(SMD_iptw)

    ind = np.argsort(SMD_orig)
    

        
    plt.rcParams.update({'font.size':18})
    fig = plt.figure(figsize=[12,12])
    plt.plot(SMD_orig[ind], np.arange(len(name_covariat)), color='m', linestyle='-', marker='o',linewidth=2, markersize=10, label='Umatched')
    plt.plot(SMD_psm[ind], np.arange(len(name_covariat)), color='darkturquoise', linestyle='-.',marker='^', linewidth=2, markersize=10, label='PSM matched')
    plt.plot(SMD_iptw[ind], np.arange(len(name_covariat)), color='orange',linestyle='--',marker='*', linewidth=2, markersize=10, label='Weighted')

    plt.yticks(np.arange(len(name_covariat)), [name_covariat[cov] for cov in np.array(list(name_covariat.keys()))[ind]])  
    plt.xlim(0,.33)
    plt.grid(color='b',
                  linestyle='--',
                  linewidth=1,
                  alpha=0.2)
    plt.legend()
    plt.axvline(x=.1, color='red', linestyle='--')
    plt.title('SMD')
    
    plt.savefig('./data_sln/SMD.tif', dpi=300, bbox_inches='tight')
    