# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 16:16:31 2022

@author: 海上的月光
"""

import pandas as pd


if __name__=="__main__":
    ##### SLNM #####
    status_name = 'sln1'
    data_file = '1369mapc'
    df_iptw = pd.read_excel(f"./data_sln/{data_file}_iptw.xlsx")
    
    mask = df_iptw[status_name]==1
    
    mean_1, std_1 = [], []
    mean_0, std_0 = [], []    
    for cov in ['age0', 'kps']:
        mean_1.append((df_iptw[mask][cov]*df_iptw[mask]['iptw']).mean())
        std_1.append((df_iptw[mask][cov]*df_iptw[mask]['iptw']).std())
    
        mean_0.append((df_iptw[~mask][cov]*df_iptw[~mask]['iptw']).mean())
        std_0.append((df_iptw[~mask][cov]*df_iptw[~mask]['iptw']).std())
        

    covariates = ['gender', 'pcsize5', 'pcloca', 'liverm','peritoneum',\
                 'lungm','bonem','spleenm', 'pain','wl10','ca199500', 'cea10','alb35',\
                 'alt60', 'tb24','alp130','ldh250','wbc10','hb120', 'bsc', 'resected']
    for cov in covariates:
        mm1 = df_iptw[mask][cov] == 1
        mean_1.append((mm1*df_iptw[mask]['iptw'])[mm1].sum())
        ratio = (mm1*df_iptw[mask]['iptw'])[mm1].sum()/df_iptw[mask]['iptw'].sum()
        std_1.append(ratio)
        
        mm0 = df_iptw[~mask][cov] == 1
        mean_0.append((mm0*df_iptw[~mask]['iptw'])[mm0].sum())
        ratio = (mm0*df_iptw[~mask]['iptw'])[mm0].sum()/df_iptw[~mask]['iptw'].sum()
        std_0.append(ratio)
     
        
    cov = 'sctr'  
    mm1 = df_iptw[mask][cov] >= 1
    mean_1.append((mm1*df_iptw[mask]['iptw'])[mm1].sum())
    ratio = (mm1*df_iptw[mask]['iptw'])[mm1].sum()/df_iptw[mask]['iptw'].sum()
    std_1.append(ratio)
    
    mm0 = df_iptw[~mask][cov] >= 1
    mean_0.append((mm0*df_iptw[~mask]['iptw'])[mm0].sum())
    ratio = (mm0*df_iptw[~mask]['iptw'])[mm0].sum()/df_iptw[~mask]['iptw'].sum()
    std_0.append(ratio)
            
    
    cov = 'sctr'    
    for j in [1,2,3,4]:
        mm1 = df_iptw[mask][cov] == j
        mean_1.append((mm1*df_iptw[mask]['iptw'])[mm1].sum())
        ratio = (mm1*df_iptw[mask]['iptw'])[mm1].sum()/df_iptw[mask]['iptw'][df_iptw[mask][cov]!=0].sum()
        std_1.append(ratio)
        
        mm0 = df_iptw[~mask][cov] == j
        mean_0.append((mm0*df_iptw[~mask]['iptw'])[mm0].sum())
        ratio = (mm0*df_iptw[~mask]['iptw'])[mm0].sum()/df_iptw[~mask]['iptw'][df_iptw[~mask][cov]!=0].sum()
        std_0.append(ratio)
    
    
    print('ITPW With SLNM:')
    for k1, k2 in zip(mean_1, std_1):
        print(f'{k1:.1f}({k2:.3f})')
    
    print('ITPW Without SLNM:')
    for k1, k2 in zip(mean_0, std_0):
        print(f'{k1:.1f}({k2:.3f})')