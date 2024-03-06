#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:56:10 2024

@author: dliu
"""

from psm import *
import pandas as pd
import numpy as np


data_file = '1257pdac'
df = pd.read_stata('{}.dta'.format(data_file))


status_name = 'tace'
feature_name = ['peritoneum', 'kps', 'pcsize5', 'ca199500', 'ldh250', 'concur', "sctr","alt60",'lnm']
                #"alb35", ]
                

if __name__=="__main__":
    ######################
    # lnm, ca199500
    T_mask = df[status_name]==1
    T_num = T_mask.sum()
    Tgroup = df[T_mask]
    Cgroup = df[~T_mask]
    
    # delet_num = 3
    # delet_idx = np.random.choice(np.arange(T_num)[Tgroup['lnm']==0],delet_num,replace=False) #np.array([5, 68, 67])
    # delet_idx = np.array([5, 68, 67])
    # T_idx = np.delete(np.arange(Tgroup.shape[0]), delet_idx)
    # Tgroup = Tgroup.iloc[T_idx]
    
    df = pd.concat([Tgroup, Cgroup], ignore_index=True)
    
    
    ### get data: feature(X) and status(Y) ###
    feature, status = get_data(df, feature_name, status_name)
    
    ### get propensity score
    propensity_score = pre_logit(feature, status)
    
    ### match the treatment and control    
    results = T_C_nearest_match(status, propensity_score, threshold=0.002, crop=T_num*4)
    
    num_T = status.sum()
    results_T = results[:num_T]
    results_C = results[num_T:]
    
    
    
    
    selected = df.iloc[results,:]
            
    mask_T = selected[f'{status_name}']==1
    selected_T = selected[mask_T]
    selected_C = selected[~mask_T]
    
            
    
    ## get t test
    from scipy import stats
    tStat, pvalue_age = stats.ttest_ind(selected_T['age0'], selected_C['age0'], equal_var = False) #run independent sample T-Test
    if pvalue_age<0.2:
        print("P-Value:{0}".format(pvalue_age)) #print the P-Value and the T-Statistic
    tStat, pvalue_pcsize = stats.ttest_ind(selected_T['pcsize'], selected_C['pcsize'], equal_var = False) #run independent sample T-Test
    if pvalue_pcsize<0.2:
        print("P-Value:{0}".format(pvalue_pcsize)) #print the P-Value and the T-Statistic
        
            
    ## get ANOVA table as R like output
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    
    anova_list = ['gender', 'kps80', 'pcloca', 'age65', 'peritoneum', 'wbc',\
                  'pain', 'ca199500', 'cea10', 'alb35', 'tb', 'ldh', 'ca24220',\
                  'sctr', 'liverm']
                  # 'exhep',  'lnm', 'lungm', 'bonem', 'bg6',]
    pvalue = {}
    pvalue_problem = {}
    for cov in anova_list:
        # Ordinary Least Squares (OLS) model
        model = ols(f'{status_name} ~ {cov}', data=selected).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        _pvalue = anova_table['PR(>F)'][cov]
        pvalue[cov] = _pvalue
    
        if _pvalue<0.1:
            pvalue_problem[cov] = _pvalue
            
    print(pvalue_problem)
    
    if len(pvalue_problem)==0 and pvalue_pcsize>0.37 and pvalue['gender']>0.3 and pvalue['age65']>0.2: #pvalue_age>0.23:
    # if len(pvalue_problem)==0 and pvalue_pcsize>0.37 and pvalue['gender']>0.24 and pvalue_age>0.23:
        flag = False
        
        

    
    ### get selected data(from original data)
    df = pd.read_stata(f'{data_file}.dta')
    df = df.drop(['_st', '_d', '_t', '_t0'], axis=1)
    selected = df.iloc[results,:]
    ### save selected data
    save_data(selected, f"{data_file}_selected.xlsx")
    # pd.DataFrame({'seed':[seed]}).to_csv('./data_sln/seed.csv')
    
    