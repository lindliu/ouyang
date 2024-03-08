# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 16:15:07 2022

@author: 海上的月光
"""

import pandas as pd
from statsmodels.stats.weightstats import ttest_ind
import scipy


def weighted_anova(df, status_name, cov, weight):
    """
    https://center-based-statistics.com/html/weightedAnovaOne.html
    """
    df_ = df[[status_name, cov]]
    mask = df_[status_name]==1
    
    x_t_mean = (df_[mask][cov]*weight[mask]).sum() / weight[mask].sum()
    x_c_mean = (df_[~mask][cov]*weight[~mask]).sum() / weight[~mask].sum()
    X_mean = (df_[cov]*weight).sum()/weight.sum()
    
    SS_total = (weight*(df_[cov]-X_mean)**2).sum()

    SS_within = (weight[mask]*(df_[mask][cov]-x_t_mean)**2).sum() + \
                (weight[~mask]*(df_[~mask][cov]-x_c_mean)**2).sum()
    SS_between = weight[mask].sum() * (x_t_mean-X_mean)**2 + \
                  weight[~mask].sum() * (x_c_mean-X_mean)**2
    
    
    n = df_.shape[0]
    k = 2
    df_total = n-1
    df_between = k-1
    df_within = n-k

    MS_total = SS_total/df_total
    MS_between = SS_between/df_between
    MS_within = SS_within/df_within
    
    F = MS_between/MS_within
    
    return F, df_between, df_within

if __name__=="__main__":
    ########################################################
    ########## ANOVA/RankSum/T-test for IPTW data ##########
    ########################################################
    
    status_name = 'sln1'
    data_file = '1369mapc'
    df_iptw = pd.read_excel(f"./data_sln/{data_file}_iptw.xlsx")

    t_list = ['age0', 'kps']
    anova_list = ['gender', 'pcsize5', 'pcloca','liverm','peritoneum',\
                  'lungm','bonem','spleenm', 'pain','wl10','ca199500', 'cea10','alb35',\
                  'alt60', 'tb24','alp130','ldh250','wbc10','hb120', 'bsc', 'resected', 'sctr' ]
        
    covariate_list = ['age0', 'kps', 'gender', 'pcsize5', 'pcloca','liverm','peritoneum',\
                      'lungm','bonem','spleenm', 'pain','wl10','ca199500', 'cea10','alb35',\
                      'alt60', 'tb24','alp130','ldh250','wbc10','hb120', 'bsc', 'resected', 'sctr' ]
    pvalue_weighted = {}   
    for cov in covariate_list:
        ##### Weighted t-test #####
        if cov in t_list:
            df_ = df_iptw[[status_name, cov, 'iptw']].dropna()
            mask_T = df_[status_name]==1
            x, y = df_[cov][~mask_T], df_[cov][mask_T]
            weights_x, weights_y = df_[~mask_T]['iptw'], df_[mask_T]['iptw']
            # weights_x, weights_y = weight[~mask_T], weight[mask_T]
            _, _pvalue, _ = ttest_ind(x, y, usevar='unequal', weights=(weights_x, weights_y))
            
            pvalue_weighted[cov] = _pvalue
        
        ##### Weighted ANOVA #####
        elif cov in anova_list:
            # Ordinary Least Squares (OLS) model
            df_ = df_iptw[[status_name,cov,'iptw']].dropna()
            F, df1, df2 = weighted_anova(df_, status_name, cov, df_['iptw'])
            _pvalue = 1-scipy.stats.f.cdf(F, df1, df2)
            
            pvalue_weighted[cov] = _pvalue
            
            
    # ##### Weighted rank sum test #####
    # cov = 'sctr'
    # df_ = df_iptw[[status_name, cov]].dropna()
    # mask_T = df_[status_name]==1
    # stat, pvalue_sctr = stats.ranksums(df_[cov][~mask_T]*weight[~mask_T], df_[cov][mask_T]*weight[mask_T])
    # pvalue_weighted[cov] = pvalue_sctr
    
    # ##### pvalue #####
    for i in list(pvalue_weighted.values()):
        print(f'{i:.2f}')