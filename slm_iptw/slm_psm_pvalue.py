# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 16:13:45 2022

@author: 海上的月光
"""
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

if __name__=="__main__":

    ###############################################################
    ########## ANOVA/RankSum/T-test for PSM matched data ##########
    ###############################################################
    status_name = 'sln1'
    data_file = '1369mapc'
    df_selected = pd.read_excel(f"./data_sln/{data_file}_selected.xlsx")

    t_list = ['age0', 'kps']
    anova_list = ['gender', 'pcsize5', 'pcloca','liverm','peritoneum',\
                  'lungm','bonem','spleenm', 'pain','wl10','ca199500', 'cea10','alb35',\
                  'alt60', 'tb24','alp130','ldh250','wbc10','hb120', 'bsc', 'resected', 'sctr' ]
        
    covariate_list = ['age0', 'kps', 'gender', 'pcsize5', 'pcloca','liverm','peritoneum',\
                      'lungm','bonem','spleenm', 'pain','wl10','ca199500', 'cea10','alb35',\
                      'alt60', 'tb24','alp130','ldh250','wbc10','hb120', 'bsc', 'resected', 'sctr' ]
    pvalue_psm = {}   
    for cov in covariate_list:
        ## t-test
        if cov in t_list:
            df_ = df_selected[[status_name, cov]].dropna()    
            tStat, _pvalue = stats.ttest_ind(df_[cov][df_[status_name] == 0], df_[cov][df_[status_name] == 1], equal_var = False) #run independent sample T-Test
            pvalue_psm[cov] = _pvalue
        ## ANOVA
        elif cov in anova_list:
            df_ = df_selected[[status_name,cov]].dropna()
            # Ordinary Least Squares (OLS) model
            model = ols(f'{status_name} ~ {cov}', data=df_).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            _pvalue = anova_table['PR(>F)'][cov]
            pvalue_psm[cov] = _pvalue
            
    # ##### rank sum test #####
    # cov = 'sctr'
    # df_ = df_selected[[status_name, cov]].dropna()    
    # stat, pvalue_sctr = stats.ranksums(df_[cov][df_[status_name] == 0], df_[cov][df_[status_name] == 1])
    # pvalue_psm[cov] = pvalue_sctr

    # ##### pvalue #####
    for i in list(pvalue_psm.values()):
        print(f'{i:.2f}')
    
    # np.savetxt('./data_sln/pvalue_psm.txt', np.array(list(pvalue_psm.items())))