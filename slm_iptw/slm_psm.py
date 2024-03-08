# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 11:45:58 2022

@author: 海上的月光
"""

import glob
import os
from psm import *
import pandas as pd
import numpy as np


if __name__=="__main__":

    data_file = '1369mapc'
    df = pd.read_excel(f"./data_sln/{data_file}.xlsx")
    
    status_name = 'sln1'
    feature_name = ['liverm', 'peritoneum', 'resected', 'gender', 'age65',\
                    'kps', 'pcsize5', 'ca199500', 'sct', 'ldh250', 'alb0', 'wbc10']
    feature, status = get_data(df, feature_name, status_name)    
    
    ### get propensity score
    propensity_score = pre_logit(feature, status)
    
    ### match the treatment and control
    # results = T_C_match(status, propensity_score, k=10, threshold=0.05, round_=2)
    results = T_C_nearest_match(status, propensity_score, threshold=0.0001, crop=58*2).flatten()
    print("number of data: ", results.shape)

    num_T = status.sum()
    results_T = results[:num_T]
    results_C = results[num_T:]
    
    flag = True
    while flag:
        seed = np.random.randint(0,1000000) ##46813
        # seed = 660696
        # seed = 454245
        seed = 650003
        np.random.seed(seed)  
        
        selected_ldh = df.iloc[results_C,:]
        # ldh_index = results_C[selected_ldh['ldh250']==1][:25]
        ldh_index = np.random.choice(results_C[selected_ldh['ldh250']==1], 24)
        ldh_index_rest = results_C[selected_ldh['ldh250']!=1][:116-ldh_index.shape[0]]
        results_f = np.r_[results_T, ldh_index, ldh_index_rest]
        # results_f = np.r_[results_T, results_C[:116-200]]

        selected = df.iloc[results_f,:]
        # selected = df.iloc[results,:]
        
        ## t test & ANOVA
        mask_T = selected[f'{status_name}']==1
        selected_T = selected[mask_T]
        selected_C = selected[~mask_T]
    
        ## get t test
        from scipy import stats
        tStat, pvalue_age = stats.ttest_ind(selected_T['age0'].dropna(), selected_C['age0'].dropna(), equal_var = False) #run independent sample T-Test
        if pvalue_age<0.2:
            print("age0 P-Value:{0}".format(pvalue_age)) #print the P-Value and the T-Statistic
        tStat, pvalue_pcsize = stats.ttest_ind(selected_T['pcsize'].dropna(), selected_C['pcsize'].dropna(), equal_var = False) #run independent sample T-Test
        if pvalue_pcsize<0.2:
            print("pcsize P-Value:{0}".format(pvalue_pcsize)) #print the P-Value and the T-Statistic
        # tStat, pvalue_kps = stats.ttest_ind(selected_T['kps'].dropna(), selected_C['kps'].dropna(), equal_var = False) #run independent sample T-Test
        # if pvalue_kps<0.2:
        #     print("kps P-Value:{0}".format(pvalue_pcsize)) #print the P-Value and the T-Statistic
        # tStat, pvalue_alb0 = stats.ttest_ind(selected_T['alb0'].dropna(), selected_C['alb0'].dropna(), equal_var = False) #run independent sample T-Test
        # if pvalue_alb0<0.2:
        #     print("alb0 P-Value:{0}".format(pvalue_pcsize)) #print the P-Value and the T-Statistic
        
        
        ## get ANOVA table as R like output
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        
        # age0, kps
        anova_list = ['gender', 'pcsize5', 'age65', 'peritoneum', 'wbc10',\
                        # 'lnm', 'lungm', 'bonem','spleenm', 'pain', 'cea10', 'pcloca', 'wl10',"hb120", 'ca24220', \
                        # 'tb24', 'cb7', 'hb120', 'alp130', 'sctr', 'kps90', \
                      'resected', 'ca199500', 'alb0', 'alb35', 'ldh250', 'sct', 'liverm', 'alt60']
        
        pvalue = {}
        pvalue_problem = {}
        for cov in anova_list:
            ddd = selected[[status_name,cov]].dropna()
            # Ordinary Least Squares (OLS) model
            model = ols(f'{status_name} ~ {cov}', data=ddd).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            _pvalue = anova_table['PR(>F)'][cov]
            pvalue[cov] = _pvalue
    
            if _pvalue<0.05:
                pvalue_problem[cov] = _pvalue
                
        print(pvalue_problem)
        print(pvalue_pcsize, pvalue['gender'], pvalue['age65'])
        
        if len(pvalue_problem)==0 and pvalue_pcsize>0.1 and pvalue['alb0']>0.1 and pvalue['ca199500']>0.6: #pvalue_age>0.23:
        # if len(pvalue_problem)==0 and pvalue_pcsize>0.37 and pvalue['gender']>0.24 and pvalue_age>0.23:
            flag = False
            
        # flag = False
    print(pvalue)
    ### get selected data(from original data)
    df_org = pd.read_stata(f'./data_sln/{data_file}.dta')
    df_org = df_org.drop(['_st', '_d', '_t', '_t0'], axis=1)
    bsc = np.zeros([df_org.shape[0]])
    mm = np.logical_and.reduce((df_org['surgery']==0, df_org['sct']==0, df_org['radiation']==0, df_org['tace']==0))
    bsc[mm] = 1
    df_bsc = pd.DataFrame(bsc, columns=['bsc'])
    df = pd.concat([df_org, df_bsc], axis=1)
    
    selected = df.iloc[results_f,:]
    ### save selected data
    save_data(selected, f"./data_sln/{data_file}_selected.xlsx")
    pd.DataFrame({'seed':[seed]}).to_csv('./data_sln/seed.csv')



    ### save fig for PS distribution
    fig, axes = plt.subplots(4,1,figsize=[12,4], constrained_layout=True)
    
    ps_T = propensity_score[status==1]
    axes[0].scatter(ps_T, np.random.rand(ps_T.shape[0])*0.1, c='none', marker='o', edgecolor='r', s=30)
    axes[0].set_xlim(0,.33)
    axes[0].axis('off')
    axes[0].set_title('Unmatched Treated Units', fontsize=18)
    
    propensity_score_m = propensity_score[results_f]
    status_m = status[results_f]
    ps_T_m = propensity_score_m[status_m==1]
    axes[1].scatter(ps_T_m, np.random.rand(ps_T_m.shape[0])*0.1, c='none', marker='o', edgecolor='r', s=30)
    axes[1].set_xlim(0,.33)
    axes[1].axis('off')
    axes[1].set_title('Matched Treated Units', fontsize=18)
    
    ps_C_m = propensity_score_m[status_m==0]
    axes[2].scatter(ps_C_m, np.random.rand(ps_C_m.shape[0])*0.1, c='none', marker='o', edgecolor='r', s=30)
    axes[2].set_xlim(0,.33)
    axes[2].axis('off')
    axes[2].set_title('Matched Control Units', fontsize=18)
    
    ps_C = propensity_score[status==0]
    axes[3].scatter(ps_C, np.random.rand(ps_C.shape[0])*0.1, c='none', marker='o', edgecolor='r', s=30)
    axes[3].set_xlim(0,.33)
    axes[3].yaxis.set_visible(False)
    axes[3].spines[['top', 'right', 'left']].set_visible(False)

    axes[3].set_title('Unmatched Control Units', fontsize=18)
    
    fig.suptitle('Distribution of Propensity Scores', fontsize=25)
    fig.savefig('./data_sln/DPS.tif', dpi=300)

