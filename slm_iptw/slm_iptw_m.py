# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 16:50:05 2022

@author: 海上的月光
"""

import pandas as pd
from lifelines import CoxPHFitter
from psm import *

if __name__=="__main__":
    data_file = '1369mapc'
    
    df = pd.read_excel(f"./data_sln/{data_file}.xlsx")
    covariates = ['month', 'peritoneum', 'gender', 'age60', 'kps90', 'liverm', 'lungm', \
                  'ca199500', 'cea10', 'tb24', 'alt60', 'alb35', 'ggt54', 'alp130', \
                  'ldh250', 'pcsize5', 'wbc10', 'hb120', 'wl10', 'resected', 'sct', 'sln1', 'status']
    cph_org = CoxPHFitter()
    cph_org.fit(df[covariates].dropna(), 'month', 'status')
    cph_org.print_summary()
    
    
    
    
    
    df_iptw = pd.read_excel(f"./data_sln/{data_file}_iptw.xlsx")
    
    covariates = ['month', 'peritoneum', 'gender', 'age60', 'kps90', 'liverm', 'lungm', \
                  'ca199500', 'cea10', 'tb24', 'alt60', 'alb35', 'ggt54', 'alp130', \
                  'ldh250', 'pcsize5', 'wbc10', 'hb120', 'wl10', 'resected', 'sct', 'sln1', 'iptw', 'status']
        
    cph = CoxPHFitter()
    # cph.fit(df_iptw[covariates].dropna(), 'month', 'status')
    cph.fit(df_iptw[covariates].dropna(), 'month', 'status', weights_col='iptw', robust=True)
    cph.print_summary()
    
    
    