# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 10:58:18 2022

@author: 海上的月光
"""

import pandas as pd
from psm import *
    

if __name__=="__main__":
    ### get data: feature(X) and status(Y) ###
    data_file = '592pclm-sct'
    df = pd.read_stata('{}.dta'.format(data_file))
    status_name = 'age65'
    feature_name = ['concur', 'kps', 'wl10', 'ca199500', 'ldh250', 'cea10']
    feature, status = get_data(df, feature_name, status_name)
    
    ### get propensity score
    propensity_score = pre_logit(feature, status)
    
    ### match the treatment and control
    results = T_C_match(status, propensity_score, k=3, threshold=0.05, round_=2)
    
    ### save selected data
    selected = df.iloc[results.flatten(),:]
    save_data(selected, "{}.xlsx".format(data_file+'_selected'))
