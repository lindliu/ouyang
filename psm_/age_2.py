# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 09:54:30 2022

@author: 海上的月光
"""

from psm import *
import pandas as pd
import numpy as np

data_file = '467bsc'
bsc_origin = pd.read_stata('{}.dta'.format(data_file))
sct_selected = pd.read_excel('592pclm-sct'+'_selected.xlsx')

mask = sct_selected['age0']>=65
sct_larger_65 = sct_selected[mask]
Y_T = np.ones(sct_larger_65.shape[0], np.int32)

mask = bsc_origin['age0']>=65
bsc_larger_65 = bsc_origin[mask]
Y_C = np.zeros(bsc_larger_65.shape[0], np.int32)

Y = np.r_[Y_T, Y_C]
df = pd.concat([sct_larger_65, bsc_larger_65])

df = df.assign(Y=Y)


if __name__=="__main__":
    # df = pd.read_stata('{}.dta'.format(data_file))

    ### get data: feature(X) and status(Y) ###
    status_name = 'Y'
    feature_name = ['peritoneum', 'kps', 'pcsize5', 'ca199500', 'ldh250',  'wl10',\
                    "concur","tb24","cea10",'alt60'] ##????
    feature, status = get_data(df, feature_name, status_name)
    
    ### get propensity score
    propensity_score = pre_logit(feature, status)
    
    ### match the treatment and control    
    results = T_C_nearest_match(status, propensity_score, threshold=0.01)
    
    ### save selected data
    selected = df.iloc[results.flatten(),:]
    selected = selected.drop(status_name, axis=1)
    save_data(selected, "{}.xlsx".format(data_file+'_selected'))
