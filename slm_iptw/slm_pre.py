# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:15:57 2022

@author: 海上的月光
"""

import glob
import os
from psm import *
import pandas as pd
import numpy as np


if __name__=="__main__":
    data_pathes = glob.glob('./data_sln/*.dta')
    
    df1244 = pd.read_stata(data_pathes[0])
    df1369= pd.read_stata(data_pathes[1])
    df661 = pd.read_stata(data_pathes[2])
    
    df1244 = df1244.drop(['_st', '_d', '_t', '_t0'], axis=1)
    df1369 = df1369.drop(['_st', '_d', '_t', '_t0'], axis=1)
    df661 = df661.drop(['_st', '_d', '_t', '_t0'], axis=1)
    
    df_insert = pd.DataFrame({'sctno':[0]*df1244.shape[0], 'sctr':[0]*df1244.shape[0]})
    df1244_new = pd.concat([df1244, df_insert], axis=1)
    for hos in df1244_new['hospital']:
        index_ = df661.index[df661['hospital'] == hos].tolist()
        if len(index_)>=1:
            df1244_new.loc[df1244_new['hospital']==hos,'sctno'] = df661.iloc[index_[0]]['sctno']
            df1244_new.loc[df1244_new['hospital']==hos,'sctr'] = df661.iloc[index_[0]]['sctr']
    
    save_data(df1244_new, "./data_sln/1244_new.xlsx")

    # df_insert = pd.DataFrame({'sctno':[0]*df1369.shape[0], 'sctr':[0]*df1369.shape[0]})
    # df1369_new = pd.concat([df1369, df_insert], axis=1)
    # for hos in df1369_new['hospital']:
    #     index_ = df661.index[df661['hospital'] == hos].tolist()
    #     if len(index_)>=1:            
    #         df1369_new.loc[df1369_new['hospital']==hos,'sctno'] = df661.iloc[index_[0]]['sctno']
    #         df1369_new.loc[df1369_new['hospital']==hos,'sctr'] = df661.iloc[index_[0]]['sctr']

    # save_data(df1369_new, "./data_sln/1369_new.xlsx")
    
    
    
    df_insert = pd.DataFrame({'sln':[0]*df661.shape[0], 'sln1':[0]*df661.shape[0]})
    df661_new = pd.concat([df661, df_insert], axis=1)
    for hos in df661_new['hospital']:
        index_ = df1244.index[df1244['hospital'] == hos].tolist()
        if len(index_)>=1:            
            df661_new.loc[df661_new['hospital']==hos,'sln'] = df1244.iloc[index_[0]]['sln']
            df661_new.loc[df661_new['hospital']==hos,'sln1'] = df1244.iloc[index_[0]]['sln1']

    save_data(df661_new, "./data_sln/661_new.xlsx")
    