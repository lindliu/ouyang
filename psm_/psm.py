# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 08:53:52 2022

@author: 海上的月光
"""

import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing  ## scale features
from sklearn.linear_model import LogisticRegression

def get_data(df, X, Y):
    df[X] = df[X].fillna(df[X].median())

    m_x = np.array([i in df.columns for i in X])
    assert np.all(m_x), '{} not in columns'.format(np.array(X)[~m_x])
    assert Y in df.columns, '{} not in columns'.format(Y)

    status = df[Y]
    feature = df[X]
    return feature, status

def pre_logit(feature, status):
    """
    get propensity score by features and status
    """
    ### preprocessing
    scaler = preprocessing.StandardScaler().fit(feature)
    # scaler = preprocessing.Normalizer().fit(feature)
    feature = scaler.transform(feature)
    
    ### fit logit regression model to obtain score
    ps_model = LogisticRegression(C=1e5, max_iter=200).fit(feature, status)
    propensity_score = ps_model.predict_proba(feature)[:,1]
    return propensity_score

def T_C_match(status, propensity_score, k=2, threshold=0.01, round_=2):
    """
    k: 1:k
    threshold: distance for propensity score
    round: iteration times
    """
    mask_C = status==0
    mask_T = status==1
    org_ind = np.arange(status.shape[0])
    
    ### sorted score ###
    T_ps_idx = np.argsort(propensity_score[mask_T])
    T_ps = propensity_score[mask_T][T_ps_idx]
    
    C_ps_idx = np.argsort(propensity_score[mask_C])
    C_ps = propensity_score[mask_C][C_ps_idx]

    plt.figure()
    plt.scatter(np.zeros(T_ps.shape[0]), T_ps, label='Treatment score')
    plt.scatter(np.ones(C_ps.shape[0]), C_ps, label='Control score')
    plt.legend()
    plt.title('Scatter plot for scores')
    ####################
    
    ### match nearst score ###
    def get_match(match, iidx, mask_curr):
        C_ps_ = copy.deepcopy(C_ps)
        C_ps_[match[mask_curr].astype(np.int32).flatten()] = 1000

        for j in range(k):
            for i in iidx:
                dist = np.abs(T_ps[i] - C_ps_)
                if dist.min()<threshold:
                    idx = np.argmin(dist)
                    match[i, j] = idx
                    match_dist[i,j] = dist.min()
                    C_ps_[idx] = 1000
                else:
                    match[i, j] = None
        
        
        mask_and = (~pd.isnull(match)).any(axis=1)
        mask_or = (pd.isnull(match)).any(axis=1)
        mask_xor = np.logical_and(mask_and, mask_or)
        match[mask_xor] = None
        
        iidx = np.arange(match.shape[0])[mask_xor]
        iidx_k = iidx.shape[0]//k
    
        mask_curr = (~pd.isnull(match)).all(axis=1)
        return match, iidx[:iidx_k], mask_curr
    
    match = np.zeros([status[mask_T].shape[0], k])
    match_dist = np.zeros([status[mask_T].shape[0], k])
    iidx = np.arange(T_ps.shape[0])
    mask_curr = np.zeros_like(match, dtype=bool)
    
    for _ in range(round_):
        match, iidx, mask_curr = get_match(match, iidx, mask_curr)
        
    match_C_idx = match[mask_curr].astype(np.int32).flatten()
    org_C_idx = org_ind[mask_C][C_ps_idx[match_C_idx]].reshape([-1,k])
    org_T_idx = org_ind[mask_T][T_ps_idx[mask_curr]]
    
    ### 1st col: treatment; left col: control
    results = np.c_[org_T_idx, org_C_idx]
    
    print(results)
    print('total number of treatment is {}; control is {}; total {}'.\
          format(results.shape[0], results.shape[0]*k, results.shape[0]+results.shape[0]*k))
    
    return results

def T_C_match_(status, propensity_score, k=2, threshold=0.01):
    mask_C = status==0
    mask_T = status==1
    org_ind = np.arange(status.shape[0])
    
    ### sorted score ###
    T_ps_idx = np.argsort(propensity_score[mask_T])
    T_ps = propensity_score[mask_T][T_ps_idx]
    
    C_ps_idx = np.argsort(propensity_score[mask_C])
    C_ps = propensity_score[mask_C][C_ps_idx]

    plt.figure()
    plt.scatter(np.zeros(T_ps.shape[0]), T_ps, label='Treatment score')
    plt.scatter(np.ones(C_ps.shape[0]), C_ps, label='Control score')
    plt.legend()
    plt.title('Scatter plot for scores')
    ####################
    
    ### match nearst score ###
    C_ps_ = copy.deepcopy(C_ps)
    match = np.zeros([status[mask_T].shape[0], k])
    match_dist = np.zeros([status[mask_T].shape[0], k])
    
    for j in range(k):
        for i in range(T_ps.shape[0]):
            dist = np.abs(T_ps[i] - C_ps_)
            if dist.min()<threshold:
                idx = np.argmin(dist)
                match[i, j] = idx
                match_dist[i,j] = dist.min()
                C_ps_[idx] = 1000
            else:
                match[i, j] = None
    
    
    mask_and = (~pd.isnull(match)).any(axis=1)
    mask_or = (pd.isnull(match)).any(axis=1)
    mask_xor = np.logical_and(mask_and, mask_or)
    iidx = np.arange(match.shape[0])[mask_xor]
    iidx_k = iidx.shape[0]//k

    mask_curr = (~pd.isnull(match)).all(axis=1)
    C_ps__ = copy.deepcopy(C_ps)
    C_ps__[match[mask_curr].astype(np.int32).flatten()] = 1000
    for j in range(k):
        for i in iidx[:iidx_k]:
            dist = np.abs(T_ps[i] - C_ps__)
            if dist.min()<threshold:
                idx = np.argmin(dist)
                match[i, j] = idx
                match_dist[i,j] = dist.min()
                C_ps__[idx] = 1000
            else:
                match[i, j] = None
    
    mask_curr = (~pd.isnull(match)).all(axis=1)
    match = match[mask_curr]
    
    match_C_idx = match.astype(np.int32).flatten()
    org_C_idx = org_ind[mask_C][C_ps_idx[match_C_idx]].reshape([-1,k])
    org_T_idx = org_ind[mask_T][T_ps_idx[mask_curr]]
    
    ### 1st col: treatment; left col: control
    results = np.c_[org_T_idx, org_C_idx]
    
    print(results)
    print('total number of treatment is {}; control is {}; total {}'.\
          format(results.shape[0], results.shape[0]*k, results.shape[0]+results.shape[0]*k))
    
    return results


def T_C_nearest_match(status, propensity_score, threshold=0.01, crop=None):
    mask_C = status==0
    mask_T = status==1
    org_ind = np.arange(status.shape[0])
    
    T_ps = propensity_score[mask_T]
    C_ps = propensity_score[mask_C]
    
    dist_ = np.abs(T_ps.reshape([-1,1]) - C_ps)
    dist = np.min(dist_, axis=0)
    idx_sort = np.argsort(dist)
    dist_sort = dist[idx_sort]
    
    C_idx = org_ind[mask_C]
    mask = dist_sort<threshold
    
    if crop:
        if crop>mask.sum():
            C_idx_selected = C_idx[idx_sort[mask]]
        else:
            mask[crop:]=False
            C_idx_selected = C_idx[idx_sort[mask]]
            
    else:
        C_idx_selected = C_idx[idx_sort[mask]]
        
    print('Treatment group: {}; Control group: {}'.format(T_ps.shape[0], C_idx_selected.shape[0]))
    
    org_T_idx = org_ind[mask_T]
    org_C_idx = C_idx_selected
    results = np.r_[org_T_idx, org_C_idx]
    
    return results
    

def T_C_nearest_match_(status, propensity_score, ratio=3, crop=250):
    mask_C = status==0
    mask_T = status==1
    org_ind = np.arange(status.shape[0])
    
    T_ps = propensity_score[mask_T]
    C_ps = propensity_score[mask_C]
    
    dist_ = np.abs(T_ps.reshape([-1,1]) - C_ps)
    
    
    
    x,y = np.meshgrid(np.arange(dist_.shape[0]), np.arange(dist_.shape[1]))
    x,y = x.T, y.T
    idx_sort = np.argsort(dist_.flatten())
    x_sort = x.flatten()[idx_sort]
    y_sort = y.flatten()[idx_sort]
    
    coor_sort = np.c_[x_sort,y_sort]
    
    # mean_dist = []
    # for crop in range(110, mask_T.sum()):
    #     xx = np.unique(coor_sort[:crop,0])
    #     yy = np.unique(coor_sort[:crop,1])
        
    #     mask = np.array([i in xx for i in coor_sort[:,0]])
        
    #     flag = True
    #     k = 10
    #     while flag:
    #         select = coor_sort[mask,:][:k]
            
    #         if np.unique(select[:,1]).shape[0]<xx.shape[0]*ratio:
    #             k = k+1
    #         else:
    #             flag = False
        
            
    #     selected_x = np.unique(select[:,0])
    #     selected_y = np.unique(select[:,1])
        
    #     dd = np.min(dist_[selected_x][:,selected_y], axis=0).mean()
    #     mean_dist.append(dd)
    
    
    xx = np.unique(coor_sort[:crop,0])
    yy = np.unique(coor_sort[:crop,1])
    
    mask = np.array([i in xx for i in coor_sort[:,0]])
    
    
    
    if yy.shape[0]>xx.shape[0]*ratio:
        k = 10
        while True:
            select = coor_sort[mask,:][:k]
            if np.unique(select[:,0]).shape[0]<xx.shape[0]:
                k = k+1
                continue
            else:
                break
    
        while True:
            select = coor_sort[mask,:][:k]
            
            if np.unique(select[:,1]).shape[0]>np.unique(select[:,0]).shape[0]*ratio:
                k = k-1
            else:
                break
            
        
    else:
        k = 10
        while True:
            select = coor_sort[mask,:][:k]
            if np.unique(select[:,1]).shape[0]<xx.shape[0]*ratio:
                k = k+1
            else:
                break
        
    selected_x = np.unique(select[:,0])
    selected_y = np.unique(select[:,1])
    
    final_idx_T = org_ind[mask_T][selected_x]
    final_idx_C = org_ind[mask_C][selected_y]
    
    
    results = np.r_[final_idx_T, final_idx_C]
    print(f'number of T: {selected_x.shape[0]}, number of C: {selected_y.shape[0]}')
    return results#final_idx_T, final_idx_C

def save_data(data, save_path):
    if os.path.exists(save_path):
        os.remove(save_path)
        print("Successfully! The File has been removed")
    
    data.to_excel(save_path, index=False)
    