import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



data_ = pd.read_excel('test.xlsx')
# print(data_.columns)

data = data_.to_numpy()
idex = data[1:,0]

### get data: feature(X) and status(Y)
status_ = data[1:,9]
status = np.ones([status_.shape[0]], dtype=np.int32)
mask_C = pd.isnull(status_)
mask_T = ~pd.isnull(status_)
status[mask_C] = 0

col = [11,12,13,14,16,17,18,19,20,21,22,23,25,26,27,28,29,30,31,32,33,38,40,41,56,57]
feature = data[1:,col].astype(np.float32)
from sklearn import preprocessing  ## scale features
scaler = preprocessing.StandardScaler().fit(feature)
feature_scaled = scaler.transform(feature)

### fit logit regression model to obtain score
from sklearn.linear_model import LogisticRegression
ps_model = LogisticRegression(C=1e5, max_iter=200).fit(feature_scaled, status)
propensity_score = ps_model.predict_proba(feature_scaled)[:,1]

### sorted score
T_ps_idx = np.argsort(propensity_score[mask_T])
T_ps = propensity_score[mask_T][T_ps_idx]

C_ps_idx = np.argsort(propensity_score[mask_C])
C_ps = propensity_score[mask_C][C_ps_idx]

plt.figure()
plt.scatter(np.zeros(T_ps.shape[0]), T_ps, label='Treatment score')
plt.scatter(np.ones(C_ps.shape[0]), C_ps, label='Control score')
plt.legend()
plt.title('Scatter plot for scores')

C_ps_ = copy.deepcopy(C_ps)

k=3
match = np.zeros([status[mask_T].shape[0], k])
match_dist = np.zeros([status[mask_T].shape[0], k])

# dist = np.abs(T_ps.reshape(-1,1) - C_ps)
# np.argmin(dist, axis=1)

for j in range(k):
    for i in range(T_ps.shape[0]):
        dist = np.abs(T_ps[i] - C_ps_)
        if dist.min()<0.2:
            idx = np.argmin(dist)
            match[i, j] = idx
            match_dist[i,j] = dist.min()
            C_ps_[idx] = 1000
        else:
            match[i, j] = None

print(match)
print(match_dist)
    
extra_ = (~np.isnan(match)).any(axis=1) & np.isnan(match).any(axis=1)
if extra_.sum()!=0:
    extra_id = np.where(extra_ == True)[0][0]
    C_ps[match[:extra_id,:].flatten().astype(np.int32)] = 1000
    
    dist = np.abs(T_ps[extra_id] - C_ps)
    if (np.sort(dist)[:k]<0.2).all():
        match[extra_id,:] = np.argsort(dist)[:k]
        match_dist[extra_id,:] = np.sort(dist)[:k]
        
    match[extra_id+1:,:] = None
    match_dist[extra_id+1:,:] = 0
    
print(match)
print(match_dist)

# ####https://www.youtube.com/watch?v=ACVyPp1Fy6Y&t=182s&ab_channel=DougMcKee
# X = np.array([[.5, .01], [.6, .02], [.7, .01], [.6, .02], 
#               [.6, .01], [.5, .02], [.1, .04], [.3, .05], [.2, .04]])
# Y = np.array([1,1,1,1,0,0,0,0,0])
# ps_model = LogisticRegression(C=1e6).fit(X, Y)
# propensity_score = ps_model.predict_proba(X)[:,1]







# -*- coding: utf-8 -*-

"""

Spyder Editor



This is a temporary script file.

"""



import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing  ## scale features
from sklearn.linear_model import LogisticRegression


def pre_logit(feature, status):
    """
    get propensity score by features and status
    """
    ### preprocessing
    scaler = preprocessing.StandardScaler().fit(feature)
    feature_scaled = scaler.transform(feature)
    
    ### fit logit regression model to obtain score
    ps_model = LogisticRegression(C=1e5, max_iter=200).fit(feature_scaled, status)
    propensity_score = ps_model.predict_proba(feature_scaled)[:,1]
    return propensity_score

def T_C_match(status, propensity_score, k=2, threshold=0.1):
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
    
    num = (~pd.isnull(match)[:,-1]).sum()
    # print(match, num)
    
    match_C_idx = match[:num,:].astype(np.int32).flatten()
    org_C_idx = org_ind[mask_C][C_ps_idx[match_C_idx]].reshape([-1,k])
    org_T_idx = org_ind[mask_T][T_ps_idx[np.arange(num)]]
    
    ### 1st col: treatment; left col: control
    results = np.c_[org_T_idx, org_C_idx]
    
    print(results)
    print('total number of treatment is {}; control is {}; total {}'.format(results.shape[0], results.shape[0]*k, results.shape[0]+results.shape[0]*k))
    
    return results



if __name__=="__main__":
    ### get data: feature(X) and status(Y) ###
    df = pd.read_stata('932new.dta')
    df.fillna(df.median(), inplace=True)
    
    status = df['peritoneum']
    feature = df[['tace', 'kps', 'lnm', 'pcloca', 'pcsize5', 'lungm', 'ca199500', 'lmextent1', 'alb35',"tb24","alt60"]]
    
    ### get propensity score
    propensity_score = pre_logit(feature, status)
    
    ### match the treatment and control
    results = T_C_match(status, propensity_score, k=2, threshold = 0.1)

    ### save selected data
    selected = df.iloc[results.flatten(),:]
    selected.to_excel("932new_selected_.xlsx")
    
