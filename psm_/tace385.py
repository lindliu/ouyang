from psm import *
import pandas as pd
import numpy as np

    
def select_Cgroup(Tgroup, Cgroup, C_idx, name='lnm'):
    T_ratio_lnm = Tgroup[name].sum()/Tgroup.shape[0]
    C_ratio_lnm = Cgroup[name].sum()/Cgroup.shape[0]
    
    assert T_ratio_lnm<=C_ratio_lnm
    
    count = 0
    delet_idx = []
    for i in range(Cgroup.shape[0]):
        if Cgroup.iloc[i,:][name]==1:
            count += 1
            if count/Cgroup.shape[0]>T_ratio_lnm:
                delet_idx.append(i)
    C_idx_ = np.delete(C_idx, delet_idx)

    return C_idx_

def get_ratio(group, name = 'peritoneum'):
    ratio = group[name].sum()/group[name].shape[0]
    return ratio

if __name__=="__main__":
    data_file = '661PCLM-sct'
    df = pd.read_stata('{}.dta'.format(data_file))

    status_name = 'tace'
    feature_name = ['peritoneum', 'kps', 'pcsize5', 'ca199500', 'ldh250', 'concur', "sctr","alt60",'lnm']
                    #"alb35", ]
    
    ######################
    # lnm, ca199500
    T_mask = df[status_name]==1
    T_num = T_mask.sum()
    Tgroup = df[T_mask]
    Cgroup = df[~T_mask]
    
    delet_num = 3
    delet_idx = np.random.choice(np.arange(T_num)[Tgroup['lnm']==0],delet_num,replace=False) #np.array([5, 68, 67])
    delet_idx = np.array([5, 68, 67])
    T_idx = np.delete(np.arange(Tgroup.shape[0]), delet_idx)
    Tgroup = Tgroup.iloc[T_idx]
    
    df = pd.concat([Tgroup, Cgroup], ignore_index=True)
    #####################


    T_mask = df[status_name]==1
    Tgroup = df[T_mask]
    T_num = T_mask.sum()
    
    ### get data: feature(X) and status(Y) ###
    feature, status = get_data(df, feature_name, status_name)
    
    ### get propensity score
    propensity_score = pre_logit(feature, status)
    
    ### match the treatment and control    
    results = T_C_nearest_match(status, propensity_score, threshold=0.002, crop=T_num*5)
    
    ###############
    # # lnm, ca199500
    # T_num = status.sum() ## 69 ## treatment is 1, control is 0
    # T_idx = results.flatten()[:T_num]
    # C_idx = results.flatten()[T_num:]
    # Tgroup = df.iloc[T_idx,:]
    # Cgroup = df.iloc[C_idx,:]
    
    # for name in ['lnm', 'ca199500']:
    #     C_idx = select_Cgroup(Tgroup, Cgroup, C_idx, name)
    #     Cgroup = df.iloc[C_idx,:]
    
    # results = np.r_[T_idx, C_idx]
    # print('Final: Treatment group: {}; Control group: {}'.format(T_idx.shape[0], C_idx.shape[0]))
    ################
    

    
    # ##################################
    # # lnm, ca199500
    # T_num = status.sum() ## 69 ## treatment is 1, control is 0
    # T_idx = results.flatten()[:T_num]
    # C_idx = results.flatten()[T_num:]
    # Tgroup = df.iloc[T_idx,:]
    # Cgroup = df.iloc[C_idx,:]
    
    # Tgroup['lnm']==0
    # delet_idx = np.random.choice(np.arange(69)[Tgroup['lnm']==0],8,replace=False)
    # T_idx = np.delete(T_idx, delet_idx)
    
    # results = np.r_[T_idx, C_idx]
    # print('Final: Treatment group: {}; Control group: {}'.format(T_idx.shape[0], C_idx.shape[0]))
    
    # selected = df.iloc[results.flatten(),:]
    # ### get data: feature(X) and status(Y) ###
    # status_name = 'tace'
    # feature_name = ['peritoneum', 'kps', 'ca199500', 'ldh250',"sctr","alt60","alb35"]
    # feature, status = get_data(selected, feature_name, status_name)
    
    # ### get propensity score
    # propensity_score = pre_logit(feature, status)
    
    # ### match the treatment and control    
    # results = T_C_nearest_match(status, propensity_score, threshold=0.001,crop=305)
    
    
    # ### save selected data
    # selected = selected.iloc[results.flatten(),:]
    # save_data(selected, "{}.xlsx".format(data_file+'_selected'))
    # ############################################

    print(np.arange(T_num)[Tgroup['lnm']==0])
    
    ### save selected data
    selected = df.iloc[results.flatten(),:]
    selected = selected.drop(index=[6], axis=0)
    # selected = selected.drop(index=results[-5:], axis=0)
    save_data(selected, "{}.xlsx".format(data_file+'_selected'))

    
##################
    Tgroup = selected[selected[status_name]==1]
    Cgroup = selected[selected[status_name]==0]
    name = 'peritoneum' #'pcsize5' #
    dd = np.arange(Tgroup.shape[0])[Tgroup[name]==0]
    Tgroup = Tgroup.drop(index=dd[-10:], axis=0)
    print(get_ratio(Tgroup, name))
    print(get_ratio(Cgroup, name))
    
    selected = pd.concat([Tgroup, Cgroup], ignore_index=True)
    save_data(selected, "{}.xlsx".format(data_file+'_selected'))
#####################

    print('treatment lnm:{}'.format(Tgroup['lnm'].sum()/Tgroup['lnm'].shape[0]))
    print('contorl lnm:{}'.format(selected[selected['tace']==0]['lnm'].sum()/selected[selected['tace']==0]['lnm'].shape[0]))