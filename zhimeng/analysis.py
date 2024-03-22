import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np



import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# np.random.seed(42)



def plot_corr(df,size=10):
    '''Plot a graphical correlation matrix for a dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''
    
    # %matplotlib inline
    import matplotlib.pyplot as plt

    # Compute the correlation matrix for the received dataframe
    corr = df.corr()
    
    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr, cmap='RdYlGn')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);
    
    # Add the colorbar legend
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)

# plot_corr(selected)

import scipy.cluster.hierarchy as sch
def feature_extraction(X_train, y_train, X_test, y_test, X):

    assert X_train.shape[1]==149
    remove_group_num = np.array([50,32,15, 7,3,1,1,1,1,1,1])
    remove_order_num = np.array([10, 8, 5, 3,2,1,1,1,1,1,1])  #60,40,20,10,5,2,2,2,2,2,2
    remove = 149-np.cumsum(remove_group_num+remove_order_num)


    from model import get_importance_index
    unique = np.arange(X_train.shape[1])  

    F1_list, idx_list = [], []
    for epoch in range(remove_order_num.shape[0]):

        ################ feature importance ranking #################
        scoring = 'roc_auc'
        ranked_idx, importance_score, F1 = get_importance_index(X_train.iloc[:,unique], y_train, X_test.iloc[:,unique], y_test, scoring)
        #############################################################


        ############### remove features by importance ranking ##################
        if (F1>.8).sum()>=2:
            ranked_idx = ranked_idx[:,F1>.8]  ### use model that F1 score larger than .8
        else:
            ranked_idx = ranked_idx[:,np.argsort(F1)[-2:]]  ### if all model F1 score less than .8, than use top 2 models

        ranked_idx_ = unique[ranked_idx.flatten()]

        unique_rank = []
        for i in ranked_idx_.flatten():
            if i not in unique_rank:
                unique_rank.append(i)

        unique_rank = np.array(unique_rank)[:-remove_order_num[epoch]]
        # unique_rank, unique_rank.shape
        ########################################################################
        F1_list.append(F1)
        idx_list.append(unique_rank)

        ############### group selected feature by correlation values ################

        selected = X.iloc[:, unique_rank]

        corr_values = selected.corr().values
        d = sch.distance.pdist(corr_values)   # vector of ('55' choose 2) pairwise distances
        L = sch.linkage(d, method='complete')
        ind = sch.fcluster(L, 0.5*d.max(), 'distance')
        columns = [selected.columns.tolist()[i] for i in list((np.argsort(ind)))]
        selected_sorted = selected.reindex(columns, axis=1)

        selected_columns_new = dict([(name, f'group{i}_'+name) for i,name in zip(np.sort(ind),selected_sorted.columns)])
        selected_sorted.rename(columns=selected_columns_new, inplace=True)
        # plot_corr(selected_sorted, size=15)
        ############################################################################


        ########################### remove features by groups(avoid selected features highly correlated) ###############
        group_member = []
        for i in np.unique(ind):
            group_member.append(unique_rank[ind==i])
        # for i in group_member:
        #     print(i)

        ### select percent(per) of current feature, spread in each group has num features
        num = remove_group_num[epoch]//np.unique(ind).shape[0]+1
        # print(num)

        unique_ = []
        for member in group_member:
            unique_.extend(list(member[:-num]))
        unique_ = np.array(unique_)[:remove[epoch]]
        # print(unique_.shape)

        extra = []
        for i in unique_rank:
            if i not in unique_:
                extra.append(i)
            if len(extra) == remove[epoch]-unique_.shape[0]:
                break
            
        unique_ = np.r_[unique_, extra]
        # print(unique_.shape)
        ################################################################################################################

        unique = unique_

    return F1_list, idx_list


def get_data(df_):
    # kPa = '22kPa'
    # df_ = df[df['kPa']==kPa]
    X = df_.drop(["ID", "cluster","kPa"], axis=1) # Independent variables
    # X = X.iloc[:,feature_selected]
    y_ = df_.cluster # Dependent variable

    y = np.zeros(y_.shape)
    y[y_=='TIF'] = 1
    y = pd.DataFrame(y, columns=['cluster'], dtype=float)
    # y_.loc[y_=='CAF'] = 0
    # y_.loc[y_=='TIF'] = 1
    # y = y_.astype(float)


    X_columns_new = dict([(name, f'ind_{i}_'+name) for i,name in enumerate(X.columns)])
    X.rename(columns=X_columns_new, inplace=True)
    features = np.array(X.columns)

    # Split into train and test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train.iloc[:,:] = scaler.transform(X_train)
    X_test.iloc[:,:] = scaler.transform(X_test)

    X_train.shape

    return X_train, y_train, X_test, y_test, X, features


df = pd.read_csv('Total.csv', index_col = None)
columns_new = dict([(name, name.replace('.','__')) for name in df.columns])
df.rename(columns=columns_new, inplace=True)

kPa_list = df['kPa'].unique()
kPa_list

if __name__=="__main__":
    import pickle

    F1_kPa, idx_kPa = [], []
    for kPa in kPa_list:
        df_ = df[df['kPa']==kPa]
        X_train, y_train, X_test, y_test, X, features = get_data(df_)
        F1_list, idx_list = feature_extraction(X_train, y_train, X_test, y_test, X)
        
        F1_kPa.append(F1_list)
        idx_kPa.append(idx_list)


        with open(f'F1_{kPa}', 'wb') as file:
            pickle.dump(F1_list, file)

        with open(f'idx_{kPa}', 'wb') as file:
            pickle.dump(idx_list, file)


    # with open('F1_kPa', 'wb') as file:
    #     pickle.dump(F1_kPa, file)
        
    # with open('idx_kPa', 'wb') as file:
    #     pickle.dump(idx_kPa, file)

    # # with open('F1_kPa', 'rb') as file:
    # #     F1_kPa = pickle.load(file)

    # # with open('idx_kPa', 'rb') as file:
    # #     idx_kPa = pickle.load(file)