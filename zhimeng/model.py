import numpy as np
from sklearn.metrics import classification_report, f1_score
import shap

from fun import get_feature_importance
from fun import get_model_NB, get_model_LR, get_model_XGBoost, get_model_RF, get_model_SVC

def get_importance_index(X_train, y_train, X_test, y_test, scoring):

    # ######################## Navier Bayes ##########################
    # ### get model
    # model_NB = get_model_NB(X_train, y_train['cluster'], scoring=scoring)
    # print('Naiver Bayes: ')
    # print(classification_report(y_test, model_NB.predict(X_test)))
    # ### explain model
    # explainer = shap.KernelExplainer(model_NB.predict_proba, shap.sample(X_train, 50))
    # importance_NB = get_feature_importance(explainer, X_test)
    # index_NB = np.argsort(importance_NB)[::-1]

    # ########################## SVM ########################
    # ### get model
    # model_SVC = get_model_SVC(X_train, y_train['cluster'], scoring=scoring)
    # print('SVC: ')
    # print(classification_report(y_test, model_SVC.predict(X_test)))
    # ### explain model
    # explainer = shap.KernelExplainer(model_SVC.predict_proba, shap.sample(X_train,50))
    # importance_SVC = get_feature_importance(explainer, X_test)
    # index_SVC = np.argsort(importance_SVC)[::-1]


    ######################### Random Forest ##################################
    ### get model
    model_RF = get_model_RF(X_train, y_train['cluster'], scoring=scoring)
    print('Random Forest: ')
    print(classification_report(y_test, model_RF.predict(X_test)))
    print(f1_score(y_test, model_RF.predict(X_test), average='weighted'))
    ### explain model
    explainer = shap.TreeExplainer(model_RF, X_train)
    importance_RF = get_feature_importance(explainer, X_test)
    index_RF = np.argsort(importance_RF)[::-1]


    ######################## Logistic regression #########################
    ### get model
    model_LR = get_model_LR(X_train, y_train['cluster'], scoring=scoring)
    print('Logistic regression: ')
    print(classification_report(y_test, model_LR.predict(X_test)))
    ### explain model
    explainer = shap.LinearExplainer(model_LR, X_train, feature_perturbation="correlation_dependent")
    importance_LR = get_feature_importance(explainer, X_test)
    index_LR = np.argsort(importance_LR)[::-1]


    ######################### XGBoost #######################################
    ### get model
    model_XGBoost = get_model_XGBoost(X_train, y_train['cluster'], scoring=scoring)
    print('XGBoost: ')
    print(classification_report(y_test, model_XGBoost.predict(X_test)))
    ### explain model
    explainer = shap.TreeExplainer(model_XGBoost, X_train)
    importance_XGBoost = get_feature_importance(explainer, X_test)
    index_XGBoost = np.argsort(importance_XGBoost)[::-1]


    # return np.c_[index_NB, index_SVC, index_LR, index_XGBoost, index_RF], \
            # np.c_[importance_NB, importance_SVC, importance_LR, importance_XGBoost, importance_RF]
    return np.c_[index_LR, index_XGBoost, index_RF], np.c_[importance_LR, importance_XGBoost, importance_RF]



