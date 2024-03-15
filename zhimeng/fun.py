# https://github.com/Explainable-Hospital-Mortality/individual-SHAP-comparison/tree/main
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
import numpy as np


def get_feature_importance(explainer, X_test):
    shap_values = explainer.shap_values(X_test)
    shap_values = np.r_[shap_values]
    if len(shap_values.shape)==3:
        shap_values = np.sum(np.abs(shap_values), axis=0)

    importance = np.abs(shap_values).mean(axis=0)
    # shap.summary_plot(shap_values, X_test, plot_type='bar')
    return importance


def get_best_model(model_best, search, X_train, y_train):
    model_best.set_params(**search.best_params_)
    print(search.best_params_)

    model_best.fit(X_train, y_train)
    return model_best



from sklearn.naive_bayes import GaussianNB
def get_model_NB(X_train, y_train, scoring='roc_auc'):

    model = GaussianNB()
    param = {'var_smoothing': np.logspace(0,-9, num=100)}

    clf = GridSearchCV(model, param, n_jobs=-1, cv=5, scoring=scoring)
    # clf = RandomizedSearchCV(model, param, n_jobs=-1, cv=5, scoring=scoring, random_state=42, n_iter=5)
    search = clf.fit(X_train, y_train)
    model_best = get_best_model(model, search, X_train, y_train)

    return model_best



from scipy.stats import uniform
from sklearn.linear_model import LogisticRegression
def get_model_LR(X_train, y_train, scoring='roc_auc'):

    model = LogisticRegression(class_weight="balanced", solver="liblinear", random_state=42)
    param = {'penalty': ['l2', 'l1'],
            'tol': [1e-5, 1e-3, 1e-2],
            'C': [0.1, 0.5, 1.0, 2.0], 
            # 'C': uniform(loc=0, scale=4),
            'fit_intercept': [True, False],
            'max_iter': [50, 100, 250, 500]}

    clf = GridSearchCV(model, param, n_jobs=-1, cv=5, scoring=scoring)
    # clf = RandomizedSearchCV(model, param, n_jobs=-1, cv=5, scoring=scoring, random_state=42, n_iter=5)
    search = clf.fit(X_train, y_train)
    model_best = get_best_model(model, search, X_train, y_train)

    return model_best



from xgboost import XGBClassifier
def get_model_XGBoost(X_train, y_train, scoring='roc_auc'):

    model = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic', nthread=1)
    param = {'n_estimators': [100, 250, 600, 1000],
                            'min_child_weight': [1, 5, 10],
                            'gamma': [0.5, 1, 1.5, 2, 5],
                            'subsample': [0.6, 0.8, 1.0],
                            'colsample_bytree': [0.6, 0.8, 1.0],
                            'max_depth': [3, 4, 5]
                            }

    # clf = GridSearchCV(model, param, n_jobs=-1, cv=5, scoring=scoring)
    clf = RandomizedSearchCV(model, param, n_jobs=-1, cv=5, scoring=scoring, random_state=42, n_iter=5)
    search = clf.fit(X_train, y_train)
    model_best = get_best_model(model, search, X_train, y_train)

    return model_best



from sklearn.ensemble import RandomForestClassifier
def get_model_RF(X_train, y_train, scoring='roc_auc'):

    model = RandomForestClassifier()
    param = {
            'bootstrap': [True, False], #, False
            'max_depth': [10, 70, None],
    #         'max_features': ['sqrt', 'auto', None], #'auto',
            'min_samples_leaf': [5, 10], #4
            'min_samples_split': [5, 10], #5
            'n_estimators': [200, 500]  #200, 2000
            }

    clf = GridSearchCV(model, param, n_jobs=-1, cv=5, scoring=scoring)
    # clf = RandomizedSearchCV(model, param, n_jobs=-1, cv=5, scoring=scoring, random_state=42, n_iter=5)
    search = clf.fit(X_train, y_train)
    model_best = get_best_model(model, search, X_train, y_train)

    return model_best