#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 19:02:30 2024

@author: dliu
"""

# import pandas as pd

# df = pd.read_excel("Total.xlsx", index_col=None)

# cluster = df['Cluster'].str.split(' ', n=1, expand=True)
# cluster = cluster.rename(columns={0:'cluster', 1:'kPa'})
# features = df.drop('Cluster', axis=1)

# df = pd.concat([features, cluster], axis=1)
# df.to_csv('Total.csv', index=False)
# print(df)





import shap
import pandas as pd
import numpy as np
shap.initjs()

df = pd.read_csv('Total.csv', index_col = None)



from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


mask = df['kPa']=='22kPa'
df = df[mask]
X = df.drop(["ID", "cluster","kPa"], axis=1) # Independent variables
y = df.cluster # Dependent variable

features = X.columns





# Split into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Train a machine learning model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make prediction on the testing data
y_pred = clf.predict(X_test)

# Classification Report
print(classification_report(y_pred, y_test))


explainer = shap.Explainer(clf)
shap_values = explainer.shap_values(X_test)

# https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability
shap.summary_plot(shap_values, X_test)
shap.summary_plot(shap_values[0], X_test)


shap.dependence_plot("Intensity_IntegratedIntensityEdge_YAP", shap_values[0], X_test,interaction_index="Intensity_IntegratedIntensity_YAP")
shap.plots.force(explainer.expected_value[0], shap_values[0][0,:], X_test.iloc[0, :], matplotlib = True)
shap.decision_plot(explainer.expected_value[1], shap_values[1], X_test.columns)










# Initialize and train a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
feature_importances = rf.feature_importances_

# Print feature importances
for i, importance in enumerate(feature_importances):
    print(f"Feature {i+1}: {importance}")
    
import matplotlib.pyplot as plt
plt.plot(np.sort(feature_importances))
    
features[np.argsort(feature_importances)[::-1][:10]]















from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# Initialize and fit LDA
lda = LinearDiscriminantAnalysis()
X_train_lda = lda.fit_transform(X_train, y_train)

# Make prediction on the testing data
y_pred = lda.predict(X_test)

# Classification Report
print(classification_report(y_pred, y_test))

# Access the coefficients or weights of the features
feature_importance = lda.coef_
print("Feature Importance:")
for i, importance in enumerate(feature_importance[0]):
    print(f"Feature {i+1}: {importance}")
    
    
    


from sklearn.linear_model import LogisticRegression
# Train a logistic regression model on the transformed data
lr = LogisticRegression()
lr.fit(X_train_lda, y_train)

# Explain the model predictions using SHAP
explainer = shap.Explainer(lr, X_train_lda)
shap_values = explainer(X_test)


shap.summary_plot(shap_values, X_test)





model = LogisticRegression()
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values)
