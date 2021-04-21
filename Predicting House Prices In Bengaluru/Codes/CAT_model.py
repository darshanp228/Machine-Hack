# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 18:55:34 2020

@author: Darshan
"""
from sklearn.model_selection import train_test_split
from Preprocessing import PreProcessing
from catboost import Pool, CatBoostRegressor
import pandas as pd
import numpy as np

#Get Trainging data
dataset = pd.read_csv('Train.csv')
X_train = dataset.iloc[:, :-1].values
y_train = dataset.iloc[:, 8].values

#Get Testing data
dataset = pd.read_csv('Test.csv')
X_test = dataset.iloc[:, :-1].values

#Preprocessing Training data
X_train = pd.DataFrame(X_train)
X_train = PreProcessing(X_train)

#Preprocessing Testing data
X_test = pd.DataFrame(X_test)
X_test = PreProcessing(X_test)

features = ['Area_Type', 'Location']
train_data = X_train
train_label = y_train

train_dataset = Pool(data=train_data,
                     label=train_label,
                     cat_features=features)

# Initialize CatBoostClassifier
model = CatBoostRegressor(
                        iterations = 2000,
                        eval_metric='RMSE',         
                        #         task_type="GPU",
                        nan_mode='Min',
                        verbose=False)
   
# Fit model with `use_best_model=True`
model.fit(train_dataset,
          use_best_model=True,
          plot=True)

#Predict outputs for Test data
y_pred = model.predict(X_test)

#Calculate results
CATresults = 1 - np.sqrt(np.square(np.log10(y_pred +1) - np.log10(y_test +1)).mean())
print(CATresults)

# Create Submission CSV
df_T = pd.DataFrame(y_pred,columns = ['price'])
df_T.to_csv("CAT.csv",index=None)
