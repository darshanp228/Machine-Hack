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


dataset = pd.read_csv('Train.csv')
X_train = dataset.iloc[:, :-1].values
y_train = dataset.iloc[:, 8].values
#X = dataset.iloc[:, :-1].values
#y = dataset.iloc[:, 8].values


dataset = pd.read_csv('Test.csv')
X_test = dataset.iloc[:, :-1].values

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train = pd.DataFrame(X_train)
X_train = PreProcessing(X_train)

X_test = pd.DataFrame(X_test)
X_test = PreProcessing(X_test)

features = ['Area_Type', 'Location']
train_data = X_train
train_label = y_train
#eval_data = X_test
#eval_label = y_test

train_dataset = Pool(data=train_data,
                     label=train_label,
                     cat_features=features)

#eval_dataset = Pool(data=eval_data,
#                    label=eval_label,
#                    cat_features=features)


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

y_pred = model.predict(X_test)

CATresults = 1 - np.sqrt(np.square(np.log10(y_pred +1) - np.log10(y_test +1)).mean())
print(CATresults)

# Create Submission CSV
df_T = pd.DataFrame(y_pred,columns = ['price'])
df_T.to_csv("CAT.csv",index=None)
