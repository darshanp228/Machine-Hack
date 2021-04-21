# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 12:20:19 2020

@author: Darshan
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Preprocessing import PreProcessing
import category_encoders as ce

# Importing the dataset
dataset = pd.read_csv('Train.csv')


# Standard Scaling Price
#from sklearn.preprocessing import StandardScaler
#scaled_features = dataset.copy()
#col_names = ['price']
#features = scaled_features[col_names]
#price_scaler = StandardScaler().fit(features.values)
#features = price_scaler.transform(features.values)
#scaled_features[col_names] = features
#dataset = scaled_features

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

#Handling Categorical Data of Area Type
#encoder_at = ce.BaseNEncoder(cols=['Area_Type'],return_df=True,base=2)
encoder_at = ce.OneHotEncoder(cols='Area_Type',handle_unknown='return_nan',return_df=True,use_cat_names=True)
#X_train = pd.get_dummies(data=X_train,columns=['Area_Type'],drop_first=True)
#X_test = pd.get_dummies(data=X_test,columns=['Area_Type'],drop_first=True)
X_train = encoder_at.fit_transform(X_train)
X_test = encoder_at.transform(X_test)


#Handling Categorical Data of Location        
encoder_loc = ce.BaseNEncoder(cols=['Location'],return_df=True,base=2)
#encoder_loc = ce.BinaryEncoder(cols=['Location'],return_df=True)
X_train = encoder_loc.fit_transform(X_train)
X_test = encoder_loc.transform(X_test)


#Liner Regressor
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
LRresults = 1 - np.sqrt(np.square(np.log10(y_pred_lr +1) - np.log10(y_test +1)).mean())
print(LRresults)

#XG Boost
import xgboost as xgb
xg =xgb.XGBRegressor()
xg.fit(X_train, y_train)
y_pred_xg = xg.predict(X_test)
XGresults = 1 - np.sqrt(np.square(np.log10(y_pred_xg +1) - np.log10(y_test +1)).mean())
print(XGresults)

#Decisionn Tree
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)
r_sq = regressor.score(X_train, y_train)
y_pred = regressor.predict(X_test)
DTresults = 1 - np.sqrt(np.square(np.log10(y_pred +1) - np.log10(y_test +1)).mean())
print(DTresults)
# saving the dataframe  
#df.to_csv('DT_DUMMY_BIN_output.csv', index=False)

#Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 50, random_state = 0)
regressor.fit(X_train, y_train)
r_sq = regressor.score(X_train, y_train)
y_pred = regressor.predict(X_test)
RFresults = 1 - np.sqrt(np.square(np.log10(y_pred +1) - np.log10(y_test +1)).mean())
print(RFresults)
# saving the dataframe  
#df.to_csv('RF_DUMMY_BIN_output.csv', index=False)

#Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
regressor = GradientBoostingRegressor()
regressor.fit(X_train, y_train)
r_sq = regressor.score(X_train, y_train)
y_pred = regressor.predict(X_test)
df = pd.DataFrame(y_pred)      
GBRresults = 1 - np.sqrt(np.square(np.log10(y_pred +1) - np.log10(y_test +1)).mean())
print(GBRresults)
# saving the dataframe  
#df.to_csv('GBR_OHE_BN_output.csv', index=False)

#ElasticNet
from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet()
elasticnet.fit(X, y)
y_pred_enet = elasticnet.predict(X_test)
ENresults = 1 - np.sqrt(np.square(np.log10(y_pred_enet +1) - np.log10(y_test +1)).mean())  

#Lasso
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(X, y)
y_pred_lasso = lasso.predict(X_test)
Lassoresults = 1 - np.sqrt(np.square(np.log10(y_pred_lasso +1) - np.log10(y_test +1)).mean())

#KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()
knn.fit(X, y)
y_pred_knn = knn.predict(X_test)
KNNresults = 1 - np.sqrt(np.square(np.log10(y_pred_knn +1) - np.log10(y_test +1)).mean())
# saving the dataframe  
#df.to_csv('KNN_OHE_BN_output.csv', index=False)

#Bagging Regressor
from sklearn.ensemble import BaggingRegressor
from sklearn import tree
br = BaggingRegressor(tree.DecisionTreeRegressor(random_state=1))
br.fit(X_train, y_train)
y_pred_br = br.predict(X_test)
BRresults = 1 - np.sqrt(np.square(np.log10(y_pred_br +1) - np.log10(y_test +1)).mean())
print(BRresults)

#SVR
from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)
SVRresults = 1 - np.sqrt(np.square(np.log10(y_pred_svr +1) - np.log10(y_test +1)).mean())
print(SVRresults)

#RidgeCV
from sklearn.linear_model import RidgeCV
rcv = RidgeCV()
rcv.fit(X_train, y_train)
y_pred_rcv = rcv.predict(X_test)
RCVresults = 1 - np.sqrt(np.square(np.log10(y_pred_rcv +1) - np.log10(y_test +1)).mean())
print(RCVresults)

#Voting Regressor
from sklearn.ensemble import VotingRegressor
estimatorsvr = [('dt',DecisionTreeRegressor(random_state = 0)),
                ('rf',RandomForestRegressor(n_estimators = 50, random_state = 0)),
              ('gbr',GradientBoostingRegressor()),
#              ('en',ElasticNet()),
#              ('lasso',Lasso()),
              ('knn',KNeighborsRegressor()),
              ('br',BaggingRegressor(tree.DecisionTreeRegressor(random_state=1))),
#              ('svr',SVR(kernel = 'rbf'))
              ]
#[7,12,11,3,3,11,12,3]
vr = VotingRegressor(estimatorsvr, weights=[2,5,3,5,3])
vr.fit(X_train, y_train)    
y_pred_vr = vr.predict(X_test)
VRresults = 1 - np.sqrt(np.square(np.log10(y_pred_vr +1) - np.log10(y_test +1)).mean())
print(VRresults)

#Stacking Regressor
from sklearn.ensemble import StackingRegressor
estimatorssr = [('dt',DecisionTreeRegressor(random_state = 0)),
              ('rf',RandomForestRegressor(n_estimators = 50, random_state = 0)),
              ('gbr',GradientBoostingRegressor()),
              ('knn',KNeighborsRegressor()),
              ('br',BaggingRegressor(tree.DecisionTreeRegressor(random_state=1))),
              ]
sr = StackingRegressor(estimators=estimatorssr,final_estimator=RandomForestRegressor(n_estimators=50,random_state=42))
sr.fit(X_train, y_train)    
y_pred_sr = sr.predict(X_test)
SRresults = 1 - np.sqrt(np.square(np.log10(y_pred_sr +1) - np.log10(y_test +1)).mean())
print(SRresults)

# Create Submission CSV
df_T = pd.DataFrame(y_pred_vr,columns = ['price'])
df_T.to_csv("VR_9.csv",index=None)

score = 1 - np.sqrt(np.square(np.log10(y_pred_lr +1) - np.log10(y_test +1)).mean())
