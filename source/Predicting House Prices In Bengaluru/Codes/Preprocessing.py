# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:42:04 2020

@author: Darshan
"""

# Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd
import re
from sklearn.impute import SimpleImputer 
import category_encoders as ce

def PreProcessing(X):
    
    #Standardise Bedrooms
    for i in range(X.shape[0]):
        if type(X.iloc[i, 3]) == str:
            temp = X.iloc[i, 3].split()
            if temp[1][0] == 'B':
                X.iloc[i, 3] = int(temp[0]) + 1
            else:
                X.iloc[i, 3] = int(temp[0])
    
    # Taking care of missing data in Bedrooms
    X = X.iloc[:, :].values
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(X[:, 3:4])
    X[:, 3:4] = imputer.transform(X[:, 3:4])
    
    # Taking care of missing data in Balcony in Bath
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(X[:, 6:8])
    X[:, 6:8] = imputer.transform(X[:, 6:8])
    
    X = pd.DataFrame(X)
    
    #Drop Society column
    X = X.drop(X.columns[4], axis=1)
    X = X.reset_index(drop=True)
    
    #Give column names
    X.columns = ['Area_Type', 'Availability', 'Location', 'Size', 'Total_Area', 'Bath', 'Balcony']        
    
    
    #Drop row with nan value in Location
#    for i in range(X.shape[0]):
#        if type(X.iloc[i, 2]) == float:
#            X = X.drop(i)
#            X = X.reset_index(drop=True)
#            break
    
    #Taking mean for range of Sq. Feet
    for i in range(X.shape[0]):
        if re.findall("-", X.iloc[i, 4]):
            temp = X.iloc[i, 4].split('-')
            X.iloc[i, 4] = (float(temp[0].strip()) + float(temp[1].strip()))/2
            
    #Converting all values in Sq. Feet
    for i in range(X.shape[0]):
        temp = X.iloc[i, 4]
        if re.findall('Acres', str(temp)):
            temp = temp.replace('Acres','')
            temp = float(temp) * 43560
            X.iloc[i, 4] = temp
            continue
    #        break
        elif re.findall('Cents', str(temp)):
            temp = temp.replace('Cents','')
            temp = float(temp) * 435.6
            X.iloc[i, 4] = temp
            continue
    #        break
        elif re.findall('Perch', str(temp)):
            temp = temp.replace('Perch','')
            temp = float(temp) * 272.25
            X.iloc[i, 4] = temp
            continue
    #        break
        elif re.findall('Sq. Meter', str(temp)):
            temp = temp.replace('Sq. Meter','')
            temp = float(temp) * 10.7639
            X.iloc[i, 4] = temp
            continue
    #        break
        elif re.findall('Sq. Yards', str(temp)):
            temp = temp.replace('Sq. Yards','')
            temp = float(temp) * 9
            X.iloc[i, 4] = temp
            continue
    #        break
        elif re.findall('Grounds', str(temp)):
            temp = temp.replace('Grounds','')
            temp = float(temp) * 2400
            X.iloc[i, 4] = temp
            continue
    #        break
        elif re.findall('Guntha', str(temp)):
            temp = temp.replace('Guntha','')
            temp = float(temp) * 1089
            X.iloc[i, 4] = temp
            continue
    #        break
        else:
            X.iloc[i, 4] = float(temp)
    
    #Create ranking for date
    date_feature = { 'Jan': 1,
                  'Feb': 2,
                  'Mar': 3,
                  'Apr': 4,
                  'May': 5,
                  'Jun': 6,
                  'Jul': 7,
                  'Aug': 8,
                  'Sep': 9,
                  'Oct': 10,
                  'Nov': 11,
                  'Dec': 12
                  }
    
    #Handling Availability
    for i in range(X.shape[0]):
        if X.iloc[i, 1] == 'Ready To Move' or X.iloc[i, 1] == 'Immediate Possession':
            X.iloc[i, 1] = 0
        else:
            X.iloc[i, 1] = date_feature[X.iloc[i, 1].split('-')[1]]
            
#    #Handling Categorical Data of Area Type
#    encoder_at = ce.BaseNEncoder(cols=['Area_Type'],return_df=True,base=2)
#    X = encoder_at.fit_transform(X)
#    
#    #Handling Categorical Data of Location        
#    encoder_loc = ce.BaseNEncoder(cols=['Location'],return_df=True,base=2)
#    X = encoder_loc.fit_transform(X)
    
    return X
