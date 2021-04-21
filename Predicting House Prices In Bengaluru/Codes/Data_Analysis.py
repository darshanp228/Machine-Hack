# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 14:39:14 2020

@author: Darshan
"""

import pandas as pd
data = pd.read_csv('Train.csv',encoding='ISO-8859-1',dtype=str)

type(data.iloc[6, 0])

def CountFrequency(my_list): 
    # Creating an empty dictionary  
    freq = {} 
    for items in my_list: 
        freq[items] = my_list.count(items) 
    return freq

# Find Area Type
lst = []
for i in range(data.shape[0]):
    lst.append(data.iloc[i, 0])
area_type_count = CountFrequency(lst)
area_type_list = list(set(lst))

#Find availability
lst = []
for i in range(data.shape[0]):
    lst.append(data.iloc[i, 1])
availability_count = CountFrequency(lst)
availability_list = list(set(lst))

#Find location
lst = []
for i in range(data.shape[0]):
    lst.append(data.iloc[i, 2])
location_count = CountFrequency(lst)
location_list = list(set(lst))

#Find size
lst = []
for i in range(data.shape[0]):
    lst.append(data.iloc[i, 3])
size_count = CountFrequency(lst)
size_list = list(set(lst))


#Find society
lst = []
for i in range(data.shape[0]):
    lst.append(data.iloc[i, 4])
society_count = CountFrequency(lst)
society_list = list(set(lst))

#Find sqft
lst = []
for i in range(data.shape[0]):
    lst.append(data.iloc[i, 5])
sqft_count = CountFrequency(lst)
sqft_list = list(set(lst))

#Find bath
lst = []
for i in range(data.shape[0]):
    lst.append(data.iloc[i, 6])
bath_count = CountFrequency(lst)
bath_list = list(set(lst))

#Find balcony
lst = []
for i in range(data.shape[0]):
    lst.append(data.iloc[i, 7])
balcony_count = CountFrequency(lst)
balcony_list = list(set(lst))

import re

lst = []
for i in range(data.shape[0]):
    if re.findall("[a-zA-Z]", data.iloc[i, 5]):
        lst.append(data.iloc[i, 5])
x = list(set(lst))
#
#sqft = acres * 43560
#sqft = cents * 435.6
#sqft = perch * 272.25
#sqft = sqmt * 10.7639
#sqft = sqyd * 9
#sqft = ground * 2400
#sqft = guntha * 1089 
#Acres
#Cents
#Perch
#Sq. Meter
#Sq. Yards
#Grounds
#Guntha
