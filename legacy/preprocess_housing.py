#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:56:48 2023

Preprocess Housing

@author: max
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("housing/data/housing.csv")
print(df.head())
print(df.info())

def get_numerical_summary(df):
    total = df.shape[0]
    missing_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
    missing_percent = {}
    for col in missing_cols:
        null_count = df[col].isnull().sum()
        per = (null_count/total)*100
        missing_percent[col] = per
        print("{} : {} ({})".format(col, null_count, round(per, 3)))
    return missing_percent

get_numerical_summary(df)

df.dropna(inplace=True)
print(df.head())
print(df.info())

cols_to_encode = ['Suburb', 'Address', 'Type', 'Method', 'SellerG', 'CouncilArea', 'Regionname']
le = LabelEncoder()
for col in cols_to_encode:
    df[col] = le.fit_transform(df[col])
    
cols_not_price = [col for col in df.columns if col != "Price"]
df = df[cols_not_price + ["Price"]]
    
print(df.head())
print(df.info())
df.to_csv("housing/data/housing.csv", index=False)
