#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 00:39:09 2023

Impute Mean

@author: max
"""

import os
import numpy as np
import pandas as pd
from copy import deepcopy 
import pickle 
import cloudpickle 

SEED = 73

def delete_randomly_data(df, delete_percent, random_state=None):
    num_values_to_delete = int(df.size * delete_percent)
    np.random.seed(random_state)
    indices = np.random.choice(df.size, num_values_to_delete, replace=False)
    
    df_copy = df.copy(deep=True).values.flatten()
    if df_copy.dtype.kind == 'f':
        df_copy[indices] = np.nan
    else:
        df_copy = df_copy.astype(float)
        df_copy[indices] = np.nan
    
    return pd.DataFrame(df_copy.reshape(df.shape), columns=df.columns)

def impute_mean(data):
    imputed_data = deepcopy(data)
    
    for column in imputed_data.columns:
        if imputed_data[column].isnull().any():
            mean = imputed_data[column].mean()
            imputed_data[column].fillna(mean, inplace=True)
            
    return imputed_data

if __name__ == "__main__":
    
    dataset = None
    
    
    PLATZHALTER = 'iris'
    MISSING = 30
    if dataset == None:
        iris = np.load('iris/data/iris.npy')
        X = iris[:, :-1]
        y = iris[:, -1]
    
    X = pd.DataFrame(X)
    
    X_complete = deepcopy(X)
    
    filename=""
    
    
     
#---------------------------------------------------------------------------------------------------
    
    cols_to_delete = [0, 1, 3]
    X[cols_to_delete] = delete_randomly_data(X[cols_to_delete], MISSING*0.01, SEED)
    
#-------------------------------------------------------------------------------------------------------
# # #---------------------------------------------------------------------------------------------------------------
    
    #impute here
    X = impute_mean(X)
    
    if dataset == None:
        X[X.columns.size] = y
# -----------------------------------------------------------------------------------------------------
    #save results - kde and dataframe with imputed data
    
    if dataset == None:
        dataset = PLATZHALTER
    
    try:
        with open(f"{dataset}/data/{dataset}_data_{MISSING}.pkl", 'wb') as outp:
            pickle.dump(X, outp, pickle.HIGHEST_PROTOCOL)
    except:
        with open(f"{dataset}/data/{dataset}_data_{MISSING}.cp.pkl", 'wb') as outp:
            cloudpickle.dump(X, outp)
    