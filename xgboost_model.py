#!/usr/bin/env python

import sys
from xgboost import XGBRegressor
import pandas as pd
from sklearn.model_selection import train_test_split,ShuffleSplit
from scipy.stats import pearsonr
from math import sqrt
from sklearn.metrics import mean_squared_error
import numpy as np
from glob import glob
from pandas_desc import PandasDescriptors


def rmse(y_actual, y_predicted):
    return sqrt(mean_squared_error(y_actual, y_predicted))

def train_ng_model(infile_name):
    res = []
    pandas_descriptors = PandasDescriptors(['morgan2', 'descriptors'])
    
    ng_df = pd.read_csv(infile_name)
    ng_df = ng_df.dropna()
    rd_df = ng_df[ng_df.columns[0:3]]
    
    rd_df = pandas_descriptors.from_dataframe(input_df=rd_df)
    desc_cols = [x for x in rd_df.columns[3:] if not x.startswith("B_")]
    desc_df = rd_df[['Name'] + desc_cols]
    ng_df = ng_df.merge(desc_df,on="Name")
    
    ng_X = ng_df.values[0::, 3::]
    rd_X = rd_df.values[0::, 3::]
    Y = ng_df.Act.values

    rms_list = []

    rows = len(Y)
    shuffle_split = ShuffleSplit(10,test_size=0.3)
    for train, test in shuffle_split.split(range(0,rows)):
        train = list(train)
        test = list(test)
        ng_x_train = ng_X[train]
        rd_x_train = rd_X[train]
        y_train = Y[train]
        
        ng_x_test = ng_X[test]
        rd_x_test = rd_X[test]
        y_test = Y[test]

        ng_xgb = XGBRegressor()
        ng_xgb.fit(ng_x_train, y_train)
        ng_pred = ng_xgb.predict(ng_x_test)
        ng_r2 = pearsonr(ng_pred, y_test)[0] ** 2
        ng_rmse = rmse(ng_pred, y_test)

        rd_xgb = XGBRegressor()
        rd_xgb.fit(rd_x_train, y_train)
        rd_pred = rd_xgb.predict(rd_x_test)
        rd_r2 = pearsonr(rd_pred, y_test)[0] ** 2
        rd_rmse = rmse(rd_pred, y_test)
        res.append([infile_name.replace(".csv","").replace("ngfp_data/",""),ng_r2,rd_r2, ng_rmse, rd_rmse])
    return res

output = []
for filename in glob("ngfp_data/*.csv"):
    print(filename)
    output += train_ng_model(filename)

output_df = pd.DataFrame(output,columns=["DATASET","NG_R2","RD_R2","NG_RMSE","RD_RMSE"])
output_df.to_csv("results.csv",index=False)


