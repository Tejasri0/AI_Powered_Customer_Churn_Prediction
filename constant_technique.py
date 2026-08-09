import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import logging
from log_code import setup_logging
logger = setup_logging("constant_technique")

def missing_value_handle(X_train,X_test):
    try:
        logger.info(f"Before Train_data : \n : {X_train.isnull().sum()}")
        logger.info(f"Before Test_data : \n : {X_test.isnull().sum()}")
        # Applying Constant Technique
        for i in X_train.columns:
            if X_train[i].isnull().sum() > 0:
                X_train[i+"_constant"]=np.where(X_train[i].isnull(),0,X_train[i])
                X_train = X_train.drop([i],axis=1)

        for i in X_test.columns:
            if X_test[i].isnull().sum() > 0:
                X_test[i+"_constant"] = np.where(X_test[i].isnull(), 0, X_test[i])
                X_test = X_test.drop([i],axis=1)

        logger.info(f"After Train_data : \n : {X_train.isnull().sum()}")
        logger.info(f"After Test_data : \n : {X_test.isnull().sum()}")

        return X_train,X_test
    
    except Exception as e:
        er_ty, er_msg, er_lineno = sys.exc_info()
        logger.info(f"Error in line no : {er_lineno.tb_lineno} : due to : {er_ty} : reason : {er_msg}")
