import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold
from log_code import setup_logging
logger = setup_logging("feature_selection_file")


def column_selection(X_train,X_test,y_train,y_test):
    try:
        logger.info(f"Before Constant Technique : {X_train.shape} : \n : {X_train.columns}")
        logger.info(f"Before Constant Technique : {X_test.shape} : \n : {X_test.columns}")

        # constant Technique
        constant_reg = VarianceThreshold(threshold=0.0)
        constant_reg.fit(X_train)
        logger.info(f"From Constant Technique we can remove below columns : \n : {X_train.columns[~constant_reg.get_support()]}")
        logger.info(f"After Constant Technique : {X_train.shape} : \n : {X_train.columns}")
        logger.info(f"After Constant Technique : {X_test.shape} : \n : {X_test.columns}")

        # quasi Constant
        quasi_reg = VarianceThreshold(threshold=0.1)
        quasi_reg.fit(X_train)  # 7
        logger.info(
            f"From Quasi Constant Technique we can remove below columns : \n : {X_train.columns[~quasi_reg.get_support()]}")
        logger.info(f"After Quasi Constant Technique : {X_train.shape} : \n : {X_train.columns}")
        logger.info(f"After Quasi Constant Technique : {X_test.shape} : \n : {X_test.columns}")

        # Hypothesis Testing  -> 5
        values = []
        for i in X_train.columns:
            values.append(pearsonr(X_train[i] , y_train))
        values_array = np.array(values)
        p_value = pd.Series(values_array[: , 1] , index = X_train.columns)
        plt.figure(figsize = (5,3))
        p_value.plot.bar()
        plt.show()
        logger.info(f"After Hypothesis Technique : {X_train.shape} : \n : {X_train.columns}")
        logger.info(f"After Hypothesis Technique : {X_test.shape} : \n : {X_test.columns}")

        return X_train , X_test
    except Exception as e:
        er_ty, er_msg, er_lineno = sys.exc_info()
        logger.info(f"Error in line no : {er_lineno.tb_lineno} : due to : {er_ty} : reason : {er_msg}")
