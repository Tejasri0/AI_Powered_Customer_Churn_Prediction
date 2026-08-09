import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy.stats import yeojohnson
from log_code import setup_logging
logger = setup_logging("var_outlier")

def transformation_outlier(X_train, X_test):
    try:
        logger.info("===== Train Dataset =====")
        logger.info(f"{X_train.shape}\n{X_train.columns}")
        logger.info("===== Test Dataset =====")
        logger.info(f"{X_test.shape}\n{X_test.columns}")

        # Columns on which Yeo-Johnson should be applied
        cols = ["tenure", "MonthlyCharges","TotalCharges_constant"]

        # Distribution before transformation
        for col in cols:
            plt.figure(figsize=(5,3))
            plt.title(col)
            X_train[col].plot(kind="kde")
            plt.show()

        # Apply Yeo-Johnson only to selected columns
        for col in cols:
            # Train
            X_train[col + "_yeo"], lam = yeojohnson(X_train[col])
            # Test
            X_test[col + "_yeo"], _ = yeojohnson(X_test[col])
            # Remove original columns (Optional)
            X_train.drop(columns=[col], inplace=True)
            X_test.drop(columns=[col], inplace=True)

        # Distribution after transformation
        for col in cols:
            plt.figure(figsize=(5, 3))
            plt.title(col+"_yeo")
            X_train[col+"_yeo"].plot(kind="kde")
            plt.show()

        logger.info("===== Train Dataset After Transformation =====")
        logger.info(f"{X_train.shape}\n{X_train.columns}")
        logger.info("===== Test Dataset After Transformation =====")
        logger.info(f"{X_test.shape}\n{X_test.columns}")

        return X_train, X_test

    except Exception as e:
        er_ty, er_msg, er_lineno = sys.exc_info()
        logger.error(
            f"Error in line {er_lineno.tb_lineno}: {er_ty} : {er_msg}"
        )