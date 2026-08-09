'''
In this file we are going to developing end to end ML pipeline so this
file is the main source to call other functions
'''

import os
import sys
import logging
import sklearn
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from log_code import setup_logging
logger = setup_logging("main")
from sklearn.model_selection import train_test_split
from constant_technique import missing_value_handle
from var_outlier import transformation_outlier
from feature_selection import column_selection
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from imblearn.over_sampling import SMOTE
from all_models import feature_scaling

class CUSTOMER_RETENTION_APPLICATION:
    def __init__(self , path):  #path = 'creditcard.csv
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            # Shape of the dataset
            logger.info(f"Total dataset Size was : {self.df.shape}")   # (7043, 22)
            # Checking null values in the dataset
            logger.info(f"Checking Null values : \n {self.df.isnull().sum()}")
            # Column value datatypes in dataset
            logger.info(f"Column Datatypes : \n  {self.df.dtypes}")
            # Column names in dataset
            logger.info(f"Dataset Column Names : \n : {self.df.columns}")
            # We don't need customerId column so i am removing the column
            self.df.drop(["customerID"], axis=1, inplace=True)
            # We are having blank space in the TotalCharges column in dataset
            if self.df['TotalCharges'].dtype == 'object':
                logger.info(
                    f"Blank Spaces : {(self.df['TotalCharges'].str.strip() == '').sum()}"
                )
            # Changing the blank spaces into null values in the TotalCharges column
            self.df['TotalCharges'] = self.df['TotalCharges'].replace(' ', np.nan)
            # Changing TotalCharges column in str to numeric
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'])
            # After changing the datatype checking for null values
            logger.info(f"After Changing the datatype Checking Null values : \n {self.df.isnull().sum()}")
            # Checking the unique values in Churn Column
            logger.info(f"{self.df['Churn'].unique()}")
            # Changing Yes into 1 and No into 0
            self.df['Churn'] = self.df['Churn'].map({"Yes": 1, "No": 0}).astype(int)
            # Dividing the Independent and Dependent columns
            self.X = self.df.iloc[:, :-1]  # indepedent (20)
            self.y = self.df.iloc[:, -1]  # dependent (1)
            # Shape of Independent Columns
            logger.info(f"Independent columns : {len(self.X.columns)} :\n : {self.X.columns}")
            # Spliting the Independent and dependent datasets into train and test
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                    random_state=42)
            # Shape of Training and testing datasets
            logger.info(f"Train Dataset Size : \n : {self.X_train.shape} : \n : {self.y_train.shape}")
            logger.info(f"Test Dataset Size : \n : {self.X_test.shape} : \n : {self.y_test.shape}")

        except Exception as e:
            er_ty, er_msg, er_lineno = sys.exc_info()
            logger.info(f"Error in line no : {er_lineno.tb_lineno} : due to : {er_ty} : reason : {er_msg}")

    # We have 11 null values in TotalCharges column so we want to handle
    def handling_missing_values(self):
        try:
            logger.info(f"Before Function Train_data : \n : {self.X_train.isnull().sum()}")
            logger.info(f"Before Function Test_data : \n : {self.X_test.isnull().sum()}")
            # Calling
            self.X_train,self.X_test = missing_value_handle(self.X_train,self.X_test)
            logger.info(f"After Function Train_data : \n : {self.X_train.isnull().sum()}")
            logger.info(f"After Function Test_data : \n : {self.X_test.isnull().sum()}")
        except Exception as e:
            er_ty , er_msg,er_lineno = sys.exc_info()
            logger.info(f"Error in line no : {er_lineno.tb_lineno} : due to : {er_ty} : reason : {er_msg}")


    # Split the data into 2 types numerical and categorical
    def data_seperation(self):
        try:
            self.X_train_num = self.X_train.select_dtypes(exclude='str')
            self.X_train_cat = self.X_train.select_dtypes(include = 'str')
            self.X_test_num = self.X_test.select_dtypes(exclude = 'str')
            self.X_test_cat = self.X_test.select_dtypes(include='str')
            logger.info("=========Train data details===========")
            logger.info(f"Train Numberical columns : {len(self.X_train_num.columns)} : \n : {self.X_train_num.columns}")
            logger.info(f"Train Cateogorical columns : {len(self.X_train_cat.columns)} : \n : {self.X_train_cat.columns}")
            logger.info("=========Test data details===========")
            logger.info(f"Test Numberical columns : {len(self.X_test_num.columns)} : \n : {self.X_test_num.columns}")
            logger.info(
                f"Test Cateogorical columns : {len(self.X_test_cat.columns)} : \n : {self.X_test_cat.columns}")
        except Exception as e:
            er_ty, er_msg, er_lineno = sys.exc_info()
            logger.info(f"Error in line no : {er_lineno.tb_lineno} : due to : {er_ty} : reason : {er_msg}")


    # Changing numerical columns into normal distribution
    def vt_out(self):
        try:
            logger.info(f"=====Train Dataset======")
            logger.info(f"{self.X_train_num.shape} : \n : {self.X_train_num.columns}")
            logger.info(f"=====Test Dataset======")
            logger.info(f"{self.X_test_num.shape} : \n : {self.X_test_num.columns}")
            self.X_train_num,self.X_test_num = transformation_outlier(self.X_train_num , self.X_test_num)
            logger.info(f"=====Train Dataset VT Outlier ======")
            logger.info(f"{self.X_train_num.shape} : \n : {self.X_train_num.columns} : \n : {self.X_train_num.isnull().sum()}")
            logger.info(f"=====Test Dataset VT Outlier ======")
            logger.info(f"{self.X_test_num.shape} : \n : {self.X_test_num.columns} : \n : {self.X_test_num.isnull().sum()}")
        except Exception as e:
            er_ty, er_msg, er_lineno = sys.exc_info()
            logger.info(f"Error in line no : {er_lineno.tb_lineno} : due to : {er_ty} : reason : {er_msg}")

    # remove unwanted columns
    def fs(self):
        try:
            logger.info(f"Before Going to Feature selection : \n : {self.X_train_num.columns} :\n {self.X_train_num.shape}")
            logger.info(
                f"Before Going to Feature selection : \n : {self.X_test_num.columns} :\n {self.X_test_num.shape}")
            self.X_train_num,self.X_test_num = column_selection(self.X_train_num,self.X_test_num,self.y_train,self.y_test)
            logger.info(
                f"After Going to Feature selection : \n : {self.X_train_num.columns} :\n {self.X_train_num.shape}")
            logger.info(
                f"After Going to Feature selection : \n : {self.X_test_num.columns} :\n {self.X_test_num.shape}")
        except Exception as e:
            er_ty, er_msg, er_lineno = sys.exc_info()
            logger.info(f"Error in line no : {er_lineno.tb_lineno} : due to : {er_ty} : reason : {er_msg}")

    def cat_to_numerical(self):
        try:
            logger.info(f"Before OneHot : {self.X_train_cat.columns} : {self.X_train_cat.isnull().sum()}")
            logger.info(f"Before OneHot : {self.X_test_cat.columns} : {self.X_test_cat.isnull().sum()}")
            # OneHotEncoding on Gender and Region column
            one_hot = OneHotEncoder(drop='first')
            one_hot.fit(self.X_train_cat[ [
                'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'PaperlessBilling',
                'PaymentMethod', 'sim_type'
            ]])
            result_1 = one_hot.transform(self.X_train_cat[ [
                'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'PaperlessBilling',
                'PaymentMethod', 'sim_type'
            ]]).toarray()
            result_2 = one_hot.transform(self.X_test_cat[ [
                'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'PaperlessBilling',
                'PaymentMethod', 'sim_type'
            ]]).toarray()
            t1 = pd.DataFrame(data=result_1, columns=one_hot.get_feature_names_out())
            t2 = pd.DataFrame(data=result_2, columns=one_hot.get_feature_names_out())
            self.X_train_cat.reset_index(drop=True, inplace=True)
            self.X_test_cat.reset_index(drop=True, inplace=True)
            t1.reset_index(drop=True, inplace=True)
            t2.reset_index(drop=True, inplace=True)
            self.X_train_cat = pd.concat([self.X_train_cat, t1], axis=1)
            self.X_test_cat = pd.concat([self.X_test_cat, t2], axis=1)
            self.X_train_cat = self.X_train_cat.drop( [
                'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'PaperlessBilling',
                'PaymentMethod', 'sim_type'
            ], axis=1)
            self.X_test_cat = self.X_test_cat.drop( [
                'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'PaperlessBilling',
                'PaymentMethod', 'sim_type'
            ], axis=1)
            logger.info(f"After OneHot : {self.X_train_cat.columns} : {self.X_train_cat.isnull().sum()}")
            logger.info(f"After OneHot : {self.X_test_cat.columns} : {self.X_test_cat.isnull().sum()}")

            # Ordinal Encoding
            ord_enc = OrdinalEncoder()
            ord_enc.fit(self.X_train_cat[['Contract']])
            r1 = ord_enc.transform(self.X_train_cat[['Contract']])
            r2 = ord_enc.transform(self.X_test_cat[['Contract']])

            p1 = pd.DataFrame(data=r1, columns=ord_enc.get_feature_names_out() + "_res")
            p2 = pd.DataFrame(data=r2, columns=ord_enc.get_feature_names_out() + "_res")

            self.X_train_cat.reset_index(drop=True, inplace=True)
            self.X_test_cat.reset_index(drop=True, inplace=True)

            p1.reset_index(drop=True, inplace=True)
            p2.reset_index(drop=True, inplace=True)

            self.X_train_cat = pd.concat([self.X_train_cat, p1], axis=1)
            self.X_test_cat = pd.concat([self.X_test_cat, p2], axis=1)

            self.X_train_cat = self.X_train_cat.drop(['Contract'], axis=1)
            self.X_test_cat = self.X_test_cat.drop(['Contract'], axis=1)

            logger.info(f"After Odinal : {self.X_train_cat.columns} : {self.X_train_cat.isnull().sum()}")
            logger.info(f"After Odinal : {self.X_test_cat.columns} : {self.X_test_cat.isnull().sum()}")

            self.X_train_num.reset_index(drop=True, inplace=True)
            self.X_train_cat.reset_index(drop=True, inplace=True)

            self.X_test_num.reset_index(drop=True, inplace=True)
            self.X_test_cat.reset_index(drop=True, inplace=True)

            self.final_training_data = pd.concat([self.X_train_num, self.X_train_cat], axis=1)
            self.final_testing_data = pd.concat([self.X_test_num, self.X_test_cat], axis=1)

            logger.info(
                f'final training dataset : {self.final_training_data.shape} :\n {self.final_training_data.columns} : \n {self.final_training_data.isnull().sum()}')
            logger.info(
                f'final testing dataset : {self.final_testing_data.shape} :\n {self.final_testing_data.columns} : \n {self.final_testing_data.isnull().sum()}')

        except Exception as e:
            er_ty, er_msg, er_lineno = sys.exc_info()
            logger.info(f"Error in line no : {er_lineno.tb_lineno} : due to : {er_ty} : reason : {er_msg}")


    def data_balancing(self):
        try:
            logger.info("=================Checking Data Balancing=============")
            logger.info(f"Number of Rows for : {1} : Yes : {sum(self.y_train == 1)}")
            logger.info(f"Number of Rows for : {0} : No : {sum(self.y_train == 0)}")

            sm_obj = SMOTE(random_state=42)
            self.final_training_data_balanced , self.y_train_balanced = sm_obj.fit_resample(self.final_training_data , self.y_train)

            logger.info(f"Number of Rows for : {1} : Yes : {sum(self.y_train_balanced == 1)}")
            logger.info(f"Number of Rows for : {0} : No : {sum(self.y_train_balanced == 0)}")

            logger.info(f"After Balancing : \n : {self.final_training_data_balanced.shape} : \n : {self.y_train_balanced.shape}")
            feature_scaling(self.final_training_data_balanced,self.y_train_balanced,self.final_testing_data,self.y_test)
        except Exception as e:
            er_ty, er_msg, er_lineno = sys.exc_info()
            logger.info(f"Error in line no : {er_lineno.tb_lineno} : due to : {er_ty} : reason : {er_msg}")



if __name__ == "__main__":
    obj = CUSTOMER_RETENTION_APPLICATION("Customer_Information.csv")
    obj.handling_missing_values()
    obj.data_seperation()
    obj.vt_out()
    obj.fs()
    obj.cat_to_numerical()
    obj.data_balancing()


