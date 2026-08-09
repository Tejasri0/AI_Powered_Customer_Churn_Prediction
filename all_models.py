import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from log_code import setup_logging
logger = setup_logging("all_models")
from sklearn.preprocessing import StandardScaler  # z_score = (x - mean) / std
import pickle
from sklearn.model_selection import GridSearchCV
from scipy.stats.distributions import nbinom
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import roc_curve, roc_auc_score

def knn(X_train,y_train,X_test,y_test):
  global knn_reg
  knn_reg = KNeighborsClassifier(n_neighbors=5)
  knn_reg.fit(X_train,y_train)
  global knn_predictions
  knn_predictions = knn_reg.predict(X_test)
  logger.info(f"KNN Test Accuracy \n : {accuracy_score(y_test , knn_predictions)}")
  logger.info(f"KNN Confusion Matrix \n : {confusion_matrix(y_test,knn_predictions)}")
  logger.info(f"KNN Classification report \n : {classification_report(y_test,knn_predictions)}")

def nb(X_train,y_train,X_test,y_test):
  global nb_reg
  nb_reg = GaussianNB()
  nb_reg.fit(X_train,y_train)
  global nb_predictions
  nb_predictions = nb_reg.predict(X_test)
  logger.info(f"Naive Bayes Test Accuracy \n : {accuracy_score(y_test , nb_predictions)}")
  logger.info(f"Naive Bayes Confusion Matrix \n : {confusion_matrix(y_test,nb_predictions)}")
  logger.info(f"Naive Bayes Classification report \n : {classification_report(y_test,nb_predictions)}")


def lr(X_train,y_train,X_test,y_test):
  global lr_reg
  lr_reg = LogisticRegression(
      max_iter=1000,
      random_state=42
  )
  lr_reg.fit(X_train,y_train)
  global lr_predictions
  lr_predictions = lr_reg.predict(X_test)
  logger.info(f"Logistic Regression Test Accuracy \n : {accuracy_score(y_test , lr_predictions)}")
  logger.info(f"Logistic Regression Confusion Matrix \n : {confusion_matrix(y_test,lr_predictions)}")
  logger.info(f"Logistic Regression Classification report \n : {classification_report(y_test,lr_predictions)}")


def dt(X_train,y_train,X_test,y_test):
  global dt_reg
  dt_reg = DecisionTreeClassifier(
      criterion="entropy",
      random_state=42
  )
  dt_reg.fit(X_train,y_train)
  global dt_predictions
  dt_predictions = dt_reg.predict(X_test)
  logger.info(f"Decision Tree Test Accuracy \n : {accuracy_score(y_test , dt_predictions)}")
  logger.info(f"Decision Tree Confusion Matrix \n : {confusion_matrix(y_test,dt_predictions)}")
  logger.info(f"Decision Tree Classification report \n : {classification_report(y_test,dt_predictions)}")



def rf(X_train,y_train,X_test,y_test):
  global rf_reg
  rf_reg = RandomForestClassifier(
      criterion="entropy",
      n_estimators=100,
      random_state=42
  )
  rf_reg.fit(X_train,y_train)
  global rf_predictions
  rf_predictions = rf_reg.predict(X_test)
  logger.info(f"Random Forest Test Accuracy \n : {accuracy_score(y_test , rf_predictions)}")
  logger.info(f"Random Forest Confusion Matrix \n : {confusion_matrix(y_test,rf_predictions)}")
  logger.info(f"Random Forest Classification report \n : {classification_report(y_test,rf_predictions)}")


def ada(X_train,y_train,X_test,y_test):
  t = LogisticRegression()
  global ada_reg
  ada_reg = AdaBoostClassifier(
      estimator=t,
      n_estimators=100,
      random_state=42
  )
  ada_reg.fit(X_train,y_train)
  global ada_predictions
  ada_predictions = ada_reg.predict(X_test)
  logger.info(f"Adaboost Test Accuracy \n : {accuracy_score(y_test , ada_predictions)}")
  logger.info(f"Adaboost Confusion Matrix \n : {confusion_matrix(y_test,ada_predictions)}")
  logger.info(f"Adaboost Classification report \n : {classification_report(y_test,ada_predictions)}")

def GB(X_train,y_train,X_test,y_test):
  global gb_reg
  gb_reg = GradientBoostingClassifier(
      n_estimators=100,
      random_state=42
  )
  gb_reg.fit(X_train,y_train)
  global gb_predictions
  gb_predictions = gb_reg.predict(X_test)
  logger.info(f"Gradient Boosting Test Accuracy \n : {accuracy_score(y_test , gb_predictions)}")
  logger.info(f"Gradient Boosting Confusion Matrix \n : {confusion_matrix(y_test,gb_predictions)}")
  logger.info(f"Gradient Boosting Classification report \n : {classification_report(y_test,gb_predictions)}")

def xgb_(X_train,y_train,X_test,y_test):
  global xgb_reg
  xgb_reg = XGBClassifier(
      n_estimators=100,
      random_state=42,
      eval_metric="logloss"
  )
  xgb_reg.fit(X_train,y_train)
  global xgb_predictions
  xgb_predictions = xgb_reg.predict(X_test)
  logger.info(f"Xtreme Gradient Boosting Test Accuracy \n : {accuracy_score(y_test , xgb_predictions)}")
  logger.info(f"Xtreme Gradient Boosting Confusion Matrix \n : {confusion_matrix(y_test,xgb_predictions)}")
  logger.info(f"Xtreme Gradient Boosting Classification report \n : {classification_report(y_test,xgb_predictions)}")

def svm(X_train,y_train,X_test,y_test):
    global svm_reg
    svm_reg = SVC(kernel="rbf", probability=True, random_state=42)
    svm_reg.fit(X_train,y_train)

    global svm_predictions
    svm_predictions = svm_reg.predict(X_test)

    logger.info(f"SVM Accuracy \n : {accuracy_score(y_test,svm_predictions)}")
    logger.info(f"SVM Confusion Matrix \n : {confusion_matrix(y_test,svm_predictions)}")
    logger.info(f"SVM Classification Report \n : {classification_report(y_test,svm_predictions)}")

def common(X_train,y_train,X_test,y_test):
  logger.info("============KNN==========")
  knn(X_train,y_train,X_test,y_test)
  logger.info("============Naive Bayes==========")
  nb(X_train,y_train,X_test,y_test)
  logger.info("============Logistic Regression==========")
  lr(X_train,y_train,X_test,y_test)
  logger.info("============Decision Tree==========")
  dt(X_train,y_train,X_test,y_test)
  logger.info("============Random Forest==========")
  rf(X_train,y_train,X_test,y_test)
  logger.info("============Adaboost==========")
  ada(X_train,y_train,X_test,y_test)
  logger.info("============Gradient Boosting==========")
  GB(X_train,y_train,X_test,y_test)
  logger.info("============Xtreme Gradient Boosting==========")
  xgb_(X_train,y_train,X_test,y_test)
  logger.info("============Support Vector Machine==========")
  svm(X_train, y_train, X_test, y_test)



def feature_scaling(X_train,y_train,X_test,y_test):
    try:
        logger.info(f"Before scaling : \n : {X_train.head(3)}")
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_scaled = sc.transform(X_train)
        X_test_scaled = sc.transform(X_test)
        logger.info(f"After scaling : {X_train_scaled}")
        with open("Standard_scaler.pkl","wb") as f:
            pickle.dump(sc,f)

        logistic_regression_model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        logistic_regression_model.fit(X_train_scaled, y_train)
        logger.info(f"Model Test Accuracy : {accuracy_score(y_test, logistic_regression_model.predict(X_test_scaled))}")
        logger.info(f"Confusion Matrix : {confusion_matrix(y_test, logistic_regression_model.predict(X_test_scaled))}")
        logger.info(
            f"Classification Report : {classification_report(y_test, logistic_regression_model.predict(X_test_scaled))}")

        # Already Logistic Regression is working fine wth 76 % of test accuracy
        # But using Tuning Techniques we can Improve the Model Performance
        # we are going to Use Gridsearch CV

        common(X_train_scaled,y_train,X_test_scaled,y_test)

        knn_fpr, knn_tpr, knn_threshold = roc_curve(y_test, knn_predictions)
        nb_fpr, nb_tpr, nb_threshold = roc_curve(y_test, nb_predictions)
        lr_fpr, lr_tpr, lr_threshold = roc_curve(y_test, lr_predictions)
        dt_fpr, dt_tpr, dt_threshold = roc_curve(y_test, dt_predictions)
        rf_fpr, rf_tpr, rf_threshold = roc_curve(y_test, rf_predictions)
        ada_fpr, ada_tpr, ada_threshold = roc_curve(y_test, ada_predictions)
        gb_fpr, gb_tpr, gb_threshold = roc_curve(y_test, gb_predictions)
        xgb_fpr, xgb_tpr, xgb_threshold = roc_curve(y_test, xgb_predictions)
        svm_fpr, svm_tpr, svm_threshold = roc_curve(y_test, svm_predictions)


        plt.figure(figsize=(5, 3))
        plt.title("Best Model selection")

        plt.plot(knn_fpr, knn_tpr, label='knn', color='r')
        plt.plot(nb_fpr, nb_tpr, label='NB', color='g')
        plt.plot(lr_fpr, lr_tpr, label='LR', color='black')
        plt.plot(dt_fpr, dt_tpr, label='DT', color='blue')
        plt.plot(rf_fpr, rf_tpr, label='RF', color='yellow')
        plt.plot(ada_fpr, ada_tpr, label='ADA')
        plt.plot(gb_fpr, gb_tpr, label='GB')
        plt.plot(xgb_fpr, xgb_tpr, label='XGB')
        plt.plot(svm_fpr, svm_tpr, label='SVM')


        plt.legend()
        plt.show()

        # logger.info(f"Knn_roc_auc_score : {roc_auc_score(y_test, knn_predictions)}")
        # logger.info(f"Nb_roc_auc_score : {roc_auc_score(y_test, nb_predictions)}")
        # logger.info(f"Lr_roc_auc_score : {roc_auc_score(y_test, lr_predictions)}")
        # logger.info(f"Dt_roc_auc_score : {roc_auc_score(y_test, dt_predictions)}")
        # logger.info(f"Rf_roc_auc_score : {roc_auc_score(y_test, rf_predictions)}")
        # logger.info(f"ada_roc_auc_score : {roc_auc_score(y_test, ada_predictions)}")
        # logger.info(f"gb_roc_auc_score : {roc_auc_score(y_test, gb_predictions)}")
        # logger.info(f"xgb_roc_auc_score : {roc_auc_score(y_test, xgb_predictions)}")
        # logger.info(f"svm_roc_auc_score : {roc_auc_score(y_test, svm_predictions)}")


       #Hyper Parameter Tuning
        parameter_of_logistic_regression = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'class_weight': [None, 'balanced'],
            'max_iter': [100, 500, 1000],
            'tol': [1e-4, 1e-3]
        }

        logistic_reg = LogisticRegression(random_state=42)
        logistic_obj = GridSearchCV(
            estimator=logistic_reg,
            param_grid=parameter_of_logistic_regression,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=3
        )
        result = logistic_obj.fit(X_train_scaled, y_train)
        logger.info(f"Best Parameters for Logistic Regression : \n{result.best_params_}")
        logger.info(f"Best Cross Validation Accuracy : {result.best_score_}")


        # Train Logistic Regression with Best Parameters
        logistic_regression_model = LogisticRegression(
            C=result.best_params_['C'],
            penalty=result.best_params_['penalty'],
            solver=result.best_params_['solver'],
            class_weight=result.best_params_['class_weight'],
            max_iter=result.best_params_['max_iter'],
            tol=result.best_params_['tol'],
            random_state=42
        )

        logistic_regression_model.fit(X_train_scaled, y_train)
        # Predictions
        y_pred = logistic_regression_model.predict(X_test_scaled)
        logger.info(f"Model Test Accuracy : {accuracy_score(y_test, y_pred)}")
        logger.info(f"Confusion Matrix : \n{confusion_matrix(y_test, y_pred)}")
        logger.info(f"Classification Report : \n{classification_report(y_test, y_pred)}")

        # Saving the model
        logger.info("=======Saving the Model===================")
        with open("Customer_Churn.pkl", "wb") as t:
            pickle.dump(logistic_regression_model, t)

    except Exception as e:
        er_ty, er_msg, er_lineno = sys.exc_info()
        logger.info(f"Error in line no : {er_lineno.tb_lineno} : due to : {er_ty} : reason : {er_msg}")
