import os
import sys

import numpy as np
import pandas as pd
import dill

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from logger import logging
from exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(obj, file_path):
    """
    This function is used to save the object to the file path.
    :param obj: object to be saved
    :param file_path: file path where the object will be saved
    :return: None
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException("Error occurred in save_object", e)

def evaluate_models(models, hyperparameters, X_train, y_train, X_test, y_test):
    """
    This function is used to evaluate the models using the training and testing data.
    :param models: models to be evaluated
    :param X_train: training data
    :param y_train: training target
    :param X_test: testing data
    :param y_test: testing target
    :return: model_report: report of the models
    """
    try:
        logging.info("Evaluating the models")
        model_report = {}
        for model_name, model in models.items():
            param = hyperparameters[model_name]
            gs = GridSearchCV(model, param, cv=5, n_jobs=-1)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train) # Train the model
            y_train_pred = model.predict(X_train) # Predict the training data
            y_test_pred = model.predict(X_test) # Predict the testing data
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            model_report[model_name] = {
                "train_score": train_model_score,
                "test_score": test_model_score
            }

        return model_report
    
    except Exception as e:
        raise CustomException("Error occurred in evaluate_models", e)
    


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)