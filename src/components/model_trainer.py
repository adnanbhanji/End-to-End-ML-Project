import os
import sys
from dataclasses import dataclass

# from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomException
from logger import logging

from utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("artifacts", "trained_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Initiating Model Trainer")
            logging.info("Splitting train and test data - arr")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            models = {
                "LinearRegression": LinearRegression(),
                "RandomForestRegressor": RandomForestRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "XGBRegressor": XGBRegressor(),
            }

            model_report: dict = evaluate_models(
                models, X_train, y_train, X_test, y_test
            )

            best_model_name = max(
                model_report, key=lambda x: model_report[x]["test_score"]
            )
            best_model_score = model_report[best_model_name]["test_score"]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException(
                    "Best model score is less than 0.7, hence not saving the model"
                )
            
            logging.info(f"Best Model: {best_model_name}")

            #preprocessing_obj = # we can get the preprocessing object from the preprocessor_path however for now we don't need it
            save_object(
                file_path=self.config.trained_model_path, obj=best_model,
            )

            predicted_values = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted_values)
            return r2

        except Exception as e:
            raise CustomException("Error occurred in initiate_model_trainer", e)
