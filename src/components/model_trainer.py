import os
import sys
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    model_obj_file_path = os.path.join("artifacts", "model.pkl")

class ModelTainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def Find_Best_Model(self, train_data, test_data):
        try:
            x_train = train_data[:, :-1]
            y_train = train_data[:,-1]
            x_test = test_data[:,:-1]
            y_test = test_data[:,-1]
            logging.info("Splitted the data into x and y")
            
            models = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "RandomForestRegressor": RandomForestRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor()
            }
            
            params = {
                "LinearRegression": {
                    "fit_intercept": [True, False],
                    "positive": [True, False],
                },
                "Lasso": {
                    "alpha": [0.001, 0.01, 0.1, 1, 10],
                    "fit_intercept": [True, False],
                    "max_iter": [1000, 5000, 10000],
                    "selection": ["cyclic", "random"],
                },
                "Ridge": {
                    "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
                    "fit_intercept": [True, False],
                    "solver": ["auto", "svd", "cholesky", "lsqr", "sag"],
                },
                "RandomForestRegressor": {
                    "n_estimators": [100, 200, 500],
                    #"max_depth": [None, 5, 10, 20, 30],
                    #"min_samples_split": [2, 5, 10],
                    #min_samples_leaf": [1, 2, 4],
                    #"max_features": ["sqrt", "log2", None],
                    #"bootstrap": [True, False],
                },
                "KNeighborsRegressor": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    #"weights": ["uniform", "distance"],
                    #"algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                    #"leaf_size": [20, 30, 40],
                    #"p": [1, 2],
                },
                "DecisionTreeRegressor": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                    #"max_depth": [None, 5, 10, 20, 30],
                    #"min_samples_split": [2, 5, 10],
                    #"min_samples_leaf": [1, 2, 4],
                    #"max_features": ["sqrt", "log2", None],
                },
                "XGBRegressor": {
                    "n_estimators": [100, 200, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    #"max_depth": [3, 5, 7, 10],
                    #"subsample": [0.6, 0.8, 1.0],
                    #"colsample_bytree": [0.6, 0.8, 1.0],
                    #"gamma": [0, 0.1, 0.3],
                    #"reg_alpha": [0, 0.01, 0.1],
                    #"reg_lambda": [1, 2, 5],
                },
                "CatBoostRegressor": {
                    "iterations": [100, 300, 500],
                    "learning_rate": [0.01, 0.05, 0.1],
                    #"depth": [4, 6, 8, 10],
                    #"l2_leaf_reg": [1, 3, 5, 7],
                    #"loss_function": ["RMSE"],
                },
                "AdaBoostRegressor": {
                    "n_estimators": [50, 100, 200, 500],
                    #"learning_rate": [0.01, 0.05, 0.1, 1.0],
                    #"loss": ["linear", "square", "exponential"],
                },
                "GradientBoostingRegressor": {
                    "n_estimators": [100, 200, 500],
                    "learning_rate": [0.01, 0.05, 0.1],
                    #"max_depth": [3, 5, 7],
                    #"min_samples_split": [2, 5, 10],
                    #"min_samples_leaf": [1, 2, 4],
                    #"subsample": [0.8, 1.0],
                },
            }
            
            model_report, best_param = evaluate_model(x_test=x_test, x_train= x_train, y_test=y_test, y_train=y_train, models=models, params=params)
            
            ## give the best score
            best_model_score = max(sorted(model_report.values())) 
            
            ## Model with best score
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            best_model_param = best_param[best_model_name]

            if best_model_score<0.6:
                raise CustomException("All the models you tried has r2_score of less then 0.6 so no best model found")
            logging.info("Got the best model")
            best_model.set_params(**best_model_param)
            best_model.fit(x_train, y_train)
            
            save_object(obj=best_model,file_path=self.model_trainer_config.model_obj_file_path)
            
            y_pred_test = best_model.predict(x_test)
            score = r2_score(y_test, y_pred_test)
            return score
        
        except Exception as e:
            raise CustomException(e,sys)
