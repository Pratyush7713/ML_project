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
            
            model_report: dict = evaluate_model(x_test=x_test, x_train= x_train, y_test=y_test, y_train=y_train, models=models)
            
            ## give the best score
            best_model_score = max(sorted(model_report.values())) 
            
            ## Model with best score
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("All the models you tried has r2_score of less then 0.6 so no best model found")
            logging.info("Got the best model")
            best_model.fit(x_train, y_train)
            
            save_object(obj=best_model,file_path=self.model_trainer_config.model_obj_file_path)
            
            y_pred_test = best_model.predict(x_test)
            score = r2_score(y_test, y_pred_test)
            return score
        
        except Exception as e:
            raise CustomException(e,sys)
