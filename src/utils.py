import os
import sys
from sklearn.metrics import r2_score
import dill

from src.exception import CustomException
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(x_train, y_train,x_test, y_test, models, params):
    try:
        report = {}
        best_params = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            grid = GridSearchCV(estimator=model,  param_grid=param, scoring='r2',n_jobs=-1, cv=5)
            
            grid.fit(x_train, y_train)
            
            y_test_pred = grid.predict(x_test)
            score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = score
            best_params[list(models.keys())[i]] = grid.best_params_
        
        return report, best_params
    
    except Exception as e:
        raise CustomException(e, sys)