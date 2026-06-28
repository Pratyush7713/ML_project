import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
import os
import sys
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils import save_object

@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation_obj(self):
        try:
            
            cat_column = ["gender","race/ethnicity","parental level of education","lunch","test preparation course"]
            num_column = ["reading score","writing score"]
            
            logging.info(f"Categorical features: {cat_column}")
            logging.info(f"Numerical features: {num_column}")
            
            num_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy='median')),
                    ("scale", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )
            
            logging.info("Created both num and Cat pipeline")
            
            preprocessing = ColumnTransformer(
                [
                    ("cat_feature", cat_pipeline, cat_column),
                    ("num_feature", num_pipeline, num_column)
                ],
                remainder="passthrough",
            )

            logging.info("created the preprocessor")
            return preprocessing
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            preprocessing = self.get_data_transformation_obj()
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Got the train and Test dataframes")
            
            x_train = train_df.drop(columns='math score')
            y_train = train_df['math score']
            x_test = test_df.drop(columns='math score')
            y_test = test_df['math score']
            logging.info("Split the dependent and independent features")
            
            x_train = preprocessing.fit_transform(x_train)
            x_test = preprocessing.transform(x_test)

            train_arr = np.c_[x_train, np.array(y_train)]
            test_arr = np.c_[x_test, np.array(y_test)]
            logging.info("done preprocessing and merging dependent and independent feature")

            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessing
            )
            logging.info("Saved the preprocessing as a pickel file")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)