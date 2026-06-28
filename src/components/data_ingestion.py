import pandas as pd
import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation

'''
@dataclass is a decorator in Python that automatically generates common methods for classes that are mainly used to store data.
Without @dataclass, you often write a lot of boilerplate code.

Without @dataclass
class Student:
    def __init__(self, name, age, marks):
        self.name = name
        self.age = age
        self.marks = marks

s1 = Student("Pratyush", 19, 95)
print(s1.name)

Here, you manually wrote the __init__() method.

With @dataclass
from dataclasses import dataclass

@dataclass
class Student:
    name: str
    age: int
    marks: float

s1 = Student("Pratyush", 19, 95)
print(s1.name)

Python automatically creates:

def __init__(self, name, age, marks):
    self.name = name
    self.age = age
    self.marks = marks

for you.
'''

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class dataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initial_data_ingestion(self):
        try:
            df = pd.read_csv('notebooks/data/StudentsPerformance.csv')
            logging.info('Read the dataset and put into Dataframe formate')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)  ## "Create the folder in which train.csv will be stored, and don't complain if the folder already exists."
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved")
            
            train_set, test_set = train_test_split(df, test_size= 0.2, random_state= 42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logging.info("Train data saved")
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Test data saved")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
    
if __name__ == "__main__":
    obj = dataIngestion()
    train_path, test_path = obj.initial_data_ingestion()
    data_tranformation = DataTransformation()
    data_tranformation.initiate_data_transformation(train_path=train_path, test_path=test_path)