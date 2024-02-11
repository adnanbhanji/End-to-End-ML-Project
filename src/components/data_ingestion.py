import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomException
from logger import logging
import pandas as pd
from data_transformation import DataTransformation
from data_transformation import DataTransformationConfig

from sklearn.model_selection import train_test_split
from dataclasses import dataclass 
from model_trainer import ModelTrainer
from model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            logging.info("Initiating data ingestion")
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Read dataset as dataframe")
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")
            train, test = train_test_split(df, test_size=0.2)
            train.to_csv(self.config.train_data_path, index=False, header=True)
            test.to_csv(self.config.test_data_path, index=False, header=True)
            logging.info("Data ingestion completed")
            return (self.config.train_data_path, self.config.test_data_path)
        
        except Exception as e:
            logging.error(f"Error in data ingestion: {str(e)}")
            raise CustomException(f"Error in data ingestion: {str(e)}")

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))